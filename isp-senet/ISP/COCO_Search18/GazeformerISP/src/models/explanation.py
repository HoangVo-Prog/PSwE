"""Differentiable hierarchical explanations for the COCO-Search18 predictor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.explanation_llm import SemanticEncoderWrapper, SharedExplanationLLM, cosine_alignment_loss


def final_spatial_fixation_distribution(action_logits: Tensor, already_probabilities=False) -> Tensor:
    if action_logits.ndim != 3 or action_logits.shape[-1] < 2:
        raise ValueError("action_logits must be [batch,length,1+spatial_cells]")
    spatial = action_logits[..., 1:]
    return spatial / spatial.sum(-1, keepdim=True).clamp_min(torch.finfo(spatial.dtype).eps) if already_probabilities else F.softmax(spatial, -1)


def compose_joint_supervised_loss(scan_loss, explanation_output=None):
    if explanation_output is None:
        return scan_loss
    return scan_loss + explanation_output["loss_explanation"]


def duration_parameter_adapter(duration_mu: Tensor, duration_param2: Tensor) -> Tensor:
    if duration_mu.shape != duration_param2.shape:
        raise ValueError("duration_mu and duration_param2 must have identical shapes")
    return torch.stack((duration_mu, duration_param2), -1)


def _batch_first(value, batch, length, name):
    if value.ndim != 3:
        raise ValueError(f"{name} must be rank 3")
    if value.shape[:2] == (batch, length):
        return value
    if value.shape[:2] == (length, batch):
        return value.permute(1, 0, 2)
    raise ValueError(f"{name} must be [N,L,D] or [L,N,D], got {tuple(value.shape)}")


class FixationReasoningEncoder(nn.Module):
    def __init__(self, decoder_dim, visual_dim, latent_dim, spatial_dim, duration_hidden_dim=None, position_hidden_dim=None):
        super().__init__()
        self.spatial_dim = tuple(int(v) for v in spatial_dim)
        height, width = self.spatial_dim
        rows = torch.arange(height, dtype=torch.float32) / max(height - 1, 1)
        cols = torch.arange(width, dtype=torch.float32) / max(width - 1, 1)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        self.register_buffer("grid_xy", torch.stack((xx.reshape(-1), yy.reshape(-1)), -1))
        position_hidden_dim = int(position_hidden_dim or max(32, latent_dim // 2))
        duration_hidden_dim = int(duration_hidden_dim or max(32, latent_dim // 2))
        self.decoder_projection = nn.Linear(decoder_dim, latent_dim)
        self.visual_projection = nn.Linear(visual_dim, latent_dim)
        self.position_mlp = nn.Sequential(nn.Linear(2, position_hidden_dim), nn.GELU(), nn.Linear(position_hidden_dim, latent_dim))
        self.duration_mlp = nn.Sequential(nn.Linear(2, duration_hidden_dim), nn.GELU(), nn.Linear(duration_hidden_dim, latent_dim))
        self.fixation_mlp = nn.Sequential(nn.Linear(4 * latent_dim, 2 * latent_dim), nn.GELU(), nn.Linear(2 * latent_dim, latent_dim))
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, decoder_states, visual_memory, action_logits, duration_mu, duration_param2):
        batch, length, actions = action_logits.shape
        cells = self.spatial_dim[0] * self.spatial_dim[1]
        if actions != cells + 1:
            raise ValueError(f"expected {cells + 1} final actions, got {actions}")
        decoder_states = _batch_first(decoder_states, batch, length, "decoder_states")
        if visual_memory.shape[:2] == (cells, batch):
            visual_memory = visual_memory.permute(1, 0, 2)
        if visual_memory.shape[:2] != (batch, cells):
            raise ValueError("visual_memory is not aligned with the action grid")
        duration_mu, duration_param2 = duration_mu.reshape(batch, length), duration_param2.reshape(batch, length)
        fixation_prob = final_spatial_fixation_distribution(action_logits)
        soft_visual = torch.einsum("nlg,ngd->nld", fixation_prob, visual_memory)
        expected_xy = torch.einsum("nlg,gc->nlc", fixation_prob, self.grid_xy.to(fixation_prob))
        duration_feature = self.duration_mlp(duration_parameter_adapter(duration_mu, duration_param2))
        combined = torch.cat((self.decoder_projection(decoder_states), self.visual_projection(soft_visual), self.position_mlp(expected_xy), duration_feature), -1)
        return {"fixation_prob": fixation_prob, "soft_visual": soft_visual, "expected_xy": expected_xy, "duration_feature": duration_feature, "z": self.norm(self.fixation_mlp(combined))}


class WhyR0Router(nn.Module):
    def __init__(self, latent_dim, query_dim, kmax, max_length, temporal_dim=None, hidden_dim=None):
        super().__init__()
        if kmax <= 0: raise ValueError("router_kmax must be positive")
        temporal_dim, hidden_dim = int(temporal_dim or max(16, latent_dim // 4)), int(hidden_dim or max(64, latent_dim))
        self.kmax = int(kmax)
        self.max_length, self.temporal_dim = int(max_length), temporal_dim
        self.register_buffer("temporal_frequency", torch.exp(-torch.arange((temporal_dim + 1) // 2, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / temporal_dim)))
        self.query_projection = nn.Linear(query_dim, latent_dim)
        self.mlp = nn.Sequential(nn.Linear(2 * latent_dim + temporal_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, kmax))

    def forward(self, z, query):
        batch, length, _ = z.shape
        if length > self.max_length: raise ValueError("sequence length exceeds router max_length")
        positions = torch.arange(length, device=z.device).unsqueeze(0).expand(batch, -1)
        query_context = self.query_projection(query).unsqueeze(1).expand(-1, length, -1)
        normalized_time = positions.to(z.dtype) / float(max(length - 1, 1))
        temporal = z.new_zeros((batch, length, self.temporal_dim))
        temporal[..., 0::2] = torch.sin(normalized_time.unsqueeze(-1) * self.temporal_frequency[:temporal[..., 0::2].shape[-1]])
        temporal[..., 1::2] = torch.cos(normalized_time.unsqueeze(-1) * self.temporal_frequency[:temporal[..., 1::2].shape[-1]])
        logits = self.mlp(torch.cat((z, temporal, query_context), -1))
        return logits, logits.softmax(-1)

    @staticmethod
    def routing_loss(router_prob, why_membership, fixation_mask):
        if router_prob.shape != why_membership.shape: raise ValueError("why_membership must match router_prob")
        mask = fixation_mask.to(device=router_prob.device, dtype=router_prob.dtype)
        if mask.sum() == 0: return router_prob.sum() * 0
        target = why_membership.to(device=router_prob.device, dtype=router_prob.dtype)
        if torch.any(target[mask.bool()].sum(-1).sub(1).abs() > 1e-4) or torch.any(target[~mask.bool()].abs() > 1e-6):
            raise ValueError("real rows need one gold episode and padded rows need zero membership")
        return (-(target * router_prob.clamp_min(torch.finfo(router_prob.dtype).tiny).log()).sum(-1) * mask).sum() / mask.sum()

    @staticmethod
    def soft_episode_aggregation(z, router_prob, fixation_mask):
        masked = router_prob * fixation_mask.to(z.dtype).unsqueeze(-1)
        mass = masked.sum(1)
        return torch.einsum("nlk,nld->nkd", masked, z) / (mass.unsqueeze(-1) + 1e-8), mass


class HowR0Aggregator(nn.Module):
    def __init__(self, latent_dim, query_dim, global_dim, hidden_dim=None):
        super().__init__()
        self.query_projection = nn.Linear(query_dim, global_dim)
        self.mlp = nn.Sequential(nn.Linear(latent_dim + global_dim, int(hidden_dim or max(64, global_dim))), nn.GELU(), nn.Linear(int(hidden_dim or max(64, global_dim)), global_dim))
        self.norm = nn.LayerNorm(global_dim)

    def forward(self, episode_tokens, query, episode_count):
        batch, kmax, _ = episode_tokens.shape
        counts = torch.as_tensor(episode_count, device=episode_tokens.device, dtype=torch.long).reshape(-1)
        if counts.numel() != batch or torch.any(counts < 0) or torch.any(counts > kmax): raise ValueError("episode_count must have one value in [0,Kmax]")
        active = torch.arange(kmax, device=episode_tokens.device).unsqueeze(0) < counts.unsqueeze(1)
        pooled = (episode_tokens * active.unsqueeze(-1).to(episode_tokens.dtype)).sum(1) / counts.clamp_min(1).to(episode_tokens.dtype).unsqueeze(-1)
        pooled = pooled * (counts > 0).to(episode_tokens.dtype).unsqueeze(-1)
        return {"global_token": self.norm(self.mlp(torch.cat((pooled, self.query_projection(query)), -1))), "active_episode_mask": active, "pooled_episode": pooled}


def _zero(value): return value.sum() * 0


def _text_rows(values, batch, length):
    if values is None: return [["" for _ in range(length)] for _ in range(batch)]
    rows = list(values)
    if len(rows) != batch: raise ValueError("text batch size mismatch")
    return [[str(row[index]) if index < len(row) else "" for index in range(length)] for row in rows]


def _slot_rows(values, batch, kmax):
    if values is None: return [["" for _ in range(kmax)] for _ in range(batch)]
    rows = list(values)
    if len(rows) != batch: raise ValueError("WHY text batch size mismatch")
    return [[str(row[index]) if index < len(row) else "" for index in range(kmax)] for row in rows]


class HierarchicalExplanationModule(nn.Module):
    def __init__(self, decoder_dim, visual_dim, latent_dim, spatial_dim, max_length, kmax, semantic_encoder=None, semantic_encoder_name=None, semantic_encoder_dim=None, freeze_semantic_encoder=True, causal_lm=None, causal_lm_name=None, causal_lm_hidden_dim=None, causal_lm_tokenizer=None, freeze_causal_lm=True, global_dim=None, lambda_exp_what=1, lambda_exp_why=1, lambda_exp_how=1, lambda_what_txt=1, lambda_what_align=1, lambda_route=1, lambda_why_txt=1, lambda_why_align=1, lambda_how_txt=1, lambda_how_align=1):
        super().__init__()
        self.kmax = int(kmax)
        self.semantic_encoder = SemanticEncoderWrapper(semantic_encoder, semantic_encoder_name, freeze_semantic_encoder, semantic_encoder_dim)
        global_dim = int(global_dim or latent_dim)
        self.fixation_encoder = FixationReasoningEncoder(decoder_dim, visual_dim, latent_dim, spatial_dim)
        self.why_router = WhyR0Router(latent_dim, self.semantic_encoder.output_dim, kmax, max_length)
        self.how_aggregator = HowR0Aggregator(latent_dim, self.semantic_encoder.output_dim, global_dim)
        self.shared_llm = SharedExplanationLLM(self.semantic_encoder.output_dim, latent_dim, global_dim, causal_lm, causal_lm_name, causal_lm_hidden_dim, freeze_causal_lm, causal_lm_tokenizer)
        self.what_alignment = nn.Linear(latent_dim, self.semantic_encoder.output_dim)
        self.why_alignment = nn.Linear(latent_dim, self.semantic_encoder.output_dim)
        self.how_alignment = nn.Linear(global_dim, self.semantic_encoder.output_dim)
        self.loss_weights = {"exp_what": lambda_exp_what, "exp_why": lambda_exp_why, "exp_how": lambda_exp_how, "what_txt": lambda_what_txt, "what_align": lambda_what_align, "route": lambda_route, "why_txt": lambda_why_txt, "why_align": lambda_why_align, "how_txt": lambda_how_txt, "how_align": lambda_how_align}

    def _align(self, projected, texts):
        return _zero(projected) if not texts else cosine_alignment_loss(projected, self.semantic_encoder.encode(texts).to(projected))

    def forward(self, decoder_states, decoder_memory, action_logits, duration_mu, duration_param2, query_text, fixation_mask, what_targets=None, why_membership=None, why_text_targets=None, episode_count=None, how_target=None, compute_generation=True, compute_alignment=True):
        if isinstance(query_text, str): query_text = [query_text]
        batch, length, _ = action_logits.shape
        if len(query_text) != batch: raise ValueError("query_text batch size mismatch")
        fixation_mask = fixation_mask.to(device=action_logits.device, dtype=torch.bool)
        query = self.semantic_encoder.encode(query_text).to(device=action_logits.device, dtype=action_logits.dtype)
        fixation = self.fixation_encoder(decoder_states, decoder_memory, action_logits, duration_mu, duration_param2)
        z = fixation["z"]
        router_logits, router_prob = self.why_router(z, query)
        if why_membership is None: membership, route_loss = torch.zeros_like(router_prob), _zero(z)
        else:
            membership = torch.as_tensor(why_membership, device=z.device)
            route_loss = WhyR0Router.routing_loss(router_prob, membership, fixation_mask)
        episode_tokens, episode_mass = WhyR0Router.soft_episode_aggregation(z, router_prob, fixation_mask)
        if episode_count is None and why_membership is not None: episode_count = (membership.sum(1) > 0).sum(-1).long()
        how = None if episode_count is None else self.how_aggregator(episode_tokens, query, episode_count)
        what_rows, why_rows = _text_rows(what_targets, batch, length), _slot_rows(why_text_targets, batch, self.kmax)
        how_rows = ["" for _ in range(batch)] if how_target is None else ([how_target] if isinstance(how_target, str) else [str(x) for x in how_target])
        if len(how_rows) != batch: raise ValueError("HOW text batch size mismatch")
        loss_what_txt = loss_what_align = _zero(z)
        active_what = [(n, t) for n in range(batch) for t in range(length) if fixation_mask[n, t] and what_rows[n][t].strip()]
        if active_what:
            latents, queries = torch.stack([z[n, t] for n, t in active_what]), torch.stack([query[n] for n, _ in active_what])
            texts = [what_rows[n][t] for n, t in active_what]
            if compute_generation: loss_what_txt = self.shared_llm.teacher_forced_loss("what", queries, latents, texts)
            if compute_alignment: loss_what_align = self._align(self.what_alignment(latents), texts)
        loss_why_txt = loss_why_align = _zero(z)
        if how is not None:
            counts = torch.as_tensor(episode_count).reshape(-1)
            active_why = [(n, k) for n in range(batch) for k in range(self.kmax) if k < int(counts[n].item()) and why_rows[n][k].strip()]
            if active_why:
                latents, queries = torch.stack([episode_tokens[n, k] for n, k in active_why]), torch.stack([query[n] for n, _ in active_why])
                texts = [why_rows[n][k] for n, k in active_why]
                if compute_generation: loss_why_txt = self.shared_llm.teacher_forced_loss("why", queries, latents, texts)
                if compute_alignment: loss_why_align = self._align(self.why_alignment(latents), texts)
        loss_how_txt = loss_how_align = _zero(z)
        if how is not None:
            active_how = [n for n, text in enumerate(how_rows) if text.strip()]
            if active_how:
                latents, queries = how["global_token"][active_how], query[active_how]
                texts = [how_rows[n] for n in active_how]
                if compute_generation: loss_how_txt = self.shared_llm.teacher_forced_loss("how", queries, latents, texts)
                if compute_alignment: loss_how_align = self._align(self.how_alignment(latents), texts)
        loss_what = self.loss_weights["what_txt"] * loss_what_txt + self.loss_weights["what_align"] * loss_what_align
        loss_why = self.loss_weights["route"] * route_loss + self.loss_weights["why_txt"] * loss_why_txt + self.loss_weights["why_align"] * loss_why_align
        loss_how = self.loss_weights["how_txt"] * loss_how_txt + self.loss_weights["how_align"] * loss_how_align
        loss_exp = self.loss_weights["exp_what"] * loss_what + self.loss_weights["exp_why"] * loss_why + self.loss_weights["exp_how"] * loss_how
        return {**fixation, "query_embedding": query, "router_logits": router_logits, "router_prob": router_prob, "episode_tokens": episode_tokens, "episode_mass": episode_mass, "global_token": None if how is None else how["global_token"], "active_episode_mask": None if how is None else how["active_episode_mask"], "loss_what_txt": loss_what_txt, "loss_what_align": loss_what_align, "loss_what": loss_what, "loss_route": route_loss, "loss_why_txt": loss_why_txt, "loss_why_align": loss_why_align, "loss_why": loss_why, "loss_how_txt": loss_how_txt, "loss_how_align": loss_how_align, "loss_how": loss_how, "loss_explanation": loss_exp}


def normalize_explanation_annotation(raw: Mapping[str, Any], max_length: int, kmax: int):
    missing = [key for key in ("query_text", "what_texts", "why_texts", "how_text") if key not in raw]
    if missing: raise ValueError("Explanation annotation missing: " + ", ".join(missing))
    what = [str(x) for x in raw["what_texts"]]
    if "why_membership" in raw:
        source = torch.as_tensor(raw["why_membership"], dtype=torch.float32)
        if source.ndim != 2 or source.shape[1] > kmax: raise ValueError("why_membership must be [fixations,K<=Kmax]")
        if torch.any((source.sum(-1) - 1).abs() > 1e-4): raise ValueError("Each real fixation must have exactly one gold WHY episode")
        ids, count = [int(row.argmax()) if row.sum() else -1 for row in source], int((source.sum(0) > 0).sum())
    else:
        if "why_episode_ids" not in raw: raise ValueError("Explanation annotation needs why_episode_ids or why_membership")
        source_ids, ordered = [int(x) for x in raw["why_episode_ids"]], []
        for episode in source_ids:
            if episode >= 0 and episode not in ordered: ordered.append(episode)
        if len(ordered) > kmax: raise ValueError("gold WHY episode count exceeds Kmax")
        remap = {episode: index for index, episode in enumerate(ordered)}
        ids, count = [remap.get(episode, -1) for episode in source_ids], len(ordered)
    if len(what) != len(ids): raise ValueError("WHAT and WHY annotations must cover the same fixations")
    what, ids = what[:max_length], ids[:max_length]
    membership = torch.zeros((max_length, kmax), dtype=torch.float32)
    for timestep, episode in enumerate(ids):
        if episode >= 0: membership[timestep, episode] = 1
    why = [str(x) for x in raw["why_texts"]]
    return {"query_text": str(raw["query_text"]), "what_texts": what + [""] * (max_length - len(what)), "why_membership": membership, "episode_count": count, "why_texts": why + [""] * (kmax - len(why)), "how_text": str(raw["how_text"])}


def load_explanation_annotations(path):
    with Path(path).open("r", encoding="utf-8") as handle: return json.load(handle)


def lookup_explanation_annotation(payload, fixation):
    if all(key in fixation for key in ("query_text", "what_texts", "how_text")): return fixation
    records = payload.get("annotations", payload) if isinstance(payload, Mapping) else payload
    if isinstance(records, Mapping):
        for key in ("id", "sample_id", "name"):
            if key in fixation and (str(fixation[key]) in records or fixation[key] in records): return records.get(str(fixation[key]), records.get(fixation[key]))
    elif isinstance(records, list):
        for record in records:
            if isinstance(record, Mapping) and (("id" in fixation and record.get("id") == fixation.get("id")) or (record.get("name") == fixation.get("name") and ("subject" not in record or record.get("subject") == fixation.get("subject")))): return record
    raise KeyError(f"No explanation annotation matched fixation {fixation.get('name', '<unnamed>')!r}")


def load_model_state_with_explanation_migration(model, state_dict, explanation_enabled, logger=print):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    prefix = "explanation_module."
    allowed_missing = [key for key in missing if key.startswith(prefix)]
    allowed_unexpected = [key for key in unexpected if key.startswith(prefix)]
    unrelated_missing = [key for key in missing if key not in allowed_missing]
    unrelated_unexpected = [key for key in unexpected if key not in allowed_unexpected]
    logger(f"checkpoint migration: explanation_enabled={explanation_enabled}, missing_keys={missing}, unexpected_keys={unexpected}")
    if unrelated_missing or unrelated_unexpected: raise RuntimeError(f"Checkpoint incompatibility outside explanation keys: missing={unrelated_missing}, unexpected={unrelated_unexpected}")
    if allowed_missing: logger("checkpoint migration: explanation parameters freshly initialized")
    return missing, unexpected

