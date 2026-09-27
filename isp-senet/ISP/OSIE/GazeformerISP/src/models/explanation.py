"""Differentiable hierarchical WHAT/WHY-R0/HOW-R0 supervision.

This module owns the latent hierarchy and its losses.  It deliberately consumes
the predictor's continuous tensors before any scanpath sampling occurs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.explanation_llm import (
    SemanticEncoderWrapper,
    SharedExplanationLLM,
    cosine_alignment_loss,
)


def final_spatial_fixation_distribution(action_logits: Tensor, already_probabilities: bool = False) -> Tensor:
    """Return ``m_t`` from final action logits, excluding the STOP slot."""

    if action_logits.ndim != 3 or action_logits.shape[-1] < 2:
        raise ValueError(
            "Final action logits must have shape [batch, length, 1 + spatial_cells]"
        )
    spatial = action_logits[..., 1:]
    if already_probabilities:
        return spatial / spatial.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(spatial.dtype).eps)
    return F.softmax(spatial, dim=-1)


def compose_joint_supervised_loss(scan_loss: Tensor, explanation_output: Optional[Mapping[str, Tensor]] = None) -> Tensor:
    """Compose ``L_total = L_scan + L_EXP`` without affecting RL callers."""

    if explanation_output is None:
        return scan_loss
    return scan_loss + explanation_output["loss_explanation"]


def duration_parameter_adapter(duration_mu: Tensor, duration_param2: Tensor) -> Tensor:
    """Adapt the two predictor duration outputs without resolving their meaning.

    The existing predictor calls the second quantity ``sigma2`` in its loss but
    uses it as a sampling scale elsewhere.  Explanation code therefore carries
    it under the neutral ``duration_param2`` name and feeds the two native
    quantities directly to its duration MLP.  No baseline duration semantics
    are changed here.
    """

    if duration_mu.shape != duration_param2.shape:
        raise ValueError(
            "duration_mu and duration_param2 must have identical shapes, got "
            f"{tuple(duration_mu.shape)} and {tuple(duration_param2.shape)}"
        )
    return torch.stack((duration_mu, duration_param2), dim=-1)


def _batch_first(value: Tensor, batch: int, length: int, name: str) -> Tensor:
    if value.ndim != 3:
        raise ValueError(f"{name} must be rank 3, got {tuple(value.shape)}")
    if value.shape[0] == batch and value.shape[1] == length:
        return value
    if value.shape[0] == length and value.shape[1] == batch:
        return value.permute(1, 0, 2)
    raise ValueError(
        f"{name} must be [N,L,D] or [L,N,D] with N={batch}, L={length}; "
        f"got {tuple(value.shape)}"
    )


class FixationReasoningEncoder(nn.Module):
    """Build ``z_t`` from decoder state, soft visual evidence, position, duration."""

    def __init__(
        self,
        decoder_dim: int,
        visual_dim: int,
        latent_dim: int,
        spatial_dim: tuple[int, int],
        duration_hidden_dim: Optional[int] = None,
        position_hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.decoder_dim = int(decoder_dim)
        self.visual_dim = int(visual_dim)
        self.latent_dim = int(latent_dim)
        self.spatial_dim = tuple(int(v) for v in spatial_dim)
        height, width = self.spatial_dim
        if height <= 0 or width <= 0:
            raise ValueError("spatial_dim must contain positive dimensions")
        rows = torch.arange(height, dtype=torch.float32)
        cols = torch.arange(width, dtype=torch.float32)
        if height > 1:
            rows = rows / float(height - 1)
        if width > 1:
            cols = cols / float(width - 1)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        self.register_buffer("grid_xy", torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1))

        position_hidden_dim = int(position_hidden_dim or max(32, latent_dim // 2))
        duration_hidden_dim = int(duration_hidden_dim or max(32, latent_dim // 2))
        self.decoder_projection = nn.Linear(decoder_dim, latent_dim)
        self.visual_projection = nn.Linear(visual_dim, latent_dim)
        self.position_mlp = nn.Sequential(
            nn.Linear(2, position_hidden_dim), nn.GELU(), nn.Linear(position_hidden_dim, latent_dim)
        )
        self.duration_mlp = nn.Sequential(
            nn.Linear(2, duration_hidden_dim), nn.GELU(), nn.Linear(duration_hidden_dim, latent_dim)
        )
        self.fixation_mlp = nn.Sequential(
            nn.Linear(4 * latent_dim, 2 * latent_dim), nn.GELU(), nn.Linear(2 * latent_dim, latent_dim)
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        decoder_states: Tensor,
        visual_memory: Tensor,
        action_logits: Tensor,
        duration_mu: Tensor,
        duration_param2: Tensor,
        action_probabilities: bool = False,
    ) -> dict[str, Tensor]:
        if action_logits.ndim != 3:
            raise ValueError("action_logits must have shape [N,L,A]")
        batch, length, action_count = action_logits.shape
        expected_cells = self.spatial_dim[0] * self.spatial_dim[1]
        if action_count != expected_cells + 1:
            raise ValueError(
                f"action logits have {action_count} actions but spatial grid requires "
                f"{expected_cells + 1} including STOP"
            )
        decoder_states = _batch_first(decoder_states, batch, length, "decoder_states")
        if visual_memory.ndim != 3:
            raise ValueError("visual_memory must be rank 3")
        if visual_memory.shape[0] == batch and visual_memory.shape[1] == expected_cells:
            visual_memory = visual_memory
        elif visual_memory.shape[0] == expected_cells and visual_memory.shape[1] == batch:
            visual_memory = visual_memory.permute(1, 0, 2)
        else:
            raise ValueError(
                "visual_memory must be [N,G,D] or [G,N,D] aligned with the action grid; "
                f"got {tuple(visual_memory.shape)}"
            )
        duration_mu = duration_mu.reshape(batch, length)
        duration_param2 = duration_param2.reshape(batch, length)
        fixation_prob = final_spatial_fixation_distribution(action_logits, already_probabilities=action_probabilities)
        soft_visual = torch.einsum("nlg,ngd->nld", fixation_prob, visual_memory)
        expected_xy = torch.einsum("nlg,gc->nlc", fixation_prob, self.grid_xy.to(fixation_prob))
        position = self.position_mlp(expected_xy)
        duration = self.duration_mlp(duration_parameter_adapter(duration_mu, duration_param2))
        combined = torch.cat(
            (
                self.decoder_projection(decoder_states),
                self.visual_projection(soft_visual),
                position,
                duration,
            ),
            dim=-1,
        )
        z = self.norm(self.fixation_mlp(combined))
        return {
            "fixation_prob": fixation_prob,
            "soft_visual": soft_visual,
            "expected_xy": expected_xy,
            "duration_feature": duration,
            "z": z,
        }


class WhyR0Router(nn.Module):
    """Simple per-fixation MLP router; no contextual router transformer."""

    def __init__(
        self,
        latent_dim: int,
        query_dim: int,
        kmax: int,
        max_length: int,
        temporal_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if kmax <= 0:
            raise ValueError("router_kmax must be positive")
        self.kmax = int(kmax)
        self.max_length = int(max_length)
        temporal_dim = int(temporal_dim or max(16, latent_dim // 4))
        hidden_dim = int(hidden_dim or max(64, latent_dim))
        self.temporal_dim = temporal_dim
        self.register_buffer(
            "temporal_frequency",
            torch.exp(-torch.arange((temporal_dim + 1) // 2, dtype=torch.float32) * (torch.log(torch.tensor(10000.0)) / temporal_dim)),
        )
        self.query_projection = nn.Linear(query_dim, latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + temporal_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.kmax),
        )

    def forward(self, z: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        batch, length, _ = z.shape
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds router max_length {self.max_length}")
        positions = torch.arange(length, device=z.device).unsqueeze(0).expand(batch, -1)
        normalized_time = positions.to(z.dtype) / float(max(length - 1, 1))
        temporal = z.new_zeros((batch, length, self.temporal_dim))
        temporal[..., 0::2] = torch.sin(normalized_time.unsqueeze(-1) * self.temporal_frequency[: temporal[..., 0::2].shape[-1]])
        temporal[..., 1::2] = torch.cos(normalized_time.unsqueeze(-1) * self.temporal_frequency[: temporal[..., 1::2].shape[-1]])
        query_context = self.query_projection(query).unsqueeze(1).expand(-1, length, -1)
        logits = self.mlp(torch.cat((z, temporal, query_context), dim=-1))
        return logits, logits.softmax(dim=-1)

    @staticmethod
    def routing_loss(
        router_prob: Tensor,
        why_membership: Tensor,
        fixation_mask: Tensor,
    ) -> Tensor:
        if router_prob.shape != why_membership.shape:
            raise ValueError(
                f"why_membership must match router_prob, got {tuple(why_membership.shape)} "
                f"and {tuple(router_prob.shape)}"
            )
        mask = fixation_mask.to(device=router_prob.device, dtype=router_prob.dtype)
        if mask.sum() == 0:
            return router_prob.sum() * 0.0
        target = why_membership.to(device=router_prob.device, dtype=router_prob.dtype)
        if torch.any(target[mask.bool()].sum(dim=-1).sub(1.0).abs() > 1e-4):
            raise ValueError("Each real fixation must belong to exactly one gold WHY episode")
        if torch.any(target[~mask.bool()].abs() > 1e-6):
            raise ValueError("Padded fixation rows must have zero WHY membership")
        log_prob = router_prob.clamp_min(torch.finfo(router_prob.dtype).tiny).log()
        per_fixation = -(target * log_prob).sum(dim=-1)
        return (per_fixation * mask).sum() / mask.sum()

    @staticmethod
    def soft_episode_aggregation(z: Tensor, router_prob: Tensor, fixation_mask: Tensor) -> tuple[Tensor, Tensor]:
        mask = fixation_mask.to(device=z.device, dtype=z.dtype).unsqueeze(-1)
        masked_prob = router_prob * mask
        mass = masked_prob.sum(dim=1)
        weighted = torch.einsum("nlk,nld->nkd", masked_prob, z)
        episode_tokens = weighted / (mass.unsqueeze(-1) + 1e-8)
        return episode_tokens, mass


class HowR0Aggregator(nn.Module):
    """Gold-active-slot mean pooling followed by query-conditioned global token."""

    def __init__(
        self,
        latent_dim: int,
        query_dim: int,
        global_dim: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or max(64, global_dim))
        self.query_projection = nn.Linear(query_dim, global_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + global_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, global_dim)
        )
        self.norm = nn.LayerNorm(global_dim)

    def forward(self, episode_tokens: Tensor, query: Tensor, episode_count: Tensor) -> dict[str, Tensor]:
        batch, kmax, _ = episode_tokens.shape
        counts = torch.as_tensor(episode_count, device=episode_tokens.device, dtype=torch.long).reshape(-1)
        if counts.numel() != batch:
            raise ValueError(f"episode_count must have one value per example ({batch}), got {counts.numel()}")
        if torch.any(counts < 0) or torch.any(counts > kmax):
            raise ValueError(f"episode_count must be in [0, {kmax}]")
        slot_ids = torch.arange(kmax, device=episode_tokens.device).unsqueeze(0)
        active = slot_ids < counts.unsqueeze(1)
        pooled = (episode_tokens * active.unsqueeze(-1).to(episode_tokens.dtype)).sum(dim=1)
        pooled = pooled / counts.clamp_min(1).to(episode_tokens.dtype).unsqueeze(-1)
        # Zero-count examples have no defined R0 active episode; retaining a
        # zero pooled representation is explicit and avoids inventing a slot.
        pooled = pooled * (counts > 0).to(episode_tokens.dtype).unsqueeze(-1)
        global_token = self.norm(self.mlp(torch.cat((pooled, self.query_projection(query)), dim=-1)))
        return {"global_token": global_token, "active_episode_mask": active, "pooled_episode": pooled}


def _zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def _normalise_text_rows(values: Any, batch: int, length: int, default: str = "") -> list[list[str]]:
    if values is None:
        return [[default for _ in range(length)] for _ in range(batch)]
    rows = list(values)
    if len(rows) != batch:
        raise ValueError(f"Expected {batch} text rows, got {len(rows)}")
    result: list[list[str]] = []
    for row in rows:
        if isinstance(row, str):
            row = [row]
        row = list(row)
        result.append([str(row[index]) if index < len(row) else default for index in range(length)])
    return result


def _normalise_slot_texts(values: Any, batch: int, kmax: int) -> list[list[str]]:
    if values is None:
        return [["" for _ in range(kmax)] for _ in range(batch)]
    rows = list(values)
    if len(rows) != batch:
        raise ValueError(f"Expected {batch} WHY text rows, got {len(rows)}")
    return [[str(row[index]) if index < len(row) else "" for index in range(kmax)] for row in rows]


class HierarchicalExplanationModule(nn.Module):
    """Full differentiable latent hierarchy and optional language supervision."""

    def __init__(
        self,
        decoder_dim: int,
        visual_dim: int,
        latent_dim: int,
        spatial_dim: tuple[int, int],
        max_length: int,
        kmax: int,
        semantic_encoder: Optional[nn.Module] = None,
        semantic_encoder_name: Optional[str] = None,
        semantic_encoder_dim: Optional[int] = None,
        freeze_semantic_encoder: bool = True,
        causal_lm: Optional[nn.Module] = None,
        causal_lm_name: Optional[str] = None,
        causal_lm_hidden_dim: Optional[int] = None,
        causal_lm_tokenizer: Any = None,
        freeze_causal_lm: bool = True,
        global_dim: Optional[int] = None,
        lambda_exp_what: float = 1.0,
        lambda_exp_why: float = 1.0,
        lambda_exp_how: float = 1.0,
        lambda_what_txt: float = 1.0,
        lambda_what_align: float = 1.0,
        lambda_route: float = 1.0,
        lambda_why_txt: float = 1.0,
        lambda_why_align: float = 1.0,
        lambda_how_txt: float = 1.0,
        lambda_how_align: float = 1.0,
    ) -> None:
        super().__init__()
        self.kmax = int(kmax)
        self.max_length = int(max_length)
        self.semantic_encoder = SemanticEncoderWrapper(
            encoder=semantic_encoder,
            model_name=semantic_encoder_name,
            freeze=freeze_semantic_encoder,
            output_dim=semantic_encoder_dim,
        )
        global_dim = int(global_dim or latent_dim)
        self.fixation_encoder = FixationReasoningEncoder(
            decoder_dim=decoder_dim,
            visual_dim=visual_dim,
            latent_dim=latent_dim,
            spatial_dim=spatial_dim,
        )
        self.why_router = WhyR0Router(
            latent_dim=latent_dim,
            query_dim=self.semantic_encoder.output_dim,
            kmax=self.kmax,
            max_length=max_length,
        )
        self.how_aggregator = HowR0Aggregator(
            latent_dim=latent_dim,
            query_dim=self.semantic_encoder.output_dim,
            global_dim=global_dim,
        )
        self.shared_llm = SharedExplanationLLM(
            semantic_dim=self.semantic_encoder.output_dim,
            latent_dim=latent_dim,
            how_dim=global_dim,
            llm=causal_lm,
            model_name=causal_lm_name,
            hidden_dim=causal_lm_hidden_dim,
            freeze=freeze_causal_lm,
            tokenizer=causal_lm_tokenizer,
            query_dim=self.semantic_encoder.output_dim,
        )
        self.what_alignment = nn.Linear(latent_dim, self.semantic_encoder.output_dim)
        self.why_alignment = nn.Linear(latent_dim, self.semantic_encoder.output_dim)
        self.how_alignment = nn.Linear(global_dim, self.semantic_encoder.output_dim)
        self.loss_weights = {
            "exp_what": float(lambda_exp_what),
            "exp_why": float(lambda_exp_why),
            "exp_how": float(lambda_exp_how),
            "what_txt": float(lambda_what_txt),
            "what_align": float(lambda_what_align),
            "route": float(lambda_route),
            "why_txt": float(lambda_why_txt),
            "why_align": float(lambda_why_align),
            "how_txt": float(lambda_how_txt),
            "how_align": float(lambda_how_align),
        }

    def _alignment_for_texts(self, projected: Tensor, texts: Sequence[str], device: torch.device) -> Tensor:
        if not texts:
            return _zero(projected)
        target = self.semantic_encoder.encode(list(texts)).to(device=device, dtype=projected.dtype)
        return cosine_alignment_loss(projected, target)

    def forward(
        self,
        decoder_states: Tensor,
        decoder_memory: Tensor,
        action_logits: Tensor,
        duration_mu: Tensor,
        duration_param2: Tensor,
        query_text: Sequence[str] | str,
        fixation_mask: Tensor,
        what_targets: Any = None,
        why_membership: Optional[Tensor] = None,
        why_text_targets: Any = None,
        episode_count: Optional[Tensor] = None,
        how_target: Any = None,
        compute_generation: bool = True,
        compute_alignment: bool = True,
        action_probabilities: bool = False,
    ) -> dict[str, Any]:
        if isinstance(query_text, str):
            query_text = [query_text]
        batch, length, _ = action_logits.shape
        if len(query_text) != batch:
            raise ValueError(f"Expected {batch} query texts, got {len(query_text)}")
        fixation_mask = fixation_mask.to(device=action_logits.device, dtype=torch.bool)
        if fixation_mask.shape != (batch, length):
            raise ValueError(f"fixation_mask must be {(batch, length)}, got {tuple(fixation_mask.shape)}")
        query = self.semantic_encoder.encode(query_text).to(device=action_logits.device, dtype=action_logits.dtype)
        fixation = self.fixation_encoder(
            decoder_states=decoder_states,
            visual_memory=decoder_memory,
            action_logits=action_logits,
            duration_mu=duration_mu,
            duration_param2=duration_param2,
            action_probabilities=action_probabilities,
        )
        z = fixation["z"]
        router_logits, router_prob = self.why_router(z, query)
        if why_membership is None:
            membership = torch.zeros_like(router_prob)
            route_loss = _zero(z)
        else:
            membership = torch.as_tensor(why_membership, device=z.device)
            if membership.shape != router_prob.shape:
                raise ValueError(
                    f"why_membership must have shape {tuple(router_prob.shape)}, got {tuple(membership.shape)}"
                )
            route_loss = self.why_router.routing_loss(router_prob, membership, fixation_mask)
        episode_tokens, episode_mass = self.why_router.soft_episode_aggregation(z, router_prob, fixation_mask)
        if episode_count is None and why_membership is not None:
            episode_count = (membership.sum(dim=1) > 0).sum(dim=-1).long()
        how = None
        if episode_count is not None:
            how = self.how_aggregator(episode_tokens, query, episode_count)

        what_rows = _normalise_text_rows(what_targets, batch, length)
        why_rows = _normalise_slot_texts(why_text_targets, batch, self.kmax)
        if how_target is None:
            how_rows = ["" for _ in range(batch)]
        elif isinstance(how_target, str):
            how_rows = [how_target]
        else:
            how_rows = [str(value) for value in how_target]
        if len(how_rows) != batch:
            raise ValueError(f"Expected {batch} HOW targets, got {len(how_rows)}")

        loss_what_txt = _zero(z)
        loss_what_align = _zero(z)
        active_what: list[tuple[int, int]] = [
            (sample, step)
            for sample in range(batch)
            for step in range(length)
            if bool(fixation_mask[sample, step]) and bool(what_rows[sample][step].strip())
        ]
        if active_what:
            what_latents = torch.stack([z[sample, step] for sample, step in active_what])
            what_query = torch.stack([query[sample] for sample, _ in active_what])
            what_texts = [what_rows[sample][step] for sample, step in active_what]
            if compute_generation:
                loss_what_txt = self.shared_llm.teacher_forced_loss("what", what_query, what_latents, what_texts)
            if compute_alignment:
                loss_what_align = self._alignment_for_texts(self.what_alignment(what_latents), what_texts, z.device)

        loss_why_txt = _zero(z)
        loss_why_align = _zero(z)
        if how is not None:
            active_why: list[tuple[int, int]] = [
                (sample, slot)
                for sample in range(batch)
                for slot in range(self.kmax)
                if slot < int(torch.as_tensor(episode_count)[sample].item()) and bool(why_rows[sample][slot].strip())
            ]
            if active_why:
                why_latents = torch.stack([episode_tokens[sample, slot] for sample, slot in active_why])
                why_query = torch.stack([query[sample] for sample, _ in active_why])
                why_texts = [why_rows[sample][slot] for sample, slot in active_why]
                if compute_generation:
                    loss_why_txt = self.shared_llm.teacher_forced_loss("why", why_query, why_latents, why_texts)
                if compute_alignment:
                    loss_why_align = self._alignment_for_texts(self.why_alignment(why_latents), why_texts, z.device)

        loss_how_txt = _zero(z)
        loss_how_align = _zero(z)
        if how is not None:
            active_how = [index for index, text in enumerate(how_rows) if text.strip()]
            if active_how:
                how_latents = how["global_token"][active_how]
                how_query = query[active_how]
                how_texts = [how_rows[index] for index in active_how]
                if compute_generation:
                    loss_how_txt = self.shared_llm.teacher_forced_loss("how", how_query, how_latents, how_texts)
                if compute_alignment:
                    loss_how_align = self._alignment_for_texts(self.how_alignment(how_latents), how_texts, z.device)

        loss_what = self.loss_weights["what_txt"] * loss_what_txt + self.loss_weights["what_align"] * loss_what_align
        loss_why = (
            self.loss_weights["route"] * route_loss
            + self.loss_weights["why_txt"] * loss_why_txt
            + self.loss_weights["why_align"] * loss_why_align
        )
        loss_how = self.loss_weights["how_txt"] * loss_how_txt + self.loss_weights["how_align"] * loss_how_align
        loss_explanation = (
            self.loss_weights["exp_what"] * loss_what
            + self.loss_weights["exp_why"] * loss_why
            + self.loss_weights["exp_how"] * loss_how
        )
        return {
            **fixation,
            "query_embedding": query,
            "router_logits": router_logits,
            "router_prob": router_prob,
            "episode_tokens": episode_tokens,
            "episode_mass": episode_mass,
            "global_token": None if how is None else how["global_token"],
            "active_episode_mask": None if how is None else how["active_episode_mask"],
            "loss_what_txt": loss_what_txt,
            "loss_what_align": loss_what_align,
            "loss_what": loss_what,
            "loss_route": route_loss,
            "loss_why_txt": loss_why_txt,
            "loss_why_align": loss_why_align,
            "loss_why": loss_why,
            "loss_how_txt": loss_how_txt,
            "loss_how_align": loss_how_align,
            "loss_how": loss_how,
            "loss_explanation": loss_explanation,
        }


def normalize_explanation_annotation(raw: Mapping[str, Any], max_length: int, kmax: int) -> dict[str, Any]:
    """Normalize one source annotation to the predictor's padded contract."""

    required = ("query_text", "what_texts", "why_texts", "how_text")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Explanation annotation is missing required fields: {', '.join(missing)}")
    query_text = str(raw["query_text"])
    what_source = [str(value) for value in raw["what_texts"]]
    if "why_membership" in raw:
        membership_source = torch.as_tensor(raw["why_membership"], dtype=torch.float32)
        if membership_source.ndim != 2:
            raise ValueError("why_membership must be rank 2")
        if torch.any((membership_source.sum(dim=-1) - 1.0).abs() > 1e-4):
            raise ValueError("Each real fixation must have exactly one gold WHY episode")
        episode_count = int((membership_source.sum(dim=0) > 0).sum().item())
        if membership_source.shape[1] > kmax:
            raise ValueError(f"Gold WHY episode count exceeds Kmax={kmax}")
        why_ids = membership_source.argmax(dim=-1).tolist()
    else:
        if "why_episode_ids" not in raw:
            raise ValueError("Explanation annotation requires why_episode_ids or why_membership")
        why_ids_source = [int(value) for value in raw["why_episode_ids"]]
        ordered_ids: list[int] = []
        for episode_id in why_ids_source:
            if episode_id >= 0 and episode_id not in ordered_ids:
                ordered_ids.append(episode_id)
        if len(ordered_ids) > kmax:
            raise ValueError(f"Gold WHY episode count exceeds Kmax={kmax}")
        remap = {episode_id: index for index, episode_id in enumerate(ordered_ids)}
        why_ids = [remap.get(episode_id, -1) for episode_id in why_ids_source]
        episode_count = len(ordered_ids)
    if len(what_source) != len(why_ids):
        raise ValueError("what_texts and WHY membership must cover the same real fixations")
    if len(what_source) > max_length:
        what_source = what_source[:max_length]
        why_ids = why_ids[:max_length]
    membership = torch.zeros((max_length, kmax), dtype=torch.float32)
    for timestep, episode_id in enumerate(why_ids):
        if episode_id < 0:
            continue
        if episode_id >= kmax:
            raise ValueError(f"WHY episode id {episode_id} is outside Kmax={kmax}")
        membership[timestep, episode_id] = 1.0
    why_text_source = [str(value) for value in raw["why_texts"]]
    why_texts = [why_text_source[index] if index < len(why_text_source) else "" for index in range(kmax)]
    return {
        "query_text": query_text,
        "what_texts": what_source + [""] * (max_length - len(what_source)),
        "why_membership": membership,
        "episode_count": episode_count,
        "why_texts": why_texts,
        "how_text": str(raw["how_text"]),
    }


def load_explanation_annotations(path: str | Path) -> Any:
    """Load a user-configured annotation artifact without assuming its schema."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def lookup_explanation_annotation(payload: Any, fixation: Mapping[str, Any]) -> Mapping[str, Any]:
    """Find an annotation by an explicit record id or image/subject identity."""

    if all(key in fixation for key in ("query_text", "what_texts", "how_text")):
        return fixation
    records = payload.get("annotations", payload) if isinstance(payload, Mapping) else payload
    candidates = []
    if isinstance(records, Mapping):
        for key in ("id", "sample_id", "name"):
            if key in fixation:
                value = records.get(str(fixation[key]), records.get(fixation[key]))
                if value is not None:
                    candidates.append(value)
        if candidates:
            return candidates[0]
        raise KeyError(
            f"No explanation annotation matched fixation keys for {fixation.get('name', '<unnamed>')!r}"
        )
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            if "id" in fixation and record.get("id") == fixation.get("id"):
                return record
            if record.get("name") == fixation.get("name") and (
                "subject" not in record or record.get("subject") == fixation.get("subject")
            ):
                return record
    raise KeyError(
        f"No explanation annotation matched fixation {fixation.get('name', '<unnamed>')!r}; "
        "provide an explicit annotation record id/name mapping."
    )


def load_model_state_with_explanation_migration(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    explanation_enabled: bool,
    logger=print,
):
    """Controlled base<->explanation checkpoint migration.

    Only explanation-prefixed incompatibilities are allowed.  Any unrelated
    missing/unexpected key remains an error instead of being hidden by a global
    ``strict=False``.
    """

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    explanation_prefix = "explanation_module."
    allowed_missing = [key for key in missing if key.startswith(explanation_prefix)]
    allowed_unexpected = [key for key in unexpected if key.startswith(explanation_prefix)]
    unrelated_missing = [key for key in missing if key not in allowed_missing]
    unrelated_unexpected = [key for key in unexpected if key not in allowed_unexpected]
    logger(
        "checkpoint migration: explanation_enabled={}, missing_keys={}, unexpected_keys={}".format(
            explanation_enabled, missing, unexpected
        )
    )
    if unrelated_missing or unrelated_unexpected:
        raise RuntimeError(
            "Checkpoint incompatibility outside explanation keys: "
            f"missing={unrelated_missing}, unexpected={unrelated_unexpected}"
        )
    if allowed_missing:
        logger("checkpoint migration: explanation parameters freshly initialized")
    return missing, unexpected

