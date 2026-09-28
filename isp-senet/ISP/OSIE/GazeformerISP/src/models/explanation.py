"""Differentiable hierarchical WHAT/WHY-R0/HOW-R0 supervision.

This module owns the latent hierarchy and its losses.  It deliberately consumes
the predictor's continuous tensors before any scanpath sampling occurs.
"""

from __future__ import annotations

import json
from pathlib import Path
from numbers import Integral
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

    def forward(self, z: Tensor, query: Tensor, fixation_mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        batch, length, _ = z.shape
        if length > self.max_length:
            raise ValueError(f"sequence length {length} exceeds router max_length {self.max_length}")
        positions = torch.arange(length, device=z.device).unsqueeze(0).expand(batch, -1)
        lengths = (fixation_mask.sum(-1) if fixation_mask is not None
                   else torch.full((batch,), length, device=z.device))
        normalized_time = positions.to(z.dtype) / (lengths - 1).clamp_min(1).to(z.dtype).unsqueeze(-1)
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
        if not torch.all((target == 0) | (target == 1)):
            raise ValueError("Gold WHY membership must be one-hot")
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
        target = self.semantic_encoder.encode_target(list(texts)).to(device=device, dtype=projected.dtype)
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
        router_logits, router_prob = self.why_router(z, query, fixation_mask)
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
        if why_membership is not None:
            active = membership.sum(dim=1) > 0
            counts = active.sum(dim=-1).long()
            expected = torch.arange(self.kmax, device=z.device).unsqueeze(0) < counts.unsqueeze(-1)
            if not torch.equal(active, expected):
                raise ValueError("Gold WHY membership must use contiguous active slots from zero")
            if episode_count is not None and not torch.equal(torch.as_tensor(episode_count, device=z.device).long(), counts):
                raise ValueError("episode_count disagrees with gold WHY membership")
            episode_count = counts
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
                if bool(how["active_episode_mask"][sample, slot]) and bool(why_rows[sample][slot].strip())
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
            active_how = [index for index, text in enumerate(how_rows) if text.strip() and bool(how["active_episode_mask"][index].any())]
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


def _annotation_identity(raw: Mapping[str, Any]) -> str:
    """Return enough sample identity to make malformed-data errors actionable."""

    return (
        f"dataset={raw.get('dataset', '<unknown>')!r}, "
        f"name={raw.get('name', raw.get('image_id', '<unnamed>'))!r}, "
        f"subject={raw.get('subject', raw.get('subject_idx', '<unknown>'))!r}"
    )


def _fixation_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"Fixation/episode index must be an integer, got {value!r}")
    return int(value)


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _inline_query_text(raw: Mapping[str, Any]) -> str:
    for key in ("question", "query_text", "task_text", "task"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise ValueError(f"Augmented explanation sample ({_annotation_identity(raw)}) has no task/question text")


def _validate_inline_annotation(raw: Mapping[str, Any]) -> tuple[str, list[str], list[tuple[list[int], str]], str]:
    """Validate the raw augmented format before any model-side truncation."""

    identity = _annotation_identity(raw)
    try:
        x = list(raw["X"])
        y = list(raw["Y"])
        t = list(raw["T"])
    except KeyError as exc:
        raise ValueError(f"Augmented explanation sample ({identity}) is missing raw field {exc.args[0]!r}") from exc
    if not (len(x) == len(y) == len(t)):
        raise ValueError(
            f"Augmented explanation sample ({identity}) violates len(X)==len(Y)==len(T): "
            f"{len(x)}, {len(y)}, {len(t)}"
        )
    if not x:
        raise ValueError(f"Augmented explanation sample ({identity}) has a zero-length scanpath")
    prediction = raw.get("prediction")
    if not isinstance(prediction, Mapping):
        raise ValueError(f"Augmented explanation sample ({identity}) has no object-valued prediction field")

    what_entries = prediction.get("fixations")
    if not isinstance(what_entries, list):
        raise ValueError(f"Augmented explanation sample ({identity}) prediction.fixations must be a list")
    what_ids = []
    what_texts_by_id: dict[int, str] = {}
    for entry_index, entry in enumerate(what_entries):
        if not isinstance(entry, Mapping) or "fixation" not in entry or "what" not in entry:
            raise ValueError(
                f"Augmented explanation sample ({identity}) prediction.fixations[{entry_index}] "
                "must contain fixation and what"
            )
        try:
            fixation_id = _fixation_id(entry["fixation"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Augmented explanation sample ({identity}) prediction.fixations[{entry_index}].fixation "
                f"is not an integer: {entry.get('fixation')!r}"
            ) from exc
        what_ids.append(fixation_id)
        what_texts_by_id[fixation_id] = _required_text(entry["what"], f"{identity} WHAT {fixation_id}")
    expected_ids = list(range(1, len(x) + 1))
    if len(what_entries) != len(x) or what_ids != expected_ids:
        raise ValueError(
            f"Augmented explanation sample ({identity}) prediction.fixations IDs must be exactly "
            f"1..{len(x)} in order; got {what_ids!r}"
        )

    regions = prediction.get("regions")
    if not isinstance(regions, list):
        raise ValueError(f"Augmented explanation sample ({identity}) prediction.regions must be a list")
    parsed_regions: list[tuple[list[int], str]] = []
    covered: list[int] = []
    for region_index, region in enumerate(regions):
        if not isinstance(region, Mapping) or "fixations" not in region or "why" not in region:
            raise ValueError(
                f"Augmented explanation sample ({identity}) prediction.regions[{region_index}] "
                "must contain fixations and why"
            )
        ids = list(region["fixations"])
        if not ids:
            raise ValueError(f"Augmented explanation sample ({identity}) prediction.regions[{region_index}] is empty")
        parsed_ids: list[int] = []
        for id_index, value in enumerate(ids):
            try:
                fixation_id = _fixation_id(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Augmented explanation sample ({identity}) prediction.regions[{region_index}].fixations"
                    f"[{id_index}] is not an integer: {value!r}"
                ) from exc
            if not 1 <= fixation_id <= len(x):
                raise ValueError(
                    f"Augmented explanation sample ({identity}) prediction.regions[{region_index}] references "
                    f"raw fixation {fixation_id}; valid range is 1..{len(x)}"
                )
            if fixation_id in parsed_ids:
                raise ValueError(
                    f"Augmented explanation sample ({identity}) prediction.regions[{region_index}] repeats "
                    f"raw fixation {fixation_id}"
                )
            parsed_ids.append(fixation_id)
            covered.append(fixation_id)
        parsed_regions.append((parsed_ids, _required_text(region["why"], f"{identity} WHY {region_index}")))
    if sorted(covered) != expected_ids:
        raise ValueError(
            f"Augmented explanation sample ({identity}) WHY regions must partition raw fixation IDs "
            f"1..{len(x)} exactly once; got {covered!r}"
        )
    parsed_regions.sort(key=lambda region: min(region[0]))
    if "how" not in prediction:
        raise ValueError(f"Augmented explanation sample ({identity}) prediction is missing how")
    return _inline_query_text(raw), [what_texts_by_id[i] for i in expected_ids], parsed_regions, _required_text(prediction["how"], f"{identity} HOW")


def _normalise_model_indices(
    raw_length: int,
    max_length: int,
    model_raw_indices: Optional[Sequence[int]] = None,
    raw_to_model_idx: Optional[Mapping[int, int]] = None,
) -> tuple[list[int], dict[int, int]]:
    """Validate the preprocessing-produced 1-based raw -> 0-based model map."""

    if raw_to_model_idx is not None and model_raw_indices is not None:
        raise ValueError("Pass either model_raw_indices or raw_to_model_idx, not both")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if raw_to_model_idx is not None:
        pairs = [(_fixation_id(raw_id), _fixation_id(token)) for raw_id, token in raw_to_model_idx.items()]
        pairs.sort(key=lambda item: item[1])
        if [token for _, token in pairs] != list(range(len(pairs))):
            raise ValueError("raw-to-model mapping must have contiguous model indices from zero")
        indices = [raw_id for raw_id, _ in pairs]
    elif model_raw_indices is None:
        indices = list(range(1, min(raw_length, max_length) + 1))
        mapping = {raw_id: token for token, raw_id in enumerate(indices)}
    else:
        indices = [_fixation_id(value) for value in model_raw_indices]
    if len(set(indices)) != len(indices):
        raise ValueError(f"raw-to-model mapping contains duplicate raw IDs: {indices!r}")
    if indices != sorted(indices):
        raise ValueError("raw-to-model mapping must preserve temporal fixation order")
    if any(raw_id < 1 or raw_id > raw_length for raw_id in indices):
        raise ValueError(f"raw-to-model mapping contains an out-of-range raw fixation: {indices!r}")
    indices = indices[:max_length]
    mapping = {raw_id: token for token, raw_id in enumerate(indices)}
    if list(mapping.values()) != list(range(len(mapping))):
        raise ValueError(f"raw-to-model mapping must have contiguous model indices from zero: {mapping!r}")
    return indices, mapping


def normalize_augmented_explanation_annotation(
    raw: Mapping[str, Any],
    max_length: int,
    kmax: int,
    model_raw_indices: Optional[Sequence[int]] = None,
    raw_to_model_idx: Optional[Mapping[int, int]] = None,
    allow_truncated_how: bool = False,
) -> dict[str, Any]:
    """Normalize an inline augmented benchmark record without positional guessing."""

    query_text, raw_what, raw_regions, how_text = _validate_inline_annotation(raw)
    return _align_explanation_targets(raw, query_text, raw_what, raw_regions, how_text,
                                      max_length, kmax, model_raw_indices, raw_to_model_idx,
                                      allow_truncated_how)


def _align_explanation_targets(raw, query_text, raw_what, raw_regions, how_text,
                               max_length, kmax, model_raw_indices, raw_to_model_idx,
                               allow_truncated_how):
    raw_length = len(raw_what)
    if kmax <= 0:
        raise ValueError("Kmax must be positive")
    kept_raw_ids, mapping = _normalise_model_indices(
        raw_length, max_length, model_raw_indices=model_raw_indices, raw_to_model_idx=raw_to_model_idx
    )
    truncated = len(kept_raw_ids) != raw_length
    if not kept_raw_ids:
        raise ValueError("Explanation supervision requires at least one model-visible fixation")
    if truncated and how_text.strip() and not allow_truncated_how:
        raise ValueError(
            f"Augmented explanation sample ({_annotation_identity(raw)}) has a HOW target for a raw trajectory "
            "that is truncated by preprocessing. Set allow_truncated_how=True only when that target is known "
            "to describe the model-visible trajectory."
        )

    membership = torch.zeros((max_length, kmax), dtype=torch.float32)
    what_texts = [""] * max_length
    for raw_id in kept_raw_ids:
        what_texts[mapping[raw_id]] = raw_what[raw_id - 1]

    active_regions: list[tuple[list[int], str]] = []
    for raw_ids, why_text in raw_regions:
        surviving = [raw_id for raw_id in raw_ids if raw_id in mapping]
        if not surviving:
            continue
        active_regions.append((surviving, why_text))
    active_regions.sort(key=lambda region: min(mapping[raw_id] for raw_id in region[0]))
    if len(active_regions) > kmax:
        raise ValueError(
            f"Augmented explanation sample ({_annotation_identity(raw)}) has {len(active_regions)} surviving WHY "
            f"episodes, exceeding Kmax={kmax}"
        )
    why_texts = [""] * kmax
    for slot, (raw_ids, why_text) in enumerate(active_regions):
        why_texts[slot] = why_text
        for raw_id in raw_ids:
            membership[mapping[raw_id], slot] = 1.0
    return {
        "query_text": query_text,
        "what_texts": what_texts,
        "why_membership": membership,
        "episode_count": len(active_regions),
        "why_texts": why_texts,
        "how_text": how_text if (not truncated or allow_truncated_how) else "",
        "raw_to_model_idx": mapping,
        "model_raw_indices": kept_raw_ids,
        "raw_length": raw_length,
        "how_truncated": truncated,
    }


def normalize_explanation_annotation(
    raw: Mapping[str, Any],
    max_length: int,
    kmax: int,
    model_raw_indices: Optional[Sequence[int]] = None,
    raw_to_model_idx: Optional[Mapping[int, int]] = None,
    allow_truncated_how: bool = False,
) -> dict[str, Any]:
    """Normalize inline augmented data, retaining the historical normalized form."""

    if "prediction" in raw or "X" in raw or "Y" in raw or "T" in raw:
        return normalize_augmented_explanation_annotation(
            raw,
            max_length=max_length,
            kmax=kmax,
            model_raw_indices=model_raw_indices,
            raw_to_model_idx=raw_to_model_idx,
            allow_truncated_how=allow_truncated_how,
        )

    required = ("query_text", "what_texts", "why_texts", "how_text")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Explanation annotation is missing required fields: {', '.join(missing)}")
    query_text = _required_text(raw["query_text"], "query_text")
    what_source = [_required_text(value, "WHAT") for value in raw["what_texts"]]
    why_texts = list(raw["why_texts"])
    how_text = _required_text(raw["how_text"], "HOW")
    if not what_source:
        raise ValueError("Explanation annotation has a zero-length scanpath")
    if "why_membership" in raw:
        membership = torch.as_tensor(raw["why_membership"], dtype=torch.float32)
        if membership.ndim != 2 or membership.shape[0] != len(what_source):
            raise ValueError("why_membership must cover the same real fixations as WHAT")
        if not torch.all((membership == 0) | (membership == 1)) or not torch.all(membership.sum(-1) == 1):
            raise ValueError("Each real fixation must have exactly one gold WHY episode")
        why_ids = membership.argmax(-1).tolist()
        active_ids = list(dict.fromkeys(why_ids))
        if len(why_texts) != membership.shape[1]:
            raise ValueError("why_texts must be indexed by membership column")
        text_by_id = dict(enumerate(why_texts))
    else:
        if "why_episode_ids" not in raw:
            raise ValueError("Explanation annotation requires why_episode_ids or why_membership")
        why_ids = [_fixation_id(value) for value in raw["why_episode_ids"]]
        if len(why_ids) != len(what_source) or any(value < 0 for value in why_ids):
            raise ValueError("Each real fixation must have exactly one gold WHY episode")
        active_ids = list(dict.fromkeys(why_ids))
        if len(why_texts) != len(active_ids):
            raise ValueError("why_texts must match episodes in first-occurrence order")
        text_by_id = dict(zip(active_ids, why_texts))
    regions = [([step + 1 for step, value in enumerate(why_ids) if value == episode],
                _required_text(text_by_id[episode], "WHY")) for episode in active_ids]
    return _align_explanation_targets(raw, query_text, what_source, regions, how_text,
                                      max_length, kmax, model_raw_indices, raw_to_model_idx,
                                      allow_truncated_how)


def load_explanation_annotations(path: str | Path) -> Any:
    """Load a user-configured annotation artifact without assuming its schema."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def lookup_explanation_annotation(payload: Any, fixation: Mapping[str, Any]) -> Mapping[str, Any]:
    """Find an annotation by an explicit record id or image/subject identity."""

    if "prediction" in fixation:
        return fixation
    if all(key in fixation for key in ("query_text", "what_texts", "how_text")):
        return fixation
    records = payload.get("annotations", payload) if isinstance(payload, Mapping) else payload
    explicit_id = fixation.get("sample_id", fixation.get("id"))
    if isinstance(records, Mapping):
        keyed = records.get(str(explicit_id), records.get(explicit_id)) if explicit_id is not None else None
        if keyed is not None:
            records = [dict(keyed, sample_id=explicit_id)]
        else:
            records = list(records.values())
    if not isinstance(records, list):
        raise KeyError(f"No explanation annotation for {_annotation_identity(fixation)}")
    def identity(record):
        return (record.get("name", record.get("image_id")),
                record.get("explanation_source_subject", record.get("subject_idx", record.get("subject"))),
                record.get("question_id", record.get("task")), record.get("condition"))
    matches = [record for record in records if isinstance(record, Mapping) and (
        record.get("sample_id", record.get("id")) == explicit_id if explicit_id is not None
        else identity(record) == identity(fixation))]
    if len(matches) != 1:
        raise ValueError(f"Expected one explanation annotation for {_annotation_identity(fixation)}, found {len(matches)}")
    annotation = matches[0]
    for field in ("X", "Y", "T"):
        if field in annotation and field in fixation and list(annotation[field]) != list(fixation[field]):
            raise ValueError(f"Sidecar {field} differs from raw scanpath for {_annotation_identity(fixation)}")
    if "what_texts" in annotation and len(annotation["what_texts"]) != len(fixation["X"]):
        raise ValueError("Sidecar WHAT count does not match the raw scanpath")
    return annotation


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

    expected = set(model.state_dict())
    supplied = set(state_dict)
    prefix = "explanation_module."
    missing = sorted(expected - supplied)
    unexpected = sorted(supplied - expected)
    fresh = explanation_enabled and not any(key.startswith(prefix) for key in supplied)
    unrelated = [key for key in missing if not (fresh and key.startswith(prefix))]
    unrelated += [key for key in unexpected if not (not explanation_enabled and key.startswith(prefix))]
    logger(f"checkpoint migration: explanation_enabled={explanation_enabled}, missing_keys={missing}, unexpected_keys={unexpected}")
    if unrelated:
        raise RuntimeError(f"Checkpoint incompatibility outside explanation keys or partial explanation checkpoint: {unrelated}")
    result = model.load_state_dict(state_dict, strict=False)
    if fresh and missing:
        logger("checkpoint migration: explanation parameters freshly initialized")
    return result


def restore_training_checkpoint(model, optimizer, checkpoint, explanation_enabled, logger=print):
    missing, unexpected = load_model_state_with_explanation_migration(
        model, checkpoint["model"], explanation_enabled, logger=logger)
    migrated = bool(missing or unexpected)
    if migrated:
        optimizer.state.clear()
        logger("checkpoint migration: optimizer and schedule reset for changed parameter set")
    elif "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return migrated
