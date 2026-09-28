"""Optional semantic and causal-language-model adapters for explanations.

The predictor does not import this module unless explanations are enabled.  All
model loading is therefore lazy and dependency-injection friendly.  Small fake
encoders and causal LMs can be supplied by CPU/offline tests.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _as_text_list(texts: Any) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return [str(text) for text in texts]


def _extract_tensor(value: Any) -> Optional[Tensor]:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for key in ("sentence_embedding", "pooler_output", "last_hidden_state", "embeddings", "logits"):
            if key in value and torch.is_tensor(value[key]):
                return value[key]
    for key in ("sentence_embedding", "pooler_output", "last_hidden_state", "embeddings", "logits"):
        candidate = getattr(value, key, None)
        if torch.is_tensor(candidate):
            return candidate
    if isinstance(value, (tuple, list)):
        for candidate in value:
            tensor = _extract_tensor(candidate)
            if tensor is not None:
                return tensor
    return None


class SemanticEncoderWrapper(nn.Module):
    """Explicit RoBERTa-role semantic encoder.

    ``encoder`` is the preferred path for tests and experiments.  When it is
    omitted, ``model_name`` must be supplied and the Hugging Face dependency is
    imported only at construction time.  The wrapper accepts either an
    ``encode(texts)`` object or a tokenizer/model pair.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        model_name: Optional[str] = None,
        freeze: bool = True,
        output_dim: Optional[int] = None,
        tokenizer: Any = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        if encoder is None:
            if not model_name:
                raise ValueError(
                    "Explanation is enabled but no semantic encoder was configured. "
                    "Pass an injected encoder or --semantic_encoder_name."
                )
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "Explanation requires transformers for the configured semantic "
                    "encoder; install it or inject an offline encoder."
                ) from exc
            self.encoder = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.encoder = encoder

        inferred_dim = output_dim
        if inferred_dim is None:
            inferred_dim = getattr(self.encoder, "output_dim", None)
        if inferred_dim is None:
            inferred_dim = getattr(getattr(self.encoder, "config", None), "hidden_size", None)
        if inferred_dim is None:
            raise ValueError(
                "Could not determine semantic encoder output dimension. "
                "Pass semantic_encoder_dim or expose encoder.output_dim/config.hidden_size."
            )
        self.output_dim = int(inferred_dim)
        self.freeze = bool(freeze)
        if self.freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            self.encoder.eval()

    def _tokenized_encode(self, texts: list[str]) -> Tensor:
        if self.tokenizer is None:
            raise ValueError(
                "The configured semantic encoder has no encode method or tokenizer. "
                "Inject an encoder with encode(texts) for offline use."
            )
        device = next(self.encoder.parameters(), torch.empty(0)).device
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(device) if torch.is_tensor(value) else value for key, value in encoded.items()}
        output = self.encoder(**encoded)
        hidden = _extract_tensor(output)
        if hidden is None:
            raise TypeError("Semantic encoder output does not contain a tensor representation")
        if hidden.ndim == 3:
            mask = encoded.get("attention_mask")
            if mask is None:
                hidden = hidden.mean(dim=1)
            else:
                weights = mask.to(hidden.dtype).unsqueeze(-1)
                hidden = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return hidden

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def encode_target(self, texts):
        training = self.encoder.training
        self.encoder.eval()
        try:
            with torch.no_grad():
                return self.encode(texts).detach()
        finally:
            self.encoder.train(training and not self.freeze)

    def encode(self, texts: Sequence[str] | str) -> Tensor:
        text_list = _as_text_list(texts)
        encode_method = getattr(self.encoder, "encode", None)
        context = torch.no_grad() if self.freeze else nullcontext()
        with context:
            if callable(encode_method):
                try:
                    value = encode_method(text_list)
                except TypeError:
                    value = encode_method(text_list, convert_to_tensor=True)
                result = _extract_tensor(value)
            else:
                result = self._tokenized_encode(text_list)
        if result is None:
            raise TypeError("Semantic encoder encode() did not return a tensor representation")
        if result.ndim == 1:
            result = result.unsqueeze(0)
        if result.ndim != 2:
            raise ValueError(f"Semantic encoder must return [batch, dim], got {tuple(result.shape)}")
        if result.shape[-1] != self.output_dim:
            raise ValueError(
                f"Semantic encoder dimension changed from {self.output_dim} to {result.shape[-1]}"
            )
        return result

    def forward(self, texts: Sequence[str] | str) -> Tensor:
        return self.encode(texts)


def _get_hidden_size(llm: nn.Module, configured: Optional[int]) -> int:
    if configured is not None:
        return int(configured)
    config_size = getattr(getattr(llm, "config", None), "hidden_size", None)
    if config_size is None:
        config_size = getattr(getattr(llm, "config", None), "n_embd", None)
    if config_size is None:
        embedding = llm.get_input_embeddings() if hasattr(llm, "get_input_embeddings") else None
        config_size = getattr(embedding, "embedding_dim", None)
    if config_size is None:
        raise ValueError(
            "Could not determine causal LLM hidden size. "
            "Pass explanation_llm_hidden_dim or use a model exposing config.hidden_size."
        )
    return int(config_size)


class SharedExplanationLLM(nn.Module):
    """One causal LLM with branch-specific latent prefixes."""

    BRANCHES = ("what", "why", "how")

    def __init__(
        self,
        semantic_dim: int,
        latent_dim: int,
        how_dim: int,
        llm: Optional[nn.Module] = None,
        model_name: Optional[str] = None,
        hidden_dim: Optional[int] = None,
        freeze: bool = True,
        tokenizer: Any = None,
        query_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if llm is None:
            if not model_name:
                raise ValueError(
                    "Explanation is enabled but no causal LLM was configured. "
                    "Pass an injected causal_lm or --explanation_llm_name."
                )
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - environment dependent
                raise ImportError(
                    "Explanation requires transformers for the configured causal LLM; "
                    "install it or inject an offline causal_lm."
                ) from exc
            llm = AutoModelForCausalLM.from_pretrained(model_name)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = llm
        self.tokenizer = tokenizer
        if tokenizer is not None and hasattr(tokenizer, "pad_token_id"):
            if tokenizer.pad_token_id is None:
                if getattr(tokenizer, "eos_token_id", None) is None:
                    raise ValueError("Causal tokenizer requires a padding token or an EOS token")
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        self.hidden_dim = _get_hidden_size(llm, hidden_dim)
        self.query_dim = int(query_dim or semantic_dim)
        self.branch_tokens = nn.ParameterDict({
            branch: nn.Parameter(torch.empty(self.hidden_dim)) for branch in self.BRANCHES
        })
        self.query_projections = nn.ModuleDict({
            branch: nn.Linear(self.query_dim, self.hidden_dim) for branch in self.BRANCHES
        })
        self.latent_projections = nn.ModuleDict({
            "what": nn.Linear(latent_dim, self.hidden_dim),
            "why": nn.Linear(latent_dim, self.hidden_dim),
            "how": nn.Linear(how_dim, self.hidden_dim),
        })
        for parameter in self.branch_tokens.values():
            nn.init.normal_(parameter, std=0.02)
        self.freeze = bool(freeze)
        if self.freeze:
            for parameter in self.llm.parameters():
                parameter.requires_grad_(False)
            self.llm.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.llm.eval()
        return self

    def _prefix(self, branch: str, query: Tensor, latent: Tensor) -> Tensor:
        if branch not in self.BRANCHES:
            raise KeyError(f"Unknown explanation branch {branch!r}")
        token = self.branch_tokens[branch].to(device=latent.device, dtype=latent.dtype)
        token = token.view(1, 1, -1).expand(latent.shape[0], -1, -1)
        query_token = self.query_projections[branch](query)
        latent_token = self.latent_projections[branch](latent)
        return torch.cat((token, query_token.unsqueeze(1), latent_token.unsqueeze(1)), dim=1)

    def _tokenize(self, texts: Sequence[str], device: torch.device) -> tuple[Tensor, Tensor]:
        if self.tokenizer is None:
            tokenize_method = getattr(self.llm, "tokenize", None)
            if callable(tokenize_method):
                result = tokenize_method(list(texts))
            else:
                raise ValueError(
                    "The configured causal LLM has no tokenizer. Inject tokenizer or "
                    "a causal_lm.tokenize(texts) method for teacher forcing."
                )
        else:
            result = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        if isinstance(result, Mapping):
            input_ids = result["input_ids"]
            attention_mask = result.get("attention_mask", torch.ones_like(input_ids))
        elif isinstance(result, (tuple, list)) and len(result) >= 2:
            input_ids, attention_mask = result[0], result[1]
        else:
            input_ids = result
            attention_mask = torch.ones_like(input_ids)
        return input_ids.to(device=device, dtype=torch.long), attention_mask.to(device=device)

    def teacher_forced_loss(
        self,
        branch: str,
        query: Tensor,
        latent: Tensor,
        texts: Sequence[str],
    ) -> Tensor:
        if len(texts) == 0:
            return latent.sum() * 0.0
        prefix = self._prefix(branch, query, latent)
        input_ids, attention_mask = self._tokenize(texts, prefix.device)
        embedding_layer = self.llm.get_input_embeddings()
        token_embeds = embedding_layer(input_ids)
        inputs_embeds = torch.cat((prefix, token_embeds), dim=1)
        prefix_mask = torch.ones(
            (prefix.shape[0], prefix.shape[1]), dtype=attention_mask.dtype, device=prefix.device
        )
        full_attention_mask = torch.cat((prefix_mask, attention_mask), dim=1)
        ignored_prefix = torch.full(
            (input_ids.shape[0], prefix.shape[1]), -100, dtype=torch.long, device=prefix.device
        )
        labels = torch.cat((ignored_prefix, input_ids.masked_fill(attention_mask == 0, -100)), dim=1)
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_attention_mask, labels=labels)
        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, Mapping):
            loss = outputs.get("loss")
        if loss is None:
            logits = _extract_tensor(outputs)
            if logits is None or logits.ndim != 3:
                raise TypeError("Causal LLM must return loss or [batch, sequence, vocabulary] logits")
            shifted_logits = logits[:, :-1].contiguous()
            shifted_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]),
                shifted_labels.view(-1),
                ignore_index=-100,
            )
        return loss


def cosine_alignment_loss(predicted: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Cosine alignment with an explicit stop-gradient target contract."""

    if predicted.numel() == 0:
        return predicted.sum() * 0.0
    target = target.detach()
    predicted = F.normalize(predicted, dim=-1)
    target = F.normalize(target, dim=-1)
    values = 1.0 - (predicted * target).sum(dim=-1)
    if mask is not None:
        mask = mask.to(device=values.device, dtype=values.dtype)
        if mask.sum() == 0:
            return values.sum() * 0.0
        return (values * mask).sum() / mask.sum()
    return values.mean()

