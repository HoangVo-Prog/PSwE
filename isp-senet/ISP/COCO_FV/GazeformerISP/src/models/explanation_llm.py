"""Optional semantic and causal-language-model adapters for COCO-FV."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping, Optional, Sequence

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
    def __init__(self, encoder=None, model_name=None, freeze=True, output_dim=None, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer
        if encoder is None:
            if not model_name:
                raise ValueError("Explanation requires an injected semantic encoder or --semantic_encoder_name")
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError("Install transformers or inject an offline semantic encoder") from exc
            self.encoder = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.encoder = encoder
        inferred = output_dim or getattr(self.encoder, "output_dim", None)
        inferred = inferred or getattr(getattr(self.encoder, "config", None), "hidden_size", None)
        if inferred is None:
            raise ValueError("Pass semantic_encoder_dim or expose encoder.output_dim/config.hidden_size")
        self.output_dim = int(inferred)
        self.freeze = bool(freeze)
        if self.freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)

    def encode(self, texts: Sequence[str] | str) -> Tensor:
        texts = _as_text_list(texts)
        context = torch.no_grad() if self.freeze else nullcontext()
        with context:
            encode_method = getattr(self.encoder, "encode", None)
            if callable(encode_method):
                try:
                    output = encode_method(texts)
                except TypeError:
                    output = encode_method(texts, convert_to_tensor=True)
            else:
                if self.tokenizer is None:
                    raise ValueError("Semantic encoder needs tokenizer or encode(texts)")
                device = next(self.encoder.parameters(), torch.empty(0)).device
                encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                encoded = {key: value.to(device) if torch.is_tensor(value) else value for key, value in encoded.items()}
                output = self.encoder(**encoded)
                hidden = _extract_tensor(output)
                if hidden is not None and hidden.ndim == 3:
                    mask = encoded.get("attention_mask")
                    weights = torch.ones(hidden.shape[:2], device=hidden.device) if mask is None else mask.to(hidden.dtype)
                    output = (hidden * weights.unsqueeze(-1)).sum(1) / weights.sum(1).clamp_min(1).unsqueeze(-1)
            result = _extract_tensor(output)
        if result is None:
            raise TypeError("Semantic encoder did not return a tensor")
        if result.ndim == 1:
            result = result.unsqueeze(0)
        if result.ndim != 2 or result.shape[-1] != self.output_dim:
            raise ValueError(f"Semantic encoder must return [batch,{self.output_dim}], got {tuple(result.shape)}")
        return result

    forward = encode


def _hidden_size(llm, configured):
    if configured is not None:
        return int(configured)
    size = getattr(getattr(llm, "config", None), "hidden_size", None)
    size = size or getattr(getattr(llm, "config", None), "n_embd", None)
    if size is None and hasattr(llm, "get_input_embeddings"):
        size = getattr(llm.get_input_embeddings(), "embedding_dim", None)
    if size is None:
        raise ValueError("Pass explanation_llm_hidden_dim or expose causal LLM hidden size")
    return int(size)


class SharedExplanationLLM(nn.Module):
    BRANCHES = ("what", "why", "how")

    def __init__(self, semantic_dim, latent_dim, how_dim, llm=None, model_name=None,
                 hidden_dim=None, freeze=True, tokenizer=None, query_dim=None):
        super().__init__()
        if llm is None:
            if not model_name:
                raise ValueError("Explanation requires an injected causal_lm or --explanation_llm_name")
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError("Install transformers or inject an offline causal LLM") from exc
            llm = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.llm = llm
        self.tokenizer = tokenizer
        self.hidden_dim = _hidden_size(llm, hidden_dim)
        query_dim = int(query_dim or semantic_dim)
        self.branch_tokens = nn.ParameterDict({branch: nn.Parameter(torch.empty(self.hidden_dim)) for branch in self.BRANCHES})
        self.query_projections = nn.ModuleDict({branch: nn.Linear(query_dim, self.hidden_dim) for branch in self.BRANCHES})
        self.latent_projections = nn.ModuleDict({
            "what": nn.Linear(latent_dim, self.hidden_dim),
            "why": nn.Linear(latent_dim, self.hidden_dim),
            "how": nn.Linear(how_dim, self.hidden_dim),
        })
        for token in self.branch_tokens.values():
            nn.init.normal_(token, std=0.02)
        self.freeze = bool(freeze)
        if self.freeze:
            for parameter in self.llm.parameters():
                parameter.requires_grad_(False)

    def _prefix(self, branch, query, latent):
        token = self.branch_tokens[branch].to(device=latent.device, dtype=latent.dtype).view(1, 1, -1).expand(latent.shape[0], -1, -1)
        return torch.cat((token, self.query_projections[branch](query).unsqueeze(1), self.latent_projections[branch](latent).unsqueeze(1)), 1)

    def _tokenize(self, texts, device):
        if self.tokenizer is not None:
            result = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        else:
            tokenize_method = getattr(self.llm, "tokenize", None)
            if not callable(tokenize_method):
                raise ValueError("The configured causal LLM needs an injected tokenizer or tokenize(texts) method")
            result = tokenize_method(list(texts))
        if isinstance(result, Mapping):
            ids = result["input_ids"]
            mask = result.get("attention_mask", torch.ones_like(ids))
        elif isinstance(result, (tuple, list)):
            ids, mask = result[0], result[1]
        else:
            ids, mask = result, torch.ones_like(result)
        return ids.to(device=device, dtype=torch.long), mask.to(device=device)

    def teacher_forced_loss(self, branch, query, latent, texts):
        if not texts:
            return latent.sum() * 0
        prefix = self._prefix(branch, query, latent)
        ids, mask = self._tokenize(texts, prefix.device)
        embeds = self.llm.get_input_embeddings()(ids)
        inputs = torch.cat((prefix, embeds), 1)
        prefix_mask = torch.ones((prefix.shape[0], prefix.shape[1]), dtype=mask.dtype, device=prefix.device)
        labels = torch.cat((torch.full((ids.shape[0], prefix.shape[1]), -100, dtype=torch.long, device=prefix.device), ids.masked_fill(mask == 0, -100)), 1)
        outputs = self.llm(inputs_embeds=inputs, attention_mask=torch.cat((prefix_mask, mask), 1), labels=labels)
        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, Mapping):
            loss = outputs.get("loss")
        if loss is None:
            logits = _extract_tensor(outputs)
            shifted_logits, shifted_labels = logits[:, :-1], labels[:, 1:]
            loss = F.cross_entropy(shifted_logits.reshape(-1, shifted_logits.shape[-1]), shifted_labels.reshape(-1), ignore_index=-100)
        return loss


def cosine_alignment_loss(predicted, target, mask=None):
    target = target.detach()
    values = 1 - (F.normalize(predicted, dim=-1) * F.normalize(target, dim=-1)).sum(-1)
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    return values.sum() * 0 if mask.sum() == 0 else (values * mask).sum() / mask.sum()

