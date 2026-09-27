import sys
from pathlib import Path

import torch
from torch import nn


SRC = Path(__file__).parents[1] / "ISP" / "OSIE" / "GazeformerISP" / "src"
sys.path.insert(0, str(SRC))

from models.explanation import (  # noqa: E402
    FixationReasoningEncoder,
    HierarchicalExplanationModule,
    HowR0Aggregator,
    WhyR0Router,
    final_spatial_fixation_distribution,
    normalize_explanation_annotation,
)


class FakeSemanticEncoder(nn.Module):
    output_dim = 6

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return torch.tensor(
            [[float(len(text) + offset) for offset in range(self.output_dim)] for text in texts],
            dtype=torch.float32,
        )


class FakeTokenizer:
    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        rows = [[(ord(char) % 7) + 1 for char in text[:4]] or [1] for text in texts]
        width = max(len(row) for row in rows)
        ids = torch.zeros((len(rows), width), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row_index, row in enumerate(rows):
            ids[row_index, : len(row)] = torch.tensor(row)
            mask[row_index, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class FakeCausalLM(nn.Module):
    class Config:
        hidden_size = 8

    config = Config()

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.readout = nn.Linear(8, 16)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask=None, labels=None):
        logits = self.readout(inputs_embeds)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return type("Output", (), {"loss": loss, "logits": logits})()


def test_spatial_probability_excludes_stop_and_backpropagates():
    logits = torch.randn(2, 3, 1 + 6, requires_grad=True)
    probability = final_spatial_fixation_distribution(logits)
    assert probability.shape == (2, 3, 6)
    assert torch.allclose(probability.sum(-1), torch.ones(2, 3), atol=1e-6)
    probability[..., 0].sum().backward()
    assert logits.grad is not None
    assert logits.grad[..., 1:].abs().sum() > 0
    assert torch.all(logits.grad[..., 0] == 0)


def test_soft_fixation_evidence_and_reasoning_token_gradients():
    encoder = FixationReasoningEncoder(5, 4, 7, spatial_dim=(2, 3))
    states = torch.randn(2, 4, 5, requires_grad=True)
    visual = torch.randn(2, 6, 4, requires_grad=True)
    action_logits = torch.randn(2, 4, 7, requires_grad=True)
    duration_mu = torch.randn(2, 4, requires_grad=True)
    duration_param2 = torch.randn(2, 4, requires_grad=True)
    output = encoder(states, visual, action_logits, duration_mu, duration_param2)
    assert output["soft_visual"].shape == (2, 4, 4)
    assert output["z"].shape == (2, 4, 7)
    output["z"].square().mean().backward()
    for value in (states, visual, action_logits, duration_mu, duration_param2):
        assert value.grad is not None and value.grad.abs().sum() > 0


def test_padding_is_excluded_from_route_and_episode_pooling():
    router = WhyR0Router(latent_dim=4, query_dim=3, kmax=3, max_length=4)
    z = torch.randn(1, 4, 4)
    query = torch.randn(1, 3)
    _, probability = router(z, query)
    mask = torch.tensor([[True, True, False, False]])
    membership = torch.zeros(1, 4, 3)
    membership[0, 0, 0] = 1
    membership[0, 1, 1] = 1
    base_tokens, _ = router.soft_episode_aggregation(z, probability, mask)
    altered = z.clone()
    altered[:, 2:] = 1000
    altered_tokens, _ = router.soft_episode_aggregation(altered, probability, mask)
    assert torch.allclose(base_tokens, altered_tokens)
    assert router.routing_loss(probability, membership, mask).isfinite()


def test_how_r0_uses_only_gold_active_prefix_slots():
    aggregator = HowR0Aggregator(latent_dim=4, query_dim=3, global_dim=5)
    query = torch.randn(1, 3)
    episodes = torch.randn(1, 4, 4)
    first = aggregator(episodes, query, torch.tensor([2]))["global_token"]
    changed = episodes.clone()
    changed[:, 2:] = changed[:, 2:] + 1000
    second = aggregator(changed, query, torch.tensor([2]))["global_token"]
    assert torch.allclose(first, second)


def test_full_hierarchy_has_shared_llm_and_how_gradient_path():
    causal_lm = FakeCausalLM()
    module = HierarchicalExplanationModule(
        decoder_dim=5,
        visual_dim=4,
        latent_dim=7,
        spatial_dim=(2, 3),
        max_length=4,
        kmax=3,
        semantic_encoder=FakeSemanticEncoder(),
        causal_lm=causal_lm,
        causal_lm_tokenizer=FakeTokenizer(),
        causal_lm_hidden_dim=8,
        lambda_how_txt=0.0,
        lambda_how_align=1.0,
    )
    assert module.shared_llm.llm is causal_lm
    assert module.shared_llm.latent_projections["what"] is not module.shared_llm.latent_projections["why"]
    states = torch.randn(2, 4, 5, requires_grad=True)
    visual = torch.randn(2, 6, 4, requires_grad=True)
    actions = torch.randn(2, 4, 7, requires_grad=True)
    mu = torch.randn(2, 4, requires_grad=True)
    param2 = torch.randn(2, 4, requires_grad=True)
    output = module(
        states,
        visual,
        actions,
        mu,
        param2,
        query_text=["query one", "query two"],
        fixation_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
        why_membership=torch.tensor(
            [
                [[1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]],
            ],
            dtype=torch.float32,
        ),
        why_text_targets=[["reason one", "reason two"], ["reason one", "reason two"]],
        episode_count=torch.tensor([2, 2]),
        how_target=["overall strategy", "overall strategy"],
        compute_generation=False,
        compute_alignment=True,
    )
    output["loss_how_align"].backward()
    assert module.how_aggregator.mlp[0].weight.grad is not None
    assert module.why_router.mlp[0].weight.grad is not None
    assert module.fixation_encoder.fixation_mlp[0].weight.grad is not None
    assert states.grad is not None and states.grad.abs().sum() > 0


def test_teacher_forced_generation_loss_is_available_for_all_branches():
    module = HierarchicalExplanationModule(
        decoder_dim=3,
        visual_dim=2,
        latent_dim=4,
        spatial_dim=(1, 2),
        max_length=2,
        kmax=2,
        semantic_encoder=FakeSemanticEncoder(),
        causal_lm=FakeCausalLM(),
        causal_lm_tokenizer=FakeTokenizer(),
        causal_lm_hidden_dim=8,
    )
    output = module(
        torch.randn(1, 2, 3),
        torch.randn(1, 2, 2),
        torch.randn(1, 2, 3),
        torch.randn(1, 2),
        torch.randn(1, 2),
        ["query"],
        torch.tensor([[True, True]]),
        what_targets=[["what", ""]],
        why_membership=torch.tensor([[[1, 0], [1, 0]]], dtype=torch.float32),
        why_text_targets=[["why", ""]],
        episode_count=torch.tensor([1]),
        how_target=["how"],
        compute_generation=True,
        compute_alignment=False,
    )
    assert output["loss_what_txt"].isfinite()
    assert output["loss_why_txt"].isfinite()
    assert output["loss_how_txt"].isfinite()


def test_annotation_normalization_canonicalizes_and_pads():
    annotation = normalize_explanation_annotation(
        {
            "query_text": "a query",
            "what_texts": ["w0", "w1", "w2"],
            "why_episode_ids": [7, 3, 7],
            "why_texts": ["episode seven", "episode three"],
            "how_text": "a plan",
        },
        max_length=4,
        kmax=3,
    )
    assert annotation["episode_count"] == 2
    assert annotation["why_membership"].shape == (4, 3)
    assert annotation["why_membership"][0].tolist() == [1, 0, 0]
    assert annotation["why_membership"][1].tolist() == [0, 1, 0]
    assert annotation["what_texts"][-1] == ""

