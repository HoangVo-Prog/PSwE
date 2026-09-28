import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_air_and_augmented import augmented_record
from test_explanation_core import FakeSemanticEncoder, FakeCausalLM, FakeTokenizer
from models.explanation import (
    normalize_explanation_annotation, lookup_explanation_annotation,
    WhyR0Router, HierarchicalExplanationModule,
    load_model_state_with_explanation_migration, restore_training_checkpoint,
)
from models.explanation_llm import SemanticEncoderWrapper, SharedExplanationLLM


@pytest.mark.parametrize("bad_id", [True, 1.9, "1"])
def test_raw_ids_are_not_silently_coerced(bad_id):
    record = augmented_record(3)
    record["prediction"]["fixations"][0]["fixation"] = bad_id
    with pytest.raises(ValueError, match="integer"):
        normalize_explanation_annotation(record, 4, 3)


@pytest.mark.parametrize("branch", ["what", "why", "how"])
def test_empty_annotations_fail(branch):
    record = augmented_record(3)
    if branch == "what":
        record["prediction"]["fixations"][0]["what"] = " "
    elif branch == "why":
        record["prediction"]["regions"][0]["why"] = None
    else:
        record["prediction"]["how"] = ""
    with pytest.raises(ValueError, match="nonempty"):
        normalize_explanation_annotation(record, 4, 3)


@pytest.mark.parametrize("mutation", ["overlap", "bounds", "missing", "empty", "length"])
def test_invalid_partition_and_lengths_fail(mutation):
    record = augmented_record(3)
    regions = record["prediction"]["regions"]
    if mutation == "overlap":
        regions[1]["fixations"].append(1)
    elif mutation == "bounds":
        regions[0]["fixations"].append(9)
    elif mutation == "missing":
        regions.pop()
    elif mutation == "empty":
        regions[0]["fixations"] = []
    else:
        record["T"].pop()
    with pytest.raises(ValueError):
        normalize_explanation_annotation(record, 4, 3)


def test_zero_length_and_invalid_discarded_mapping_fail():
    record = augmented_record(3)
    for indices in ([1, 2, 2], [1, 2, 9]):
        with pytest.raises(ValueError):
            normalize_explanation_annotation(record, 2, 3, model_raw_indices=indices)
    record["X"], record["Y"], record["T"] = [], [], []
    with pytest.raises(ValueError, match="zero-length"):
        normalize_explanation_annotation(record, 4, 3)


def test_surviving_episodes_reorder_by_first_model_fixation():
    result = normalize_explanation_annotation(augmented_record(5), 3, 3,
                                              model_raw_indices=[2, 3, 4], allow_truncated_how=True)
    assert result["why_texts"] == ["second", "first", ""]
    assert result["why_membership"].argmax(-1).tolist() == [0, 1, 0]
    assert result["what_texts"] == ["what-2", "what-3", "what-4"]


def test_legacy_sidecars_use_mapping_and_recompute_surviving_slots():
    record = {"query_text": "query", "what_texts": ["a", "b", "c"],
              "why_episode_ids": [7, 3, 9], "why_texts": ["seven", "three", "nine"], "how_text": "all"}
    with pytest.raises(ValueError, match="truncated"):
        normalize_explanation_annotation(record, 2, 3)
    result = normalize_explanation_annotation(record, 2, 3, model_raw_indices=[2, 3], allow_truncated_how=True)
    assert result["what_texts"] == ["b", "c"]
    assert result["episode_count"] == 2
    assert result["why_texts"] == ["three", "nine", ""]


def test_sidecar_identity_includes_question_subject_and_condition():
    records = [{"image_id": "same.jpg", "question_id": question, "subject_idx": subject}
               for question in (10, 20) for subject in (0, 1)]
    fixation = dict(records[-1], X=[1])
    assert lookup_explanation_annotation(records, fixation) is records[-1]
    with pytest.raises(ValueError, match="found 2"):
        lookup_explanation_annotation(records + [records[-1]], fixation)


def test_router_time_is_independent_of_batch_padding():
    router = WhyR0Router(4, 3, 3, 6)
    latents = torch.randn(1, 3, 4)
    query = torch.randn(1, 3)
    short = router(latents, query, torch.ones(1, 3, dtype=torch.bool))[1]
    padded = torch.cat((latents, torch.randn(1, 3, 4)), dim=1)
    long = router(padded, query, torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.bool))[1]
    torch.testing.assert_close(short, long[:, :3])


def test_episode_pool_matches_soft_weighted_mean():
    latents = torch.tensor([[[1., 2.], [5., 6.], [900., 900.]]])
    routing = torch.tensor([[[.2, .8], [.6, .4], [.5, .5]]])
    pooled, mass = WhyR0Router.soft_episode_aggregation(latents, routing, torch.tensor([[1, 1, 0]]))
    torch.testing.assert_close(mass, torch.tensor([[.8, 1.2]]))
    torch.testing.assert_close(pooled[0, 0], torch.tensor([4., 5.]))


class DropoutEncoder(nn.Module):
    output_dim = 6

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(6))
        self.dropout = nn.Dropout(.8)

    def encode(self, texts):
        return self.dropout(self.weight.expand(len(texts), -1))


def test_frozen_semantics_and_gold_targets_are_deterministic():
    frozen = SemanticEncoderWrapper(encoder=DropoutEncoder(), freeze=True)
    frozen.train()
    assert not frozen.encoder.training
    torch.testing.assert_close(frozen.encode(["gold"]), frozen.encode(["gold"]))
    trainable = SemanticEncoderWrapper(encoder=DropoutEncoder(), freeze=False)
    trainable.train()
    target = trainable.encode_target(["gold"])
    assert not target.requires_grad and trainable.encoder.training
    assert trainable.encode(["query"]).requires_grad


def test_frozen_llm_keeps_prefix_gradients_and_masks_padding():
    class RecordingLM(FakeCausalLM):
        def forward(self, **kwargs):
            self.labels = kwargs["labels"]
            return super().forward(**kwargs)
    causal = RecordingLM()
    wrapper = SharedExplanationLLM(6, 7, 7, llm=causal, tokenizer=FakeTokenizer(), freeze=True)
    wrapper.train()
    query = torch.randn(2, 6, requires_grad=True)
    latent = torch.randn(2, 7, requires_grad=True)
    loss = wrapper.teacher_forced_loss("what", query, latent, ["a", "long"])
    loss.backward()
    assert not causal.training
    assert torch.all(causal.labels[:, :3] == -100)
    assert torch.all(causal.labels[0, 4:] == -100)
    assert latent.grad.abs().sum() > 0 and query.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in causal.parameters())


def test_partial_explanation_checkpoint_is_not_fresh_migration():
    model = nn.Module()
    model.base = nn.Linear(2, 2)
    model.explanation_module = nn.Linear(2, 2)
    state = dict(model.state_dict())
    del state["explanation_module.weight"]
    with pytest.raises(RuntimeError, match="partial explanation"):
        load_model_state_with_explanation_migration(model, state, True)


def test_optimizer_resets_only_for_base_to_explanation_migration():
    base = nn.Module()
    base.base = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(base.parameters())
    checkpoint = {"model": base.state_dict(), "optimizer": optimizer.state_dict()}
    base.explanation_module = nn.Linear(2, 2)
    extended = torch.optim.AdamW(base.parameters())
    assert restore_training_checkpoint(base, extended, checkpoint, True)
    assert not extended.state
    checkpoint = {"model": base.state_dict(), "optimizer": extended.state_dict()}
    assert not restore_training_checkpoint(base, extended, checkpoint, True)


@pytest.mark.parametrize("zero_weights", [False, True])
def test_weighted_hierarchical_objective_and_zero_weights(zero_weights):
    weights = dict(lambda_exp_what=2., lambda_exp_why=3., lambda_exp_how=4.,
                   lambda_what_txt=5., lambda_what_align=6., lambda_route=7.,
                   lambda_why_txt=8., lambda_why_align=9., lambda_how_txt=10., lambda_how_align=11.)
    if zero_weights:
        weights = {name: 0. for name in weights}
    module = HierarchicalExplanationModule(4, 4, 8, (2, 2), 3, 3,
                                           semantic_encoder=FakeSemanticEncoder(), causal_lm=FakeCausalLM(),
                                           causal_lm_tokenizer=FakeTokenizer(), **weights)
    states = torch.randn(2, 3, 4, requires_grad=True)
    membership = torch.zeros(2, 3, 3)
    membership[:, 0, 0], membership[:, 1, 1] = 1, 1
    output = module(states, torch.randn(2, 4, 4), torch.randn(2, 3, 5),
                    torch.randn(2, 3), torch.ones(2, 3), ["query", "task"],
                    torch.tensor([[1, 1, 0], [1, 1, 0]]),
                    what_targets=[["one", "two", "ignored"], ["one", "two", "ignored"]],
                    why_membership=membership, why_text_targets=[["why1", "why2", "ignored"], ["why1", "why2", "ignored"]],
                    episode_count=torch.tensor([2, 2]), how_target=["whole", "all"])
    expected = (weights["lambda_exp_what"] * (weights["lambda_what_txt"] * output["loss_what_txt"] + weights["lambda_what_align"] * output["loss_what_align"])
                + weights["lambda_exp_why"] * (weights["lambda_route"] * output["loss_route"] + weights["lambda_why_txt"] * output["loss_why_txt"] + weights["lambda_why_align"] * output["loss_why_align"])
                + weights["lambda_exp_how"] * (weights["lambda_how_txt"] * output["loss_how_txt"] + weights["lambda_how_align"] * output["loss_how_align"]))
    torch.testing.assert_close(output["loss_explanation"], expected)
    output["loss_explanation"].backward()
    assert torch.all(states.grad[:, 2] == 0)
    if zero_weights:
        assert output["loss_explanation"] == 0 and torch.all(states.grad == 0)
    else:
        assert states.grad[:, :2].abs().sum() > 0
