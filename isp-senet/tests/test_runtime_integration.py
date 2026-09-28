import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("dataset", ["OSIE", "COCO_FV", "COCO_Search18", "AiR"])
def test_real_predictor_hierarchy_and_disabled_parity(dataset, tmp_path):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "ISP" / dataset / "GazeformerISP" / "src")
    result = subprocess.run([sys.executable, __file__, dataset, str(tmp_path)],
                            env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


def run_predictor(dataset, directory):
    sys.modules["transformers"] = None
    sys.modules["torchvision"] = None
    from models.models import Transformer
    from models.gazeformer import gazeformer
    from models.loss import CrossEntropyLoss, MLPLogNormalDistribution
    from models.sampling import Sampling
    from models.explanation import load_model_state_with_explanation_migration
    from test_explanation_core import FakeSemanticEncoder, FakeCausalLM, FakeTokenizer

    torch.manual_seed(0)
    torch.set_num_threads(1)
    table_path = directory / "subjects.pt"
    torch.save(torch.randn(2, 4), table_path)
    args = SimpleNamespace(enable_explanation=False, user_emb_path=str(table_path), nhead=2,
                           hidden_dim=8, subject_feature_dim=4, im_h=2, im_w=2,
                           encoder_dropout=0., decoder_dropout=0., return_explanation_latents=False,
                           generate_explanations=False)
    def build():
        transformer = Transformer(d_model=8, img_hidden_dim=6, subject_feature_dim=4,
                                  lm_dmodel=6, nhead=2, num_encoder_layers=1, num_decoder_layers=1,
                                  dim_feedforward=8, encoder_dropout=0., decoder_dropout=0.,
                                  device="cpu", args=args)
        return gazeformer(transformer, (2, 2), args, 2, 4, 2, dropout=0., max_len=3, device="cpu")
    base = build()
    images, subjects, task = torch.randn(2, 4, 6), torch.tensor([0, 1]), torch.randn(2, 6)
    disabled = base(images, subjects, task)
    assert set(disabled) == {"actions", "log_normal_mu", "log_normal_sigma2", "action_map"}
    assert not any(name.startswith("explanation_module.") for name in base.state_dict())
    state = base.state_dict()
    args.enable_explanation = True
    args.explanation_semantic_encoder = FakeSemanticEncoder()
    args.explanation_causal_lm = FakeCausalLM()
    args.explanation_llm_tokenizer = FakeTokenizer()
    args.explanation_dim = 8
    args.router_kmax = 3
    args.how_hidden_dim = 8
    enabled = build()
    load_model_state_with_explanation_migration(enabled, state, True)
    membership = torch.zeros(2, 3, 3)
    membership[:, 0, 0] = 1
    membership[:, 1, 1] = 1
    inputs = dict(query_text=["query", "another query"], fixation_mask=torch.tensor([[1, 1, 0], [1, 1, 0]]),
                  what_texts=[["first", "second", ""], ["first", "second", ""]],
                  why_membership=membership, why_texts=[["why a", "why b", ""], ["why a", "why b", ""]],
                  episode_count=torch.tensor([2, 2]), how_text=["whole", "whole"])
    output = enabled(images, subjects, task, explanation_inputs=inputs)
    for name in disabled:
        torch.testing.assert_close(disabled[name], output[name])
    torch.testing.assert_close(output["explanation"]["fixation_prob"], output["actions"][..., 1:].softmax(-1))
    output["explanation"]["loss_how"].backward()
    for module in (enabled.transformer.encoder, enabled.transformer.decoder,
                   enabled.explanation_module.fixation_encoder, enabled.explanation_module.why_router,
                   enabled.explanation_module.how_aggregator):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())
    base.eval()
    enabled.eval()
    base_output = base(images, subjects, task)
    enabled_output = enabled(images, subjects, task)
    for name in base_output:
        torch.testing.assert_close(base_output[name], enabled_output[name])
    enabled.zero_grad()
    enabled_output["all_actions_prob"][..., 1].sum().backward()
    assert any(parameter.grad is not None for parameter in enabled.transformer.parameters())
    assert all(parameter.grad is None for parameter in enabled.explanation_module.parameters())
    sampler = Sampling(convLSTM_length=3, min_length=1, map_width=2, map_height=2, width=32, height=32)
    sampled = sampler.random_sample(base_output["all_actions_prob"], base_output["log_normal_mu"], base_output["log_normal_sigma2"])
    assert sampled["durations"].shape == (2, 3)
    predicted, _, _ = sampler.generate_scanpath(images, sampled["selected_actions_probs"], sampled["durations"], sampled["selected_actions"])
    assert predicted[0].dtype.names == ("start_x", "start_y", "duration")
    if dataset == "AiR":
        run_air_lifecycle(directory, enabled)
    else:
        run_ofc_dataset(dataset, directory)


def run_ofc_dataset(dataset, directory):
    sys.path.insert(0, str(ROOT / "ISP" / dataset / "GazeformerISP/src"))
    from dataset import dataset as loaders
    task = "free-viewing" if dataset == "OSIE" else "target"
    features = directory / "features"
    image_directory = features if dataset == "OSIE" else features / task
    image_directory.mkdir(parents=True)
    torch.save(torch.randn(4, 6), image_directory / "image.pth")
    np.save(directory / "embeddings.npy", {"free-viewing": np.ones(6, dtype=np.float32),
                                          "target": np.ones(6, dtype=np.float32)})
    records = [dict(name="image.jpg", subject=subject, task=task, condition="absent", split="train",
                    X=[1., 17., 20.], Y=[1., 1., 20.], T=[50., 100., 150.],
                    prediction={"fixations": [{"fixation": index, "what": str(index)} for index in (1, 2, 3)],
                                "regions": [{"fixations": [1, 3], "why": "one"}, {"fixations": [2], "why": "two"}],
                                "how": "all"}) for subject in (0, 1)]
    path = directory / "fixations.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    args = SimpleNamespace(enable_explanation=False, ex_subject=[-1], fewshot_subject=[-1], router_kmax=3,
                           explanation_annotations=None, allow_truncated_how=False)
    dataset_type = loaders.OSIE if dataset == "OSIE" else loaders.COCOSearch
    data = dataset_type(args, "", str(features), str(path), str(directory / "embeddings.npy"),
                        origin_size=(32, 32), resize=(32, 32), action_map=(2, 2), max_length=4, blur_sigma=None)
    base = data[0]
    assert base["target_scanpath"][0].argmax(-1).tolist() == [1, 2, 4, 0]
    data.explanation_enabled = True
    enabled = data[0]
    for key in ("target_scanpath", "duration", "action_mask", "duration_mask", "subject", "task_embedding"):
        np.testing.assert_array_equal(base[key], enabled[key])
    annotation = enabled["explanation_annotations"][0]
    assert annotation["raw_to_model_idx"] == {1: 0, 2: 1, 3: 2}
    assert annotation["what_texts"] == ["1", "2", "3", ""]
    assert annotation["why_membership"][3].sum() == 0
    batch = data.collate_func([enabled, enabled])
    assert batch["images"].shape == (1, 4, 4, 6)
    assert len(batch["explanation_annotations"]) == 2
    data.max_length = 2
    try:
        data[0]
    except ValueError as error:
        assert "truncated" in str(error)
    else:
        raise AssertionError("Implicit full-trajectory HOW on truncation")
    args.allow_truncated_how = True
    assert data[0]["explanation_annotations"][0]["raw_to_model_idx"] == {1: 0, 2: 1}
    data.fixations[0]["T"].pop()
    try:
        data[0]
    except ValueError as error:
        assert "len(X)" in str(error)
    else:
        raise AssertionError("Malformed raw sequence was accepted")


def run_air_lifecycle(directory, explanation_model):
    sys.path.insert(0, str(ROOT / "ISP/AiR/GazeformerISP/src"))
    from train import main, supervised_step
    from test import run
    from dataset.dataset import AiR, AiR_evaluation
    feature_dir = directory / "features"
    feature_dir.mkdir()
    torch.save(torch.randn(4, 6), feature_dir / "image.pth")
    np.save(directory / "embeddings.npy", {10: np.ones(6, dtype=np.float32)})
    records = []
    for split in ("train", "validation", "test"):
        for subject in (1, 0):
            records.append(dict(image_id="image.jpg", question_id=10, question="where?", subject_idx=subject,
                                subject_answer="yes", answer="yes", width=32, height=32,
                                X=[4., 20.], Y=[5., 22.], T_start=[0, 100], T_end=[50, 200], split=split))
    for record in records:
        record["prediction"] = {"fixations": [{"fixation": 1, "what": "first"}, {"fixation": 2, "what": "second"}],
                                "regions": [{"fixations": [1, 2], "why": "episode"}], "how": "trajectory"}
    fixation_path = directory / "fixations.json"
    fixation_path.write_text(json.dumps(records), encoding="utf-8")
    data = AiR("", feature_dir, fixation_path, args=SimpleNamespace(enable_explanation=False),
               task_emb_dir=directory / "embeddings.npy", action_map=(2, 2), max_length=3, blur_sigma=None)
    sample = data[0]
    assert sample["subject"].tolist() == [0, 1]
    assert sample["target_scanpath"][0].argmax(-1).tolist() == [1, 4, 0]
    evaluation = AiR_evaluation("", feature_dir, fixation_path, task_emb_dir=directory / "embeddings.npy", type="test")
    assert evaluation[0]["fix_vectors"][0].dtype.names == ("start_x", "start_y", "duration")
    explanation_data = AiR("", feature_dir, fixation_path, args=SimpleNamespace(enable_explanation=True, router_kmax=3),
                           task_emb_dir=directory / "embeddings.npy", action_map=(2, 2), max_length=3, blur_sigma=None)
    batch = explanation_data.collate_func([explanation_data[0]])
    explanation_model.train()
    explanation_model.zero_grad()
    loss, output = supervised_step(explanation_model, batch,
                                   SimpleNamespace(enable_explanation=True, max_length=3, im_h=2, im_w=2, lambda_1=1.),
                                   torch.device("cpu"))
    assert loss > output["explanation"]["loss_explanation"]
    loss.backward()
    assert explanation_model.explanation_module.why_router.mlp[0].weight.grad.abs().sum() > 0
    options = ["--feat_dir", str(feature_dir), "--fix_dir", str(fixation_path),
               "--emb_dir", str(directory / "embeddings.npy"), "--user_emb_path", str(directory / "subjects.pt"),
               "--log_root", str(directory / "run"), "--hidden_dim", "8", "--subject_feature_dim", "4",
               "--img_hidden_dim", "6", "--lm_hidden_dim", "6", "--nhead", "2", "--num_encoder", "1",
               "--num_decoder", "1", "--im_h", "2", "--im_w", "2", "--width", "32", "--height", "32",
               "--max_length", "3", "--subject_num", "2", "--epoch", "1", "--skip_metrics"]
    main(options)
    checkpoint = directory / "run/checkpoints/checkpoint.pth"
    assert checkpoint.is_file()
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["epoch"] == 0 and "scheduler" in saved
    model, predictions = run(options + ["--checkpoint", str(checkpoint), "--eval_repeat_num", "2"])
    assert len(predictions) == 4
    assert {item["subject_idx"] for item in predictions} == {0, 1}
    for key, tensor in saved["model"].items():
        torch.testing.assert_close(model.state_dict()[key], tensor)
    main(options + ["--checkpoint", str(checkpoint), "--epoch", "2"])
    assert torch.load(checkpoint, weights_only=False)["epoch"] == 1
    try:
        run(options)
    except ValueError as error:
        assert "requires --checkpoint" in str(error)
    else:
        raise AssertionError("Untrained inference was accepted")


if __name__ == "__main__":
    run_predictor(sys.argv[1], Path(sys.argv[2]))
