import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).parents[1]
SRC = ROOT / "ISP" / "OSIE" / "GazeformerISP" / "src"
sys.path.insert(0, str(SRC))

from models.explanation import compose_joint_supervised_loss, load_model_state_with_explanation_migration  # noqa: E402
from opts import parse_opt  # noqa: E402


def test_explanation_cli_defaults_disabled_and_keeps_legacy_parser_surface(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = parse_opt()
    assert args.enable_explanation is False
    assert args.max_length == 16
    assert hasattr(args, "user_emb_path")
    assert args.semantic_encoder_name is None


def test_explanation_checkpoint_migration_allows_only_explanation_keys():
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(2, 2)
            self.explanation_module = nn.Linear(2, 2)

    model = Model()
    base_state = {key: value for key, value in model.state_dict().items() if not key.startswith("explanation_module.")}
    messages = []
    load_model_state_with_explanation_migration(model, base_state, True, logger=messages.append)
    assert any("freshly initialized" in message for message in messages)

    unrelated = dict(base_state)
    unrelated.pop("base.weight")
    with pytest.raises(RuntimeError, match="outside explanation keys"):
        load_model_state_with_explanation_migration(model, unrelated, True, logger=messages.append)


def test_joint_loss_adds_explanations_only_in_supervised_composition():
    scan = torch.tensor(2.0, requires_grad=True)
    exp = torch.tensor(3.0, requires_grad=True)
    assert compose_joint_supervised_loss(scan, {"loss_explanation": exp}).item() == 5.0
    assert compose_joint_supervised_loss(scan).item() == 2.0


def test_disabled_predictor_does_not_construct_explanation_module():
    from models.gazeformer import gazeformer

    class Transformer(nn.Module):
        d_model = 4

    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = str(Path(directory) / "subject.pt")
        torch.save(torch.randn(1, 3), checkpoint_path)
        args = types.SimpleNamespace(
            enable_explanation=False,
            user_emb_path=checkpoint_path,
            nhead=1,
            hidden_dim=4,
        )
        model = gazeformer(
            Transformer(), (1, 3), args, subject_num=1, subject_feature_dim=3,
            action_map_num=2, dropout=0.0, max_len=2, device="cpu",
        )
    assert model.explanation_module is None
    assert not any(key.startswith("explanation_module.") for key in model.state_dict())

