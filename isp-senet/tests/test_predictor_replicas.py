import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_f_and_c_explanation_interfaces_are_offline_importable():
    code = """
from models.explanation import FixationReasoningEncoder, WhyR0Router, HowR0Aggregator
import torch
encoder = FixationReasoningEncoder(4, 3, 5, (2, 2))
out = encoder(torch.randn(1, 2, 4), torch.randn(1, 4, 3), torch.randn(1, 2, 5), torch.randn(1, 2), torch.randn(1, 2))
assert out['z'].shape == (1, 2, 5)
assert torch.allclose(out['fixation_prob'].sum(-1), torch.ones(1, 2), atol=1e-6)
"""
    for relative in ("ISP/COCO_FV/GazeformerISP/src", "ISP/COCO_Search18/GazeformerISP/src"):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / relative)
        completed = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr


def test_f_and_c_cli_switch_defaults_false_without_optional_dependencies():
    code = "import sys; sys.argv=['opts.py']; from opts import parse_opt; args=parse_opt(); assert args.enable_explanation is False; assert args.explanation_dim > 0"
    for relative in ("ISP/COCO_FV/GazeformerISP/src", "ISP/COCO_Search18/GazeformerISP/src"):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / relative)
        completed = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr

