import random
import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_offline_tests(monkeypatch):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
