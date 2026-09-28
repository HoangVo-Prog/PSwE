"""Small loader for the deliberately shared predictor implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_canonical(filename: str, module_name: str):
    path = Path(__file__).resolve().parents[4] / "OSIE" / "GazeformerISP" / "src" / "models" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"Unable to load canonical Air-D dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

