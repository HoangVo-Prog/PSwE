"""Canonical hierarchical explanation implementation for COCO-Freeview.

The predictor copies intentionally keep their local model and dataset files,
but the mathematical R0 explanation implementation is shared with the
canonical OSIE copy.  Loading it by source path avoids making the baseline
execution path import optional language dependencies.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_CANONICAL = Path(__file__).resolve().parents[4] / "OSIE" / "GazeformerISP" / "src" / "models" / "explanation.py"
_SPEC = importlib.util.spec_from_file_location("_isp_senet_osie_explanation", _CANONICAL)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - packaging error
    raise ImportError(f"Unable to load canonical explanation module: {_CANONICAL}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

for _name, _value in vars(_MODULE).items():
    if not _name.startswith("_"):
        globals()[_name] = _value

