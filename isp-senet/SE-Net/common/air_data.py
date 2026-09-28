"""Air-D data normalization used by the existing SE-Net pipeline.

The reference AiR preprocessing stores one record per subject/question with
``question_id``, ``image_id``, ``subject_idx``, raw ``X/Y`` and millisecond
``T_start/T_end`` arrays.  This module translates that contract to the common
SE-Net trajectory fields without changing the UserEmbeddingNet API.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


AIR_SPLITS = ("train", "validation", "test")


def _duration_values(record: Mapping[str, Any]) -> list[float]:
    if "T" in record:
        values = record["T"]
    elif "T_start" in record and "T_end" in record:
        starts = list(record["T_start"])
        ends = list(record["T_end"])
        if len(starts) != len(ends):
            raise ValueError("AiR T_start and T_end lengths differ")
        values = [float(end) - float(start) for start, end in zip(starts, ends)]
    else:
        raise ValueError("AiR record needs T or T_start/T_end duration fields")
    return [float(value) for value in values]


def _record_error(record: Mapping[str, Any], message: str) -> ValueError:
    return ValueError(
        "Air-D record {} (question_id={!r}, subject={!r}): {}".format(
            record.get("name", record.get("image_id", "<unnamed>")),
            record.get("question_id"),
            record.get("subject_idx", record.get("subject")),
            message,
        )
    )


def normalize_air_record(
    record: Mapping[str, Any],
    target_width: int | None = None,
    target_height: int | None = None,
) -> dict[str, Any]:
    """Return one Air-D record in the field vocabulary expected by SE-Net."""

    result = copy.deepcopy(dict(record))
    x = [float(value) for value in result.get("X", [])]
    y = [float(value) for value in result.get("Y", [])]
    durations = _duration_values(result)
    if not (len(x) == len(y) == len(durations)):
        raise _record_error(result, "len(X), len(Y), and len(T) must be equal")
    if not x:
        raise _record_error(result, "empty scanpaths are not valid Air-D support examples")

    image_name = result.get("name", result.get("image_id"))
    if image_name is None:
        raise _record_error(result, "missing image_id/name")
    question_id = result.get("question_id")
    if question_id is None:
        question_id = result.get("task_id", result.get("task"))
    if question_id is None:
        raise _record_error(result, "missing question_id/task key for task embedding lookup")
    question_text = result.get("question", result.get("query_text", result.get("task_text", result.get("task"))))
    if question_text is None:
        raise _record_error(result, "missing natural-language question/task text")

    subject = result.get("subject_idx", result.get("subject"))
    if subject is None:
        raise _record_error(result, "missing subject_idx/subject")
    if isinstance(subject, bool) or not isinstance(subject, (int, np.integer)) or subject < 0:
        raise _record_error(result, "subject_idx must be a nonnegative integer")
    width = result.get("width")
    height = result.get("height")
    if not width or not height or width <= 0 or height <= 0:
        raise _record_error(result, "positive per-record width and height are required")
    if not np.isfinite(x + y + durations).all() or any(value <= 0 for value in durations):
        raise _record_error(result, "coordinates must be finite and durations positive")
    if any(value < 0 or value >= width for value in x) or any(value < 0 or value >= height for value in y):
        raise _record_error(result, "out-of-image fixations cannot be silently discarded")
    if target_width is not None and target_height is not None and width and height:
        x = [value * float(target_width) / float(width) for value in x]
        y = [value * float(target_height) / float(height) for value in y]

    result.update({
        "name": str(image_name),
        "image_id": str(image_name),
        # Keep the source key for exact embedding-dictionary lookup.  The
        # shared SE-Net tuple uses a string task label separately because it
        # builds composite image/task keys.
        "question_id": question_id,
        "question": str(question_text),
        # `task` is the stable embedding key.  The natural-language text is
        # retained separately and is consumed by the explanation semantic path.
        "task": str(question_id),
        "task_key": question_id,
        "task_text": str(question_text),
        "condition": result.get("condition", "qa"),
        "X": x,
        "Y": y,
        "T": durations,
        "length": len(x),
        "subject": int(subject),
        "subject_idx": int(subject),
        "dataset": "Air-D",
    })
    return result


def load_air_scanpaths(
    dataset_root: str | Path,
    fix_path: str = "fixations",
    fixation_file_pattern: str = "AiR_fixations_{}.json",
) -> list[dict[str, Any]]:
    """Load train/validation/test Air-D files in deterministic split order."""

    root = Path(dataset_root)
    configured = root / fix_path
    if configured.is_file():
        with configured.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return list(payload)
    records: list[dict[str, Any]] = []
    for split in AIR_SPLITS:
        path = configured / fixation_file_pattern.format(split)
        if not path.is_file():
            raise FileNotFoundError(f"Missing Air-D {split} fixation file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            split_records = json.load(handle)
        for record in split_records:
            item = dict(record)
            item.setdefault("split", split)
            records.append(item)
    return records


def normalize_air_scanpaths(
    records: Sequence[Mapping[str, Any]],
    target_width: int,
    target_height: int,
) -> list[dict[str, Any]]:
    return [normalize_air_record(record, target_width, target_height) for record in records]


def save_subject_embeddings(embeddings, path, hparams):
    import torch

    if hparams.Data.name in ("Air-D", "AiR", "Air"):
        subject_ids = list(hparams.Data.subject_ids)
        if len(subject_ids) != embeddings.shape[0] or len(set(subject_ids)) != len(subject_ids):
            raise ValueError("Air-D subject table rows do not match their source subject IDs")
        torch.save(embeddings, path)
        Path(str(path) + ".subjects.json").write_text(json.dumps(subject_ids), encoding="utf-8")
    else:
        torch.save(embeddings, path)
