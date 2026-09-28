"""Air-D/AiR Gazeformer-ISP data contract.

This is a project-local translation of the reference AiR loader.  Questions
are the grouping key, image features are flat by image ID, and durations are
``T_end - T_start`` in milliseconds before conversion to seconds.  The
optional explanation branch augments each original record; it never changes
the disabled base sample fields.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from models.explanation import lookup_explanation_annotation, normalize_explanation_annotation


def _duration(record: Mapping[str, Any]) -> np.ndarray:
    if "T" in record:
        values = np.asarray(record["T"], dtype=np.float32)
    else:
        starts = np.asarray(record["T_start"], dtype=np.float32)
        ends = np.asarray(record["T_end"], dtype=np.float32)
        if starts.shape != ends.shape:
            raise ValueError(f"Air-D record {record.get('image_id', record.get('name'))!r} has mismatched T_start/T_end")
        values = ends - starts
    return values


def _feature_path(feature_dir: str | Path, image_id: str) -> Path:
    return Path(feature_dir) / str(image_id).replace(".jpg", ".pth")


def _load_feature(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"Air-D feature file is missing: {path}")
    value = torch.load(path, map_location="cpu")
    if not torch.is_tensor(value):
        raise TypeError(f"Air-D feature file {path} did not contain a tensor")
    return value.unsqueeze(0) if value.ndim == 2 else value


def _lookup_embedding(embedding_dict: Mapping[Any, Any], question_id: Any) -> Any:
    candidates = [question_id, str(question_id)]
    try:
        candidates.append(int(question_id))
    except (TypeError, ValueError):
        pass
    for key in candidates:
        if key in embedding_dict:
            return embedding_dict[key]
    raise KeyError(f"Air-D question embedding is missing for question_id={question_id!r}")


class AiR(Dataset):
    """Question-grouped supervised Air-D data, matching the reference keys."""

    def __init__(
        self,
        stimuli_dir,
        feature_dir,
        fixations_dir,
        action_map=(24, 32),
        origin_size=(600, 800),
        resize=(384, 512),
        max_length=16,
        blur_sigma=1,
        type="train",
        transform=None,
        args=None,
        task_emb_dir=None,
    ):
        self.stimuli_dir = stimuli_dir
        self.feature_dir = feature_dir
        self.fixations_dir = fixations_dir
        self.action_map = tuple(action_map)
        self.origin_size = origin_size
        self.resize = resize
        self.max_length = int(max_length)
        self.blur_sigma = blur_sigma
        self.type = type
        self.transform = transform
        self.args = args
        self.explanation_enabled = bool(getattr(args, "enable_explanation", False)) and type == "train" and self.__class__ is AiR
        annotation_path = getattr(args, "explanation_annotations", None) if args is not None else None
        self.explanation_store = None
        if self.explanation_enabled and annotation_path:
            with Path(annotation_path).open("r", encoding="utf-8") as handle:
                self.explanation_store = json.load(handle)
        if task_emb_dir is None:
            task_emb_dir = Path(stimuli_dir).parent / "embeddings.npy"
        self.embedding_dict = np.load(task_emb_dir, allow_pickle=True).item()

        path = Path(fixations_dir)
        if path.is_dir():
            path = path / f"AiR_fixations_{type}.json"
        with path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        self.fixations = [dict(record) for record in records if record.get("split", type) == type]
        if not self.fixations:
            raise ValueError(f"No Air-D records found for split {type!r} in {path}")
        self.subject_ids = None
        table_path = getattr(args, "user_emb_path", None)
        metadata = Path(str(table_path) + ".subjects.json") if table_path else None
        if metadata is not None and metadata.is_file():
            self.subject_ids = json.loads(metadata.read_text(encoding="utf-8"))
            if not self.subject_ids or any(isinstance(value, bool) or not isinstance(value, int) for value in self.subject_ids) or len(set(self.subject_ids)) != len(self.subject_ids):
                raise ValueError("Subject table metadata must list unique integer source IDs in row order")
            mapping = {subject: row for row, subject in enumerate(self.subject_ids)}
            self.fixations = [dict(record, explanation_source_subject=record["subject_idx"],
                                   subject_idx=mapping[record["subject_idx"]])
                              for record in self.fixations if record["subject_idx"] in mapping]
            if not self.fixations:
                raise ValueError("No Air-D records match the subject table row IDs")
        self.qid_to_sub: OrderedDict[Any, list[int]] = OrderedDict()
        for index, record in enumerate(self.fixations):
            self.qid_to_sub.setdefault(record["question_id"], []).append(index)
        for qid, indices in self.qid_to_sub.items():
            if len({self.fixations[index]["image_id"] for index in indices}) != 1:
                raise ValueError(f"Question {qid!r} maps to multiple images")
            subjects = [int(self.fixations[index]["subject_idx"]) for index in indices]
            if len(subjects) != len(set(subjects)):
                raise ValueError(f"Question {qid!r} contains duplicate subject records")
            self.qid_to_sub[qid] = sorted(indices, key=lambda index: int(self.fixations[index]["subject_idx"]))
        self.qid = list(self.qid_to_sub)

    def __len__(self):
        return len(self.qid)

    def _annotation(self, record, kept_length):
        if self.explanation_store is None and "prediction" not in record:
            raise ValueError(
                "Air-D explanation is enabled but the sample has no inline prediction and no sidecar: "
                f"image={record.get('image_id')!r}, subject={record.get('subject_idx')!r}, "
                f"question_id={record.get('question_id')!r}"
            )
        raw = lookup_explanation_annotation(self.explanation_store, record)
        # AiR's original benchmark vocabulary stores durations as start/end
        # timestamps.  The augmented explanation contract uses T, so expose
        # the exact derived duration to the validator without changing the
        # base record or its predictor semantics.
        if "T" not in raw and "T_start" in raw and "T_end" in raw:
            raw = dict(raw)
            raw["T"] = (np.asarray(raw["T_end"], dtype=np.float32) - np.asarray(raw["T_start"], dtype=np.float32)).tolist()
        return normalize_explanation_annotation(
            raw,
            max_length=self.max_length,
            kmax=getattr(self.args, "router_kmax", 4),
            model_raw_indices=list(range(1, kept_length + 1)),
            allow_truncated_how=getattr(self.args, "allow_truncated_how", False),
        )

    def __getitem__(self, index):
        qid = self.qid[index]
        grouped = [self.fixations[item] for item in self.qid_to_sub[qid]]
        image_id = grouped[0]["image_id"]
        image_features = _load_feature(_feature_path(self.feature_dir, image_id))
        images, subjects, durations, action_masks, duration_masks = [], [], [], [], []
        task_embeddings, target_scanpaths, performances, explanations = [], [], [], []
        for record in grouped:
            x = np.asarray(record["X"], dtype=np.float32)
            y = np.asarray(record["Y"], dtype=np.float32)
            t = _duration(record)
            if not (len(x) == len(y) == len(t)):
                raise ValueError(
                    f"Air-D record image={record.get('image_id')!r}, subject={record.get('subject_idx')!r}, "
                    f"question_id={record.get('question_id')!r} violates len(X)==len(Y)==len(T)"
                )
            if not len(x) or not np.isfinite(np.concatenate((x, y, t))).all() or (t <= 0).any():
                raise ValueError("Air-D requires nonempty, finite scanpaths and positive durations")
            if float(record["width"]) <= 0 or float(record["height"]) <= 0:
                raise ValueError("Air-D width and height must be positive")
            if (x < 0).any() or (x >= record["width"]).any() or (y < 0).any() or (y >= record["height"]).any():
                raise ValueError("Air-D fixation is outside the image; refusing to silently clamp it")
            kept = min(len(x), self.max_length)
            target = np.zeros((self.max_length, self.action_map[0] * self.action_map[1] + 1), dtype=np.float32)
            duration = np.zeros(self.max_length, dtype=np.float32)
            action_mask = np.zeros(self.max_length, dtype=np.float32)
            duration_mask = np.zeros(self.max_length, dtype=np.float32)
            cell_w = float(record["width"]) / self.action_map[1]
            cell_h = float(record["height"]) / self.action_map[0]
            for step in range(kept):
                col = int(x[step] / cell_w)
                row = int(y[step] / cell_h)
                spatial = np.zeros(self.action_map, dtype=np.float32)
                spatial[row, col] = 1.0
                if self.blur_sigma:
                    from scipy.ndimage import gaussian_filter
                    spatial = gaussian_filter(spatial, self.blur_sigma)
                    spatial /= spatial.sum()
                target[step, 1:] = spatial.reshape(-1)
                duration[step] = t[step] / 1000.0
                action_mask[step] = 1.0
                duration_mask[step] = 1.0
            if kept < self.max_length:
                target[kept:, 0] = 1.0
                action_mask[kept] = 1.0
            images.append(image_features)
            subjects.append(int(record["subject_idx"]))
            durations.append(duration)
            action_masks.append(action_mask)
            duration_masks.append(duration_mask)
            task_embeddings.append(_lookup_embedding(self.embedding_dict, qid))
            target_scanpaths.append(target)
            subject_answer = record.get("subject_answer")
            performances.append(subject_answer is not None and subject_answer == record.get("answer") and subject_answer != "faild")
            if self.explanation_enabled:
                explanations.append(self._annotation(record, kept))

        result = {
            "image": torch.cat(images),
            "subject": np.asarray(subjects),
            "img_name": image_id,
            "qid": qid,
            "duration": np.asarray(durations),
            "action_mask": np.asarray(action_masks),
            "duration_mask": np.asarray(duration_masks),
            "performance": np.asarray(performances),
            "task_embedding": np.asarray(task_embeddings),
            "target_scanpath": np.asarray(target_scanpaths),
        }
        if self.explanation_enabled:
            result["explanation_annotations"] = explanations
        return result

    def collate_func(self, batch):
        data = {
            "images": torch.cat([item["image"] for item in batch]),
            "subjects": np.concatenate([item["subject"] for item in batch]),
            "img_names": [item["img_name"] for item in batch],
            "group_sizes": [len(item["subject"]) for item in batch],
            "durations": np.concatenate([item["duration"] for item in batch]),
            "action_masks": np.concatenate([item["action_mask"] for item in batch]),
            "duration_masks": np.concatenate([item["duration_mask"] for item in batch]),
            "qids": [item["qid"] for item in batch],
            "task_embeddings": np.concatenate([item["task_embedding"] for item in batch]),
            "target_scanpaths": np.concatenate([item["target_scanpath"] for item in batch]),
            "performances": np.concatenate([item["performance"] for item in batch]),
        }
        if self.explanation_enabled:
            data["explanation_annotations"] = [item["explanation_annotations"] for item in batch]
        converted = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            converted[key] = value.unsqueeze(0) if torch.is_tensor(value) else value
        return converted


class AiR_rl(AiR):
    """Evaluation data retains variable-length continuous scanpaths."""

    def __getitem__(self, index):
        qid = self.qid[index]
        grouped = [self.fixations[item] for item in self.qid_to_sub[qid]]
        image_id = grouped[0]["image_id"]
        image_features = _load_feature(_feature_path(self.feature_dir, image_id))
        fix_vectors, subjects, qids, embeddings, performances = [], [], [], [], []
        for record in grouped:
            x = np.asarray(record["X"], dtype=np.float32) / float(record["width"]) * self.resize[1]
            y = np.asarray(record["Y"], dtype=np.float32) / float(record["height"]) * self.resize[0]
            duration = _duration(record) / 1000.0
            if not (len(x) == len(y) == len(duration)):
                raise ValueError("Air-D evaluation coordinates and durations must align")
            fix_vectors.append(np.asarray(list(zip(x, y, duration)), dtype=[
                ("start_x", "f8"), ("start_y", "f8"), ("duration", "f8")]))
            subjects.append(int(record["subject_idx"]))
            qids.append(qid)
            embeddings.append(_lookup_embedding(self.embedding_dict, qid))
            answer = record.get("subject_answer")
            performances.append(answer is not None and answer == record.get("answer") and answer != "faild")
        return {"image": image_features.expand(len(grouped), -1, -1), "fix_vectors": fix_vectors,
                "img_name": image_id, "subject": np.asarray(subjects), "qid": qids,
                "task_embedding": np.asarray(embeddings), "performance": performances}

    def collate_func(self, batch):
        data = {
            "images": torch.cat([item["image"] for item in batch]),
            "fix_vectors": [item["fix_vectors"] for item in batch],
            "subjects": np.concatenate([item["subject"] for item in batch]),
            "img_names": [item["img_name"] for item in batch],
            "group_sizes": [len(item["subject"]) for item in batch],
            "qids": [item["qid"] for item in batch],
            "task_embeddings": np.concatenate([item["task_embedding"] for item in batch]),
            "performances": np.asarray([value for item in batch for value in item["performance"]]),
        }
        converted = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            converted[key] = value.unsqueeze(0) if torch.is_tensor(value) else value
        return converted


class AiR_evaluation(AiR_rl):
    pass

