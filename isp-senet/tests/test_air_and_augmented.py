import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = Path(__file__).parents[1]
OSIE_SRC = ROOT / "ISP" / "OSIE" / "GazeformerISP" / "src"
AIR_SRC = ROOT / "ISP" / "AiR" / "GazeformerISP" / "src"
SE_ROOT = ROOT / "SE-Net"
sys.path.insert(0, str(OSIE_SRC))
sys.path.insert(0, str(AIR_SRC))
sys.path.insert(0, str(SE_ROOT))

from models.explanation import normalize_augmented_explanation_annotation  # noqa: E402
from dataset.dataset import AiR  # noqa: E402
from common.air_data import normalize_air_record  # noqa: E402


def augmented_record(length=5):
    first = [value for value in (1, 3, 5) if value <= length]
    second = [value for value in (2, 4) if value <= length]
    regions = [{"fixations": first, "why": "first"}]
    if second:
        regions.append({"fixations": second, "why": "second"})
    return {
        "name": "000000095707.jpg",
        "subject": 1,
        "task": "bottle",
        "condition": "absent",
        "X": [float(i) for i in range(length)],
        "Y": [float(i + 10) for i in range(length)],
        "T": [100 for _ in range(length)],
        "answer": "absent",
        "prediction": {
            "fixations": [{"fixation": i, "what": f"what-{i}"} for i in range(1, length + 1)],
            "regions": regions,
            "how": "the whole scanpath",
        },
    }


def test_air_record_converts_reference_duration_and_question_fields():
    normalized = normalize_air_record(
        {
            "image_id": "img.jpg",
            "question_id": 17,
            "question": "Where is the bottle?",
            "subject_idx": 3,
            "width": 800,
            "height": 600,
            "X": [100, 200],
            "Y": [120, 220],
            "T_start": [10, 100],
            "T_end": [60, 250],
        },
        target_width=400,
        target_height=300,
    )
    assert normalized["task"] == "17"
    assert normalized["task_key"] == 17
    assert normalized["question"] == "Where is the bottle?"
    assert normalized["X"] == [50.0, 100.0]
    assert normalized["Y"] == [60.0, 110.0]
    assert normalized["T"] == [50.0, 150.0]


def test_inline_identity_alignment_preserves_raw_fixation_order():
    annotation = normalize_augmented_explanation_annotation(
        augmented_record(3), max_length=4, kmax=3, model_raw_indices=[1, 2, 3], allow_truncated_how=True
    )
    assert annotation["raw_to_model_idx"] == {1: 0, 2: 1, 3: 2}
    assert annotation["what_texts"][:3] == ["what-1", "what-2", "what-3"]
    assert annotation["episode_count"] == 2


def test_inline_removed_fixation_maps_why_to_surviving_tokens():
    annotation = normalize_augmented_explanation_annotation(
        augmented_record(5), max_length=4, kmax=3, model_raw_indices=[1, 3, 4], allow_truncated_how=True
    )
    assert annotation["raw_to_model_idx"] == {1: 0, 3: 1, 4: 2}
    assert annotation["what_texts"][:3] == ["what-1", "what-3", "what-4"]
    # Raw [1,3,5] leaves [1,3] in the first active group.  Raw 5 was not
    # attached to the final token by positional coincidence.
    assert annotation["why_membership"][:3, 0].tolist() == [1.0, 1.0, 0.0]
    assert annotation["why_membership"][:3, 1].tolist() == [0.0, 0.0, 1.0]


def test_inline_truncation_drops_targets_beyond_model_length_explicitly():
    annotation = normalize_augmented_explanation_annotation(
        augmented_record(5), max_length=3, kmax=3, model_raw_indices=[1, 2, 3], allow_truncated_how=True
    )
    assert annotation["model_raw_indices"] == [1, 2, 3]
    assert annotation["what_texts"] == ["what-1", "what-2", "what-3"]
    assert annotation["episode_count"] == 2
    assert annotation["how_truncated"] is True


def test_malformed_inline_annotation_fails_with_sample_context():
    record = augmented_record(3)
    record["prediction"]["fixations"][1]["fixation"] = 4
    with pytest.raises(ValueError, match="name=.*000000095707.*prediction.fixations"):
        normalize_augmented_explanation_annotation(record, max_length=3, kmax=3)


def test_air_dataset_base_and_explanation_modes(tmp_path):
    features = tmp_path / "features"
    features.mkdir()
    torch.save(torch.randn(2, 4), features / "img.jpg".replace(".jpg", ".pth"))
    embeddings = {17: np.arange(6, dtype=np.float32)}
    np.save(tmp_path / "embeddings.npy", embeddings, allow_pickle=True)
    records = []
    for subject in (0, 1):
        record = {
            "question_id": 17,
            "image_id": "img.jpg",
            "subject_idx": subject,
            "question": "Where is the bottle?",
            "answer": "yes",
            "subject_answer": "yes",
            "height": 600,
            "width": 800,
            "X": [10.0, 20.0],
            "Y": [30.0, 40.0],
            "T_start": [0, 100],
            "T_end": [50, 250],
            "split": "train",
        }
        inline = augmented_record(2)
        inline.pop("T")
        record.update(inline)
        record.update({"question_id": 17, "image_id": "img.jpg", "subject_idx": subject, "split": "train"})
        records.append(record)
    fixation_path = tmp_path / "AiR_fixations_train.json"
    fixation_path.write_text(json.dumps(records), encoding="utf-8")

    base_args = SimpleNamespace(enable_explanation=False, explanation_annotations=None)
    base = AiR(str(tmp_path / "stimuli"), str(features), str(fixation_path), action_map=(2, 2), max_length=4, blur_sigma=None,
               type="train", args=base_args, task_emb_dir=str(tmp_path / "embeddings.npy"))
    sample = base[0]
    assert sample["qid"] == 17
    assert sample["subject"].tolist() == [0, 1]
    assert sample["duration"][0, :2].tolist() == pytest.approx([0.05, 0.15])
    assert "explanation_annotations" not in sample

    enabled_args = SimpleNamespace(
        enable_explanation=True,
        explanation_annotations=None,
        router_kmax=3,
        allow_truncated_how=True,
    )
    enabled = AiR(str(tmp_path / "stimuli"), str(features), str(fixation_path), action_map=(2, 2), max_length=4, blur_sigma=None,
                  type="train", args=enabled_args, task_emb_dir=str(tmp_path / "embeddings.npy"))
    explanation_sample = enabled[0]
    assert explanation_sample["explanation_annotations"][0]["query_text"] == "Where is the bottle?"
    assert explanation_sample["explanation_annotations"][0]["how_text"] == "the whole scanpath"


def test_air_cli_explanation_defaults_offline_and_false():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(AIR_SRC)
    completed = subprocess.run(
        [sys.executable, "-c", "from opts import parse_opt; assert parse_opt([]).enable_explanation is False; assert parse_opt(['--enable_explanation', '--semantic_encoder_name', 'offline-semantic', '--explanation_llm_name', 'offline-causal']).enable_explanation is True"],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr



def test_subject_export_metadata_preserves_table_row_identity(tmp_path):
    from common.air_data import save_subject_embeddings
    table = tmp_path / "table.pt"
    config = SimpleNamespace(Data=SimpleNamespace(name="Air-D", subject_ids=[9, 4]))
    save_subject_embeddings(torch.randn(2, 4), table, config)
    assert json.loads((tmp_path / "table.pt.subjects.json").read_text()) == [9, 4]
    torch.save(torch.randn(4, 6), tmp_path / "image.pth")
    np.save(tmp_path / "embeddings.npy", {10: np.ones(6, dtype=np.float32)})
    records = [dict(question_id=10, image_id="image.jpg", subject_idx=subject, X=[1], Y=[1], T=[100],
                    width=32, height=32, question="q", answer="yes", subject_answer="yes") for subject in (4, 9, 1)]
    path = tmp_path / "fixations.json"
    path.write_text(json.dumps(records))
    data = AiR("", tmp_path, path, task_emb_dir=tmp_path / "embeddings.npy", blur_sigma=None,
               args=SimpleNamespace(enable_explanation=False, user_emb_path=str(table)))
    assert data.subject_ids == [9, 4]
    assert data[0]["subject"].tolist() == [0, 1]
    assert [data.fixations[index]["explanation_source_subject"] for index in data.qid_to_sub[10]] == [9, 4]


@pytest.mark.parametrize("field,value", [("width", 0), ("height", None), ("subject_idx", "raw"), ("X", [-1])])
def test_air_normalizer_rejects_unestablished_coordinate_and_subject_contract(field, value):
    record = dict(image_id="image.jpg", question_id=10, question="where?", subject_idx=0,
                  width=32, height=32, X=[1], Y=[1], T=[100])
    record[field] = value
    with pytest.raises(ValueError):
        normalize_air_record(record, 32, 32)


def test_air_gaussian_targets_match_reference(tmp_path):
    scipy = pytest.importorskip("scipy.ndimage", reason="Gaussian target parity requires the existing SciPy dependency")
    torch.save(torch.randn(16, 6), tmp_path / "image.pth")
    np.save(tmp_path / "embeddings.npy", {10: np.ones(6, dtype=np.float32)})
    record = dict(image_id="image.jpg", question_id=10, subject_idx=0, X=[9], Y=[9], T=[100], width=32, height=32)
    path = tmp_path / "fixations.json"
    path.write_text(json.dumps([record]))
    data = AiR("", tmp_path, path, task_emb_dir=tmp_path / "embeddings.npy", action_map=(4, 4), max_length=3, blur_sigma=1)
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[1, 1] = 1
    expected = scipy.gaussian_filter(expected, 1)
    expected /= expected.sum()
    np.testing.assert_allclose(data[0]["target_scanpath"][0, 0, 1:], expected.reshape(-1))
    assert data[0]["target_scanpath"][0, 1:, 0].tolist() == [1, 1]
