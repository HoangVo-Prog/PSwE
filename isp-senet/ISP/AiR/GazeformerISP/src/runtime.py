"""Air-D sampling, reference metrics, and checkpoint reconstruction."""

import json
from pathlib import Path

import numpy as np
import torch

from models.explanation import load_model_state_with_explanation_migration
from models.sampling import Sampling


MODEL_OPTIONS = (
    "subject_num", "subject_feature_dim", "action_map_num", "num_encoder", "num_decoder",
    "hidden_dim", "nhead", "img_hidden_dim", "lm_hidden_dim", "encoder_dropout",
    "decoder_dropout", "cls_dropout", "max_length", "patch_size", "im_h", "im_w",
    "enable_explanation", "explanation_dim", "router_kmax", "how_hidden_dim",
    "semantic_encoder_name", "semantic_encoder_dim", "freeze_semantic_encoder",
    "explanation_llm_name", "explanation_llm_hidden_dim", "freeze_explanation_llm",
)


def sampler_for(args):
    return Sampling(convLSTM_length=args.max_length, min_length=args.min_length,
                    map_width=args.im_w, map_height=args.im_h, width=args.width, height=args.height)


def predictor_inputs(batch, device):
    images = batch["images"].to(device).reshape(-1, *batch["images"].shape[2:])
    subjects = batch["subjects"].to(device).reshape(-1).long()
    tasks = batch["task_embeddings"].to(device).reshape(-1, batch["task_embeddings"].shape[-1])
    return images, subjects, tasks


def checkpoint_path(args):
    if args.checkpoint:
        return Path(args.checkpoint)
    if args.resume_dir:
        return Path(args.resume_dir) / "checkpoints" / "checkpoint.pth"
    return None


def read_checkpoint(args, device, required=False):
    path = checkpoint_path(args)
    if path is None:
        if required:
            raise ValueError("Inference requires --checkpoint or --resume_dir; random weights are not predictions")
        return None
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    metadata = Path(str(args.user_emb_path) + ".subjects.json")
    if "subject_ids" in config:
        if not metadata.is_file() or json.loads(metadata.read_text(encoding="utf-8")) != config["subject_ids"]:
            raise ValueError("Subject table row IDs differ from the checkpoint's personalization contract")
    requested_explanation = args.enable_explanation
    for name in MODEL_OPTIONS:
        if name in config:
            if requested_explanation and not config.get("enable_explanation", False) and (
                    name.startswith("explanation_") or name.startswith("semantic_") or name.startswith("freeze_")
                    or name in ("enable_explanation", "router_kmax", "how_hidden_dim")):
                continue
            setattr(args, name, config[name])
    for name, value in config.items():
        if name.startswith("lambda_"):
            setattr(args, name, value)
    return checkpoint


def load_weights(model, checkpoint, args):
    return load_model_state_with_explanation_migration(
        model, checkpoint["model"], explanation_enabled=args.enable_explanation)


def evaluate(model, loader, args, device, metrics=True):
    model.eval()
    sampler = sampler_for(args)
    targets, repeated_predictions, records = [], [[] for _ in range(args.eval_repeat_num)], []
    with torch.no_grad():
        for batch in loader:
            images, subjects, tasks = predictor_inputs(batch, device)
            output = model(images, subjects, tasks)
            offset = 0
            if metrics:
                for size in batch["group_sizes"]:
                    observed = subjects[offset:offset + size].tolist()
                    if observed != list(range(args.subject_num)):
                        raise ValueError("Air-D subject metrics require one ordered record per subject for each question")
                    offset += size
            targets.extend(batch["fix_vectors"])
            for repeat in range(args.eval_repeat_num):
                sampled = sampler.random_sample(output["all_actions_prob"], output["log_normal_mu"], output["log_normal_sigma2"])
                predicted, _, _ = sampler.generate_scanpath(images, sampled["selected_actions_probs"],
                                                           sampled["durations"], sampled["selected_actions"])
                offset = 0
                for group, size in enumerate(batch["group_sizes"]):
                    repeated_predictions[repeat].append(predicted[offset:offset + size])
                    for within_group in range(size):
                        index = offset + within_group
                        vector = predicted[index]
                        records.append({"question_id": batch["qids"][group][within_group],
                                        "image_id": batch["img_names"][group],
                                        "subject_idx": (loader.dataset.subject_ids[int(subjects[index])]
                                                        if loader.dataset.subject_ids is not None else int(subjects[index])),
                                        "subject_row": int(subjects[index]), "repeat": repeat,
                                        "X": vector["start_x"].tolist(), "Y": vector["start_y"].tolist(),
                                        "T": vector["duration"].tolist(), "duration_unit": "seconds"})
                    offset += size
    result = None
    if metrics:
        from utils.evaluation import comprehensive_evaluation_by_subject
        measured = [comprehensive_evaluation_by_subject(targets, predictions, args)[0]
                    for predictions in repeated_predictions]
        result = {group: {name: float(np.mean([item[group][name] for item in measured]))
                          for name in measured[0][group]} for group in measured[0]}
    return records, result


def write_predictions(path, records, metrics):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"predictions": records, "metrics": metrics}, indent=2), encoding="utf-8")
