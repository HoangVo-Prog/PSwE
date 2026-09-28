"""Minimal, project-local Air-D supervised training entry point.

The scanpath objective is unchanged from the existing predictor copies.  When
``--enable_explanation`` is set, normalized inline annotations are attached to
the continuous supervised forward before sampling and added through the same
joint-loss helper as O/F/C.
"""

from __future__ import annotations

import random
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.dataset import AiR, AiR_rl, AiR_evaluation
from models.explanation import compose_joint_supervised_loss, restore_training_checkpoint
from models.gazeformer import gazeformer
from models.loss import CrossEntropyLoss, MLPLogNormalDistribution, LogAction, LogDuration
from models.models import Transformer
from opts import parse_opt
from runtime import predictor_inputs, sampler_for, read_checkpoint, evaluate, write_predictions


def build_explanation_inputs(batch, duration_masks, args, device):
    if not args.enable_explanation:
        return None
    if "explanation_annotations" not in batch:
        raise ValueError("Air-D explanation training requires inline prediction annotations")
    annotations = [annotation for sample in batch["explanation_annotations"] for annotation in sample]
    if len(annotations) != duration_masks.shape[0]:
        raise ValueError(
            f"Air-D explanation annotations ({len(annotations)}) do not align with predictor batch "
            f"({duration_masks.shape[0]})"
        )
    return {
        "query_text": [item["query_text"] for item in annotations],
        "what_texts": [item["what_texts"] for item in annotations],
        "why_membership": torch.stack([item["why_membership"] for item in annotations]).to(device),
        "why_texts": [item["why_texts"] for item in annotations],
        "episode_count": torch.tensor([item["episode_count"] for item in annotations], dtype=torch.long, device=device),
        "how_text": [item["how_text"] for item in annotations],
        "fixation_mask": duration_masks.bool(),
    }


def build_model(args, device):
    transformer = Transformer(
        num_encoder_layers=args.num_encoder,
        nhead=args.nhead,
        subject_feature_dim=args.subject_feature_dim,
        d_model=args.hidden_dim,
        num_decoder_layers=args.num_decoder,
        encoder_dropout=args.encoder_dropout,
        decoder_dropout=args.decoder_dropout,
        dim_feedforward=args.hidden_dim,
        img_hidden_dim=args.img_hidden_dim,
        lm_dmodel=args.lm_hidden_dim,
        device=device,
        args=args,
    )
    return gazeformer(
        transformer,
        spatial_dim=(args.im_h, args.im_w),
        args=args,
        subject_num=args.subject_num,
        subject_feature_dim=args.subject_feature_dim,
        action_map_num=args.action_map_num,
        dropout=args.cls_dropout,
        max_len=args.max_length,
        patch_size=args.patch_size if hasattr(args, "patch_size") else 16,
        device=str(device),
    ).to(device)


def supervised_step(model, batch, args, device):
    images = batch["images"].to(device).view(-1, *batch["images"].shape[2:])
    subjects = batch["subjects"].to(device).view(-1).long()
    task_embeddings = batch["task_embeddings"].to(device).view(-1, batch["task_embeddings"].shape[-1])
    durations = batch["durations"].to(device).view(-1, args.max_length)
    action_masks = batch["action_masks"].to(device).view(-1, args.max_length)
    duration_masks = batch["duration_masks"].to(device).view(-1, args.max_length)
    target_scanpaths = batch["target_scanpaths"].to(device).view(-1, args.max_length, 1 + args.im_h * args.im_w)
    explanation_inputs = build_explanation_inputs(batch, duration_masks, args, device)
    predicts = model(images, subjects, task_embeddings, explanation_inputs=explanation_inputs)
    scan_loss = CrossEntropyLoss(predicts["actions"], target_scanpaths, action_masks)
    scan_loss = scan_loss + args.lambda_1 * MLPLogNormalDistribution(
        predicts["log_normal_mu"].reshape_as(durations), predicts["log_normal_sigma2"].reshape_as(durations), durations, duration_masks
    )
    return compose_joint_supervised_loss(scan_loss, predicts.get("explanation") if args.enable_explanation else None), predicts


def reinforcement_step(model, batch, args, device):
    from scipy.stats import hmean
    from utils.evaluation import pairs_eval
    from utils.evaltools.scanmatch import ScanMatch

    model.eval()
    images, subjects, tasks = predictor_inputs(batch, device)
    prediction = model(images, subjects, tasks)
    sampler = sampler_for(args)
    with_duration = ScanMatch(Xres=args.width, Yres=args.height, Xbin=16, Ybin=12,
                             Offset=(0, 0), TempBin=50, Threshold=3.5)
    without_duration = ScanMatch(Xres=args.width, Yres=args.height, Xbin=16, Ybin=12,
                                Offset=(0, 0), Threshold=3.5)
    targets = [vector for group in batch["fix_vectors"] for vector in group]
    rewards, actions, durations = [], [], []
    for attempt in range(args.rl_sample_number * 20):
        sampled = sampler.random_sample(prediction["all_actions_prob"], prediction["log_normal_mu"],
                                        prediction["log_normal_sigma2"])
        predicted, action_mask, duration_mask = sampler.generate_scanpath(
            images, sampled["selected_actions_probs"], sampled["durations"], sampled["selected_actions"])
        reward = pairs_eval(targets, predicted, with_duration, without_duration, width=args.width, height=args.height)
        if not np.isfinite(reward).all():
            continue
        rewards.append(torch.as_tensor(hmean(reward[:, 5:7], axis=-1), device=device))
        actions.append(-LogAction(sampled["selected_actions_probs"], action_mask))
        durations.append(-LogDuration(sampled["durations"].detach().clamp(0, 100),
                                      prediction["log_normal_mu"], prediction["log_normal_sigma2"], duration_mask))
        if len(rewards) == args.rl_sample_number:
            break
    if len(rewards) != args.rl_sample_number:
        raise RuntimeError("Reference Air-D metrics failed to produce finite RL rewards after bounded retries")
    reward = torch.stack(rewards)
    advantage = reward - reward.mean(0, keepdim=True)
    return ((torch.stack(actions) + torch.stack(durations)) * advantage).sum()


def make_dataset(args, dataset_type, split):
    return dataset_type(args.img_dir, args.feat_dir, args.fix_dir,
                        action_map=(args.im_h, args.im_w), resize=(args.height, args.width),
                        max_length=args.max_length, blur_sigma=args.blur_sigma, type=split,
                        args=args, task_emb_dir=args.emb_dir)


def main(argv=None):
    args = parse_opt(argv)
    if args.mode != "train":
        raise ValueError("Use test.py for --mode test")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.cuda) if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = read_checkpoint(args, device)
    train_dataset = make_dataset(args, AiR, "train")
    if train_dataset.subject_ids is not None:
        args.subject_num = len(train_dataset.subject_ids)
        args.subject_ids = train_dataset.subject_ids
    loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, collate_fn=train_dataset.collate_func)
    rl_dataset = make_dataset(args, AiR_rl, "train")
    rl_loader = DataLoader(rl_dataset, batch_size=args.batch, shuffle=True, collate_fn=rl_dataset.collate_func)
    model = build_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best_metric = 0, float("-inf")
    migrated = False
    if checkpoint is not None:
        migrated = restore_training_checkpoint(model, optimizer, checkpoint, args.enable_explanation)
        if not migrated:
            start_epoch = checkpoint.get("epoch", -1) + 1
            best_metric = checkpoint.get("best_metric", best_metric)
    def lr_lambda(iteration):
        warmup = len(loader) * args.warmup_epoch
        supervised = len(loader) * args.start_rl_epoch
        if iteration <= warmup:
            return iteration / max(1, warmup)
        if iteration <= supervised:
            return 1 - (iteration - warmup) / max(1, supervised - warmup)
        return args.rl_lr_initial_decay * max(0, 1 - (iteration - supervised) /
                                             max(1, len(rl_loader) * (args.epoch - args.start_rl_epoch)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if checkpoint is not None and not migrated and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        for group, learning_rate in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group["lr"] = learning_rate
    directory = Path(args.log_root)
    (directory / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = {name: value for name, value in vars(args).items()
              if value is None or isinstance(value, (str, int, float, bool, list))}
    (directory / "hparams.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for epoch in range(start_epoch, args.epoch):
        supervised = epoch < args.start_rl_epoch
        model.train(supervised)
        for batch in loader if supervised else rl_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_step(model, batch, args, device)[0] if supervised else reinforcement_step(model, batch, args, device)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
        metric = None
        if epoch > args.no_eval_epoch:
            validation = make_dataset(args, AiR_evaluation, "validation")
            validation_loader = DataLoader(validation, batch_size=args.test_batch, collate_fn=validation.collate_func)
            records, metrics = evaluate(model, validation_loader, args, device, metrics=not args.skip_metrics)
            write_predictions(directory / "validation.json", records, metrics)
            if metrics is not None:
                from scipy.stats import hmean
                metric = float(hmean(list(metrics["ScanMatch"].values())))
        improved = metric is not None and metric > best_metric
        if improved:
            best_metric = metric
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch,
                 "best_metric": best_metric, "config": config}
        torch.save(state, directory / "checkpoints" / "checkpoint.pth")
        if improved:
            torch.save(state, directory / "checkpoints" / "checkpoint_best.pth")
        if epoch == args.start_rl_epoch - 1:
            torch.save(state, directory / "checkpoints" / "checkpoint_supervised.pth")
    return model


if __name__ == "__main__":
    main()
