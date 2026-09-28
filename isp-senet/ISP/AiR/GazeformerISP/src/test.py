"""Checkpoint-backed Air-D scanpath evaluation and serialization."""

from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset.dataset import AiR_evaluation
from opts import parse_opt
from train import build_model, make_dataset
from runtime import read_checkpoint, load_weights, evaluate, write_predictions


def run(argv=None):
    args = parse_opt(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.cuda) if torch.cuda.is_available() else torch.device("cpu")
    checkpoint = read_checkpoint(args, device, required=True)
    dataset = make_dataset(args, AiR_evaluation, args.eval_split)
    if dataset.subject_ids is not None:
        args.subject_num = len(dataset.subject_ids)
    model = build_model(args, device)
    load_weights(model, checkpoint, args)
    loader = DataLoader(dataset, batch_size=args.test_batch, shuffle=False, collate_fn=dataset.collate_func)
    records, metrics = evaluate(model, loader, args, device, metrics=not args.skip_metrics)
    write_predictions(args.output or Path(args.log_root) / "predictions.json", records, metrics)
    return model, records


if __name__ == "__main__":
    run()
