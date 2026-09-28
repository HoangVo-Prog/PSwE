"""Command-line options for the Air-D Gazeformer-ISP predictor."""

from __future__ import annotations

import argparse


def parse_opt(argv=None):
    parser = argparse.ArgumentParser(description="Air-D scanpath prediction")
    parser.add_argument("--mode", default="train", choices=("train", "test"))
    parser.add_argument("--img_dir", default="data/AiR/stimuli")
    parser.add_argument("--feat_dir", default="data/AiR/image_features")
    parser.add_argument("--fix_dir", default="data/AiR/processed_data")
    parser.add_argument("--emb_dir", default="data/AiR/embeddings.npy")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--im_h", type=int, default=24)
    parser.add_argument("--im_w", type=int, default=32)
    parser.add_argument("--blur_sigma", type=float, default=None)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--test_batch", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_root", default="runs/air-d")
    parser.add_argument("--resume_dir", default="")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip_metrics", action="store_true", default=False)
    parser.add_argument("--clip", type=float, default=12.5)
    parser.add_argument("--rl_lr_initial_decay", type=float, default=0.1)
    parser.add_argument("--user_emb_path", default="../../../SE-Net/assets/Air-D-useremb/train_user_embedding_no_vsencoder.pt")
    parser.add_argument("--eval_split", default="test", choices=("validation", "test"))
    parser.add_argument("--subject_num", type=int, default=20)
    parser.add_argument("--subject_feature_dim", type=int, default=384)
    parser.add_argument("--action_map_num", type=int, default=4)
    parser.add_argument("--num_encoder", type=int, default=6)
    parser.add_argument("--num_decoder", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--img_hidden_dim", type=int, default=2048)
    parser.add_argument("--lm_hidden_dim", type=int, default=768)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--decoder_dropout", type=float, default=0.2)
    parser.add_argument("--cls_dropout", type=float, default=0.4)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=40)
    parser.add_argument("--warmup_epoch", type=int, default=1)
    parser.add_argument("--start_rl_epoch", type=int, default=25)
    parser.add_argument("--rl_sample_number", type=int, default=5)
    parser.add_argument("--no_eval_epoch", type=int, default=5)
    parser.add_argument("--eval_repeat_num", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--lambda_1", type=float, default=1.0)

    # Explanation is completely opt-in.  No semantic encoder or LLM is
    # imported by the disabled path.
    parser.add_argument("--enable_explanation", action="store_true", default=False)
    parser.add_argument("--explanation_annotations", default=None)
    parser.add_argument("--explanation_dim", type=int, default=256)
    parser.add_argument("--router_kmax", type=int, default=4)
    parser.add_argument("--how_hidden_dim", type=int, default=256)
    parser.add_argument("--semantic_encoder_name", default=None)
    parser.add_argument("--semantic_encoder_dim", type=int, default=None)
    parser.add_argument("--freeze_semantic_encoder", action="store_true", default=False)
    parser.add_argument("--explanation_llm_name", default=None)
    parser.add_argument("--explanation_llm_hidden_dim", type=int, default=None)
    parser.add_argument("--freeze_explanation_llm", action="store_true", default=False)
    parser.add_argument("--allow_truncated_how", action="store_true", default=False)
    parser.add_argument("--return_explanation_latents", action="store_true", default=False)
    parser.add_argument("--generate_explanations", action="store_true", default=False)
    for name in ("exp_what", "exp_why", "exp_how", "what_txt", "what_align", "route",
                 "why_txt", "why_align", "how_txt", "how_align"):
        parser.add_argument("--lambda_" + name, type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.generate_explanations or args.return_explanation_latents:
        parser.error("R0 explanation inference is not defined; use the model API with explicit query/mask for latent diagnostics")
    if args.enable_explanation and (not args.semantic_encoder_name or not args.explanation_llm_name):
        parser.error("--enable_explanation requires --semantic_encoder_name and --explanation_llm_name")
    if args.rl_sample_number < 2 or args.eval_repeat_num < 1:
        parser.error("RL needs at least two samples; evaluation needs at least one repeat")
    return args

