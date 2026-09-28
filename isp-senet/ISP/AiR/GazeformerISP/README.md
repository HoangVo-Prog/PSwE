# Air-D / AiR ISP-SENet

This path implements the local ISP-SENet personalization boundary on the AiR
reference data contract. It consumes precomputed image/question features and an
offline SE-Net subject table; it does not train SE-Net jointly.

## Data and coordinates

- `processed_data/AiR_fixations_{train,validation,test}.json`, or one explicitly
  split-tagged JSON file; `question_id` groups trials and `image_id` names a flat
  `.pth` feature file. Each question has one record per subject, in subject order.
- Question embeddings are keyed by `question_id` in `embeddings.npy`. Natural
  language `question` is separate and goes through the explanation semantic encoder.
- Raw X/Y are image pixels with per-record width/height. Targets use a 24x32
  row-major grid with STOP at action zero; no initial center fixation is injected.
  Raw millisecond durations are `T_end - T_start` (or explicit T), converted to
  seconds for prediction. Invalid coordinates are rejected, not silently clamped.
- `--blur_sigma` controls the same SciPy Gaussian target smoothing as the reference.
  Evaluation uses structured start_x/start_y/duration vectors grouped by question.
- The default 20x384 subject table uses native zero-based subject_idx row IDs.
  SE-Net subset/few-shot exports include `<table>.subjects.json` in source-ID row
  order. Keep this file beside the tensor: it filters/remaps trials consistently
  and its identity is checked on checkpoint resume. Predictions contain both
  original subject_idx and subject_row. Old full tables need no new artifact.

## Commands

Run from `src`. Supply your actual data/artifact paths; no weights are downloaded
by the disabled explanation path.

```sh
python train.py --img_dir DATA/stimuli --feat_dir DATA/image_features --fix_dir DATA/processed_data --emb_dir DATA/embeddings.npy --user_emb_path TABLE.pt --log_root runs/air
python train.py --resume_dir runs/air --img_dir DATA/stimuli --feat_dir DATA/image_features --fix_dir DATA/processed_data --emb_dir DATA/embeddings.npy --user_emb_path TABLE.pt --log_root runs/air
python test.py --checkpoint runs/air/checkpoints/checkpoint.pth --img_dir DATA/stimuli --feat_dir DATA/image_features --fix_dir DATA/processed_data --emb_dir DATA/embeddings.npy --user_emb_path TABLE.pt --output predictions.json
```

Training saves model, optimizer, scheduler, epoch, configuration and best metric.
There are supervised, best-metric and latest checkpoints. Test refuses random
untrained inference. Model dimensions and enabled explanation construction are
reconstructed from saved configuration; the external subject table is still required.
A base-to-explanation migration explicitly initializes the whole explanation
branch and resets the optimizer/schedule. Partial explanation checkpoints fail.

The supervised objective is the original action CE plus lambda_1 times log-normal
loss, with L_EXP added only when enabled. The RL stage starts at start_rl_epoch,
uses the reference ScanMatch harmonic-mean reward and sample-mean baseline, and
adds no explanation loss. Metric sampling retries are bounded. Evaluation repeats
are evaluated independently and then averaged, rather than exceeding the metric
implementation's subject-matrix dimensions.

`--skip_metrics` skips only validation/test metrics, not sampling, serialization,
or RL rewards. It is useful for a feature/model smoke test when metric libraries
are absent. Full metrics/RL require the existing SciPy, pandas, multimatch_gaze,
and local ScanMatch/visual-attention dependencies. The metric utilities here are
project-local copies of the read-only AiR reference; no sibling repository is
imported at runtime. `pairs_eval` additionally honors configured output dimensions.

## Explanations

Add `--enable_explanation --semantic_encoder_name LOCAL_SEMANTIC_MODEL
--explanation_llm_name LOCAL_CAUSAL_MODEL`. Supply raw question text and inline
prediction.fixations (1-based raw IDs and what), prediction.regions (partitioned
raw IDs and why), and prediction.how. A sidecar is optional, not mandatory.
The same canonical WHAT/WHY-R0/HOW-R0 math, masks, semantic targets and ten lambda
options used by O/F/C apply here. Query/LLM training is controlled by the two
freeze flags. No real model weights are fetched by tests.

Truncation removes WHAT targets, remaps WHY groups and drops empty groups.
Full-scanpath HOW is rejected after truncation unless `--allow_truncated_how`
is explicitly justified by the annotation's meaning. Do not use this flag merely
to bypass the contract. R0 has no deploy-time episode-count rule: command-line
explanation generation is rejected, not silently treated as generation of text.

## SE-Net preparation and limitations

Use `SE-Net/configs/air_useremb.json` with the existing SE-Net launcher. RGB resize
512x320 and support max length 20 are separate subject-encoder settings, not a
change to the predictor's 512x384 / 16-fixation defaults. The config is a local
integration choice inherited from the SE-Net backbone conventions, not a claim
of reference experiment parity. Task conditioning remains active even for one
question. Existing image-based few-shot support selection is preserved; no new
support-selection policy is introduced. Use the classifier dimensions/config of
the pretrained SE-Net checkpoint for few-shot embedding extraction. Unseen
few-shot exports average all selected support trials and do not claim classifier
accuracy on unseen IDs. Empty support rows fail rather than exporting zeros.

The actual RGB backbone, deformable-attention extension, datasets, pretrained
features and production text-model artifacts remain external prerequisites.
See `../../../audit/AUDIT_REPORT.md` for the audit and its executed validations.
