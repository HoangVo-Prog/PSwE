# Audit baseline evidence - September 27, 2026

Captured before remediation. The explicit user request authorizes fixes inside isp-senet only. Existing uncommitted changes are retained.

## git status --short
```text
 M isp-senet/ARCHITECTURE_MAP.md
 M isp-senet/CODEBASE_MAP.md
 M isp-senet/ISP/COCO_FV/GazeformerISP/src/dataset/dataset.py
 M isp-senet/ISP/COCO_FV/GazeformerISP/src/models/explanation.py
 M isp-senet/ISP/COCO_FV/GazeformerISP/src/opts.py
 M isp-senet/ISP/COCO_Search18/GazeformerISP/src/dataset/dataset.py
 M isp-senet/ISP/COCO_Search18/GazeformerISP/src/models/explanation.py
 M isp-senet/ISP/COCO_Search18/GazeformerISP/src/opts.py
 M isp-senet/ISP/OSIE/GazeformerISP/src/dataset/dataset.py
 M isp-senet/ISP/OSIE/GazeformerISP/src/models/explanation.py
 M isp-senet/ISP/OSIE/GazeformerISP/src/opts.py
 M isp-senet/SE-Net/common/data.py
 M isp-senet/SE-Net/common/dataset.py
 M isp-senet/SE-Net/src/builder.py
?? AGENTS.md
?? isp-senet/ISP/AiR/
?? isp-senet/SE-Net/common/air_data.py
?? isp-senet/SE-Net/configs/air_useremb.json
?? isp-senet/tests/test_air_and_augmented.py
```

## git diff --stat
```text
 isp-senet/ARCHITECTURE_MAP.md                      |  36 ++-
 isp-senet/CODEBASE_MAP.md                          |  47 ++-
 .../COCO_FV/GazeformerISP/src/dataset/dataset.py   |  17 +-
 .../GazeformerISP/src/models/explanation.py        | 351 ++-------------------
 isp-senet/ISP/COCO_FV/GazeformerISP/src/opts.py    |   1 +
 .../GazeformerISP/src/dataset/dataset.py           |  17 +-
 .../GazeformerISP/src/models/explanation.py        | 287 ++---------------
 .../ISP/COCO_Search18/GazeformerISP/src/opts.py    |   1 +
 .../ISP/OSIE/GazeformerISP/src/dataset/dataset.py  |  24 +-
 .../OSIE/GazeformerISP/src/models/explanation.py   | 224 ++++++++++++-
 isp-senet/ISP/OSIE/GazeformerISP/src/opts.py       |   2 +
 isp-senet/SE-Net/common/data.py                    |  43 ++-
 isp-senet/SE-Net/common/dataset.py                 |  11 +-
 isp-senet/SE-Net/src/builder.py                    |  21 +-
 14 files changed, 441 insertions(+), 641 deletions(-)
```

## git log --oneline --decorate -n 20
```text
fdf1f6b (HEAD -> llm-senet, origin/llm-senet) init new branch
cc7b588 (origin/main, origin/HEAD, main) add gazeformer-isp code map
61cd16b init repo
c1b6ab4 Initial commit
```

## git diff --name-status cc7b588 -- isp-senet
```text
A	isp-senet/AGENTS.md
A	isp-senet/ARCHITECTURE_MAP.md
A	isp-senet/CODEBASE_MAP.md
M	isp-senet/ISP/COCO_FV/GazeformerISP/src/dataset/dataset.py
A	isp-senet/ISP/COCO_FV/GazeformerISP/src/models/explanation.py
A	isp-senet/ISP/COCO_FV/GazeformerISP/src/models/explanation_llm.py
M	isp-senet/ISP/COCO_FV/GazeformerISP/src/models/gazeformer.py
M	isp-senet/ISP/COCO_FV/GazeformerISP/src/opts.py
M	isp-senet/ISP/COCO_FV/GazeformerISP/src/test.py
M	isp-senet/ISP/COCO_FV/GazeformerISP/src/train.py
M	isp-senet/ISP/COCO_Search18/GazeformerISP/src/dataset/dataset.py
A	isp-senet/ISP/COCO_Search18/GazeformerISP/src/models/explanation.py
A	isp-senet/ISP/COCO_Search18/GazeformerISP/src/models/explanation_llm.py
M	isp-senet/ISP/COCO_Search18/GazeformerISP/src/models/gazeformer.py
M	isp-senet/ISP/COCO_Search18/GazeformerISP/src/opts.py
M	isp-senet/ISP/COCO_Search18/GazeformerISP/src/test.py
M	isp-senet/ISP/COCO_Search18/GazeformerISP/src/train.py
M	isp-senet/ISP/OSIE/GazeformerISP/src/dataset/dataset.py
A	isp-senet/ISP/OSIE/GazeformerISP/src/models/explanation.py
A	isp-senet/ISP/OSIE/GazeformerISP/src/models/explanation_llm.py
M	isp-senet/ISP/OSIE/GazeformerISP/src/models/gazeformer.py
M	isp-senet/ISP/OSIE/GazeformerISP/src/opts.py
M	isp-senet/ISP/OSIE/GazeformerISP/src/test.py
M	isp-senet/ISP/OSIE/GazeformerISP/src/train.py
M	isp-senet/SE-Net/common/data.py
M	isp-senet/SE-Net/common/dataset.py
M	isp-senet/SE-Net/src/builder.py
A	isp-senet/architecture.md
A	isp-senet/tests/test_configuration_and_checkpoint.py
A	isp-senet/tests/test_explanation_core.py
A	isp-senet/tests/test_predictor_replicas.py
```

## git ls-files --others --exclude-standard isp-senet
```text
isp-senet/ISP/AiR/GazeformerISP/src/dataset/__init__.py
isp-senet/ISP/AiR/GazeformerISP/src/dataset/dataset.py
isp-senet/ISP/AiR/GazeformerISP/src/models/__init__.py
isp-senet/ISP/AiR/GazeformerISP/src/models/_load_canonical.py
isp-senet/ISP/AiR/GazeformerISP/src/models/explanation.py
isp-senet/ISP/AiR/GazeformerISP/src/models/explanation_llm.py
isp-senet/ISP/AiR/GazeformerISP/src/models/gazeformer.py
isp-senet/ISP/AiR/GazeformerISP/src/models/loss.py
isp-senet/ISP/AiR/GazeformerISP/src/models/models.py
isp-senet/ISP/AiR/GazeformerISP/src/models/positional_encodings.py
isp-senet/ISP/AiR/GazeformerISP/src/models/sampling.py
isp-senet/ISP/AiR/GazeformerISP/src/opts.py
isp-senet/ISP/AiR/GazeformerISP/src/test.py
isp-senet/ISP/AiR/GazeformerISP/src/train.py
isp-senet/SE-Net/common/air_data.py
isp-senet/SE-Net/configs/air_useremb.json
isp-senet/tests/test_air_and_augmented.py
```

## Final verification and scope inventory - September 28, 2026

Final source pytest: 52 passed, 1 skipped in 43.84s. See TEST_RESULTS.md for the
exact command and the SciPy-dependent Gaussian parity skip. No model weights or
packages were installed. Final tracked source changes outside isp-senet: none.

Final CRLF-aware scoped git diff --check exit code: 0.

The following inventory includes pre-existing working-tree changes plus this
remediation; it is not a claim that every file was first created in this pass.

```text
isp-senet/ARCHITECTURE_MAP.md
isp-senet/CODEBASE_MAP.md
isp-senet/ISP/AiR/GazeformerISP/README.md
isp-senet/ISP/AiR/GazeformerISP/src/dataset/__init__.py
isp-senet/ISP/AiR/GazeformerISP/src/dataset/dataset.py
isp-senet/ISP/AiR/GazeformerISP/src/models/__init__.py
isp-senet/ISP/AiR/GazeformerISP/src/models/_load_canonical.py
isp-senet/ISP/AiR/GazeformerISP/src/models/explanation.py
isp-senet/ISP/AiR/GazeformerISP/src/models/explanation_llm.py
isp-senet/ISP/AiR/GazeformerISP/src/models/gazeformer.py
isp-senet/ISP/AiR/GazeformerISP/src/models/loss.py
isp-senet/ISP/AiR/GazeformerISP/src/models/models.py
isp-senet/ISP/AiR/GazeformerISP/src/models/positional_encodings.py
isp-senet/ISP/AiR/GazeformerISP/src/models/sampling.py
isp-senet/ISP/AiR/GazeformerISP/src/opts.py
isp-senet/ISP/AiR/GazeformerISP/src/runtime.py
isp-senet/ISP/AiR/GazeformerISP/src/test.py
isp-senet/ISP/AiR/GazeformerISP/src/train.py
isp-senet/ISP/AiR/GazeformerISP/src/utils/evaltools/scanmatch.py
isp-senet/ISP/AiR/GazeformerISP/src/utils/evaltools/visual_attention_metrics.py
isp-senet/ISP/AiR/GazeformerISP/src/utils/evaluation.py
isp-senet/ISP/COCO_FV/GazeformerISP/src/dataset/dataset.py
isp-senet/ISP/COCO_FV/GazeformerISP/src/models/explanation.py
isp-senet/ISP/COCO_FV/GazeformerISP/src/models/explanation_llm.py
isp-senet/ISP/COCO_FV/GazeformerISP/src/models/gazeformer.py
isp-senet/ISP/COCO_FV/GazeformerISP/src/models/models.py
isp-senet/ISP/COCO_FV/GazeformerISP/src/models/sampling.py
isp-senet/ISP/COCO_FV/GazeformerISP/src/opts.py
isp-senet/ISP/COCO_FV/GazeformerISP/src/test.py
isp-senet/ISP/COCO_FV/GazeformerISP/src/train.py
isp-senet/ISP/COCO_Search18/GazeformerISP/src/dataset/dataset.py
isp-senet/ISP/COCO_Search18/GazeformerISP/src/models/explanation.py
isp-senet/ISP/COCO_Search18/GazeformerISP/src/models/explanation_llm.py
isp-senet/ISP/COCO_Search18/GazeformerISP/src/models/gazeformer.py
isp-senet/ISP/COCO_Search18/GazeformerISP/src/models/models.py
isp-senet/ISP/COCO_Search18/GazeformerISP/src/models/sampling.py
isp-senet/ISP/COCO_Search18/GazeformerISP/src/opts.py
isp-senet/ISP/COCO_Search18/GazeformerISP/src/test.py
isp-senet/ISP/COCO_Search18/GazeformerISP/src/train.py
isp-senet/ISP/OSIE/GazeformerISP/src/dataset/dataset.py
isp-senet/ISP/OSIE/GazeformerISP/src/models/explanation.py
isp-senet/ISP/OSIE/GazeformerISP/src/models/explanation_llm.py
isp-senet/ISP/OSIE/GazeformerISP/src/models/gazeformer.py
isp-senet/ISP/OSIE/GazeformerISP/src/models/models.py
isp-senet/ISP/OSIE/GazeformerISP/src/models/sampling.py
isp-senet/ISP/OSIE/GazeformerISP/src/opts.py
isp-senet/ISP/OSIE/GazeformerISP/src/test.py
isp-senet/ISP/OSIE/GazeformerISP/src/train.py
isp-senet/README.md
isp-senet/SE-Net/common/air_data.py
isp-senet/SE-Net/common/data.py
isp-senet/SE-Net/common/dataset.py
isp-senet/SE-Net/configs/air_useremb.json
isp-senet/SE-Net/src/builder.py
isp-senet/SE-Net/src/eval_user.py
isp-senet/SE-Net/src/models.py
isp-senet/audit/AUDIT_REPORT.md
isp-senet/audit/EVIDENCE.md
isp-senet/audit/FINDINGS.md
isp-senet/audit/TEST_RESULTS.md
isp-senet/tests/conftest.py
isp-senet/tests/test_air_and_augmented.py
isp-senet/tests/test_audit_contracts.py
isp-senet/tests/test_configuration_and_checkpoint.py
isp-senet/tests/test_explanation_core.py
isp-senet/tests/test_runtime_integration.py
```

## A-T end-to-end evidence matrix

Paths in this matrix are relative to isp-senet. O/F/C/A refer to the predictor
copies defined by CODEBASE_MAP.md. The complete source locations and defect
before/after descriptions are in FINDINGS.md.

| Required area | Producer -> transformation -> consumer -> objective/evaluation | Evidence / limit |
| --- | --- | --- |
| A Original compatibility | opts(False) -> data without labels -> disabled model -> unchanged scan/RL outputs | Actual four-copy model/data parity; production launchers external |
| B Hierarchy | outs/memory/final logits -> z -> soft episodes -> active global token | models/explanation.py and real HOW backward checks |
| C z | final spatial logits -> weighted memory/XY + native duration + state -> fixation MLP | Numerical STOP, soft evidence and upstream gradients |
| D WHAT | mapped real labels + duration mask -> z-specific prefix/semantic projection -> text/alignment terms | Prefix/padding masks and gradient tests |
| E WHY | z/time/query -> MLP softmax Kmax -> masked episode mean -> route/text/alignment | Padding invariance, partition and weighted-mean tests |
| F HOW | active canonical gold slots -> mean -> query-conditioned global token -> HOW terms | Inactive-slot perturbation and hierarchical gradients |
| G Text models | raw query/gold text -> explicit semantic wrapper -> shared causal prefixes | No cached-query substitution; eval/detach/freeze tests |
| H Objective | branch scalars/lambdas -> loss_explanation -> scan + EXP -> backward | Nonunit/zero-weight tests; real Air-D supervised step |
| I Augmented format | raw X/Y/T and prediction labels -> strict validation -> normalized annotations | Inline and legacy fixtures; no generated-scanpath interpretation |
| J Correspondence | loaded raw indices -> retained prefix/removal map -> shared WHAT/WHY transformation | O/F/C/A data fixtures; identity/removal maps; no coordinate matching |
| K Validation | raw record -> strict IDs/text/partition/length/grid validation -> clear failure | Parameterized malformed-data tests |
| L Truncation | raw-to-model map -> discard labels/drop groups/reorder slots -> guarded HOW | Both schemas reject implicit full-trajectory HOW |
| M Air-D SE-Net | AiR files -> normalized coordinates/questions/subjects -> SE-Net -> table metadata | Pure producer/artifact tests; RGB builder import blocked by torchvision |
| N Air-D base | question-grouped features/raw fixations -> reference targets/vectors -> personalized predictor -> checkpoints/sampling/export | Real CPU lifecycle; metric/RL execution blocked by SciPy/multimatch |
| O Air-D explanation | inline annotation -> collator/build_explanation_inputs -> canonical hierarchy -> joint backward | Actual enabled Air-D step, no stubbed predictor |
| P O/F/C regression | original variant loaders/heads -> unchanged local base semantics -> losses/eval | All three real model + loader fixtures; no full GPU metrics run |
| Q CLI | parser/default/config -> constructor/dataset/stage -> runtime behavior or explicit error | Flags traced; Air-D weights wired; unsupported generation rejected |
| R Checkpoint | old/new state -> whole-branch key validation -> optimizer/schedule policy -> resume/inference | Partial states fail; actual Air-D save/load/resume |
| S Optional deps | cached-feature entry -> lazy image/text construction -> baseline without transformers/torchvision | Smoke subprocesses explicitly prohibit those imports |
| T Tests | deterministic CPU fixtures -> real data/model/math paths -> assertions | 52 pass / one external skip; historical global stub removed |

## Deliberate non-redesign boundaries

The baseline second duration parameter is not reinterpreted. The original
O/F/C scanpath objective, RL reward, support selection and inference grouping
remain unchanged. Air-D consumes the same offline table boundary; row metadata
makes an existing remap explicit rather than introducing joint SE-Net training.
R0 has no deployment episode-count rule, so generation requests fail clearly.
Historical unsupported baseline fine-tune branches, SE-Net duration-position
behavior and support-policy semantics are documented rather than silently
redefined. Published metric/data-provenance claims require external assets.
