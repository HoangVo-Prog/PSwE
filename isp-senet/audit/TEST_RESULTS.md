# Test and validation results

Audit/remediation dates: September 27-28, 2026. The final source test run was on
September 28, 2026. No packages or pretrained weights were installed/downloaded.

## Environment

- Active interpreter: Python 3.13.2 (Anaconda, Windows AMD64).
- PyTorch: 2.11.0+cpu. Tests use CPU and deterministic seeds.
- torchvision, SciPy, Detectron2, fvcore, multimatch_gaze, TensorBoard and Pillow
  are absent in this interpreter. transformers is installed, but no production
  semantic/causal checkpoint was supplied or downloaded.
- The existing alternate Python 3.11 environment has SciPy/Pillow but lacks
  PyTorch and pytest. Environments were not modified or mixed.
- Model smoke subprocesses explicitly prohibit transformers/torchvision imports.
  Injected tiny causal/semantic modules supply only the language networks; the
  actual predictor Transformer, heads, datasets, collators, router, pooling,
  language-prefix wrapper, losses, sampler and checkpoint code execute normally.

## Commands and exact observed outcomes

Unless prefixed with `isp-senet/`, pytest commands below were run with working
directory `isp-senet`. Root-prefixed forms were run from the repository root.

| Stage | Command | Observed outcome |
| --- | --- | --- |
| Baseline focused | `python -m pytest -q tests/test_explanation_core.py tests/test_air_and_augmented.py` | 14 passed in 5.04s |
| Baseline full | `python -m pytest -q tests` | 20 passed in 17.23s |
| Initial repairs | `python -m pytest -q tests/test_explanation_core.py tests/test_configuration_and_checkpoint.py` | 11 passed in 4.71s |
| New audit checks, first collection | `python -m pytest -q tests/test_audit_contracts.py tests/test_explanation_core.py tests/test_air_and_augmented.py` | 2 collection errors: missing scipy after restoring the previously ignored Gaussian branch |
| New audit checks, second run | `python -m pytest -q isp-senet/tests/test_audit_contracts.py isp-senet/tests/test_explanation_core.py isp-senet/tests/test_air_and_augmented.py` | 1 failed, 34 passed in 5.22s: no_grad alone left a parameter view reporting requires_grad=True; encode_target now explicitly detaches |
| Real predictor tests, first run | `python -m pytest -q isp-senet/tests/test_runtime_integration.py` | 1 failed, 3 passed in 23.66s: a test helper changed sys.path and imported the OSIE trainer instead of Air-D; test isolation corrected |
| Real predictor tests, next run | `python -m pytest -q isp-senet/tests/test_runtime_integration.py` | 4 passed in 29.18s |
| Expanded full suite | `python -m pytest -q tests` | 45 passed in 54.97s |
| Added real O/F/C loader checks | `python -m pytest -q isp-senet/tests/test_runtime_integration.py` | 1 failed, 3 passed in 26.60s: COCO-FV imported unused sklearn/KFold; unused import removed, no dataset behavior mocked |
| Dataset/lifecycle focused | `python -m pytest -q isp-senet/tests/test_runtime_integration.py isp-senet/tests/test_air_and_augmented.py` | 16 passed, 1 skipped in 27.78s |
| Full pre-final verification | `python -m pytest -q -rs tests` | 52 passed, 1 skipped in 43.25s |
| **Final source verification** | `python -m pytest -q -rs tests` | **52 passed, 1 skipped in 43.84s** |

Final skip:

```text
SKIPPED tests/test_air_and_augmented.py:207:
Gaussian target parity requires the existing SciPy dependency
```

The no-blur data path is tested explicitly, not substituted for a passing blur
test. Gaussian parity has its own test and remains unexecuted in this environment.

## Additional executed validation

- Parsed all 114 scoped Python files with `ast.parse`, without importing them or
  writing bytecode: PASS. Existing invalid-escape SyntaxWarnings in SE-Net
  losses/utils/sinkhorn/deformable-attention docstrings remain; no syntax errors.
- CRLF-aware scoped whitespace validation from repository root:
  `git -c core.safecrlf=false -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check -- isp-senet`.
  A newly exposed final whitespace-only line in SE-Net/src/models.py was removed.
  The final scoped check passes. Git's line-ending conversion notices are not test
  failures. No repo-wide reformatting was performed.
- `python -c "import torch; print(torch.__version__); import torchvision; print(torchvision.__version__)"`:
  printed 2.11.0+cpu, then ModuleNotFoundError: torchvision.
- From `isp-senet`, `python -c "import sys; sys.path.insert(0,'SE-Net'); import src.builder"`:
  ModuleNotFoundError: torchvision in common/dataset.py. No SE-Net RGB model was constructed.
- From `isp-senet`, `python -c "import sys; sys.path.insert(0,'ISP/AiR/GazeformerISP/src'); import utils.evaluation"`:
  ModuleNotFoundError: scipy in the reference metric utility. Metric/RL numerical
  execution is blocked; no mock metric result is reported as a pass.
- Repository status, scoped diff/stat, four-entry commit history, changed-file
  lists and untracked files were inspected. Baseline details are in EVIDENCE.md.

## Verified behavior, not just file existence

- All four real predictor copies: disabled train/inference output parity after
  controlled state loading; no explanation keys in disabled state_dict; spatial
  STOP exclusion; HOW backprop to encoder, decoder, fixation encoder and router;
  eval-mode autograd without explanation loss; CPU sampling and structured output.
- O/F/C real datasets and collators: feature/task/subject grouping, same base
  targets with explanations on/off, explicit raw maps, STOP masking, truncation
  guard and malformed-length rejection, without replacing data/model logic.
- Air-D full synthetic lifecycle: dataset -> collator -> enabled joint supervised
  step -> backward; separate disabled training -> checkpoint -> loaded inference
  -> repeated serialized predictions -> optimizer/scheduler resume. Loaded model
  tensors are compared with the saved checkpoint. Missing-checkpoint inference fails.
- Membership partition, strict IDs and texts, removed fixation reordering, sidecar
  identities, per-example temporal normalization, soft weighted means, active HOW
  slot pooling, shared LLM, prefix/padding labels, frozen target behavior, nonunit
  and zero loss weights, partial checkpoint rejection, controlled optimizer reset.
- SE-Net pure Air-D normalization and subject-table metadata export/consumption.
  These are not a replacement for running the actual RGB/deformable-attention model.

## Not executed / not established

- Real pretrained RoBERTa/causal-LLM teacher forcing, accelerator/mixed-precision
  training, GPU/DataParallel O/F/C launcher runs, real old production checkpoints,
  and dataset-scale metrics: no matching external artifacts/runtime supplied.
- SE-Net RGB training/export and compiled deformable attention: external dependency
  stack unavailable. Static paths and pure data/artifact contracts were audited.
- Full Air-D RL rewards and benchmark metrics: active Python lacks SciPy and
  multimatch_gaze. The original reward math is wired, not numerically certified.
- Published experiment parity, choice of SE-Net support/image settings, support
  provenance and pretrained feature-grid provenance: require actual experiment
  assets. No accuracy or aggregate quality score is claimed.
