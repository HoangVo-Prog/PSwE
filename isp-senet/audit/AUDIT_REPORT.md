# ISP-SENet audit and remediation report

Audit baseline: September 27, 2026. Final verification: September 28, 2026.

## 1. Executive summary

The initial implementation was not compliant despite its 20 passing tests.
Thirteen evidence-backed findings were identified and addressed: **1 BLOCKER,
7 HIGH, 4 MEDIUM, 1 LOW**. Highest-impact defects were the nonfunctional Air-D
train/save/load lifecycle, incorrect legacy/removed-fixation supervision,
ambiguous sidecar identity, and unsafe checkpoint migration.

The final offline suite reports **52 passed, 1 skipped**. All four actual
predictor models and their dataset/collator paths were exercised with synthetic
features. Air-D training, checkpoint-backed inference, serialization and resume
were executed. The remaining unverified work requires external dependencies or
production assets, not invented architecture choices. In particular, **full
SE-Net RGB execution and real Air-D metric/RL evaluation are NOT VERIFIED**.
This is not a blanket certification based on the green tests.

## 2. Audit scope

Audited and modified only `isp-senet/**`. Root `AGENTS.md` and the in-scope
contract were followed, with the user's explicit remediation request overriding
the audit-only default. `gazeformer-isp/AiR/**` was read only for reference data,
training, sampling, preprocessing and metrics. Its metric utility sources were
copied into the local Air-D path so there is no sibling runtime dependency.
`gazeformer-isp` itself and `llada` were not changed or audited.

Persistent audit artifacts are `audit/AUDIT_REPORT.md`, `audit/FINDINGS.md`,
`audit/TEST_RESULTS.md`, and `audit/EVIDENCE.md`. Test fixtures use temporary
directories. No package installation, pretrained download, branch creation,
commit, or unrelated architectural redesign was performed.

## 3. Repository/git state inspected

- Baseline HEAD: `fdf1f6b`, branch `llm-senet`, also `origin/llm-senet`.
- Earlier base: `cc7b588` (`main`, `origin/main`, `origin/HEAD`); history also
  contains `61cd16b` and `c1b6ab4`. There were only four visible commits.
- Baseline had 14 tracked modified files plus untracked Air-D source, SE-Net
  Air-D parser/config, an Air-D test file and the repository-root AGENTS.md.
  This was not a clean checkout or a single completed commit.
- Inspected `git status --short`, scoped diff/stat, commit log, and changed-file
  names relative to the base. EVIDENCE.md preserves the pre-remediation state and
  inventories. Existing changes were preserved, then repaired where necessary.
- No generated checkpoints, fixture tensors, downloaded weights or audit outputs
  were added to source paths outside the intended audit directory. Runtime
  pytest/bytecode caches are not included in the deliverable.

## 4. Source-of-truth documents used

1. Repository-root `AGENTS.md`: required A-T audit areas and evidence standards.
2. `isp-senet/AGENTS.md`: plug-and-play, R0, query, duration and migration contract.
3. `architecture.md`: latent hierarchy, equations and branch supervision.
4. `ARCHITECTURE_MAP.md`: personalized memory, final action logits, duration-mask
   semantics, query encoding, losses and intended integration boundaries.
5. `CODEBASE_MAP.md`: original O/F/C differences and inherited pipeline hazards.
6. Actual source/tests, including every applicable predictor copy.
7. Read-only AiR reference code where the new path needed base-behavior evidence.

The mathematical specification was not edited. Historical map claims were not
accepted as execution proof. Current addenda distinguish historical plans and
baseline hazards from tested behavior and explicitly label integration choices.

## 5. What was implemented

The starting tree already had the z -> WHAT / soft WHY-R0 / active-slot HOW-R0
hierarchy, shared-LLM adapters, opt-in O/F/C training wiring, augmented-record
loading and initial Air-D/SE-Net support. F/C mathematical adapters and Air-D
local model adapters were already present; dataset/predictor copies were not
merged or bulk overwritten during this pass.

Remediation addressed:

- One validated alignment transformation for inline and legacy labels, stricter
  source validation, identity-safe sidecars, and post-removal episode ordering.
- Correct per-example temporal normalization and deterministic detached semantic
  targets; frozen text networks retain prefix autograd and explicit padding.
- Controlled whole-branch checkpoint migration and optimizer/scheduler reset;
  unrelated/partially missing keys fail. Subject tables remain separate artifacts.
- Air-D supervised/RL lifecycle, metric/validation wiring, checkpoint persistence,
  config reconstruction, loaded inference and question/subject-aware serialization.
- Reference Air-D targets, structured evaluation vectors and question grouping.
- Air-D SE-Net small/few-shot export safety and source-subject row metadata through
  predictor consumption; one-question task conditioning remains active.
- End-to-end CPU regression coverage, early CLI failures for unsupported requests,
  and documentation that does not imply undefined R0 inference was implemented.

## 6. What was actually verified

`tests/test_runtime_integration.py` runs each real local Transformer and predictor
head in a separate process, with optional text/backbone imports prohibited.
For O/F/C/A it checks base output parity with an explanation-enabled model loaded
from the same base state, train/inference keys and shapes, continuous hierarchical
gradients, eval-mode autograd, CPU Sampling and structured scanpaths.

The same tests use real O/F/C datasets/collators, not replacements, for target
cells, durations, subject/task grouping, raw maps, STOP masks, truncation and
malformed raw lengths. Air-D executes its enabled dataset -> collator -> joint
supervised objective -> backward path. Its disabled CLI performs training,
checkpoint save/load, repeated inference/JSON export and optimizer/scheduler
resume, with saved/loaded tensor equality checked.

Focused tests verify partition validation, strict IDs/texts, removed groups,
sidecar uniqueness, temporal-padding invariance, soft episode weighted means,
active-slot HOW pooling, prefix/padding masks, shared LLM gradients, frozen target
behavior, nonunit/zero weights, checkpoint key validation and table-row metadata.

Not verified by those tests: real pretrained text-model execution, SE-Net RGB
training/export, production checkpoints, metric-library numerical results, GPU
launchers, real-data quality or published experiment parity.

## 7. Findings by severity

All findings retain stable IDs and before/after evidence in FINDINGS.md.

| Severity | Findings | Status |
| --- | --- | --- |
| BLOCKER | AUD-008: Air-D lifecycle/inference on random weights | Fixed; CPU lifecycle executed |
| HIGH | AUD-001 legacy mapping; AUD-002 surviving episode order; AUD-003 sidecar identity; AUD-005 temporal normalization; AUD-007 checkpoint migration; AUD-009 Air-D data/evaluation; AUD-010 SE-Net/table row boundary | Fixed in source; SE-Net full runtime and Gaussian parity have external limits |
| MEDIUM | AUD-004 malformed supervision; AUD-006 frozen text/padding; AUD-011 CLI propagation; AUD-013 single-question conditioning | Fixed; production text/SE-Net runtime remains unverified |
| LOW | AUD-012 coverage/documentation mismatch | Improved with behavior tests and current documentation |

No correctness percentage, numeric quality score or aggregate grade is assigned.

## 8. Architecture compliance

### Shared fixation token

`P/models/gazeformer.py` passes `outs` and personalized `memory_task`, each
explicitly transposed from [L,N,H]/[G,N,H] to batch-first, before Sampling.
The explanation distribution is softmax of **final aggregated** spatial action
logits, excluding action zero (STOP), not a head-local attention map.
Inference diagnostics also retain the original final logits instead of relying
on a potentially underflowed full-action softmax renormalization.

`FixationReasoningEncoder` uses a differentiable weighted visual sum, row-major
expected XY, position MLP, decoder projection and both native duration outputs.
No sampled/NumPy/argmax path enters z. The neutral duration adapter deliberately
does not decide whether the baseline second parameter means variance or scale.
The original scanpath loss and sampling distribution are unchanged.

### WHAT, WHY-R0 and HOW-R0

- WHAT uses only masked real-fixation z tokens and their own branch/query prefixes.
  Gold labels exclude the prefix and padded tokens; targets use detached semantic
  encoding of the correct gold WHAT text.
- WHY input is z + per-example temporal position + semantic query, followed by a
  simple MLP and softmax over Kmax. Gold grouping is only a routing target, never
  router input. Padded rows have no route loss/mass. Episode tokens are soft
  weighted means, with no hard assignment before aggregation.
- Active canonical gold slots alone receive WHY language/alignment supervision.
  No episode-presence/count head or contextual router Transformer was introduced.
- HOW takes the mean of active gold episode tokens and query context. It is the
  trajectory-level global token, not generated WHY text. Unused episode tokens
  are excluded. HOW gradients reach episode weights, router, z and the predictor.
- There is one causal LLM per explanation module, with separate WHAT/WHY/HOW
  branch embeddings, query projections and latent projections. Raw query text
  reaches the configured semantic encoder; cached base task vectors are not
  renamed or substituted for it.
- Gold target encoding is eval/no_grad/detached; query encoding follows the
  configured freeze policy. Freezing the LLM does not suppress prefix autograd.

### Objective and stages

All top-level and sub-loss weights reach the model in O/F/C/A. The actual
supervised callers add `loss_explanation` exactly once to action CE plus weighted
native duration loss before `backward()`. Zero weights are tested. O/F/C RL
remains on the original eval-mode/autograd path without explanation loss. Air-D
uses the reference reward/advantage objective; its numerical metric execution
is externally blocked, not falsely reported as tested.

## 9. Dataset/alignment compliance

The inline `prediction` object supplies pseudo-gold WHAT/WHY/HOW, never a model
scanpath. X/Y/T (or AiR T_start/T_end converted to T) supplies the human sequence.
IDs are validated as exact 1-based raw indices, WHY regions partition real
fixations, and real text targets cannot be blank or None. Zero-length and
malformed trajectories fail rather than silently losing supervision.

| Dataset | Active supervised transformation | Raw-to-model evidence |
| --- | --- | --- |
| OSIE | Original coordinate quantization and ms->seconds; truncate/pad | Retained prefix raw IDs explicitly map to consecutive tokens; no removal/merge in this loader |
| COCO-Freeview | Original resize/downscale convention and ms->seconds; truncate/pad | Same explicit prefix mapping; original task-feature choice and grouping preserved |
| COCO-Search18 | Original task/path grouping and quantization; truncate/pad | Same explicit prefix mapping; no assumed coordinate-based correspondence |
| Air-D | Per-record dimensions to row-major action grid, ms->seconds; truncate/pad | Explicit prefix map; malformed/out-of-image fixations rejected, not clamped or silently removed |

Sidecars must uniquely identify the source record and agree with its raw data;
original subject identity survives table remapping. Mapping validation occurs
before discarding entries beyond max_length. The generic mapping helper also
covers actual removals, drops empty regions and recanonicalizes surviving slots.
No approximate coordinate matching is used.

HOW for a full raw trajectory is not automatically valid after truncation.
Both annotation formats reject that combination unless allow_truncated_how is
explicitly set for an independently justified annotation. This flag is not an
annotation repair. Raw correspondence is verified from the supplied augmented
record; provenance of externally preprocessed annotation files remains external.

## 10. Air-D SE-Net compliance

`common/air_data.py` normalizes AiR image/question IDs, source subject_idx,
per-record dimensions, raw coordinates and millisecond durations. Missing/invalid
coordinate dimensions and nonnumeric subject IDs fail. `src/builder.py` dispatches
Air-D/AiR/Air aliases, loads train/validation/test files, and uses the configurable
validation split. Question-specific cached vectors reach the existing task
projection, including single-question datasets.

No center fixation is injected for Air-D. RGB size and max support length remain
explicit SE-Net configuration choices, separate from predictor grid/length. Their
experimental suitability is not established by code alone. The existing offline
support-selection policy is retained rather than replaced by an invented one.

Air-D export retains all support batches, avoids unnecessary triplet sampling in
evaluation, handles small top-k dimensions and rejects empty subject rows.
Few-shot exports average selected support and preserve the pretrained classifier
shape; they do not claim unseen-subject classifier accuracy. Source IDs in tensor
row order are emitted beside the table and consumed by the predictor. O/F/C
export policy is not redesigned.

Pure parsing/tensor-metadata tests pass. Full builder -> RGB network -> export is
**NOT VERIFIED**: import fails at unavailable torchvision, before the additional
Detectron2/fvcore/deformable-attention stack or external backbone assets.

## 11. Air-D predictor compliance

Compared to read-only AiR reference data/train/test/sampling/metric sources:

- Question grouping, flat image features, question-ID task embeddings, per-record
  scaling, duration units, answer performance and subject ordering are explicit.
- Gaussian targets and padded STOP semantics match the reference source.
  Gaussian numerical parity has a dedicated test skipped for missing SciPy.
- The local personalized Transformer consumes the offline SE-Net table; metadata
  maps subset rows without changing the shared explanation mathematics.
- The real supervised objective, Sampling, structured evaluation vectors,
  checkpoint lifecycle and serialized source subject/question IDs are exercised.
- Reference metric utilities are local copies; per-repeat evaluation avoids
  overflowing fixed subject matrices. RL retries fail clearly instead of hanging
  forever on nonfinite rewards. Reward screen dimensions follow configured size.

Full benchmark metrics and RL rewards are **NOT VERIFIED**, because importing the
metric stack fails at SciPy and multimatch_gaze is also absent. No production
feature/cache/model compatibility or benchmark score is inferred from synthetic
fixtures.

## 12. Plug-and-play/backward compatibility

- All parsers default enable_explanation=False. Disabled data requires no
  explanation fields or sidecar, and disabled models construct no semantic/causal
  model or tokenizer. Real-model tests prohibit optional imports.
- Baseline output keys, train/inference duration-shape differences, losses,
  query-position initialization, subject-table boundary and O/F/C dataset-specific
  behavior are preserved. No whole predictor copy was replaced.
- Feature-only predictors no longer import unused image-extraction dependencies;
  actual feature extraction and Gaussian smoothing still require their libraries.
- The CPU sampler correction changes only random-tensor device placement, not
  sampling semantics. O/F/C evaluation/grouping/serialization logic was not rewritten.
- Whole base checkpoint migration is explicit. Partial new explanation states and
  unrelated missing/unexpected keys fail. Same-model optimizer state is restored;
  changed parameter sets reset optimizer/schedule and training counters.
- Air-D saved config reconstructs model dimensions and text models. Remapped-table
  row metadata is checked. Old full subject tables remain usable without metadata;
  real externally supplied tables/checkpoints were not available for inspection.
- Undefined CLI explanation generation fails early. Programmatic latent diagnostics
  require explicit query/mask. No deploy-time K* policy or chained generated text
  was invented.

Inherited baseline hazards explicitly outside a redesign include the duration
variance/scale ambiguity, historical O/F/C fine-tune assumptions and original
SE-Net duration-position/support-selection behavior. They are recorded in the
historical CODEBASE_MAP; changing their scientific semantics would violate the
preserve-original-pipeline contract and is not represented as a remediation.

## 13. Test results

Final command from `isp-senet`: `python -m pytest -q -rs tests`.

**52 passed, 1 skipped in 43.84s.** The sole skip is Gaussian target numerical
parity, requiring SciPy. All 114 scoped Python files parse successfully; existing
legacy invalid-escape warnings remain. Scoped CRLF-aware whitespace validation
passes. TEST_RESULTS.md records commands, earlier failures, corrections and exact
external import blockers. Passing tests are not substituted for unavailable
production or SE-Net validation.

## 14. Documentation consistency

README now lists Air-D and describes opt-in explanations. The Air-D README
specifies data, CLI, artifacts, remapping, lifecycle, supported/unsupported
inference and external prerequisites. Architecture/codebase map addenda identify
historical content, implemented repairs and unresolved scientific/deployment
choices. The architecture specification and original duration interpretation
remain unchanged. FINDINGS.md cites actual source symbols and observed behavior,
not merely comments or parser declarations.

## 15. Unverified items / environment limitations

1. SE-Net RGB training/export and deformable attention: missing torchvision,
   Detectron2/fvcore/compiled stack and no production backbone/checkpoint supplied.
2. Gaussian numeric parity, reference Air-D metrics and RL reward execution:
   missing SciPy/multimatch_gaze in the active interpreter.
3. Real semantic/causal pretrained models, GPU/DataParallel, production O/F/C
   launchers and actual old/new external checkpoints: matching runtime/assets
   were not supplied; tests intentionally use CPU and tiny injected text models.
4. Dataset/support/query provenance, cache grid provenance, image/support config
   choices and published experiment reproduction: require external experiment
   artifacts, not guessed architecture decisions.

No arbitrary package installation or silent data repair was used to hide these
limits. No remaining demonstrated in-scope implementation defect from this audit
is intentionally left unaddressed; this statement does not prove the absence of
undiscovered defects in unexecuted external integrations.

## 16. Final implementation status

Statuses refer to the evidence described here, not a single overall grade.

| Area | Status | Evidence / limitation |
| --- | --- | --- |
| Base pipeline compatibility | PASS WITH ISSUES | Real CPU model/data parity; original heavy launchers and external artifacts not executed |
| WHAT | PASS | Real fixation masking, prefix labels, semantic targets, gradients |
| WHY-R0 | PASS | MLP routing, per-example time, gold-target-only membership, soft pooling and masks |
| HOW-R0 | PASS | Active gold-slot mean, query-conditioned global token and hierarchical gradients |
| Joint objective | PASS | Weighted/zero terms; actual Air-D supervised step; O/F/C backward wiring traced |
| Dataset format | PASS | Both schemas validated; malformed/ambiguous data rejected |
| Raw-to-model alignment | PASS | Real O/F/C/A loader fixtures plus removal/truncation/canonical ordering tests |
| Air-D SE-Net | NOT VERIFIED | Pure contracts/static fixes checked; full RGB/export blocked externally |
| Air-D ISP predictor | PASS WITH ISSUES | Train/save/load/resume/export executed; metrics/RL numerical checks external |
| Air-D explanation integration | PASS | Actual data/collator/joint step/backward with shared hierarchy and injected text models |
| O/F/C regression | PASS WITH ISSUES | Real model and dataset tests; GPU launcher/full metrics and inherited hazards not redesigned |
| Checkpoint compatibility | PASS WITH ISSUES | Controlled synthetic migration/resume; production checkpoint/table provenance external |
| Tests | PASS WITH ISSUES | 52 pass, one explicitly external skip; no large-model/download/CUDA requirements |
| Documentation | PASS | Current contracts, historical caveats and verification boundaries documented |
