# Audit findings and remediation

Baseline: September 27, 2026 working tree; remediation completed September 28, 2026.
The explicit user request authorized source fixes inside isp-senet. All 13 findings below are addressed in source; external runtime verification limits are not marked as passing. Counts: 1 BLOCKER, 7 HIGH, 4 MEDIUM, 1 LOW.

## AUD-001 - Legacy sidecars bypassed raw-to-model and HOW truncation rules

**Severity:** HIGH

**Area:** Dataset / raw-to-model / HOW

**Classification:** VERIFIED DEFECT

**Remediation status:** FIXED IN SOURCE (see verification limits below).

**Expected**
Inline and legacy labels must follow the actual retained fixations, with recomputed active episode slots and explicit truncated-HOW semantics.

**Observed**
Before: The legacy branch sliced WHAT/IDs positionally, ignored the supplied mapping and allow_truncated_how, and retained the original episode_count and full HOW.

After: Both formats now feed _align_explanation_targets; removed/empty groups are dropped and full-trajectory HOW on a shortened sequence fails unless explicitly authorized.

**Evidence**
- `ISP/OSIE/GazeformerISP/src/models/explanation.py:801` (`def normalize_explanation_annotation`)
- `ISP/OSIE/GazeformerISP/src/models/explanation.py:746` (`def _align_explanation_targets`)

**Impact**
Training could attach descriptions to the wrong fixation and supervise HOW with invisible trajectory content.

**Reproduction / verification**
tests/test_audit_contracts.py::test_legacy_sidecars_use_mapping_and_recompute_surviving_slots; real O/F/C dataset truncation checks in test_runtime_integration.py.

**Recommended fix**
Use one validated alignment transformation for both schemas; implemented.

**Confidence**
High for the source defect and repair. This is not a claim of production-data or unavailable-dependency validation.

## AUD-002 - Surviving WHY episodes were not recanonicalized

**Severity:** HIGH

**Area:** WHY-R0 / alignment

**Classification:** VERIFIED DEFECT

**Remediation status:** FIXED IN SOURCE (see verification limits below).

**Expected**
Gold episode slots must be ordered by the first surviving model-visible fixation.

**Observed**
Before: Inline regions were sorted by the first raw fixation before removal. When raw 1 was removed, a region first observed at raw 3 could still precede the region now starting at raw 2.

After: Surviving regions are sorted by mapped model index before assigning membership columns and WHY texts.

**Evidence**
- `ISP/OSIE/GazeformerISP/src/models/explanation.py:776` (`active_regions.sort`)

**Impact**
Router targets and semantic episode ordering disagreed with the architecture after preprocessing removals.

**Reproduction / verification**
tests/test_audit_contracts.py::test_surviving_episodes_reorder_by_first_model_fixation.

**Recommended fix**
Recanonicalize after the mapping, never by floating-point coordinate matching; implemented.

**Confidence**
High for the source defect and repair. This is not a claim of production-data or unavailable-dependency validation.

## AUD-003 - Sidecar lookup could select another question or observer

**Severity:** HIGH

**Area:** Dataset identity

**Classification:** VERIFIED DEFECT

**Remediation status:** FIXED IN SOURCE (see verification limits below).

**Expected**
Select one annotation for the same image, subject and task/question, or a unique explicit sample ID.

**Observed**
Before: The old list lookup compared name/subject only. AiR records lacking name/subject could all compare as None; COCO trials with different tasks/conditions could collide. Subject-remapping also erased source lookup identity.

After: Lookup uses explicit IDs or full identity, rejects ambiguous matches and divergent raw arrays, and preserves source subject IDs across remapping.

**Evidence**
- `ISP/OSIE/GazeformerISP/src/models/explanation.py:866` (`def lookup_explanation_annotation`)
- `ISP/OSIE/GazeformerISP/src/dataset/dataset.py:75` (`explanation_source_subject`)
- `ISP/AiR/GazeformerISP/src/dataset/dataset.py:108` (`self.subject_ids = None`)

**Impact**
Plausible but unrelated pseudo-gold language could be optimized without an exception.

**Reproduction / verification**
tests/test_audit_contracts.py::test_sidecar_identity_includes_question_subject_and_condition; table metadata mapping fixture.

**Recommended fix**
Require unique identity and validate raw correspondence before normalization; implemented.

**Confidence**
High for the source defect and repair. This is not a claim of production-data or unavailable-dependency validation.

## AUD-004 - Malformed annotations were coerced or silently unsupervised

**Severity:** MEDIUM

**Area:** Dataset validation

**Classification:** VERIFIED DEFECT

**Remediation status:** FIXED IN SOURCE (see verification limits below).

**Expected**
Reject malformed fixation IDs, partitions, raw lengths, invalid coordinates, and missing supervision.

**Observed**
Before: int(value) accepted fractional and boolean IDs; str(None) became a target; blank WHAT/WHY labels were silently skipped. Zero-length records and invalid discarded mapping entries could pass early validation.

After: IDs are strict integers; real labels are nonempty strings; mappings are unique, bounded and temporal; enabled data rejects invalid real-fixation/STOP mappings and nonfinite or nonpositive durations.

**Evidence**
- `ISP/OSIE/GazeformerISP/src/models/explanation.py:579` (`def _fixation_id`)
- `ISP/OSIE/GazeformerISP/src/models/explanation.py:599` (`def _validate_inline_annotation`)
- `ISP/OSIE/GazeformerISP/src/models/explanation.py:694` (`def _normalise_model_indices`)
- `ISP/OSIE/GazeformerISP/src/dataset/dataset.py:152` (`Explanation fixation cannot map`)

**Impact**
A run could optimize incomplete or corrupted supervision while appearing enabled.

**Reproduction / verification**
Parameterized malformed-ID, empty-label, partition, zero-length and mapping tests in test_audit_contracts.py; O/F/C loader length failures.

**Recommended fix**
Fail with context rather than coercing or repairing labels; implemented. Unknown query text is not fabricated.

**Confidence**
High for the source defect and repair. This is not a claim of production-data or unavailable-dependency validation.

## AUD-005 - WHY temporal positions used padding length instead of real T

**Severity:** HIGH

**Area:** WHY-R0 mathematics

**Classification:** VERIFIED DEFECT

**Remediation status:** FIXED IN SOURCE (see verification limits below).

**Expected**
architecture.md section 13.1 defines PE((t-1)/(T-1)), with padding excluded.

**Observed**
Before: WhyR0Router normalized all examples by padded tensor length L, so changing padding changed valid-fixation routing context.

After: The router receives only a real-fixation mask for lengths (not gold grouping), and normalizes each example independently.

**Evidence**
- `ISP/OSIE/GazeformerISP/src/models/explanation.py:177` (`class WhyR0Router`)
- `ISP/OSIE/GazeformerISP/src/models/explanation.py:213` (`lengths = (fixation_mask.sum`)

**Impact**
Episode assignments depended on padding and violated the stated temporal representation.

**Reproduction / verification**
tests/test_audit_contracts.py::test_router_time_is_independent_of_batch_padding.

**Recommended fix**
Use per-example real counts while keeping the simple MLP router; implemented.

**Confidence**
High for the source defect and repair. This is not a claim of production-data or unavailable-dependency validation.

## AUD-006 - Frozen text wrappers and padded causal batches were fragile

**Severity:** MEDIUM

**Area:** Semantic targets / shared LLM

**Classification:** VERIFIED DEFECT

**Remediation status:** FIXED IN SOURCE (see verification limits below).

**Expected**
Frozen targets must be stable and detached; causal text batching must support padding while preserving prefix gradients.

**Observed**
Before: requires_grad=False did not keep frozen modules in eval mode. Targets from a trainable semantic encoder could include dropout. Auto-tokenizer padding was requested without providing a pad token for tokenizers that lack one.

After: Frozen wrappers enforce eval mode; encode_target uses temporary eval/no_grad plus explicit detach; causal padding is right-sided and reuses existing EOS when needed, otherwise fails clearly.

**Evidence**
- `ISP/OSIE/GazeformerISP/src/models/explanation_llm.py:125` (`def encode_target`)
- `ISP/OSIE/GazeformerISP/src/models/explanation_llm.py:220` (`tokenizer.pad_token = tokenizer.eos_token`)
- `ISP/COCO_FV/GazeformerISP/src/models/explanation_llm.py:70` (`def encode_target`)
- `ISP/COCO_Search18/GazeformerISP/src/models/explanation_llm.py:67` (`def encode_target`)

**Impact**
Semantic targets were stochastic and some configured causal models could not batch gold texts. Frozen LLM gradients must not be disabled globally.

**Reproduction / verification**
Frozen-target and frozen-LLM prefix/padding/gradient tests in test_audit_contracts.py; all replicas execute offline injected language models.

**Recommended fix**
Separate target encoding from trainable query encoding and make padding explicit; implemented. Production pretrained-model execution remains external.

**Confidence**
High for the source defect and repair. This is not a claim of production-data or unavailable-dependency validation.

## AUD-007 - Checkpoint migration allowed partial explanations and broke optimizer resume

**Severity:** HIGH

**Area:** Checkpoint / supervised training

**Classification:** VERIFIED DEFECT

**Remediation status:** FIXED IN SOURCE (see verification limits below).

**Expected**
Only complete base-to-explanation initialization is a migration; unrelated or partial missing keys must fail. Optimizer/scheduler state must match the parameter set.

**Observed**
Before: All missing explanation-prefixed keys were accepted, including damaged explanation checkpoints. O/F/C resumed the old optimizer even when explanation parameters had been added.

After: Key sets are checked before loading; complete base migration is logged, partial explanation states fail, and migration resets optimizer/schedule/epoch/best tracking. Same-model state is restored.

**Evidence**
- `ISP/OSIE/GazeformerISP/src/models/explanation.py:901` (`def load_model_state_with_explanation_migration`)
- `ISP/OSIE/GazeformerISP/src/models/explanation.py:931` (`def restore_training_checkpoint`)
- `ISP/OSIE/GazeformerISP/src/train.py:162` (`migrated = restore_training_checkpoint`)
- `ISP/AiR/GazeformerISP/src/runtime.py:43` (`def read_checkpoint`)

**Impact**
Old optimizer groups could abort migration; damaged new checkpoints could silently train randomly initialized explanation parameters.

**Reproduction / verification**
Partial-checkpoint rejection, unrelated-key rejection, optimizer migration tests; four real predictor migrations; Air-D saved model/scheduler resume and tensor equality.

**Recommended fix**
Use explicit whole-branch migration and controlled optimizer reset; implemented. External subject tables remain separate and are checked for shape/row identity where metadata exists.

**Confidence**
High for the source defect and repair. This is not a claim of production-data or unavailable-dependency validation.

## AUD-008 - Air-D CLI lacked a usable training-to-inference lifecycle

**Severity:** BLOCKER

**Area:** Air-D ISP predictor

**Classification:** VERIFIED DEFECT

**Remediation status:** FIXED IN SOURCE (see verification limits below).

**Expected**
Train, save/resume, supervised/RL stages, validation, checkpoint-backed inference and export must be operational.

**Observed**
Before: The original new train.py only looped supervised batches and returned a model without saving it. test.py built random weights and returned in-memory samples; resume_dir, RL/validation/scheduler options had no effect.

After: Air-D now saves model/optimizer/scheduler/config/epoch checkpoints, supports migration and resume, uses the reference RL objective and metric utilities, validates, and loads a required checkpoint before serializing predictions.

**Evidence**
- `ISP/AiR/GazeformerISP/src/train.py:139` (`def main`)
- `ISP/AiR/GazeformerISP/src/train.py:96` (`def reinforcement_step`)
- `ISP/AiR/GazeformerISP/src/test.py:13` (`def run`)
- `ISP/AiR/GazeformerISP/src/runtime.py:35` (`def checkpoint_path`)

**Impact**
The supplied command-line path could not train an artifact for subsequent inference, and could present random predictions as an evaluation.

**Reproduction / verification**
test_runtime_integration.py::run_air_lifecycle executes train -> checkpoint -> inference -> serialized predictions -> resumed training with real local predictor modules on CPU.

**Recommended fix**
Restore the reference lifecycle without changing its objective or inventing explanation inference; implemented. Full metric/RL numerical execution is blocked by external dependencies.

**Confidence**
High for the source defect and repair. This is not a claim of production-data or unavailable-dependency validation.

## AUD-009 - Air-D loader diverged from reference targets and evaluation contracts

**Severity:** HIGH

**Area:** Air-D base data / evaluation

**Classification:** VERIFIED DEFECT

**Remediation status:** FIXED IN SOURCE (see verification limits below).

**Expected**
Match reference Gaussian targets, STOP padding, structured scanpath vectors, question grouping, and stable subject ordering.

**Observed**
Before: blur_sigma was ignored; coordinates were silently clamped; only the first padded target carried STOP; evaluation vectors were ordinary Nx3 arrays and the collator flattened question groups. Missing answers could compare equal as None.

After: Gaussian smoothing is honored; invalid coordinates fail; STOP padding matches the reference; evaluation vectors are structured and grouped. Subjects are ordered explicitly and duplicate question/subject records fail. Missing answers are not counted as correct.

**Evidence**
- `ISP/AiR/GazeformerISP/src/dataset/dataset.py:61` (`class AiR(`)
- `ISP/AiR/GazeformerISP/src/dataset/dataset.py:256` (`class AiR_rl(`)
- `ISP/AiR/GazeformerISP/src/runtime.py:74` (`def evaluate`)

**Impact**
Training distributions and downstream subject metrics could differ or fail; samples could be assigned to incorrect metric rows.

**Reproduction / verification**
Air-D target-cell, duration, structured-vector, grouping, repeat/export and loader fixtures pass. Gaussian numerical parity is separately skipped because active Python lacks SciPy.

**Recommended fix**
Keep reference data semantics and reject malformed inputs; implemented. Reference evidence: gazeformer-isp/AiR/GazeformerISP/src/dataset/dataset.py, read only.

**Confidence**
High for the source defect and repair. This is not a claim of production-data or unavailable-dependency validation.

## AUD-010 - Air-D subset/few-shot table rows could not be consumed safely

**Severity:** HIGH

**Area:** Air-D SE-Net -> predictor boundary

**Classification:** VERIFIED DEFECT

**Remediation status:** FIXED IN SOURCE (see verification limits below).

**Expected**
Keep original subject identity through support selection, table export and predictor row lookup, including small/few-shot exports.

**Observed**
Before: SE-Net remapped training support IDs but validation used original IDs; export could drop incomplete support batches and small top-k calls could fail. Predictor lookup used raw subject_idx even for remapped subset tables. One-shot export bypassed averaging.

After: Air-D validation IDs follow the support map. Export does not drop support batches or sample unused triplets, rejects empty rows, averages few-shot support and keeps pretrained classifier dimensions. Optional row-ID metadata accompanies Air-D tables and is consumed/checked by the predictor.

**Evidence**
- `SE-Net/src/builder.py:58` (`hparams.Data.subject_ids`)
- `SE-Net/common/dataset.py:133` (`subject_mapping =`)
- `SE-Net/src/eval_user.py:24` (`is_air =`)
- `SE-Net/common/air_data.py:152` (`def save_subject_embeddings`)
- `ISP/AiR/GazeformerISP/src/dataset/dataset.py:108` (`self.subject_ids = None`)

**Impact**
Tables could contain missing rows, unusable indices or embeddings assigned to another observer; few-shot export could fail or be empty.

**Reproduction / verification**
Pure normalization and tensor-export/metadata/consumer tests pass. Full SE-Net builder/export execution is NOT VERIFIED because torchvision and the RGB model stack are absent.

**Recommended fix**
Preserve tensor boundary and add explicit source-ID row metadata for remapped Air-D tables; implemented. O/F/C support-selection and export semantics are not redesigned.

**Confidence**
High for the source defect and repair. This is not a claim of production-data or unavailable-dependency validation.

## AUD-011 - CLI options did not consistently reach supported behavior

**Severity:** MEDIUM

**Area:** CLI / configuration

**Classification:** VERIFIED DEFECT

**Remediation status:** FIXED IN SOURCE (see verification limits below).

**Expected**
Expose all explanation weights across copies, fail early on missing model configuration, and never label latent computation as text generation.

**Observed**
Before: Air-D omitted all ten lambda flags. Generation/latent flags in test scripts lacked raw queries and failed late or did not generate language. Sidecar help still described annotations as always required.

After: Air-D exposes the shared weight surface; enabled CLI configuration is checked early; unsupported CLI generation/diagnostics are rejected explicitly. Programmatic latent diagnostics remain query/mask-gated, and generation requests fail rather than inventing K*.

**Evidence**
- `ISP/AiR/GazeformerISP/src/opts.py:74` (`for name in`)
- `ISP/OSIE/GazeformerISP/src/opts.py:144` (`if args.generate_explanations`)
- `ISP/OSIE/GazeformerISP/src/models/gazeformer.py:97` (`Natural-language explanation generation`)

**Impact**
Experiments could silently use default losses or request a feature the entry point did not implement.

**Reproduction / verification**
Offline parser tests across O/F/C/A and actual Air-D constructor/step flow; source propagation to HierarchicalExplanationModule inspected.

**Recommended fix**
Wire supported options and fail explicitly for undefined inference behavior; implemented. No deployment episode-count policy was added.

**Confidence**
High for the source defect and repair. This is not a claim of production-data or unavailable-dependency validation.

## AUD-012 - Tests and documentation overstated runtime coverage

**Severity:** LOW

**Area:** Tests / documentation

**Classification:** TEST GAP / DOCUMENTATION MISMATCH

**Remediation status:** FIXED IN SOURCE (see verification limits below).

**Expected**
Tests must exercise producer-to-consumer behavior and docs must distinguish implemented contracts from unexecuted plans.

**Observed**
Before: The original 20 tests passed despite the unusable Air-D lifecycle and sidecar defects. Replica tests mostly checked imports/CLI; a global models.models stub leaked across tests. README omitted Air-D and maps mixed historical plans with current claims.

After: Tests now execute real Transformer/predictor/dataset/collator paths, compare disabled outputs, backpropagate HOW and Air-D joint losses, migrate/resume checkpoints, test remapped subject tables, and prohibit optional text/backbone imports in smoke processes. Documentation includes current contracts and external blockers.

**Evidence**
- `tests/test_runtime_integration.py:17` (`def test_real_predictor_hierarchy_and_disabled_parity`)
- `tests/test_audit_contracts.py:95` (`def test_router_time_is_independent_of_batch_padding`)
- `README.md:45` (`### Optional hierarchical explanations`)

**Impact**
A green unit suite was not evidence that the implementation was usable or architecturally compliant.

**Reproduction / verification**
See TEST_RESULTS.md for baseline, failing intermediate checks, fixes and final results. No production-model or SE-Net RGB result is claimed.

**Recommended fix**
Replace existence-only confidence with deterministic behavior/gradient/integration checks; implemented. Gaussian parity remains an explicit external skip.

**Confidence**
High for the source defect and repair. This is not a claim of production-data or unavailable-dependency validation.

## AUD-013 - Air-D query conditioning rejected a valid single-question dataset

**Severity:** MEDIUM

**Area:** Air-D SE-Net conditioning

**Classification:** VERIFIED DEFECT

**Remediation status:** FIXED IN SOURCE (see verification limits below).

**Expected**
The question vector must affect Air-D encoding even when a small support set has only one distinct question.

**Observed**
Before: The builder required at least two question IDs solely because the shared model skipped task conditioning when ntask==1.

After: The arbitrary two-question constraint is removed; the existing task-conditioning branch also runs for Air-D when ntask==1.

**Evidence**
- `SE-Net/src/builder.py:42` (`n_tasks = len`)
- `SE-Net/src/models.py:697` (`if self.ntask != 1 or self.pa.name`)

**Impact**
Valid small Air-D subsets were rejected, or question conditioning would disappear if the assertion were simply removed.

**Reproduction / verification**
Static producer -> task-key lookup -> task_transform -> dorsal token path inspected. Full RGB model execution remains blocked by the external stack.

**Recommended fix**
Enable the existing question-conditioning branch for Air-D, without replacing or adding an encoder; implemented.

**Confidence**
High for the source defect and repair. This is not a claim of production-data or unavailable-dependency validation.

