# AGENTS.md

Repository-level instructions for coding agents performing an **audit** of the current `isp-senet` implementation.

This file is for audit/review work. It does not authorize new feature implementation.

---

## 1. Audit Mission

Audit the complete Codex implementation accumulated so far inside:

```text
isp-senet/**
```

The audit must determine whether the current code faithfully implements the intended project while preserving the original pipeline.

The audit scope includes all work introduced for:

```text
- the plug-and-play hierarchical LLM explanation architecture;
- WHAT / WHY-R0 / HOW-R0;
- explanation CLI/configuration;
- explanation losses and gradient flow;
- augmented WHAT/WHY/HOW dataset loading;
- raw-fixation -> model-token alignment;
- Air-D / AiR support in SE-Net;
- Air-D / AiR support in ISP-SENet;
- Air-D explanation integration;
- tests added for all of the above;
- checkpoint/backward compatibility;
- O/F/C regression safety;
- documentation describing the implementation.
```

The audit is evidence-driven. Do not assume an implementation is correct because a file, class, flag, test, or comment exists. Verify the actual producer -> transformation -> consumer path.

---

## 2. Hard Scope

### 2.1 Code being audited

Audit only:

```text
isp-senet/**
```

This includes:

```text
isp-senet/SE-Net/**
isp-senet/ISP/**
isp-senet/tests/**
isp-senet/AGENTS.md
isp-senet/ARCHITECTURE_MAP.md
isp-senet/CODEBASE_MAP.md
isp-senet/architecture.md
isp-senet/README.md
```

and any other files currently present under `isp-senet/`.

### 2.2 Reference material outside audit scope

You MAY inspect:

```text
gazeformer-isp/**
```

as read-only reference evidence, especially:

```text
gazeformer-isp/AiR/**
gazeformer-isp/COCO_Search18/**
gazeformer-isp/OSIE/**
gazeformer-isp/CODEBASE_MAP.md
```

Do not audit `gazeformer-isp` itself. Do not modify it.

You MAY inspect repository-level metadata such as:

```text
git status
git diff
git log
git show
```

to reconstruct what changed in `isp-senet`.

Do not audit or modify:

```text
llada/**
```

unless a file under `isp-senet` explicitly depends on it and that dependency must be verified.

---

## 3. All Audit Artifacts Must Stay Inside `isp-senet/`

Do not create audit output at repository root.

Do not create audit output inside `gazeformer-isp/` or `llada/`.

Use:

```text
isp-senet/audit/
```

for persistent audit artifacts.

Expected structure:

```text
isp-senet/audit/
├── AUDIT_REPORT.md
├── FINDINGS.md
├── TEST_RESULTS.md
└── EVIDENCE.md
```

Create only the files that are useful, but the primary final report must be:

```text
isp-senet/audit/AUDIT_REPORT.md
```

Temporary runtime files should use a temporary directory where possible and should not be committed.

---

## 4. Audit Is Read-Only by Default

During the audit:

```text
DO NOT fix code
DO NOT refactor code
DO NOT reformat unrelated files
DO NOT rename files
DO NOT change configs to make tests pass
DO NOT weaken tests
DO NOT silently repair malformed data
DO NOT update documentation to match incorrect code
```

The audit should identify defects, not hide them.

Allowed writes are limited to:

```text
isp-senet/audit/**
```

unless the user explicitly asks for fixes after the audit.

If a test requires generated temporary fixtures, keep them temporary or place audit-only fixtures under `isp-senet/audit/`.

---

## 5. Source-of-Truth Hierarchy

Read these before evaluating implementation correctness:

```text
1. this repository-root AGENTS.md
2. isp-senet/AGENTS.md
3. isp-senet/architecture.md
4. isp-senet/ARCHITECTURE_MAP.md
5. isp-senet/CODEBASE_MAP.md
6. actual isp-senet source code
7. gazeformer-isp reference source where relevant
```

Interpret them as:

```text
architecture.md
    = intended explanation mathematics and supervision hierarchy

isp-senet/AGENTS.md
    = implementation contract previously given to Codex

ARCHITECTURE_MAP.md
    = intended mapping from architecture to implementation

CODEBASE_MAP.md
    = previously documented base/current code paths and hazards

current source code
    = evidence of what was actually implemented

gazeformer-isp/AiR/**
    = reference evidence for AiR/Air-D base behavior
```

Documentation is not proof that code implements the documented behavior.

If documentation and code disagree:

```text
- record the disagreement;
- cite the actual code;
- classify whether code is wrong, documentation is stale, or evidence is insufficient;
- do not silently reconcile them.
```

---

## 6. Audit Principle: Trace Contracts End-to-End

For every implemented feature, audit all four layers:

```text
producer
    -> transformation
    -> model/consumer
    -> objective/evaluation
```

Examples:

```text
raw explanation annotation
    -> dataset preprocessing
    -> collator/masks
    -> explanation module
    -> loss

decoder state
    -> fixation reasoning token
    -> WHY router
    -> HOW token
    -> gradient

CLI flag
    -> parser
    -> model construction
    -> dataset behavior
    -> train/test behavior

Air-D annotation
    -> preprocessing
    -> batch fields
    -> predictor/SE-Net
    -> export/evaluation
```

A field that is parsed but never consumed is not implemented.

A module that is constructed but bypassed is not implemented.

A loss that is computed but omitted from the optimized total is not implemented.

A test that mocks away the behavior under review is not valid evidence.

---

## 7. Reconstruct What Codex Changed

Before architecture-specific review, inspect repository state.

At minimum run/read the equivalent of:

```bash
git status --short
git diff --stat
git diff -- isp-senet
git log --oneline --decorate -n 20
```

Use additional `git show`, merge-base, or file history only when needed.

Determine:

```text
- which files were added;
- which files were modified;
- which changes are still uncommitted;
- whether generated artifacts accidentally entered source paths;
- whether O/F/C/Air-D copies diverged unintentionally.
```

Do not assume the current working tree reflects one clean commit.

Record the observed audit baseline in `AUDIT_REPORT.md`.

---

## 8. Mandatory Audit Areas

The audit is incomplete unless every area below is checked.

### A. Original-Pipeline Compatibility

The explanation implementation was required to be plug-and-play.

Audit behavior when:

```text
--enable_explanation = False
```

Verify:

```text
- default is False;
- original dataset fields remain sufficient;
- no explanation annotation is required;
- no RoBERTa/LLM checkpoint is required;
- no explanation tokenizer/model is instantiated unnecessarily;
- original forward output keys/shapes remain compatible;
- original scanpath loss is unchanged;
- original RL path is unchanged;
- original Sampling path is unchanged;
- original validation/test behavior remains available;
- original checkpoint loading still works as intended;
- optional explanation dependencies do not become mandatory baseline imports.
```

Check code paths, not just parser defaults.

### B. Explanation Architecture Completeness

Verify the actual hierarchy:

```text
h_t -> z_t -> r_k^R0 -> g^R0
```

and branch supervision:

```text
z_t      <-> WHAT
r_k^R0   <-> WHY-R0
g_k^R0   <-> HOW-R0
```

For the HOW branch, confirm the implemented global token is the architecture's trajectory-level `g^R0` even if a local variable uses another name.

Audit that generated natural language is NOT chained:

```text
WHAT text -> WHY
WHY text  -> HOW
```

unless the intended architecture has explicitly changed.

### C. Fixation Reasoning Token `z_t`

Verify all required inputs are actually used:

```text
decoder state h_t
soft fixated visual evidence g_t
position embedding p_t
duration representation
```

Verify:

```text
- h_t is the intended decoder representation;
- E is the intended spatial memory/evidence;
- m_t is derived from FINAL fixation action logits;
- STOP is excluded from spatial m_t;
- m_t sums to 1 over spatial cells;
- soft evidence uses a differentiable weighted sum;
- expected coordinates use the correct action-grid ordering;
- no argmax/sampling/NumPy path enters z_t;
- duration parameters are not silently reinterpreted;
- z_t retains gradients to intended upstream tensors.
```

Pay special attention to tensor transposes and `[L,N,H]` vs `[N,L,H]`.

### D. WHAT

Verify:

```text
- WHAT consumes z_t;
- only real fixation timesteps receive WHAT supervision;
- STOP does not receive WHAT supervision;
- branch-specific prefix/query projection exists;
- teacher-forced language loss excludes prefix positions;
- semantic target encoding is frozen/stop-gradient as intended;
- semantic alignment is computed against the correct gold WHAT text;
- text and alignment losses are included with configured weights.
```

Verify batching/token masking, not only single-example behavior.

### E. WHY-R0

Required R0 path:

```text
z_t + temporal position + query context
    -> simple MLP router
    -> soft A_tk
```

Audit:

```text
- no unintended contextual Router Transformer replaced R0;
- gold grouping is a TARGET, not router input;
- A_tk softmax is over Kmax;
- padded fixations do not contribute routing loss;
- padded fixations do not contribute episode mass;
- soft episode aggregation matches the architecture;
- WHY text/alignment applies only to active gold episodes;
- unused slots have no WHY semantic/text supervision;
- episode canonical ordering is respected;
- there is no accidental hard routing before aggregation;
- no episode-presence/count head was silently added to R0.
```

### F. HOW-R0

Verify:

```text
active r_k^R0
 -> mean pool active gold slots
 -> query-conditioned global token g^R0
 -> HOW
```

Audit:

```text
- only active gold episode slots affect training HOW representation;
- unused router slots cannot leak into HOW;
- no trajectory Transformer replaced the R0 mean-pool baseline;
- HOW generation target is the intended sequence-level annotation;
- HOW semantic alignment uses the intended target;
- HOW gradients can propagate through episode tokens/router/z_t.
```

### G. Shared LLM / Semantic Encoder

Verify:

```text
- WHAT/WHY/HOW use ONE shared causal LLM instance;
- branch-specific prefix tokens/projections are distinct where intended;
- the semantic encoder role is not silently replaced by the old cached task embedding;
- raw task/question text reaches the intended semantic encoder when required;
- freeze/train policy matches configuration;
- explanation-disabled mode avoids constructing heavy models;
- tests do not download model weights.
```

If implementation uses a deliberate substitute for RoBERTa, identify it and compare it to the documented architecture rather than assuming equivalence.

### H. Joint Objective and Training Stage

Verify the actual optimized objective during supervised training:

```text
L_total = L_scan + L_EXP
```

and:

```text
L_EXP = weighted WHAT + weighted WHY-R0 + weighted HOW-R0
```

Trace each scalar into the actual `backward()` call.

Verify:

```text
- no loss is merely logged but not optimized;
- no loss is double-counted;
- configured lambda values reach the correct terms;
- zero/disabled weights behave correctly;
- RL remains unchanged by default;
- eval mode vs autograd behavior does not accidentally disable gradients.
```

### I. Augmented Explanation Dataset Format

Audit support for the actual augmented source format:

```json
{
  "name": "...",
  "subject": 1,
  "task": "...",
  "condition": "...",
  "X": [],
  "Y": [],
  "T": [],
  "answer": "...",
  "prediction": {
    "fixations": [{"fixation": 1, "what": "..."}],
    "regions": [{"fixations": [1, 2], "why": "..."}],
    "how": "..."
  }
}
```

Verify the code treats:

```text
X/Y/T
    = raw human scanpath

prediction.fixations[*].what
    = pseudo-gold WHAT

prediction.regions
    = pseudo-gold WHY grouping + WHY text

prediction.how
    = pseudo-gold HOW
```

The key `prediction` must not be interpreted as a model-generated scanpath.

### J. Raw-to-Model Fixation Correspondence

This is a high-risk audit area.

Explanation fixation indices are 1-based indices into the RAW fixation sequence.

Verify the implementation does not simply assume:

```text
raw fixation t == postprocessed model token t
```

Trace every dataset-specific transformation that can alter correspondence:

```text
filtering
invalid-fixation removal
initial-fixation removal
coordinate transformation
truncation
padding
merging
dataset-specific preprocessing
```

Verify an explicit or provably equivalent mapping exists:

```text
raw fixation index
 -> postprocessed fixation index
 -> model token index
```

Audit that the SAME mapping is used for:

```text
WHAT
WHY grouping
any fixation-level auxiliary supervision
```

Do not accept approximate coordinate matching as a substitute unless explicitly designed and justified.

### K. Dataset Validation

Verify enabled explanation loading validates at least:

```text
len(X) == len(Y) == len(T)
WHAT count/index alignment
WHY index bounds
WHY partition/overlap if the data contract guarantees a partition
```

Malformed alignment must not be silently repaired.

Audit behavior for:

```text
empty WHAT text
empty WHY region
duplicate fixation IDs
missing fixation IDs
out-of-range fixation IDs
zero-length scanpath
truncation removing a complete WHY region
```

### L. Truncation Semantics

Audit what happens when model sequence length is shorter than the raw human scanpath.

Verify:

```text
- discarded raw fixations do not retain WHAT supervision;
- WHY groups are transformed through the raw-to-model mapping;
- empty groups are handled deliberately;
- active episode ordering/count is recomputed consistently if required;
- HOW semantics are not silently assumed to match a truncated trajectory.
```

If full-scanpath HOW text supervises a truncated model-visible trajectory, record this as a design mismatch unless the project explicitly defines it as intended.

### M. Air-D / AiR SE-Net

Audit the newly added Air-D support in `isp-senet/SE-Net`.

Use `gazeformer-isp/AiR/**` plus existing SE-Net dataset implementations as reference evidence.

Verify:

```text
dataset naming/dispatch
config
annotation parsing
split names
image paths
subject IDs/remapping
question/task representation
X/Y/T interpretation
coordinate preprocessing
duration handling
max trajectory length
support selection
few-shot behavior
train/eval/export route
subject embedding dimensions
exported table row semantics
```

Do not assume Air-D should copy COCO-Search or OSIE behavior.

For every Air-D-specific choice, identify source evidence. If a choice has no evidence, mark it as an assumption.

### N. Air-D Base ISP Predictor

Audit the new Air-D predictor under `isp-senet/ISP`.

Compare its base behavior with:

```text
gazeformer-isp/AiR/**
```

before evaluating explanation integration.

Verify:

```text
dataset fields
splits
feature loading
task/question embeddings
coordinates/grid
duration units
subject grouping
model construction
losses
Sampling
evaluation
serialization
CLI/config
```

The explanation-disabled Air-D implementation should preserve the reference behavior except for deliberate ISP-SENet personalization changes that are documented and evidenced.

### O. Air-D Explanation Integration

Verify Air-D uses the same explanation architecture/contract as the existing supported predictor copies.

Audit:

```text
same feature flag
same z_t construction
same WHY-R0 math
same HOW-R0 math
same shared-LLM semantics
same loss composition
same raw-to-model annotation rules
same optional-dependency behavior
```

Dataset-specific parsing may differ. Mathematical architecture should not silently diverge.

### P. O/F/C Regression

Review existing:

```text
OSIE
COCO-Freeview
COCO-Search18
```

paths after Air-D/explanation changes.

Verify no accidental changes to:

```text
split behavior
coordinate handling
duration shape/squeeze
subject-table handling
query-position behavior
evaluation grouping
serialization
CLI defaults
```

Compare duplicated predictor files carefully rather than assuming they are identical.

### Q. CLI / Configuration Audit

For every new CLI/config option, trace:

```text
definition -> parsed value -> constructor/data path -> runtime effect
```

Audit especially:

```text
--enable_explanation
explanation annotation path
semantic encoder name/path
causal LLM name/path
Kmax
explanation dimensions
freeze settings
loss weights
Air-D dataset/config paths
```

Verify train/test parsers are both updated where required. Flag dead flags and undocumented required flags.

### R. Checkpoint Compatibility

Audit:

```text
old base predictor checkpoint -> explanation disabled
old base predictor checkpoint -> explanation enabled
new explanation checkpoint -> correct reconstruction
subject embedding table dependency
strict/non-strict state_dict behavior
missing/unexpected key reporting
optimizer/scheduler resume behavior
```

Do not accept a blanket `strict=False` as adequate migration by itself.

Check whether disabled mode changed checkpoint keys or model construction in a way that breaks prior usage.

### S. Optional Dependency Safety

Audit imports as well as construction.

Verify baseline execution does not fail merely because explanation-only libraries/models are absent.

Check for unconditional imports of:

```text
transformers
tokenizer/model-specific packages
other new explanation-only dependencies
```

inside modules imported by the original pipeline.

### T. Tests

Audit the tests themselves.

Required characteristics:

```text
CPU-runnable
offline
deterministic
no real large model download
no CUDA requirement
no user-specific absolute path
small synthetic fixtures where possible
```

Check whether tests cover behavior rather than existence.

At minimum verify coverage for:

```text
CLI default/off
disabled-path compatibility
spatial fixation distribution
z_t gradients
WHY routing/masking
episode aggregation
HOW active-slot pooling
hierarchical gradients
shared LLM
loss composition
checkpoint migration
new augmented dataset validation
raw-to-model mapping
truncation
Air-D SE-Net parsing
Air-D predictor parsing
Air-D explanation path
O/F/C regression-sensitive contracts
```

Identify tests that pass only because core logic is mocked away.

---

## 9. Static Audit Before Runtime Tests

Do static inspection first.

For every finding, gather precise evidence:

```text
file path
symbol/function/class
relevant code behavior
producer/consumer relationship
```

Do not rely on comments alone.

Search all applicable predictor copies for:

```text
enable_explanation
prediction["fixations"]
prediction["regions"]
prediction["how"]
raw_to_model
Kmax
router
loss_exp
semantic encoder
LLM construction
Air-D/AiR dispatch
```

Also inspect O/F/C/Air-D equivalents rather than sampling only one copy.

---

## 10. Runtime Verification

After static inspection, run the smallest relevant tests available.

Prefer:

```bash
cd isp-senet
pytest -q tests/<focused_file>.py
```

then:

```bash
pytest -q tests
```

if the environment supports it.

If test collection fails, record the collection failure accurately.

If dependencies are unavailable, do not install arbitrary packages unless the user explicitly asks for environment modification.

Do not edit code just to make the audit run.

Never report an unexecuted test as passing.

Record commands and exact outcomes in:

```text
isp-senet/audit/TEST_RESULTS.md
```

---

## 11. Audit Finding Format

Every substantive issue should receive a stable ID:

```text
AUD-001
AUD-002
...
```

Use severity:

```text
BLOCKER
HIGH
MEDIUM
LOW
INFO
```

Severity guidance:

```text
BLOCKER
    core requested feature cannot run or corrupts the original pipeline

HIGH
    architecture/data alignment/gradient/checkpoint behavior is materially wrong

MEDIUM
    important edge case, configuration, dataset-specific mismatch, or incomplete
    validation that can produce incorrect experiments

LOW
    localized robustness/documentation/test weakness with limited experimental impact

INFO
    observation, assumption, or verification note with no demonstrated defect
```

Use this template:

```markdown
## AUD-XXX — Short title

**Severity:** HIGH

**Area:** WHY-R0 / Air-D / dataset / checkpoint / etc.

**Expected**
What the architecture/contract/reference requires.

**Observed**
What current code actually does.

**Evidence**
- `path/to/file.py::Symbol`
- exact relevant behavior
- reference path if applicable

**Impact**
Why this matters experimentally or operationally.

**Reproduction / verification**
Command, test, or minimal reasoning path.

**Recommended fix**
Smallest correct direction. Do not implement it during audit.

**Confidence**
High / Medium / Low, with reason if not High.
```

Do not duplicate the same root cause into many findings unless they have meaningfully distinct impacts.

---

## 12. Evidence Standard

A finding is strong only when supported by one or more of:

```text
actual source path and symbol
runtime test
git diff/history
architecture equation/contract
reference implementation
data-format invariant
```

Distinguish:

```text
VERIFIED DEFECT
LIKELY DEFECT
UNVERIFIED ASSUMPTION
DOCUMENTATION MISMATCH
TEST GAP
```

Do not present speculation as a verified defect.

---

## 13. Required Audit Report Structure

`isp-senet/audit/AUDIT_REPORT.md` must contain:

```text
1. Executive summary
2. Audit scope
3. Repository/git state inspected
4. Source-of-truth documents used
5. What was implemented
6. What was actually verified
7. Findings by severity
8. Architecture compliance
9. Dataset/alignment compliance
10. Air-D SE-Net compliance
11. Air-D predictor compliance
12. Plug-and-play/backward compatibility
13. Test results
14. Documentation consistency
15. Unverified items / environment limitations
16. Final implementation status
```

Do not give a vague "looks good" conclusion.

The final status should separately state whether these areas are:

```text
PASS
PASS WITH ISSUES
FAIL
NOT VERIFIED
```

for:

```text
Base pipeline compatibility
WHAT
WHY-R0
HOW-R0
Joint objective
Dataset format
Raw-to-model alignment
Air-D SE-Net
Air-D ISP predictor
Air-D explanation integration
O/F/C regression
Checkpoint compatibility
Tests
Documentation
```

---

## 14. No Aggregate Score

Do not assign:

```text
percentage correctness
numeric quality score
star rating
single overall grade
```

Use evidence-backed status and findings instead.

---

## 15. Audit Completion Criteria

The audit is complete only when:

```text
[ ] current git/worktree state is recorded
[ ] all changed/added isp-senet files are inventoried
[ ] original disabled pipeline is audited
[ ] WHAT is audited end-to-end
[ ] WHY-R0 is audited end-to-end
[ ] HOW-R0 is audited end-to-end
[ ] joint loss is traced to backward()
[ ] augmented dataset format is audited
[ ] raw-to-model mapping is audited per applicable dataset
[ ] truncation behavior is audited
[ ] Air-D SE-Net is compared with source evidence
[ ] Air-D base predictor is compared with gazeformer-isp/AiR
[ ] Air-D explanation integration is audited
[ ] O/F/C regression risks are audited
[ ] CLI/config propagation is audited
[ ] checkpoint compatibility is audited
[ ] optional dependencies are audited
[ ] tests themselves are audited
[ ] relevant tests are actually executed where possible
[ ] findings contain precise evidence
[ ] unverified claims are labeled as such
[ ] all audit artifacts are inside isp-senet/
[ ] no source code was modified during the audit
```

---

## 16. Final Agent Response

After completing the audit, respond concisely with:

```text
1. audit report path
2. number of findings by severity
3. highest-impact verified findings
4. test commands actually run and result
5. areas that remain NOT VERIFIED and why
```

Do not fix the findings unless the user explicitly asks for a remediation pass.

Do not claim the implementation is correct merely because the test suite passes.
