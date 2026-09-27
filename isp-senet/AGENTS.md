# AGENTS.md

Repository-level instructions for coding agents working in `isp-senet`.

This file is an operational contract, not an architecture specification. Keep it
short and use the project docs for details.

## 1. Mission

Implement the complete hierarchical LLM explanation architecture in
`architecture.md` as a **plug-and-play extension** of the existing ISP-SENet
predictor.

Required hierarchy:

```text
decoder/predictor tensors
  -> Fixation Reasoning Tokens z_t
  -> WHAT
  -> WHY-R0 router + Episode Reasoning Tokens r_k^R0
  -> HOW-R0 + Global Reasoning Token g^R0
  -> explanation losses
  -> joint supervised objective
```

The original scanpath pipeline is the baseline and must remain usable unchanged
when the explanation module is disabled.

---

## 2. Read Before Editing

Read these repository-root files in order:

1. `AGENTS.md`
2. `ARCHITECTURE_MAP.md`
3. `CODEBASE_MAP.md`
4. `architecture.md`

Authority:

```text
architecture.md      = intended math and supervision hierarchy
ARCHITECTURE_MAP.md  = architecture -> current tensors/files/edit points
CODEBASE_MAP.md      = current execution/data/checkpoint hazards
source code          = final authority for actual runtime behavior/shapes
```

Do not implement from `architecture.md` alone.

If docs and current source disagree about existing behavior, inspect the code and
report the discrepancy. Do not silently alter the intended architecture to fit
an implementation accident.

---

## 3. Repository Scope

Predictor aliases used by the docs:

```text
O = ISP/OSIE/GazeformerISP/src
F = ISP/COCO_FV/GazeformerISP/src
C = ISP/COCO_Search18/GazeformerISP/src
P = the relevant O/F/C predictor copy
```

The explanation module belongs on the predictor side.

Do not redesign or joint-train `SE-Net` unless explicitly requested. Preserve the
existing offline SE-Net -> subject-embedding-table boundary.

---

## 4. Non-Negotiable Plug-and-Play Contract

A CLI feature switch is mandatory:

```text
--enable_explanation
```

Default: `False`.

When disabled, preserve the original:

```text
model path
inputs
output keys/shapes
scanpath losses
supervised/RL behavior
sampling/decoding
evaluation
checkpoint behavior
dataset requirements
test/inference requirements
```

Disabled mode must not require:

```text
explanation annotations
raw explanation text
RoBERTa/LLM checkpoints
Hugging Face downloads
new explanation artifacts
```

Disabled mode must also avoid meaningful extra compute/memory. Do not instantiate
the semantic encoder or causal LLM, tokenize text, or compute explanation
latents/losses.

Prefer conditional/lazy construction so the original pipeline does not fail
merely because explanation-only dependencies are unavailable.

Do not introduce dummy explanation fields into the legacy path.

---

## 5. No Unrequested Baseline Changes

Do not change these merely to make explanation implementation easier:

```text
base Transformer architecture
scanpath target construction
subject embedding semantics
Sampling
RL reward/losses
evaluation metrics
support selection
SE-Net training
coordinate conventions
duration loss/sampling semantics
```

If an existing bug blocks the requested work, make the smallest isolated
compatibility fix and document it. No drive-by refactors.

---

## 6. Architecture Fidelity

Implement the architecture in `architecture.md` and map it using
`ARCHITECTURE_MAP.md`.

The latent hierarchy is:

```text
h_t -> z_t -> r_k^R0 -> g^R0
```

The branches supervise different scales:

```text
z_t      <-> WHAT
r_k^R0   <-> WHY-R0
g^R0     <-> HOW-R0
```

Never chain generated language:

```text
WHAT text -> WHY      # forbidden
WHY text  -> HOW      # forbidden
```

Use **one shared causal LLM** with branch-specific special tokens/projections.

A complete implementation includes all of the following.

### Shared fixation representation

```text
differentiable decoder state h_t
differentiable spatial evidence E
final spatial fixation distribution m_t
soft fixated visual feature
expected fixation coordinate
position embedding
duration representation
Fixation Reasoning Token z_t
```

### WHAT

```text
WHAT latent prefix
teacher-forced generation loss
frozen gold-text semantic encoding
semantic alignment loss
total WHAT objective
```

### WHY-R0

```text
temporal embedding
query projection
per-fixation MLP router
soft routing A_tk
gold routing supervision
soft episode aggregation
Episode Reasoning Tokens r_k^R0
WHY generation loss
WHY semantic alignment
total WHY-R0 objective
```

R0 means:

```text
no contextual Router Transformer
no episode-presence/count head
```

### HOW-R0

```text
gold active-slot selection during training
mean pooling of active episode tokens
query-conditioned global token
HOW generation loss
HOW semantic alignment
total HOW-R0 objective
```

R0 means no trajectory Transformer.

### Joint objective

During supervised scanpath training:

```text
L_total = L_scan + L_EXP
```

Keep the existing RL objective unchanged by default. Do not automatically add
explanation losses to the RL phase.

---

## 7. Differentiability Contract

Explanation supervision must attach before sampled/NumPy scanpath generation.

Do not use these as explanation-training inputs:

```text
Sampling.random_sample
Sampling.generate_scanpath
```

Do not insert these in the explanation gradient path:

```text
.detach()
.data
NumPy conversion
argmax
hard categorical sampling
```

Required paths include:

```text
WHAT -> z_t -> predictor latents
route loss -> A_tk -> router -> z_t
WHY -> r_k^R0 -> A_tk and z_t
HOW -> g^R0 -> r_k^R0 -> A_tk and z_t
```

Stop-gradient is allowed/expected on frozen gold-text semantic targets.

---

## 8. Critical Mapping Rules

Follow `ARCHITECTURE_MAP.md`; in particular:

### `h_t`

Use the decoder states currently local to the predictor forward (`outs` in the
mapped implementation). Expose them without detaching.

### `E`

Use the explicitly selected personalized spatial memory consumed by the
decoder/head, unless current source inspection proves the map is stale.

Do not silently substitute an unrelated raw feature cache.

### `m_t`

Current action space:

```text
0     = STOP
1..G  = spatial cells
```

Construct `m_t` from **final aggregated action logits**, excluding STOP:

```python
spatial_logits = actions[..., 1:]
fixation_prob = spatial_logits.softmax(dim=-1)
```

Do not use pre-aggregation multi-head `action_map` as `m_t`.

### Fixation mask

Use real-fixation semantics, not the action mask containing STOP. If current code
still matches `CODEBASE_MAP.md`, duration-mask semantics are the correct starting
point.

---

## 9. Text/Query Contract

The intended architecture uses:

```text
q = RoBERTa(Q)
```

Do not silently rename the existing cached task/SentenceTransformer vector to
RoBERTa output.

When explanation is enabled, raw query/task text must be an explicit explanation
input and must pass through the configured semantic encoder.

If a dataset truly lacks a recoverable textual query, surface the missing data
contract. Do not fabricate one.

Keep heavy text dependencies optional to explanation-disabled execution.

---

## 10. Duration Contract

`CODEBASE_MAP.md` documents ambiguity in the existing second log-normal duration
parameter.

Do not silently reinterpret or fix that baseline behavior here.

Isolate explanation-side conversion in one helper/module using a neutral
internal name until semantics are established.

Do not change the existing scanpath duration loss or sampler as part of this
architecture implementation unless explicitly requested.

---

## 11. Explanation Data Contract

Explanation annotations are required only when enabled for supervised training.

Normalize source annotations to the semantic equivalent of:

```python
{
    "query_text": str,
    "what_texts": list[str],
    "why_episode_ids": list[int],
    "why_texts": list[str],
    "how_text": str,
}
```

WHY rules:

```text
episodes are canonically ordered by first occurring fixation
K* <= Kmax
each real fixation belongs to exactly one gold episode
padded timesteps belong to no episode
unused slots receive no WHY text/alignment supervision
episodes need not be temporally contiguous
```

Pad membership internally to `[L, Kmax]`.

Do not make explanation fields mandatory when the feature is disabled.

---

## 12. CLI and Configuration

At minimum expose:

```text
--enable_explanation
```

Default `False`.

Keep explanation options clearly namespaced and configurable. Required concepts:

```text
explanation latent dimension
Kmax
semantic encoder name/path
causal LLM name/path
encoder/LLM freeze policy
WHAT/WHY/HOW top-level weights
generation/alignment/routing sub-loss weights
```

Do not hardcode local machine paths or one private checkpoint.

Train and test currently have separate parser surfaces; mirror inference-relevant
flags deliberately.

If explanation is enabled but required config is absent, fail early with an
actionable error.

---

## 13. Module Boundaries

Prefer focused modules over turning `gazeformer.py` into a monolith.

Recommended per predictor copy:

```text
P/models/explanation.py
P/models/explanation_llm.py
```

Suggested responsibilities:

```text
explanation.py
  FixationReasoningEncoder
  WhyR0Router
  HowR0Aggregator
  HierarchicalExplanationModule

explanation_llm.py
  semantic encoder wrapper
  shared causal LLM wrapper
  branch-prefix projections
  teacher-forced LM helpers
  semantic-alignment helpers
```

Routing/aggregation stays outside the LLM wrapper.

Use one shared LLM instance, not three copies.

---

## 14. O/F/C Replication

O/F/C are separate predictor copies with local differences.

Do not bulk overwrite whole files across them.

Use this workflow:

```text
1. implement one canonical copy
2. run focused tests
3. inspect equivalent O/F/C files
4. port the minimal diff deliberately
5. preserve dataset-specific behavior
6. test replicated interfaces
```

Read the local-difference notes in `CODEBASE_MAP.md` before porting.

Do not refactor O/F/C into a shared package as part of this task unless explicitly
requested.

---

## 15. Checkpoint Compatibility

New explanation parameters affect `state_dict`.

Requirements:

```text
explanation-disabled old checkpoints load as before
base checkpoint -> explanation-enabled model migration is explicit
missing/unexpected explanation keys are reported
unrelated missing keys are not silently ignored
```

Do not globally hide incompatibility with unconditional `strict=False`.

The offline subject embedding table remains a separate artifact.

---

## 16. Optional Dependencies

Explanation-only dependencies such as `transformers`, tokenizer, RoBERTa, and the
causal LLM must not become unconditional baseline requirements where avoidable.

New text modules must support dependency injection or tiny substitutes for tests.

Unit tests must never download pretrained weights.

---

## 17. Test Location and Constraints

All new automated tests for this architecture go under repository-root:

```text
tests/
```

Use pytest-style names:

```text
tests/test_*.py
```

Tests must be:

```text
CPU-runnable
deterministic
offline
fast
independent of real datasets
independent of large model downloads
independent of CUDA
independent of external SE-Net checkpoints
independent of /mnt/d/... absolute paths
```

Use synthetic tensors, tiny fake encoders/LLMs, dependency injection,
monkeypatching, and temporary files.

Do not mock away the behavior being tested.

---

## 18. Mandatory Test Matrix

At minimum cover:

### CLI / disabled path

Verify:

```text
--enable_explanation defaults False
legacy arguments still parse
disabled mode needs no explanation paths/data
explanation module is not constructed when disabled
base output contract is unchanged when disabled
base loss path receives no explanation term
```

Where practical, check numerical parity for identical model state/input.

### Spatial fixation distribution

Verify:

```text
STOP excluded
spatial probabilities sum to 1
gradients reach spatial action logits
```

### Fixation representation

Verify:

```text
soft visual feature shape
z_t shape
mask behavior
gradients reach decoder state and visual evidence
```

### WHY-R0

Verify:

```text
A_tk shape
probability sums over Kmax
padded fixations do not affect route loss/episode pooling
gold grouping is target, not router input
unused slots receive no WHY text/alignment loss
soft episode aggregation matches the specified weighted mean
```

### HOW-R0

For `K*=k`, changing slots `k..Kmax-1` must not change the training HOW
representation.

### Hierarchical gradients

Backprop from HOW-only loss and verify gradients can reach:

```text
HowR0Aggregator
WhyR0Router
FixationReasoningEncoder
upstream decoder-state input
```

### Shared LLM

Verify WHAT/WHY/HOW reference the same causal LLM parameter object.

### Loss composition

Verify:

```text
L_EXP = weighted WHAT + WHY + HOW
L_total = L_scan + L_EXP
existing RL objective unchanged by default
```

### Checkpoint migration

Verify the intended base-checkpoint -> explanation-enabled behavior without
silently accepting unrelated missing keys.

---

## 19. Test Discipline

Tests validate observable contracts, not implementation trivia.

Do not weaken tests to make a failing implementation pass.

Prefer focused contract tests over GPU/dataset-heavy end-to-end tests.

Never claim a test passed unless it was actually executed.

If full integration cannot run, execute all possible focused tests and report the
exact blocked step and reason.

Expected pytest forms, if pytest is available in the repository environment:

```bash
pytest -q tests/<relevant_test_file>.py
pytest -q tests
```

Do not invent successful commands/results.

---

## 20. Working Method

For non-trivial edits:

```text
1. inspect target symbol and direct callers
2. inspect matching ARCHITECTURE_MAP.md section
3. confirm tensor layout
4. make the smallest coherent change
5. run the most focused affected test
6. diagnose failures from evidence
7. run the relevant tests/ subset
8. inspect git diff
9. verify explanation-disabled compatibility
```

Avoid:

```text
unrelated formatting
broad renames
speculative abstractions
repo-wide refactors
hidden behavior changes
duplicated architecture logic
```

Use targeted runtime/shape assertions for new cross-module contracts, but avoid
expensive assertions in hot training loops.

---

## 21. No Silent Architectural Substitutions

Do not silently replace:

```text
RoBERTa semantic encoder -> existing cached task vector
soft fixation evidence    -> argmax crop
soft WHY routing          -> hard assignment
R0 MLP router             -> Transformer router
HOW-R0 mean pooling       -> trajectory Transformer
one shared LLM            -> three LLMs
gold semantic targets     -> generated-text targets
```

If a deviation is genuinely required, report it explicitly before calling the
implementation complete.

---

## 22. Inference Contract

Ordinary scanpath inference must work with explanation disabled.

R0 has no episode-presence/count head. Do not invent a deploy-time rule for
active WHY/HOW slots.

If optional explanation generation is added to test/inference:

```text
gate it explicitly
keep base scanpath outputs available
do not require gold explanation labels
do not invent K* for WHY/HOW
```

---

## 23. Definition of Done

Do not declare the architecture complete until:

```text
[ ] full z_t -> r_k^R0 -> g^R0 hierarchy exists
[ ] WHAT is implemented
[ ] WHY-R0 is implemented
[ ] HOW-R0 is implemented
[ ] one shared causal LLM is used
[ ] RoBERTa-role semantic encoding is explicit
[ ] all specified generation/alignment/routing losses exist
[ ] supervised joint loss is wired
[ ] existing RL behavior is unchanged by default
[ ] --enable_explanation exists and defaults False
[ ] disabled mode requires no explanation data/models
[ ] disabled base forward/inference remains compatible
[ ] explanation gradients reach intended predictor latents
[ ] checkpoint migration behavior is explicit
[ ] O/F/C changes preserve local differences
[ ] new tests live in tests/
[ ] mandatory contract/gradient/compatibility tests pass
[ ] tests require no network/GPU/real dataset
[ ] no architecture deviation is hidden
```

---

## 24. Final Agent Report

At the end of an implementation task, report only:

```text
1. files changed
2. architecture pieces implemented
3. disabled-path compatibility behavior
4. tests actually run and results
5. blocked verification, if any, with exact reason
6. remaining known limitation, if any
```

Do not report unexecuted tests as passing.

Do not claim completeness while WHAT, WHY-R0, HOW-R0, semantic alignment,
routing supervision, joint loss, CLI gating, or mandatory compatibility tests
remain stubbed.
