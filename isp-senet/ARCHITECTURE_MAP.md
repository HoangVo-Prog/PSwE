# LLM Explanation Module — Codex Implementation Map

> **Purpose**
>
> This file maps `architecture.md` onto the current `isp-senet/**` implementation described by `CODEBASE_MAP.md`.
> Do not create or maintain a separate IMPLEMENTATION_MAP.md.
> It is an **implementation/navigation contract for Codex**, not a replacement for either source document.
>
> Use:
>
> 1. `architecture.md` as the source of truth for the intended mathematics and supervision hierarchy.
> 2. `CODEBASE_MAP.md` as the source of truth for the current repository paths, active forward paths, data contracts, and implementation hazards.
> 3. The actual source code as the final execution authority if a symbol/shape differs at runtime.
>
> **Do not silently change the architecture to fit the current code.**
> If a required architectural quantity does not exist, expose or construct it explicitly.

---

## 0. Scope

Target hierarchy:

\[
\mathbf h_t
\rightarrow
\mathbf z_t
\rightarrow
\mathbf r_k^{R0}
\rightarrow
\mathbf g^{R0}
\]

with three language-supervision branches:

\[
\mathbf z_t \leftrightarrow W_t^*
\quad\text{(WHAT)}
\]

\[
\mathbf r_k^{R0} \leftrightarrow Y_k^*
\quad\text{(WHY-R0)}
\]

\[
\mathbf g^{R0} \leftrightarrow \mathcal H^*
\quad\text{(HOW-R0)}
\]

The branches share **one LLM**. They are not chained through generated text:

```text
WRONG:
WHAT text -> WHY text -> HOW text

CORRECT:
decoder latent -> fixation latent -> episode latent -> trajectory latent
                 |                  |                 |
                WHAT               WHY               HOW
```

The generated WHAT text is not an input to WHY. Generated WHY text is not an input to HOW.

---

# 1. Repository Aliases

Use the aliases already defined by `CODEBASE_MAP.md`:

```text
S = SE-Net
O = ISP/OSIE/GazeformerISP/src
F = ISP/COCO_FV/GazeformerISP/src
C = ISP/COCO_Search18/GazeformerISP/src
P = the corresponding O/F/C predictor copy
```

The LLM explanation module belongs to the **predictor side**, not the offline SE-Net export side.

Primary implementation locations:

```text
P/models/gazeformer.py
P/models/models.py
P/models/loss.py
P/dataset/dataset.py
P/train.py
P/test.py
P/opts.py
```

Recommended new shared-within-each-copy files:

```text
P/models/explanation.py
P/models/explanation_llm.py
```

If the repository is later refactored into a shared package, deduplicate these files then. For the current repository, do not assume O/F/C import a common predictor implementation.

---

# 2. Current Predictor Boundary vs Target Explanation Boundary

## 2.1 Current differentiable predictor path

Current active path:

```text
cached query visual features
    + task embedding
    + exported subject embedding row
        |
        v
P/models/models.py::Transformer
        |
        +--> personalized visual memory
        |
        +--> decoder states `outs`
                |
                v
P/models/gazeformer.py
    +--> CrossAttentionPredictor(s)
    +--> stop logits
    +--> duration parameters
    +--> subject-conditioned map prioritization
    +--> final action logits / action probabilities
```

The important fact is that the tensors needed by the explanation module already exist **inside** the differentiable predictor forward, but several are currently local variables and are not returned.

## 2.2 Do not build explanation training from sampled scanpaths

Do **not** use:

```text
Sampling.random_sample
Sampling.generate_scanpath
```

as the training interface for the explanation branch.

Those routines move toward sampled/NumPy scanpaths and break the clean differentiable path required for auxiliary language supervision.

The explanation module must consume predictor tensors before sampling.

---

# 3. Canonical Tensor Contract

Use batch-first notation inside the new explanation module:

```text
N = flattened predictor batch size
L = predictor maximum scanpath length
G = im_h * im_w
H = predictor hidden dimension
Dv = visual feature dimension used by explanation evidence
Dz = fixation reasoning dimension
Kmax = maximum WHY router slots
DL = LLM hidden dimension
```

Recommended internal shapes:

```text
decoder states h        : [N, L, H]
visual memory E         : [N, G, Dv]
spatial fixation m      : [N, L, G]
duration mu             : [N, L]
duration scale/variance : [N, L]
fixation tokens z       : [N, L, Dz]
router A                : [N, L, Kmax]
episode tokens r        : [N, Kmax, Dz]
global token g          : [N, DH]
```

Always normalize tensor layout once at the boundary. Do not spread repeated `permute/view/squeeze` conventions across multiple branches.

---

# 4. Architecture-to-Code Mapping Table

| Architecture quantity | Meaning | Current code source | Status / implementation action |
|---|---|---|---|
| \(\mathbf h_t\) | decoder representation at timestep \(t\) | local `outs` in `P/models/gazeformer.py::training_process` / `inference`, produced by `TransformerDecoderWrapper` | **Exists, not exposed.** Return/pass it to explanation module before head-specific detaching/sampling. |
| \(E\) | spatial visual evidence used for fixation reasoning | current predictor has cached `images`, encoded memory, and personalized memory | **No exact named binding exists.** Bind explicitly; recommended binding is the personalized spatial memory consumed by the decoder/head so explanation sees the same evidence as prediction. |
| \(m_t(i,j)\) | final spatial fixation probability, sum over spatial cells = 1 | final `actions` logits in train; `all_actions_prob` in inference contain stop at index 0 + spatial cells | **Construct explicitly.** Do not use candidate `action_map` directly. Remove stop slot and renormalize spatial cells. |
| \((\mu_t,\sigma_t)\) | duration statistics | `log_normal_mu`, `log_normal_sigma2` | **Exists, semantics need care.** Current second output is named/used as variance in the loss but sampling uses it as scale. Do not silently reinterpret. |
| \(Q\) | raw task/query text | dataset has `tasks`; predictor consumes cached task embeddings | **Raw-text contract is not guaranteed by current forward.** Pass raw query text explicitly if architecture keeps RoBERTa. |
| \(\mathbf q=\mathrm{RoBERTa}(Q)\) | semantic query representation for explanation branches | no current RoBERTa explanation encoder | **New component.** Do not pretend the current SentenceTransformer task vector is RoBERTa output unless the architecture is intentionally changed. |
| \(\mathbf g_t\) | soft fixated visual feature | none | **New differentiable construction** from \(m_t\) and \(E\). |
| \(\mathbf p_t\) | expected-position embedding | none | **New component** from spatial \(m_t\). |
| \(\mathbf r_t\) | duration embedding | none | **New component** from duration parameters. |
| \(\mathbf z_t\) | Fixation Reasoning Token | none | **New shared explanation representation.** |
| \(A_{tk}\) | soft fixation-to-episode routing | none | **New WHY-R0 router.** |
| \(\mathbf r_k^{R0}\) | Episode Reasoning Token | none | **New soft aggregation.** |
| \(\mathbf g^{R0}\) | Global/Trajectory Reasoning Token | none | **New HOW-R0 aggregation.** |
| WHAT/WHY/HOW LLM prefixes | latent prefix to shared LLM | no LLM branch exists | **New module.** One LLM, branch-specific tokens/projectors. |
| text/alignment losses | explanation auxiliary losses | none | **New losses** added to supervised predictor training. |

---

# 5. Bind the Visual Evidence \(E\) Explicitly

The current codebase has several possible visual tensors:

```text
1. `images`
   = cached ResNet spatial features loaded by dataset
   shape conceptually [N, G, img_hidden_dim]

2. encoder memory
   = output of TransformerEncoderWrapper

3. personalized/integrated memory
   = output after SubjectAttentionModule + IntegrationModule
   = memory actually used downstream by the personalized decoder/head
```

For this implementation, use:

\[
\boxed{
E := \text{personalized spatial memory consumed by the decoder/head}
}
\]

Reason for this binding:

```text
- it is spatial;
- it remains differentiable;
- it already incorporates task/subject-conditioned predictor context;
- it is the evidence space actually available to the fixation decoder/head;
- it avoids creating an explanation branch over an unrelated raw cache.
```

### Required code change

`P/models/models.py::Transformer.forward` or the immediate caller must make the selected personalized memory available to `gazeformer`.

Do not detach it.

Suggested semantic name at the `gazeformer` boundary:

```python
decoder_memory
```

Then the explanation module receives:

```python
decoder_states  # [N, L, H]
decoder_memory  # [N, G, H] or equivalent after one canonical transpose
```

If the runtime shape differs, adapt once in `ExplanationModule.forward`.

---

# 6. Construct the Final Fixation Spatial Distribution \(m_t\)

The current predictor action space contains:

```text
index 0   = STOP
index 1:G = spatial fixation cells
```

The architecture requires:

\[
\sum_{i,j} m_t(i,j)=1
\]

over **spatial cells only**.

Therefore, in supervised training, construct:

\[
\boxed{
m_t
=
\operatorname{softmax}
\left(
\text{actions}_{t,1:G}
\right)
}
\]

where `actions` are the **final subject-weighted action logits**, after adaptive map prioritization.

Equivalent pseudocode:

```python
spatial_logits = actions[..., 1:]              # [N, L, G]
fixation_prob = spatial_logits.softmax(-1)     # [N, L, G]
```

Do **not** use the current `action_map` as \(m_t\).

`action_map` corresponds to multiple candidate map heads before final map mixing. The architecture's \(m_t\) is one final fixation distribution for timestep \(t\).

For inference-only explanation diagnostics, if only `all_actions_prob` is available:

```python
spatial_prob = all_actions_prob[..., 1:]
fixation_prob = spatial_prob / (spatial_prob.sum(-1, keepdim=True) + eps)
```

Training should prefer logits -> spatial softmax directly.

---

# 7. Shared Fixation Representation

Create a new module, recommended:

```text
P/models/explanation.py
```

with a component such as:

```python
class FixationReasoningEncoder(nn.Module):
    ...
```

Its job is only:

```text
(h_t, E, m_t, duration stats)
    -> g_t
    -> p_t
    -> duration representation
    -> z_t
```

It does not generate text.

---

## 7.1 Soft fixated visual feature

Architecture:

\[
\mathbf g_t
=
\sum_{i,j}m_t(i,j)\mathbf E_{ij}
\]

Batch form:

```python
# fixation_prob: [N, L, G]
# visual_memory: [N, G, Dv]
soft_visual = torch.einsum("nlg,ngd->nld", fixation_prob, visual_memory)
# [N, L, Dv]
```

No argmax. No categorical sampling. No NumPy conversion.

---

## 7.2 Expected fixation coordinate

Precompute a normalized grid aligned exactly to the flattened action-cell order:

```text
grid_xy: [G, 2]
```

with values in `[0,1]`.

Then:

```python
expected_xy = torch.einsum("nlg,gc->nlc", fixation_prob, grid_xy)
# [N, L, 2]
```

and:

```python
pos_emb = MLP_pos(expected_xy)
```

### Important

Use the predictor's **action-grid flattening convention**, not image-pixel coordinates from `Sampling.generate_scanpath`.

Do not mix:

```text
feature-grid coordinates
output image coordinates
original annotation coordinates
```

---

## 7.3 Duration representation

Architecture requests:

\[
\mathbf r_t
=
\operatorname{MLP}_d([\mu_t;\log\sigma_t])
\]

Current predictor exposes:

```text
log_normal_mu
log_normal_sigma2
```

The codebase map identifies a semantic inconsistency:

```text
- supervised NLL treats the second value as variance;
- Sampling uses it directly as a noise scale.
```

Therefore **do not silently define**:

```python
sigma = log_normal_sigma2
```

or:

```python
sigma = sqrt(log_normal_sigma2)
```

without deciding which semantics the project wants.

### Safe implementation contract

In code, use a neutral name at the explanation boundary:

```python
duration_param2
```

and isolate conversion in one helper:

```python
duration_feature = duration_parameter_adapter(
    log_normal_mu,
    duration_param2,
)
```

The helper should be the only place that converts the second parameter into the quantity used by the architecture.

Until the duration semantics are corrected/confirmed, do not duplicate conversion logic elsewhere.

---

## 7.4 Fixation Reasoning Token

Architecture:

\[
\tilde{\mathbf h}_t=P_h\mathbf h_t,
\qquad
\tilde{\mathbf g}_t=P_g\mathbf g_t
\]

\[
\boxed{
\mathbf z_t
=
\operatorname{LN}
\left(
\operatorname{MLP}_{\mathrm{fix}}
[
\tilde{\mathbf h}_t;
\tilde{\mathbf g}_t;
\mathbf p_t;
\mathbf r_t
]
\right)
}
\]

Recommended API:

```python
z = fixation_reasoning_encoder(
    decoder_states=decoder_states,
    visual_memory=decoder_memory,
    fixation_prob=fixation_prob,
    duration_mu=log_normal_mu,
    duration_param2=duration_param2,
)
# [N, L, Dz]
```

`z` is the single shared latent input to WHAT and WHY-R0.

---

# 8. Task / Query Representation

This is a **real architecture/codebase mismatch**.

Architecture:

\[
\mathbf q=\operatorname{RoBERTa}(Q)
\]

Current predictor:

```text
- loads task embedding dictionaries;
- task embeddings were produced offline;
- feature_extractor.py::text_data uses SentenceTransformer;
- predictor forward receives vectors, not a tokenizer/text model.
```

Therefore Codex must not silently map:

```text
current task_embeddings == q
```

unless the architecture is deliberately revised.

## Required interface

Add raw query/task text to the explanation batch:

```python
batch["explanation_query_text"]
```

or an equivalent explicit field.

Then the explanation module owns a semantic encoder:

```python
q = frozen_roberta(query_text)
```

If raw textual `Q` is not available for a dataset, stop at the data-contract layer and surface that missing input. Do not fabricate it from the cached 768-D task vector while still calling it RoBERTa.

### Efficiency

It is acceptable to precompute/freeze RoBERTa target/query embeddings later, but the resulting cache must preserve the same semantic contract.

---

# 9. Explanation Annotation Contract

The current predictor dataset does not provide the explanation targets required by the architecture.

Add explicit fields to each training example.

Minimum semantic schema:

```python
{
    # fixation-level
    "what_texts": List[str],            # length T_fix

    # episode-level
    "why_episode_ids": List[int],       # length T_fix; canonical 0..K*-1
    "why_texts": List[str],             # length K*

    # trajectory-level
    "how_text": str,

    # query text for explanation semantic encoder
    "explanation_query_text": str,
}
```

Alternative accepted storage:

```text
gold membership matrix M*
```

but convert it once in the collator/module to the padded tensor contract.

---

## 9.1 Fixation masking

Predictor supervision distinguishes:

```text
action_mask
    = fixations + first STOP

duration_mask
    = real fixation timesteps only
```

For WHAT/WHY/HOW fixation-level latent supervision, use the **real-fixation mask**, i.e. the semantics of `duration_mask`.

Do not attach WHAT text to the STOP timestep.

Define:

```python
fixation_mask = duration_masks.bool()  # [N, L]
```

and pad explanation annotations to `L`.

---

## 9.2 WHY gold membership

For each example:

```text
T* = number of real fixations
K* = number of gold WHY episodes
K* <= Kmax
```

Gold episodes must be canonically ordered by first occurring timestep.

Build:

```text
M_star_tilde: [L, Kmax]
```

Rules:

```text
- real fixation rows: exactly one active gold episode;
- padded/non-fixation rows: all zeros;
- unused episode columns k >= K*: all zeros;
- episodes need not be temporally contiguous.
```

Also keep:

```python
episode_count = K_star
```

for masking WHY text/alignment and HOW aggregation during training.

`episode_count` is **gold training metadata**, not an inference prediction head in R0.

---

# 10. WHAT Branch

Recommended component:

```python
class WhatHead(nn.Module):
    ...
```

Input:

```text
z          [N, L, Dz]
q          [N, Dq]
fixation_mask [N, L]
gold WHAT texts
```

Architecture prefix for each active fixation:

\[
X_t^{WHAT}
=
[
e_{\langle WHAT\rangle};
P_q^{WHAT}(\mathbf q);
P_{LM}\mathbf z_t
]
\]

The shared LLM then receives this learned prefix plus teacher-forced gold tokens.

Loss:

\[
\mathcal L_{WHAT}
=
\lambda_{txt}\mathcal L_{what-txt}
+
\lambda_{align}\mathcal L_{what-align}
\]

### Text loss masking

Compute only over real fixation timesteps with available WHAT annotations.

Normalize by the total number of gold WHAT tokens, consistent with `architecture.md`.

### Semantic alignment

Gold semantic target:

\[
\mathbf e_t^W
=
\operatorname{sg}[\operatorname{RoBERTa}(W_t^*)]
\]

Predicted semantic side:

\[
\mathbf a_t^W=P_W\mathbf z_t
\]

Use cosine alignment after normalization.

The target semantic encoder is frozen / stop-gradient on the target side as specified by the architecture.

---

# 11. WHY-R0 Branch

Recommended component:

```python
class WhyR0Head(nn.Module):
    ...
```

Input:

```text
z                [N, L, Dz]
q                [N, Dq]
fixation_mask    [N, L]
M_star_tilde     [N, L, Kmax]
episode_count    [N]
gold WHY texts
```

---

## 11.1 Router input

For each timestep:

\[
\mathbf u_t^{R0}
=
[
\mathbf z_t;
\boldsymbol\tau_t;
\mathbf c_q^R
]
\]

where:

```text
tau_t = 1D temporal positional embedding
c_q^R = P_q^R(q)
```

The router is the simple R0 baseline:

```python
router_logits = MLP_R(router_input)
A = softmax(router_logits, dim=-1)
# [N, L, Kmax]
```

Do not replace R0 with a contextual Router Transformer unless architecture is explicitly changed.

---

## 11.2 Router loss

Compute only over real fixation rows:

\[
\mathcal L_{route}
=
-
\frac{1}{T}
\sum_t
\sum_k
\tilde M_{tk}^*
\log A_{tk}
\]

Implementation should mask padded timesteps before reduction.

No episode-presence/count loss in R0.

---

## 11.3 Soft episode aggregation

\[
n_k=\sum_t A_{tk}
\]

\[
\boxed{
\mathbf r_k^{R0}
=
\frac{\sum_t A_{tk}\mathbf z_t}
{\sum_t A_{tk}+\epsilon}
}
\]

### Padding rule

Padded timesteps must not contribute router mass to episode aggregation.

Use:

```python
A_masked = A * fixation_mask[..., None]
```

before computing episode masses and weighted sums.

Recommended result:

```text
r_R0: [N, Kmax, Dz]
```

---

## 11.4 WHY language supervision

For each example, supervise only slots:

```text
k < K*
```

with:

\[
X_k^{WHY}
=
[
e_{\langle WHY\rangle};
P_q^{WHY}(\mathbf q);
P_{LM}^{Y}\mathbf r_k^{R0}
]
\]

Loss:

\[
\mathcal L_{WHY-R0}
=
\lambda_{route}\mathcal L_{route}
+
\lambda_{txt}^{Y}\mathcal L_{why-txt}
+
\lambda_{align}^{Y}\mathcal L_{why-align}
\]

Unused slots receive:

```text
no WHY text loss
no WHY alignment loss
```

---

# 12. HOW-R0 Branch

Recommended component:

```python
class HowR0Head(nn.Module):
    ...
```

Input:

```text
r_R0           [N, Kmax, Dz]
episode_count  [N]
q              [N, Dq]
gold HOW text
```

R0 deliberately uses simple active-episode mean pooling.

For each example:

\[
\bar{\mathbf r}^{R0}
=
\frac{1}{K^*}
\sum_{k=1}^{K^*}
\mathbf r_k^{R0}
\]

Then:

\[
\boxed{
\mathbf g^{R0}
=
\operatorname{LN}
\left(
\operatorname{MLP}_H
[
\bar{\mathbf r}^{R0};
P_q^H(\mathbf q)
]
\right)
}
\]

No trajectory Transformer in HOW-R0.

LLM prefix:

\[
X^{HOW}
=
[
e_{\langle HOW\rangle};
P_q^{HOW}(\mathbf q);
P_{LM}^{H}\mathbf g^{R0}
]
\]

Loss:

\[
\mathcal L_{HOW-R0}
=
\lambda_{txt}^{H}\mathcal L_{how-txt}
+
\lambda_{align}^{H}\mathcal L_{how-align}
\]

No additional routing loss.

---

# 13. Shared LLM Interface

Recommended file:

```text
P/models/explanation_llm.py
```

Recommended logical object:

```python
class SharedExplanationLLM(nn.Module):
    ...
```

It owns one causal LLM and branch-specific prefix modules:

```text
shared:
    LLM

WHAT:
    <WHAT> learnable embedding
    P_q^WHAT
    P_LM^WHAT / P_LM

WHY:
    <WHY> learnable embedding
    P_q^WHY
    P_LM^Y

HOW:
    <HOW> learnable embedding
    P_q^HOW
    P_LM^H
```

The exact LLM family/model is **not specified in the architecture**. Make it configuration-driven.

Do not instantiate three independent LLMs.

---

## 13.1 Prefix implementation

The conceptual prefix is a sequence of learned embeddings in the LLM hidden space.

Example:

```text
WHAT prefix length = 3
[branch token, query token, fixation token]

WHY prefix length = 3
[branch token, query token, episode token]

HOW prefix length = 3
[branch token, query token, trajectory token]
```

Use `inputs_embeds` or the equivalent supported interface of the selected causal LLM.

Teacher-forced gold text tokens follow the latent prefix.

The loss over text tokens must exclude prefix positions from LM token loss.

---

## 13.2 Do not conflate the two RoBERTa roles

The architecture uses semantic text encoding for:

```text
1. q = RoBERTa(Q)
2. gold explanation embeddings for alignment:
   RoBERTa(W*)
   RoBERTa(Y*)
   RoBERTa(H*)
```

These may share one frozen semantic encoder if dimensions/behavior match the architecture.

The causal generation LLM is a separate role.

Do not call the current offline SentenceTransformer cache "RoBERTa" without an explicit architecture change.

---

# 14. Top-Level Explanation Module API

Recommended:

```python
class HierarchicalExplanationModule(nn.Module):
    def forward(
        self,
        decoder_states,
        decoder_memory,
        action_logits,
        duration_mu,
        duration_param2,
        query_text,
        fixation_mask,
        what_targets=None,
        why_membership=None,
        why_text_targets=None,
        episode_count=None,
        how_target=None,
        compute_generation=True,
        compute_alignment=True,
    ):
        ...
```

Recommended returned dictionary:

```python
{
    # shared latent tensors
    "fixation_prob": ...,
    "z": ...,

    # WHY
    "router_logits": ...,
    "router_prob": ...,
    "episode_tokens": ...,

    # HOW
    "global_token": ...,

    # losses
    "loss_what_txt": ...,
    "loss_what_align": ...,
    "loss_route": ...,
    "loss_why_txt": ...,
    "loss_why_align": ...,
    "loss_how_txt": ...,
    "loss_how_align": ...,
    "loss_explanation": ...,
}
```

Generated strings should be optional diagnostics, not required for each training forward.

---

# 15. Integration into `gazeformer`

Primary file:

```text
P/models/gazeformer.py
```

## 15.1 Constructor

Add the explanation module as a registered submodule:

```python
self.explanation_module = HierarchicalExplanationModule(...)
```

Only when enabled by configuration if memory is a concern.

Adding this module changes checkpoint state dicts. Old checkpoint loading therefore requires an explicit compatibility policy.

---

## 15.2 Training forward

Current supervised training needs `actions`, duration outputs, etc.

Extend the training output contract without removing existing keys:

```python
{
    # existing
    "actions": ...,
    "log_normal_mu": ...,
    "log_normal_sigma2": ...,
    "action_map": ...,

    # new latent outputs if needed
    "decoder_states": ...,
    "decoder_memory": ...,

    # new explanation losses / diagnostics
    "explanation": {...},
}
```

Prefer computing explanation losses inside the model/module and returning scalar losses plus selected diagnostics, instead of making `train.py` reconstruct internal tensors.

However, keep the base scanpath loss in the current training code.

---

## 15.3 Inference forward

Base scanpath inference must continue to work when explanation is disabled.

Recommended flag:

```text
--enable_explanation
```

and optionally:

```text
--return_explanation_latents
--generate_explanations
```

Default production scanpath inference should not require WHY/HOW.

---

# 16. Critical R0 Inference Limitation

WHY-R0/HOW-R0 use:

```text
gold K*
```

to determine which episode slots receive text/alignment supervision and which episode tokens are active for HOW mean pooling.

The architecture intentionally has:

```text
no episode-presence loss
no episode-count head
```

Therefore R0 is naturally a **training auxiliary branch**.

Do not invent a deploy-time episode-count rule.

Consequences:

```text
- scanpath inference can omit the entire explanation module;
- WHAT can in principle be generated from predicted latents;
- WHY/HOW generation without gold K* is not fully specified by R0;
- if deploy-time WHY/HOW is later required, add an explicit presence/count design as a new architecture version.
```

---

# 17. Dataset / Collator Changes

Primary files:

```text
P/dataset/dataset.py
```

for O/F/C.

Add explanation annotation loading to the active dataset classes used by supervised training.

Do not modify unused legacy dataset classes and assume training will see the fields.

The collator must preserve:

```text
query grouping behavior
subject flattening behavior
existing target scanpath shapes
```

while additionally returning padded explanation metadata.

Recommended post-flatten shapes:

```text
what_texts          : Python nested text structure aligned to [N, L]
why_membership      : [N, L, Kmax]
episode_count       : [N]
why_texts           : per-example list of up to Kmax strings
how_text            : [N] strings
query_text           : [N] strings
```

Text objects do not need to be forced into numeric tensors before the explanation tokenizer.

---

# 18. Supervised Training Integration

Primary file:

```text
P/train.py::main.train
```

Current supervised phase:

```text
scanpath forward
-> action loss
-> duration loss
-> backward
```

Extend to:

\[
\mathcal L_{total}
=
\mathcal L_{scan}
+
\mathcal L_{EXP}
\]

where:

\[
\mathcal L_{EXP}
=
\lambda_W\mathcal L_{WHAT}
+
\lambda_Y\mathcal L_{WHY-R0}
+
\lambda_H\mathcal L_{HOW-R0}
\]

and:

\[
\mathcal L_{WHAT}
=
\lambda_{txt}\mathcal L_{what-txt}
+
\lambda_{align}\mathcal L_{what-align}
\]

\[
\mathcal L_{WHY-R0}
=
\lambda_{route}\mathcal L_{route}
+
\lambda_{txt}^{Y}\mathcal L_{why-txt}
+
\lambda_{align}^{Y}\mathcal L_{why-align}
\]

\[
\mathcal L_{HOW-R0}
=
\lambda_{txt}^{H}\mathcal L_{how-txt}
+
\lambda_{align}^{H}\mathcal L_{how-align}
\]

Recommended training code structure:

```python
scan_loss = action_loss + lambda_1 * duration_loss

exp_out = model_output["explanation"]
exp_loss = exp_out["loss_explanation"]

loss = scan_loss + exp_loss
loss.backward()
```

Do not detach `z`, `A`, `r_R0`, or `g_R0` before explanation losses.

Gold text embeddings are stop-gradient targets.

---

# 19. RL Phase

The current predictor later switches into ScanMatch policy-gradient training.

The architecture document defines the explanation module as joint auxiliary supervision with the base scanpath objective, but does not define explanation supervision inside the RL phase.

Therefore:

```text
DO NOT automatically add L_EXP to RL
```

unless the project explicitly decides to do so.

Safe first implementation:

```text
supervised phase:
    L_scan + L_EXP

RL phase:
    existing RL objective unchanged
```

This minimizes interference with the existing eval-mode-with-autograd RL path.

Expose this behavior as a clear config if later experimentation is desired.

---

# 20. Configuration Additions

Primary:

```text
P/opts.py
P/test.py parser if inference options are needed
launch scripts / YAML as appropriate
```

Recommended config surface:

```text
--enable_explanation

# latent dimensions
--explanation_dim
--router_kmax
--how_hidden_dim

# semantic encoder
--semantic_encoder_name
--freeze_semantic_encoder

# causal LLM
--explanation_llm_name
--freeze_explanation_llm

# loss weights
--lambda_exp_what
--lambda_exp_why
--lambda_exp_how

--lambda_what_txt
--lambda_what_align

--lambda_route
--lambda_why_txt
--lambda_why_align

--lambda_how_txt
--lambda_how_align

# optional diagnostics
--generate_explanations
--return_explanation_latents
```

Do not hardcode one LLM checkpoint inside model code.

If train/test parsers remain separate, mirror all inference-relevant arguments explicitly.

---

# 21. Checkpoint Compatibility

Adding registered explanation modules changes predictor `state_dict`.

Current predictor checkpoint loading is strict by default.

Therefore old scanpath checkpoints may fail after adding the module.

Choose one explicit migration behavior, for example:

```text
A. construct explanation module only when --enable_explanation is set,
   and load a base checkpoint with a controlled non-strict path; or

B. load base predictor weights first, then initialize explanation modules;
   save new-format checkpoints afterward.
```

Do not globally switch every checkpoint load to `strict=False` without logging missing/unexpected keys.

Log:

```text
missing keys
unexpected keys
whether explanation modules were freshly initialized
```

The offline SE-Net subject embedding table remains a separate dependency and is not part of the predictor checkpoint.

---

# 22. Recommended Code Decomposition

```text
P/models/explanation.py
|
|-- FixationReasoningEncoder
|     |-- P_h
|     |-- P_g
|     |-- MLP_pos
|     |-- duration_parameter_adapter
|     |-- MLP_d
|     |-- MLP_fix
|     `-- LayerNorm
|
|-- WhyR0Router
|     |-- temporal PE
|     |-- P_q_R
|     `-- MLP_R
|
|-- HowR0Aggregator
|     |-- P_q_H
|     |-- MLP_H
|     `-- LayerNorm
|
`-- HierarchicalExplanationModule
      |-- fixation_encoder
      |-- why_router
      |-- how_aggregator
      `-- shared LLM wrapper

P/models/explanation_llm.py
|
|-- frozen semantic encoder (RoBERTa role)
|-- shared causal LLM
|-- <WHAT>, <WHY>, <HOW> learnable prefix embeddings
|-- branch-specific query projections
|-- latent-to-LLM projections
|-- teacher-forced LM loss helpers
`-- semantic alignment helpers
```

Keep routing/aggregation logic outside the LLM wrapper.

---

# 23. Exact Edit Routing

## 23.1 `P/models/models.py`

Inspect/change for:

```text
- expose the personalized spatial memory used by decoder/head;
- preserve existing decoder behavior;
- do not change the non-autoregressive target structure;
- keep tensor differentiable.
```

Do not redesign the base Transformer just to add explanation supervision.

---

## 23.2 `P/models/gazeformer.py`

Main integration point.

Need access to:

```text
outs
personalized decoder memory
final actions logits
log_normal_mu
log_normal_sigma2
```

Then call:

```text
HierarchicalExplanationModule
```

before any detached sampling path.

---

## 23.3 `P/dataset/dataset.py`

Add:

```text
WHAT texts
WHY episode grouping
WHY texts
HOW text
raw explanation query text
```

and collate them consistently with the existing flattened subject/query batch.

---

## 23.4 `P/train.py`

Add:

```text
explanation target extraction
L_EXP
joint supervised loss
logging
```

Do not modify the current RL objective in the first implementation.

---

## 23.5 `P/test.py`

Only needed if explanation generation/diagnostics are requested at test time.

Base scanpath test must remain functional without explanation annotations.

---

## 23.6 `P/models/loss.py`

Optional.

The explanation-specific loss code can live with the explanation module because it depends strongly on masks/text/semantic embeddings.

Use `models/loss.py` only if the project wants all scalar objectives centralized.

---

# 24. O/F/C Replication Rule

The predictor exists as three dataset-specific copies.

Implementation order:

```text
1. implement and smoke-test one canonical predictor copy;
2. diff equivalent model/train/dataset files in O/F/C;
3. port changes deliberately;
4. preserve known local differences;
5. do not bulk overwrite O/F/C files.
```

Known local differences from `CODEBASE_MAP.md` include:

```text
- OSIE handling of 1-D subject table;
- query-position initialization differences;
- duration squeeze differences;
- dataset split/coordinate behavior;
- C legacy fine-tune branch;
- O test cap and subject-ID recovery.
```

---

# 25. Gradient-Flow Contract

The implementation is correct only if these paths remain differentiable:

```text
L_WHAT
  -> z_t
  -> decoder states / visual evidence / duration pathway

L_route
  -> A_tk
  -> MLP_R
  -> z_t

L_why_txt / L_why_align
  -> r_k^R0
  -> A_tk and z_t

L_how_txt / L_how_align
  -> g^R0
  -> active r_k^R0
  -> A_tk and z_t
```

No `.detach()`, `.data`, NumPy conversion, categorical sampling, or `argmax` may occur in those paths.

Allowed stop-gradient:

```text
gold text semantic embeddings only
```

as specified by the architecture.

---

# 26. Masking Contract

Use three logically distinct masks:

```text
1. fixation_mask
   real fixation timestep
   recommended source: duration mask semantics

2. episode_slot_mask
   k < K* for each example

3. LM_token_mask
   valid gold text tokens only
```

Do not reuse the action mask as fixation mask because the action mask includes STOP.

---

# 27. Numerical / Shape Assertions

Add assertions early.

Examples:

```python
assert decoder_states.ndim == 3
assert decoder_memory.ndim == 3
assert action_logits.shape[-1] == G + 1
assert fixation_prob.shape[-1] == G
assert torch.allclose(
    fixation_prob.sum(-1),
    torch.ones_like(fixation_prob.sum(-1)),
    atol=1e-5,
)

assert z.shape[:2] == fixation_mask.shape
assert router_prob.shape[-1] == Kmax
assert why_membership.shape == router_prob.shape

assert (episode_count <= Kmax).all()
```

For each real fixation:

```python
assert gold_membership_row.sum() == 1
```

For padded rows:

```python
assert gold_membership_row.sum() == 0
```

---

# 28. Minimal Unit Tests

Create tests around the new module before full training.

## Test A — spatial probability

Input:

```text
random action logits [N,L,G+1]
```

Verify:

```text
m_t excludes STOP
sum_G m_t = 1
gradient reaches action logits spatial portion
```

## Test B — soft visual evidence

Verify:

```text
g_t shape [N,L,Dv]
gradient reaches both m_t and E
```

## Test C — padding

Verify padded timesteps contribute zero mass to episode aggregation.

## Test D — router target

Verify:

```text
routing CE only uses real fixation rows
unused slots get no language/alignment loss
```

## Test E — HOW active-slot pooling

Given `K*=2`, changing slots 2..Kmax must not change HOW token.

## Test F — hierarchy gradient

Backprop from only HOW loss and verify nonzero gradients reach:

```text
HowR0Aggregator
WhyR0Router
FixationReasoningEncoder
decoder-state input
```

## Test G — shared LLM

Verify WHAT/WHY/HOW reference the same LLM parameter object, not three copies.

## Test H — base predictor compatibility

With explanation disabled:

```text
existing scanpath output keys/shapes unchanged
existing sampling path unchanged
```

---

# 29. Implementation Stages

Implement in this order.

## Stage 1 — expose predictor tensors only

No LLM yet.

Expose:

```text
decoder states
decoder memory
final spatial fixation distribution
duration parameters
```

Verify no base metric change with explanation disabled.

## Stage 2 — fixation reasoning token

Implement:

```text
g_t
p_t
duration representation
z_t
```

Add shape/gradient tests.

## Stage 3 — WHAT

Add:

```text
query semantic encoder
shared LLM wrapper
WHAT prefix
WHAT text loss
WHAT alignment loss
```

Train with only WHAT auxiliary loss first.

## Stage 4 — WHY-R0

Add:

```text
gold grouping data
router
route loss
soft episode token
WHY text/alignment
```

## Stage 5 — HOW-R0

Add:

```text
gold K* active-slot mean
global token
HOW text/alignment
```

## Stage 6 — full joint objective

Enable:

```text
L_scan + L_EXP
```

and add logging for every sub-loss.

## Stage 7 — replicate to O/F/C

Only after one copy passes tests.

---

# 30. Logging Requirements

Log at least:

```text
scan/action loss
scan/duration loss

WHAT text loss
WHAT alignment loss

WHY route loss
WHY text loss
WHY alignment loss

HOW text loss
HOW alignment loss

weighted L_EXP
total loss
```

Useful diagnostics:

```text
mean router entropy
mean episode slot mass
fraction of examples at Kmax
mean WHAT/WHY/HOW token counts
```

Do not use generated natural-language samples as the only correctness signal.

---

# 31. Things Codex Must Not Guess

The following are **not established** by the two source documents and must remain configurable or explicitly unresolved.

### 31.1 Which causal LLM checkpoint to use

Not specified.

### 31.2 Whether the causal LLM is frozen or trainable

Not specified.

### 31.3 Exact explanation annotation file format/path

Not specified.

### 31.4 Exact raw query text source for every dataset

Current predictor uses cached vector embeddings. The architecture requires textual \(Q\) for RoBERTa.

### 31.5 Correct interpretation of `log_normal_sigma2`

Current loss/sampling semantics are inconsistent. Do not silently resolve this.

### 31.6 Deploy-time episode count for WHY/HOW

R0 intentionally has no presence/count head.

### 31.7 Explanation supervision during RL

Not defined by the architecture. Keep RL unchanged initially.

---

# 32. Hard Non-Goals for R0

Do not add these while implementing the requested baseline:

```text
- contextual Transformer router for WHY
- episode presence/count head
- trajectory Transformer for HOW
- generated WHAT -> WHY text chaining
- generated WHY -> HOW text chaining
- hard fixation argmax for visual evidence
- hard episode assignment before aggregation
- autoregressive scanpath decoder redesign
- online SE-Net execution
- joint SE-Net/predictor training
- support-set resampling changes
- new RL explanation reward
```

Those are separate architecture versions or experiments.

---

# 33. Compact End-to-End Pseudocode

```python
# ----- base predictor -----
decoder_memory, decoder_states = transformer(...)
action_logits, duration_mu, duration_param2 = predictor_heads(
    decoder_memory,
    decoder_states,
    subject_emb,
)

# action 0 = STOP; 1: = spatial
fixation_prob = softmax(action_logits[..., 1:], dim=-1)

# ----- explanation shared fixation representation -----
soft_visual = einsum("nlg,ngd->nld", fixation_prob, decoder_memory)
expected_xy = einsum("nlg,gc->nlc", fixation_prob, normalized_grid_xy)

pos_emb = MLP_pos(expected_xy)
dur_emb = duration_adapter(duration_mu, duration_param2)

z = LN(
    MLP_fix(
        concat(
            P_h(decoder_states),
            P_g(soft_visual),
            pos_emb,
            dur_emb,
        )
    )
)

q = semantic_encoder(query_text)  # architecture: RoBERTa role

# ----- WHAT -----
what_prefix = [
    WHAT_token,
    P_q_WHAT(q),
    P_LM_WHAT(z),
]
loss_what_txt = LM_teacher_forcing(what_prefix, gold_what, fixation_mask)
loss_what_align = cosine_align(P_W(z), SG(semantic_encoder(gold_what)))

# ----- WHY-R0 -----
tau = temporal_position(L)
router_input = concat(z, tau, broadcast(P_q_R(q)))
router_logits = MLP_R(router_input)
A = softmax(router_logits, dim=-1)

loss_route = routing_ce(
    A,
    gold_membership,
    fixation_mask,
)

A_masked = A * fixation_mask[..., None]
episode_mass = A_masked.sum(dim=1)
r = einsum("nlk,nld->nkd", A_masked, z)
r = r / (episode_mass[..., None] + eps)

why_prefix = [
    WHY_token,
    P_q_WHY(q),
    P_LM_Y(r),
]
loss_why_txt = LM_teacher_forcing_active_slots(
    why_prefix,
    gold_why,
    episode_count,
)
loss_why_align = cosine_align_active_slots(
    P_Y(r),
    SG(semantic_encoder(gold_why)),
    episode_count,
)

# ----- HOW-R0 -----
active_r_mean = mean_first_k(r, episode_count)

g = LN(
    MLP_H(
        concat(
            active_r_mean,
            P_q_H(q),
        )
    )
)

how_prefix = [
    HOW_token,
    P_q_HOW(q),
    P_LM_H(g),
]
loss_how_txt = LM_teacher_forcing(how_prefix, gold_how)
loss_how_align = cosine_align(
    P_H(g),
    SG(semantic_encoder(gold_how)),
)

# ----- hierarchy -----
loss_what = lambda_what_txt * loss_what_txt \
          + lambda_what_align * loss_what_align

loss_why = lambda_route * loss_route \
         + lambda_why_txt * loss_why_txt \
         + lambda_why_align * loss_why_align

loss_how = lambda_how_txt * loss_how_txt \
         + lambda_how_align * loss_how_align

loss_exp = lambda_W * loss_what \
         + lambda_Y * loss_why \
         + lambda_H * loss_how

loss_total = loss_scan + loss_exp
```

---

# 34. Final Codex Checklist

Before declaring the implementation complete:

```text
[ ] Existing predictor still runs with explanation disabled.
[ ] `outs` is exposed as differentiable decoder states.
[ ] The selected personalized spatial memory is exposed explicitly.
[ ] m_t is constructed from FINAL action logits, spatial slots only.
[ ] STOP is not a WHAT/WHY fixation.
[ ] No Sampling/NumPy path is used for explanation training.
[ ] z_t follows the architecture inputs: h_t, g_t, p_t, duration representation.
[ ] Raw query text / RoBERTa contract is explicit.
[ ] WHAT uses z_t directly.
[ ] WHY-R0 uses a simple MLP router.
[ ] Gold grouping supervises the router but is not router input.
[ ] Episode aggregation is soft and padding-masked.
[ ] HOW-R0 mean-pools only the first K* active gold episode slots.
[ ] One shared LLM is used for WHAT/WHY/HOW.
[ ] Branch-specific special tokens and projections are separate.
[ ] Gold semantic embeddings are stop-gradient targets.
[ ] L_EXP is added only to supervised training in the first implementation.
[ ] Existing RL path is unchanged.
[ ] Explanation-disabled test/inference requires no explanation labels.
[ ] Old-checkpoint compatibility behavior is explicit and logged.
[ ] Changes are ported deliberately across O/F/C.
[ ] Duration second-parameter semantics are not silently guessed.
[ ] Deploy-time WHY/HOW episode count is not invented.
```

---

# 35. Source References

Read alongside:

```text
architecture.md
CODEBASE_MAP.md
```

Most relevant `CODEBASE_MAP.md` sections:

```text
1   Architecture at a Glance
4   Core Component Map
5   Model Architecture
6   Data Pipeline
8   Training and Objectives
9   Checkpoint and Stage Dependency
10  Evaluation and Prediction
11  Configuration Surface
13  Implementation Navigation Guide
14  High-Risk Couplings and Invariants
18  Fast Navigation Index
```

Most relevant `architecture.md` sections:

```text
0      Overview
1-5    Shared Fixation Representation
6-10   WHAT
11-19  WHY-R0
20-26  HOW-R0
27-28  Joint Objective and Gradient Flow
29     Compact Notation Index
```
