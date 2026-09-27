# Hierarchical LLM Explanation Module

  

## 0. Overview

  

The explanation module attaches language supervision to three latent scales of the scanpath model:

  

$$

\boxed{

\mathbf z_t

\;\longrightarrow\;

\mathbf r_k^{R0}

\;\longrightarrow\;

\mathbf g^{R0}

}

$$

  

where

  

- $\mathbf z_t$: **Fixation Reasoning Token** — fixation level,

- $\mathbf r_k^{R0}$: **Episode Reasoning Token** — WHY episode level,

- $\mathbf g^{R0}$: **Global / Trajectory Reasoning Token** — whole-scanpath level.

  

The corresponding language targets are

  

$$

\boxed{

\begin{aligned}

\mathbf z_t &\leftrightarrow W_t^* && \text{WHAT},\\

\mathbf r_k^{R0} &\leftrightarrow Y_k^* && \text{WHY-R0},\\

\mathbf g^{R0} &\leftrightarrow \mathcal H^* && \text{HOW-R0}.

\end{aligned}

}

$$

  

The computational hierarchy is

  

$$

\boxed{

\mathbf h_t

\rightarrow

\mathbf z_t

\rightarrow

\mathbf r_k^{R0}

\rightarrow

\mathbf g^{R0}.

}

$$

  

Natural-language outputs are **not** chained as

  

$$

WHAT\rightarrow WHY\rightarrow HOW.

$$

  

Instead, WHAT, WHY, and HOW are three supervision heads attached to different latent scales of the same hierarchy. All three branches use the **same LLM**, while branch identity is indicated by branch-specific special tokens and projections.

  

---

  

# Part I. Shared Fixation Representation

  

## 1. Predicted Scanpath

  

The scanpath prediction model outputs

  

$$

\boxed{

\hat{\mathcal S}

=

\left\{

\hat{\mathbf f}_t

=

(\hat{x}_t,\hat{y}_t,\hat{d}_t)

\right\}_{t=1}^{\hat T}

}

$$

  

where

  

- $\hat T$: predicted scanpath length,

- $\hat{x}_t,\hat{y}_t$: predicted fixation coordinate at timestep $t$,

- $\hat{d}_t$: predicted fixation duration.

  

The scanpath decoder representation at timestep $t$ is

  

$$

\boxed{\mathbf h_t}.

$$

  

---

  

## 2. Visual Features and Fixation-Aware Evidence

  

The visual encoder produces a spatial feature map

  

$$

\boxed{

E=\{\mathbf E_{ij}\},

\qquad

\mathbf E_{ij}\in\mathbb R^{d_v}.

}

$$

  

At timestep $t$, the decoder produces a fixation probability map

  

$$

\boxed{m_t(i,j)},

\qquad

\sum_{i,j}m_t(i,j)=1.

$$

  

### 2.1. Soft fixated visual feature

  

Instead of hard-sampling one spatial position, use the full fixation distribution:

  

$$

\boxed{

\mathbf g_t

=

\sum_{i,j}m_t(i,j)\mathbf E_{ij}

}

$$

  

with

  

$$

\mathbf g_t\in\mathbb R^{d_v}.

$$

  

$\mathbf g_t$ represents the visual evidence attended to at fixation $t$.

  

### 2.2. Expected fixation coordinate

  

Map spatial location $(i,j)$ to normalized image coordinates

  

$$

x_{ij}=\frac{i}{W-1},

\qquad

y_{ij}=\frac{j}{H-1}.

$$

  

The expected coordinate under $m_t$ is

  

$$

\boxed{

(\tilde{x}_t,\tilde{y}_t)

=

\sum_{i,j}m_t(i,j)(x_{ij},y_{ij})

}

$$

  

and its learnable positional representation is

  

$$

\boxed{

\mathbf p_t

=

\operatorname{MLP}_{\mathrm{pos}}

\left([

\tilde{x}_t;\tilde{y}_t

]\right).

}

$$

  

---

  

## 3. Fixation Duration Representation

  

Assume the scanpath model predicts duration-distribution statistics

  

$$

\mu_t,

\qquad

\sigma_t.

$$

  

The duration embedding is

  

$$

\boxed{

\mathbf r_t

=

\operatorname{MLP}_d

\left([

\mu_t;\log\sigma_t

]\right).

}

$$

  

$\mathbf r_t$ represents temporal/duration information of fixation $t$.

  

---

  

## 4. Task Representation

  

Given task/query $Q$,

  

$$

\boxed{

\mathbf q

=

\operatorname{RoBERTa}(Q),

\qquad

\mathbf q\in\mathbb R^{d_q}.

}

$$

  

The base scanpath model uses

  

$$

\boxed{

\mathbf m_0

=

\operatorname{TaskEncoder}(\mathbf q).

}

$$

  

The explanation module keeps $\mathbf q$ as the semantic representation of the task/query.

  

---

  

## 5. Fixation Reasoning Token

  

Project decoder and visual representations:

  

$$

\tilde{\mathbf h}_t=P_h\mathbf h_t,

\qquad

\tilde{\mathbf g}_t=P_g\mathbf g_t.

$$

  

Then construct the fixation-level reasoning token

  

$$

\boxed{

\mathbf z_t

=

\operatorname{LN}

\left(

\operatorname{MLP}_{\mathrm{fix}}

\left([

\tilde{\mathbf h}_t;

\tilde{\mathbf g}_t;

\mathbf p_t;

\mathbf r_t

]\right)

\right)

}

$$

  

with

  

$$

\boxed{

\mathbf z_t\in\mathbb R^{d_z}.

}

$$

  

The full fixation-token sequence is

  

$$

\boxed{

Z=(\mathbf z_1,\ldots,\mathbf z_T).

}

$$

  

---

  

# Part II. WHAT — Fixation-Level Explanation

  

## 6. Objective

  

WHAT explains the semantic content of each fixation:

  

> **What is the observer looking at at this fixation?**

  

Its latent-to-language relation is

  

$$

\boxed{

\mathbf z_t\longrightarrow W_t.

}

$$

  

---

  

## 7. WHAT LLM Interface

  

Project the fixation token into the LLM hidden space:

  

$$

\boxed{

\mathbf c_t^{LM}

=

P_{LM}\mathbf z_t,

\qquad

\mathbf c_t^{LM}\in\mathbb R^{d_L}.

}

$$

  

Project the task/query using a WHAT-specific projector:

  

$$

\boxed{

\mathbf c_q^{\mathrm{WHAT}}

=

P_q^{\mathrm{WHAT}}(\mathbf q),

\qquad

\mathbf c_q^{\mathrm{WHAT}}\in\mathbb R^{d_L}.

}

$$

  

Let

  

$$

\boxed{

e_{\langle\mathrm{WHAT}\rangle}\in\mathbb R^{d_L}

}

$$

  

be the learnable WHAT special-token embedding. The LLM prefix for fixation $t$ is

  

$$

\boxed{

X_t^{\mathrm{WHAT}}

=

[

e_{\langle\mathrm{WHAT}\rangle};

\mathbf c_q^{\mathrm{WHAT}};

\mathbf c_t^{LM}

].

}

$$

  

---

  

## 8. Gold WHAT Target and Text Generation

  

The gold WHAT explanation is

  

$$

\boxed{

W_t^*

=

(w_{t,1}^*,\ldots,w_{t,L_t}^*)

}

$$

  

and the generated explanation is

  

$$

\boxed{

\hat W_t

=

(\hat w_{t,1},\ldots,\hat w_{t,\hat L_t}).

}

$$

  

The total number of gold WHAT tokens in one scanpath is

  

$$

\boxed{

N_W=\sum_{t=1}^{T}L_t.

}

$$

  

The autoregressive text-generation loss is

  

$$

\boxed{

\mathcal L_{\mathrm{what\text{-}txt}}

=

-

\frac{1}{N_W}

\sum_{t=1}^{T}

\sum_{l=1}^{L_t}

\log

p_\theta

\left(

w_{t,l}^*

\mid

w_{t,<l}^*,

X_t^{\mathrm{WHAT}}

\right)

}

$$

  

where

  

$$

w_{t,<l}^*=(w_{t,1}^*,\ldots,w_{t,l-1}^*).

$$

  

---

  

## 9. WHAT Semantic Alignment

  

Encode the gold explanation with the frozen RoBERTa encoder:

  

$$

\boxed{

\mathbf e_t^W

=

\operatorname{sg}

\left[

\operatorname{RoBERTa}(W_t^*)

\right].

}

$$

  

Project the fixation token to the same semantic space:

  

$$

\boxed{

\mathbf a_t^W=P_W\mathbf z_t.

}

$$

  

Normalize both representations:

  

$$

\boxed{

\bar{\mathbf a}_t^W

=

\frac{\mathbf a_t^W}{\|\mathbf a_t^W\|_2},

\qquad

\bar{\mathbf e}_t^W

=

\frac{\mathbf e_t^W}{\|\mathbf e_t^W\|_2}.

}

$$

  

The cosine-alignment loss is

  

$$

\boxed{

\mathcal L_{\mathrm{what\text{-}align}}

=

\frac{1}{T}

\sum_{t=1}^{T}

\left(

1-(\bar{\mathbf a}_t^W)^\top\bar{\mathbf e}_t^W

\right).

}

$$

  

---

  

## 10. WHAT Objective

  

$$

\boxed{

\mathcal L_{\mathrm{WHAT}}

=

\lambda_{\mathrm{txt}}

\mathcal L_{\mathrm{what\text{-}txt}}

+

\lambda_{\mathrm{align}}

\mathcal L_{\mathrm{what\text{-}align}}.

}

$$

  

Thus WHAT provides fixation-level

  

$$

\boxed{

\text{generation supervision}

+

\text{semantic alignment supervision}.

}

$$

  

---

  

# Part III. WHY-R0 — Episode-Level Explanation

  

## 11. Objective

  

WHY-R0 explains the **shared intent/reason** of a group of fixations:

  

$$

\boxed{

\{\mathbf z_t\}_{t=1}^{T}

\rightarrow

\text{episode routing}

\rightarrow

\mathbf r_k^{R0}

\rightarrow

Y_k.

}

$$

  

WHY-R0 does **not** use generated WHAT text as input. WHAT and WHY are supervised from the same latent fixation representations $\mathbf z_t$.

  

---

  

## 12. Gold Episode Structure

  

Assume the $T$ fixations are grouped into $K^*$ gold reasoning episodes

  

$$

\boxed{

\mathcal R_1^*,\ldots,\mathcal R_{K^*}^*.

}

$$

  

Each episode $j$ has a gold WHY explanation

  

$$

\boxed{

Y_j^*

=

(y_{j,1}^*,\ldots,y_{j,L_j}^*).

}

$$

  

The gold grouping is represented by

  

$$

\boxed{

M^*\in\{0,1\}^{T\times K^*}

}

$$

  

with

  

$$

\boxed{

M_{tj}^*

=

\begin{cases}

1,& t\in\mathcal R_j^*,\\

0,& \text{otherwise}.

\end{cases}

}

$$

  

Each fixation belongs to exactly one WHY episode:

  

$$

\boxed{

\sum_{j=1}^{K^*}M_{tj}^*=1.

}

$$

  

Episodes are not required to be temporally contiguous.

  

### 12.1. Canonical episode ordering

  

Because WHY-R0 uses fixed router slots, order gold episodes by their first occurring timestep:

  

$$

\boxed{

\min\mathcal R_1^*

<

\min\mathcal R_2^*

<

\cdots

<

\min\mathcal R_{K^*}^*.

}

$$

  

Choose a maximum number of episode slots

  

$$

\boxed{K_{\max}},

\qquad

K^*\le K_{\max}.

$$

  

Pad the gold membership matrix to

  

$$

\boxed{

\tilde M^*\in\{0,1\}^{T\times K_{\max}}

}

$$

  

with

  

$$

\boxed{

\tilde M_{tk}^*

=

\begin{cases}

M_{tk}^*,& k\le K^*,\\

0,& k>K^*.

\end{cases}

}

$$

  

Unused slots receive no WHY text/alignment supervision.

  

---

  

## 13. WHY-R0 Router

  

### 13.1. Routing context

  

Add a temporal positional embedding

  

$$

\boxed{

\boldsymbol\tau_t

=

PE_{1D}

\left(

\frac{t-1}{T-1}

\right),

\qquad

\boldsymbol\tau_t\in\mathbb R^{d_\tau}.

}

$$

  

Project the task/query into router space:

  

$$

\boxed{

\mathbf c_q^R

=

P_q^R(\mathbf q),

\qquad

\mathbf c_q^R\in\mathbb R^{d_R^q}.

}

$$

  

For each fixation,

  

$$

\boxed{

\mathbf u_t^{R0}

=

[

\mathbf z_t;

\boldsymbol\tau_t;

\mathbf c_q^R

].

}

$$

  

### 13.2. Episode assignment

  

The router produces

  

$$

\boxed{

\boldsymbol\ell_t

=

\operatorname{MLP}_R(\mathbf u_t^{R0}),

\qquad

\boldsymbol\ell_t\in\mathbb R^{K_{\max}}.

}

$$

  

The soft assignment probability is

  

$$

\boxed{

A_{tk}

=

\frac{\exp(\ell_{tk})}

{\sum_{k'=1}^{K_{\max}}\exp(\ell_{tk'})}

}

$$

  

so that

  

$$

\boxed{

\mathbf A\in[0,1]^{T\times K_{\max}},

\qquad

\sum_{k=1}^{K_{\max}}A_{tk}=1.

}

$$

  

Interpretation:

  

$$

\boxed{

A_{tk}

=

P(\text{fixation }t\text{ belongs to episode }k).

}

$$

  

The gold grouping $\tilde M^*$ is **not** an input to the router; it is only a supervision target.

  

---

  

## 14. Routing Supervision

  

$$

\boxed{

\mathcal L_{\mathrm{route}}

=

-

\frac{1}{T}

\sum_{t=1}^{T}

\sum_{k=1}^{K_{\max}}

\tilde M_{tk}^*

\log A_{tk}.

}

$$

  

This directly supervises fixation-to-episode organization.

  

---

  

## 15. Soft Episode Aggregation

  

Define the soft mass of slot $k$:

  

$$

\boxed{

n_k

=

\sum_{t=1}^{T}A_{tk}.

}

$$

  

The episode reasoning token is the soft weighted mean

  

$$

\boxed{

\mathbf r_k^{R0}

=

\frac{

\sum_{t=1}^{T}A_{tk}\mathbf z_t

}{

n_k+\epsilon

}

=

\frac{

\sum_tA_{tk}\mathbf z_t

}{

\sum_tA_{tk}+\epsilon

}

}

$$

  

with

  

$$

\boxed{

\mathbf r_k^{R0}\in\mathbb R^{d_z}.

}

$$

  

---

  

## 16. WHY LLM Interface

  

Project the episode token into LLM space:

  

$$

\boxed{

\mathbf c_k^{LM,Y}

=

P_{LM}^{Y}\mathbf r_k^{R0},

\qquad

\mathbf c_k^{LM,Y}\in\mathbb R^{d_L}.

}

$$

  

Project the query using a WHY-specific projector:

  

$$

\boxed{

\mathbf c_q^{\mathrm{WHY}}

=

P_q^{\mathrm{WHY}}(\mathbf q),

\qquad

\mathbf c_q^{\mathrm{WHY}}\in\mathbb R^{d_L}.

}

$$

  

Let

  

$$

\boxed{

e_{\langle\mathrm{WHY}\rangle}\in\mathbb R^{d_L}

}

$$

  

be the WHY special-token embedding. The LLM prefix for episode $k$ is

  

$$

\boxed{

X_k^{\mathrm{WHY}}

=

[

e_{\langle\mathrm{WHY}\rangle};

\mathbf c_q^{\mathrm{WHY}};

\mathbf c_k^{LM,Y}

].

}

$$

  

---

  

## 17. WHY Text Generation

  

The gold and predicted WHY explanations are

  

$$

\boxed{

Y_k^*

=

(y_{k,1}^*,\ldots,y_{k,L_k}^*)

}

$$

  

and

  

$$

\boxed{

\hat Y_k

=

(\hat y_{k,1},\ldots,\hat y_{k,\hat L_k}).

}

$$

  

Text/alignment supervision is computed only over the $K^*$ active gold episodes.

  

The total number of gold WHY tokens is

  

$$

\boxed{

N_Y

=

\sum_{k=1}^{K^*}L_k.

}

$$

  

The autoregressive text-generation loss is

  

$$

\boxed{

\mathcal L_{\mathrm{why\text{-}txt}}

=

-

\frac{1}{N_Y}

\sum_{k=1}^{K^*}

\sum_{l=1}^{L_k}

\log

p_\theta

\left(

y_{k,l}^*

\mid

y_{k,<l}^*,

X_k^{\mathrm{WHY}}

\right)

}

$$

  

where

  

$$

y_{k,<l}^*

=

(y_{k,1}^*,\ldots,y_{k,l-1}^*).

$$

  

---

  

## 18. WHY Semantic Alignment

  

Encode the gold WHY explanation:

  

$$

\boxed{

\mathbf e_k^Y

=

\operatorname{sg}

\left[

\operatorname{RoBERTa}(Y_k^*)

\right].

}

$$

  

Project the episode token:

  

$$

\boxed{

\mathbf a_k^Y

=

P_Y\mathbf r_k^{R0}.

}

$$

  

Normalize:

  

$$

\boxed{

\bar{\mathbf a}_k^Y

=

\frac{\mathbf a_k^Y}{\|\mathbf a_k^Y\|_2},

\qquad

\bar{\mathbf e}_k^Y

=

\frac{\mathbf e_k^Y}{\|\mathbf e_k^Y\|_2}.

}

$$

  

Then

  

$$

\boxed{

\mathcal L_{\mathrm{why\text{-}align}}

=

\frac{1}{K^*}

\sum_{k=1}^{K^*}

\left(

1-(\bar{\mathbf a}_k^Y)^\top\bar{\mathbf e}_k^Y

\right).

}

$$

  

---

  

## 19. WHY-R0 Objective

  

$$

\boxed{

\mathcal L_{\mathrm{WHY\text{-}R0}}

=

\lambda_{\mathrm{route}}\mathcal L_{\mathrm{route}}

+

\lambda_{\mathrm{txt}}^Y\mathcal L_{\mathrm{why\text{-}txt}}

+

\lambda_{\mathrm{align}}^Y\mathcal L_{\mathrm{why\text{-}align}}.

}

$$

  

Equivalently,

  

$$

\boxed{

\underbrace{\mathcal L_{\mathrm{route}}}_{\text{episode structure}}

+

\underbrace{

\mathcal L_{\mathrm{why\text{-}txt}}

+

\mathcal L_{\mathrm{why\text{-}align}}

}_{\text{episode semantics}}.

}

$$

  

WHY-R0 has no episode-presence loss in the current formulation. Since WHY is an auxiliary branch for joint training and need not be used at scanpath inference, no separate head is required to predict the number of active episodes.

  

---

  

# Part IV. HOW-R0 — Trajectory-Level Explanation

  

## 20. Objective

  

HOW-R0 explains the **overall strategy of the full scanpath**:

  

$$

\boxed{

\{\mathbf r_k^{R0}\}_{k=1}^{K^*}

\rightarrow

\mathbf g^{R0}

\rightarrow

\mathcal H.

}

$$

  

HOW-R0 does **not** use generated WHAT/WHY text as input. Language is used only as a supervision target.

  

---

  

## 21. Input Episodes and Simple Aggregation

  

The active WHY-R0 episode tokens are

  

$$

\boxed{

R^{R0}

=

(\mathbf r_1^{R0},\ldots,\mathbf r_{K^*}^{R0}).

}

$$

  

Each token is

  

$$

\mathbf r_k^{R0}

=

\frac{

\sum_{t=1}^{T}A_{tk}\mathbf z_t

}{

\sum_{t=1}^{T}A_{tk}+\epsilon

},

\qquad

\mathbf r_k^{R0}\in\mathbb R^{d_z}.

$$

  

HOW-R0 is intentionally simple: no trajectory Transformer and no explicit pairwise episode modeling. Mean-pool the active episode tokens:

  

$$

\boxed{

\bar{\mathbf r}^{R0}

=

\frac{1}{K^*}

\sum_{k=1}^{K^*}

\mathbf r_k^{R0},

\qquad

\bar{\mathbf r}^{R0}\in\mathbb R^{d_z}.

}

$$

  

---

  

## 22. Global Trajectory Reasoning Token

  

Project the task/query into HOW aggregation space:

  

$$

\boxed{

\mathbf c_q^H

=

P_q^H(\mathbf q),

\qquad

\mathbf c_q^H\in\mathbb R^{d_H^q}.

}

$$

  

Construct the trajectory-level reasoning token:

  

$$

\boxed{

\mathbf g^{R0}

=

\operatorname{LN}

\left(

\operatorname{MLP}_H

\left([

\bar{\mathbf r}^{R0};

\mathbf c_q^H

]\right)

\right),

\qquad

\mathbf g^{R0}\in\mathbb R^{d_H}.

}

$$

  

Thus

  

$$

\boxed{

\mathbf z_t

\rightarrow

\mathbf r_k^{R0}

\rightarrow

\mathbf g^{R0}.

}

$$

  

---

  

## 23. HOW LLM Interface

  

Project the trajectory token into LLM space:

  

$$

\boxed{

\mathbf c_H^{LM}

=

P_{LM}^{H}\mathbf g^{R0},

\qquad

\mathbf c_H^{LM}\in\mathbb R^{d_L}.

}

$$

  

Project the query using a HOW-specific projector:

  

$$

\boxed{

\mathbf c_q^{\mathrm{HOW}}

=

P_q^{\mathrm{HOW}}(\mathbf q),

\qquad

\mathbf c_q^{\mathrm{HOW}}\in\mathbb R^{d_L}.

}

$$

  

Let

  

$$

\boxed{

e_{\langle\mathrm{HOW}\rangle}\in\mathbb R^{d_L}

}

$$

  

be the HOW special-token embedding. The LLM prefix is

  

$$

\boxed{

X^{\mathrm{HOW}}

=

[

e_{\langle\mathrm{HOW}\rangle};

\mathbf c_q^{\mathrm{HOW}};

\mathbf c_H^{LM}

].

}

$$

  

---

  

## 24. HOW Text Generation

  

The gold HOW explanation is

  

$$

\boxed{

\mathcal H^*

=

(\omega_1^*,\ldots,\omega_{L_H}^*)

}

$$

  

and the generated explanation is

  

$$

\boxed{

\hat{\mathcal H}

=

(\hat\omega_1,\ldots,\hat\omega_{\hat L_H}).

}

$$

  

The autoregressive text-generation loss is

  

$$

\boxed{

\mathcal L_{\mathrm{how\text{-}txt}}

=

-

\frac{1}{L_H}

\sum_{l=1}^{L_H}

\log

p_\theta

\left(

\omega_l^*

\mid

\omega_{<l}^*,

X^{\mathrm{HOW}}

\right)

}

$$

  

where

  

$$

\omega_{<l}^*

=

(\omega_1^*,\ldots,\omega_{l-1}^*).

$$

  

---

  

## 25. HOW Semantic Alignment

  

Encode the gold HOW explanation:

  

$$

\boxed{

\mathbf e^H

=

\operatorname{sg}

\left[

\operatorname{RoBERTa}(\mathcal H^*)

\right].

}

$$

  

Project the trajectory token:

  

$$

\boxed{

\mathbf a^H

=

P_H\mathbf g^{R0}.

}

$$

  

Normalize:

  

$$

\boxed{

\bar{\mathbf a}^{H}

=

\frac{\mathbf a^H}{\|\mathbf a^H\|_2},

\qquad

\bar{\mathbf e}^{H}

=

\frac{\mathbf e^H}{\|\mathbf e^H\|_2}.

}

$$

  

The alignment loss is

  

$$

\boxed{

\mathcal L_{\mathrm{how\text{-}align}}

=

1-(\bar{\mathbf a}^{H})^\top\bar{\mathbf e}^{H}.

}

$$

  

---

  

## 26. HOW-R0 Objective

  

$$

\boxed{

\mathcal L_{\mathrm{HOW\text{-}R0}}

=

\lambda_{\mathrm{txt}}^H\mathcal L_{\mathrm{how\text{-}txt}}

+

\lambda_{\mathrm{align}}^H\mathcal L_{\mathrm{how\text{-}align}}.

}

$$

  

HOW-R0 has no additional routing loss because fixation-to-episode structure is already supervised in WHY-R0.

  

---

  

# Part V. Joint Objective and Supervision Flow

  

## 27. Explanation Objective

  

The total explanation loss is

  

$$

\boxed{

\mathcal L_{\mathrm{EXP}}

=

\lambda_W\mathcal L_{\mathrm{WHAT}}

+

\lambda_Y\mathcal L_{\mathrm{WHY\text{-}R0}}

+

\lambda_H\mathcal L_{\mathrm{HOW\text{-}R0}}.

}

$$

  

The joint-training objective with the base scanpath model is

  

$$

\boxed{

\mathcal L_{\mathrm{total}}

=

\mathcal L_{\mathrm{scan}}

+

\mathcal L_{\mathrm{EXP}}.

}

$$

  

---

  

## 28. Hierarchical Gradient Flow

  

WHAT supervision acts directly on fixation tokens:

  

$$

\boxed{

\mathcal L_{\mathrm{WHAT}}

\rightarrow

\mathbf z_t.

}

$$

  

WHY routing and semantics act through episode construction:

  

$$

\boxed{

\mathcal L_{\mathrm{route}}

\rightarrow

A_{tk}

\rightarrow

\operatorname{MLP}_R

\rightarrow

\mathbf z_t,

}

$$

  

$$

\boxed{

\mathcal L_{\mathrm{why\text{-}txt}},

\mathcal L_{\mathrm{why\text{-}align}}

\rightarrow

\mathbf r_k^{R0}

\rightarrow

A_{tk},\mathbf z_t.

}

$$

  

HOW supervision propagates through the entire hierarchy:

  

$$

\boxed{

\mathcal L_{\mathrm{how\text{-}txt}},

\mathcal L_{\mathrm{how\text{-}align}}

\rightarrow

\mathbf g^{R0}

\rightarrow

\bar{\mathbf r}^{R0}

\rightarrow

\mathbf r_k^{R0}

\rightarrow

A_{tk},\mathbf z_t.

}

$$

  

Therefore the three branches supervise three semantic scales:

  

$$

\boxed{

\begin{aligned}

\text{WHAT:}&\quad \text{fixation semantics},\\

\text{WHY-R0:}&\quad \text{episode structure + episode semantics},\\

\text{HOW-R0:}&\quad \text{trajectory semantics}.

\end{aligned}

}

$$

  

---

  

# 29. Compact Notation Index

  

## Shared representations

  

| Symbol | Meaning |

|---|---|

| $\hat{\mathcal S}$ | Predicted scanpath |

| $\hat{\mathbf f}_t$ | Predicted fixation at timestep $t$ |

| $\hat T$ | Predicted scanpath length |

| $\mathbf E_{ij}$ | Visual feature at spatial location $(i,j)$ |

| $d_v$ | Visual feature dimension |

| $\mathbf h_t$ | Decoder representation at timestep $t$ |

| $m_t(i,j)$ | Fixation probability at location $(i,j)$ |

| $\mathbf g_t$ | Soft fixated visual feature |

| $(\tilde{x}_t,\tilde{y}_t)$ | Expected fixation coordinate |

| $\mathbf p_t$ | Fixation positional embedding |

| $\mu_t,\sigma_t$ | Duration-distribution statistics |

| $\mathbf r_t$ | Fixation duration representation |

| $\mathbf q$ | RoBERTa task/query representation |

| $\mathbf m_0$ | Task encoder output used by the base scanpath model |

| $\tilde{\mathbf h}_t$ | Projected decoder representation |

| $\tilde{\mathbf g}_t$ | Projected fixated visual representation |

| $\mathbf z_t$ | Fixation Reasoning Token |

| $d_z$ | Fixation reasoning dimension |

| $d_L$ | LLM hidden dimension |

  

## WHAT

  

| Symbol | Meaning |

|---|---|

| $\mathbf c_t^{LM}$ | Fixation token projected into LLM space |

| $\mathbf c_q^{\mathrm{WHAT}}$ | WHAT-specific query representation in LLM space |

| $e_{\langle\mathrm{WHAT}\rangle}$ | WHAT special-token embedding |

| $X_t^{\mathrm{WHAT}}$ | WHAT LLM prefix |

| $W_t^*$ | Gold WHAT explanation |

| $\hat W_t$ | Predicted WHAT explanation |

| $L_t$ | Number of gold WHAT tokens for fixation $t$ |

| $N_W$ | Total number of WHAT tokens |

| $\mathbf e_t^W$ | Gold WHAT semantic embedding |

| $\mathbf a_t^W$ | Fixation-side semantic projection |

| $\mathcal L_{\mathrm{what\text{-}txt}}$ | WHAT text-generation loss |

| $\mathcal L_{\mathrm{what\text{-}align}}$ | WHAT semantic-alignment loss |

| $\mathcal L_{\mathrm{WHAT}}$ | Total WHAT loss |

  

## WHY-R0

  

| Symbol | Meaning |

|---|---|

| $K^*$ | Number of gold WHY episodes |

| $K_{\max}$ | Maximum number of router slots |

| $\mathcal R_k^*$ | Gold set of fixation indices in episode $k$ |

| $M^*$ | Gold fixation-to-episode membership matrix |

| $\tilde M^*$ | Gold membership matrix padded to $K_{\max}$ |

| $\boldsymbol\tau_t$ | Temporal positional embedding |

| $\mathbf c_q^R$ | Router-specific task/query representation |

| $\mathbf u_t^{R0}$ | R0 router input for fixation $t$ |

| $\boldsymbol\ell_t$ | Router episode logits |

| $A_{tk}$ | Soft probability that fixation $t$ belongs to slot $k$ |

| $\mathbf A$ | Soft routing matrix |

| $n_k$ | Soft mass of episode slot $k$ |

| $\mathbf r_k^{R0}$ | Episode Reasoning Token |

| $\mathbf c_k^{LM,Y}$ | Episode token projected into LLM space |

| $\mathbf c_q^{\mathrm{WHY}}$ | WHY-specific query representation in LLM space |

| $e_{\langle\mathrm{WHY}\rangle}$ | WHY special-token embedding |

| $X_k^{\mathrm{WHY}}$ | WHY LLM prefix |

| $Y_k^*$ | Gold WHY explanation |

| $\hat Y_k$ | Predicted WHY explanation |

| $L_k$ | Number of gold WHY tokens in episode $k$ |

| $N_Y$ | Total number of WHY tokens |

| $\mathbf e_k^Y$ | Gold WHY semantic embedding |

| $\mathbf a_k^Y$ | Episode-side semantic projection |

| $\mathcal L_{\mathrm{route}}$ | Episode-routing loss |

| $\mathcal L_{\mathrm{why\text{-}txt}}$ | WHY text-generation loss |

| $\mathcal L_{\mathrm{why\text{-}align}}$ | WHY semantic-alignment loss |

| $\mathcal L_{\mathrm{WHY\text{-}R0}}$ | Total WHY-R0 loss |

  

## HOW-R0

  

| Symbol | Meaning |

|---|---|

| $R^{R0}$ | Active WHY episode-token sequence |

| $\bar{\mathbf r}^{R0}$ | Mean-pooled episode representation |

| $\mathbf c_q^H$ | HOW aggregation-specific task representation |

| $\mathbf g^{R0}$ | Global / Trajectory Reasoning Token |

| $d_H$ | HOW latent dimension |

| $\mathbf c_H^{LM}$ | Trajectory token projected into LLM space |

| $\mathbf c_q^{\mathrm{HOW}}$ | HOW-specific query representation in LLM space |

| $e_{\langle\mathrm{HOW}\rangle}$ | HOW special-token embedding |

| $X^{\mathrm{HOW}}$ | HOW LLM prefix |

| $\mathcal H^*$ | Gold HOW explanation |

| $\hat{\mathcal H}$ | Predicted HOW explanation |

| $L_H$ | Number of gold HOW tokens |

| $\mathbf e^H$ | Gold HOW semantic embedding |

| $\mathbf a^H$ | Trajectory-side semantic projection |

| $\mathcal L_{\mathrm{how\text{-}txt}}$ | HOW text-generation loss |

| $\mathcal L_{\mathrm{how\text{-}align}}$ | HOW semantic-alignment loss |

| $\mathcal L_{\mathrm{HOW\text{-}R0}}$ | Total HOW-R0 loss |