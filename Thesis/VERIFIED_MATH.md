# Verified mathematics

Every formula below was read off the implementation, not reconstructed from memory. Each
entry gives the source file and line, the LaTeX, and what the choice buys — so the thesis
can state the mathematics and defend it in the same breath.

Three formulas in the draft were wrong. They are marked **CORRECTED** with the specific
error, because the difference matters and an examiner reading the code would find it.

---

## 1. Detector — recursive least squares

`*/*/anomaly_detection*.py :: _rls_update` — one implementation, AST-verified identical
across all 27 configurations.

For a target with design vector $\mathbf{x}_t$ built from its causal parents, and forgetting
factor $\lambda$:

```latex
\begin{align}
  \mathbf{k}_t &= \frac{\mathbf{P}_{t-1}\mathbf{x}_t}
                       {\lambda + \mathbf{x}_t^{\top}\mathbf{P}_{t-1}\mathbf{x}_t}, \\[0.4em]
  e_t          &= y_t - \mathbf{x}_t^{\top}\mathbf{w}_{t-1}, \\[0.4em]
  \mathbf{w}_t &= \mathbf{w}_{t-1} + \mathbf{k}_t\,e_t, \\[0.4em]
  \mathbf{P}_t &= \frac{1}{\lambda}\left(\mathbf{P}_{t-1}
                  - \mathbf{k}_t\mathbf{x}_t^{\top}\mathbf{P}_{t-1}\right).
\end{align}
```

$\lambda = 0.995$ on robotics, $\lambda = 1.0$ on healthcare and industrial — a per-domain
setting, which is a permitted axis. Effective memory is $\approx 1/(1-\lambda) = 200$
samples at $0.995$; at $\lambda = 1$ the filter never forgets and converges to ordinary
least squares.

**Both claims are measured, not asserted** (`tools/algorithm_validation.csv`):

| Check | Value |
|---|---:|
| $\max\lvert \mathbf{w}_{\text{RLS}} - \mathbf{w}_{\text{OLS}}\rvert$ at $\lambda=1$ | $7.478\times10^{-10}$ |
| Estimate of a coefficient stepping to $2.5$, $\lambda = 0.995$ | 2.5004 |
| Same coefficient, $\lambda = 1$ | 1.2522 |

---

## 2. Anomaly score — coefficient drift

The score is the drift of the online coefficients from the fault-free baseline, and the
design vector **includes the intercept** in all 27 configurations:

```latex
\mathbf{x}_t = \big[\,1,\; x_{t,p_1},\; \dots,\; x_{t,p_k}\,\big]^{\top},
\qquad
s_j(t) = \big\lVert \mathbf{w}_j(t) - \mathbf{w}_j^{(0)} \big\rVert_2
```

One configuration (`PCMCI_robotic/anomaly_detection_rbf.py`) previously omitted the
intercept from both, which was internally consistent but measured a different quantity from
the other 26. Restoring it moved AUC-ROC $0.8519 \rightarrow 0.8752$.

**Why drift and not residual magnitude** — the distinction the thesis rests on, measured:

| Scenario | Mean drift |
|---|---:|
| Relationship breaks | **4.1707** |
| Values rescaled, relationship intact | 0.0002 |
| Nothing changes | 0.0008 |

A residual-magnitude detector cannot separate rows 1 and 2. This one separates them by four
orders of magnitude.

---

## 3. Decision threshold

`reporting_helper.py`, all 27. $\mathcal{S}_{\text{ff}}$ is the score distribution on
**held-out fault-free** data; no anomaly label is consulted.

```latex
\tau = \operatorname{percentile}_{95}\!\left(\mathcal{S}_{\text{ff}}\right)
```

This fixes the false-positive rate at 5% by construction, which is what makes F1 comparable
across configurations — and also why a better-ranked score can score *worse* on F1 while
scoring better on AUC-ROC.

---

## 4. Attribution — **CORRECTED**

### 4a. Healthcare and industrial: baseline-normalised RMS

`GES_healthcare/reporting_helper.py:73-78` and the other 17.

```latex
A_i \;=\; \frac{\displaystyle\sqrt{\tfrac{1}{T_a}\sum_{t=1}^{T_a}\epsilon_{t,i}^{2}}}
                {\max\!\left(E_i,\;\delta\right)},
\qquad
E_i \;=\; \sqrt{\tfrac{1}{T_b}\sum_{t=1}^{T_b}\epsilon_{t,i}^{2}},
\qquad \delta = 10^{-8}
```

$T_a$ is the attack window, $T_b$ the fault-free baseline window.

> **Two errors in the draft.**
> 1. The numerator was written as an instantaneous residual $\lvert\epsilon_{t_{\text{eval}},i}\rvert$.
>    The code uses the **RMS over the whole attack window**. This matters: the code's own
>    comment explains that RMS is length-independent, so $A_i = 1$ means "same drift as
>    normal" and $A_i > 1$ means elevated. An instantaneous value has no such reading.
> 2. $\delta$ was written as an additive smoother, $E_i + \delta$. The code applies it as a
>    **floor**, $\max(E_i, \delta)$. For any variable with non-trivial baseline error the two
>    differ negligibly, but they are not the same operation and the code is the authority.

### 4b. Robotics: raw $L_2$ norm

`PCMCI_robotic/reporting_helper.py:59` and the other 8.

```latex
A_i^{\text{Pepper}} \;=\; \big\lVert \epsilon_{:,i} \big\rVert_2
                    \;=\; \sqrt{\sum_{t=1}^{T_a} \epsilon_{t,i}^{2}}
```

> **Error in the draft.** It was written $\lVert \epsilon_{t_{\text{eval}}} \rVert_2$ — a norm
> over *variables at one timestep*. The code takes the norm over *time for one variable*.
> The index that is free is $i$, not $t$.

The two rules differ in exactly one respect that matters: **4a is length-independent and
baseline-relative, 4b is neither.** 4b is retained on robotics character-for-character
because reproducing the XAI course laboratory is a stated objective, and a different ranking
rule would not reproduce it.

---

## 5. Neural baselines

### 5a. Training objective

`neural_baselines.py :: train` — `nn.MSELoss()` over the whole sequence:

```latex
\mathcal{L} = \frac{1}{L\,D}\sum_{t=1}^{L}\sum_{i=1}^{D}\big(\hat{x}_{t,i}-x_{t,i}\big)^{2}
```

### 5b. Scoring — **not the same as the loss**

`neural_baselines.py :: recon_error` uses only the **final timestep**:

```latex
s(t) = \frac{1}{D}\sum_{i=1}^{D}\big(\hat{x}_{L,i}-x_{L,i}\big)^{2}
```

The draft gave only the training loss. Trained over $L$ steps, scored at step $L$ alone —
worth one sentence, since a reader who assumes scoring matches training will misread the
detector.

### 5c. Early stopping

Verified exactly as drafted: 15% validation split, improvement threshold
`v < best - 1e-6`, patience 15 epochs, `MAX_EPOCHS = 300`, and best weights restored via
`m.load_state_dict(state)`.

### 5d. Search grid

$h \in \{32,64\}$, $l \in \{1,2,3\}$, $\alpha \in \{0.001,0.01\}$, Adam — 12 configurations
per architecture, selected on fault-free validation loss. $L = 10$ fixed after a prior sweep
over $\{5,10,20\}$.

> **Scope correction.** Until now the search ran on **robotics only**. Healthcare and
> industrial used a single fixed configuration $(h{=}64,\;l{=}1,\;\alpha{=}0.001)$, so the
> cross-domain comparison set a tuned causal pipeline against an untuned neural one — the
> same criticism this thesis makes of the literature. `neural_baselines_all.py --search` now
> runs the identical grid on all three domains, writing `neural_all_domains_tuned.csv`.

---

## 6. Bootstrap

`tools/bootstrap_intervals.py` — moving-block, block length $L$ from the score series'
autocorrelation, per-path CRC32 seeding for reproducibility.

An i.i.d. resample destroys serial dependence and produces intervals that are far too
narrow. Compare, for PCMCI robotics RBF:

| Method | 95% interval |
|---|---|
| Analytical Hanley–McNeil | [0.8258, 0.9246] |
| **Moving-block bootstrap** | **[0.7559, 0.9587]** |

**Use the bootstrap.** The analytical interval is 45% narrower and would let you claim
separability the data does not support.

> For comparing *two* AUCs on the same samples, neither is the right test — use **DeLong**
> (`delong1988comparing` in `refs.bib`). Hanley–McNeil gives an interval for one AUC.

---

## 7. Trivial control

`Baselines/trivial_baseline.py` — no learned parameters, same threshold rule:

```latex
s(t) = \frac{1}{n}\sum_{j=1}^{n}
       \left\lvert \frac{x_j(t)-\mu_j}{\sigma_j} \right\rvert
```

$\mu_j, \sigma_j$ from fault-free training data only. This is the floor any method must
clear to have earned its complexity — and on industrial it scores **0.9286** against the
best causal configuration's **0.9151**.

---

## Quick reference — what changed

| # | Draft | Verified |
|---|---|---|
| 4a | $\lvert\epsilon_{t_{\text{eval}},i}\rvert$ | RMS over the attack window |
| 4a | $E_i + \delta$ | $\max(E_i,\delta)$ |
| 4b | $\lVert\epsilon_{t_{\text{eval}}}\rVert_2$ | $\lVert\epsilon_{:,i}\rVert_2$ |
| 5b | (absent) | scoring uses the final timestep only |
| 5d | "each architecture searched" | robotics only, until now |
| 6 | Hanley–McNeil | bootstrap; DeLong for paired comparisons |
