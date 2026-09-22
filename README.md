# Causal Anomaly Detection from Multivariate Time Series

Master's thesis codebase. Three causal-discovery algorithms compared across three domains
under a single uniform detection pipeline, plus a neural baseline.

Every number below is produced by `tools/collect_results.py` from the `metrics_summary.csv`
files each run writes. None are transcribed by hand.

---

## 1. What this project does

A causal graph is learned from fault-free data. Each monitored variable is regressed on its
causal parents, and those coefficients are tracked online with Recursive Least Squares. An
anomaly is a **drift in the regression coefficients** away from the fault-free baseline, not a
raw prediction error. Because each score attaches to a specific target variable, the detector
explains itself: the variables whose coefficients drift most are the attributed root causes.

| Stage | Implementation |
| :--- | :--- |
| Causal discovery | PCMCI (`tigramite`) · GES (`causal-learn`) · CAM-UV (`lingam`) |
| Offline baseline | Ridge over each target's causal parents |
| Online monitoring | Recursive Least Squares coefficient tracking |
| Anomaly score | Coefficient drift vs the fault-free baseline |
| Explainability | Per-variable residual ranking, attributed per attack class |

### The three axes

Following the reviewer's constraint — *"use this same algorithm, just changing the dataset,
the causal discovery algorithm or the regression type, and nothing else"* — exactly three
things vary across the 27 configurations:

| Axis | Values |
| :--- | :--- |
| Causal discovery | PCMCI · GES · CAM-UV |
| Domain | Robotics (PEPPER) · Healthcare (PTB-XL ECG) · Industrial (Tennessee Eastman) |
| Regression | Linear/Ridge · Polynomial degree 3 · RBF + Ridge |

Everything else is identical. §7 documents how that is enforced and verified.

### Algorithms

- **PCMCI** — Peter–Clark Momentary Conditional Independence. Time-delayed; recovers a lagged
  causal tensor with `tau_max` derived from the data's frequency content.
- **GES** — Greedy Equivalence Search. Contemporaneous; recovers a CPDAG, from which only
  unambiguous directed edges (`adj[i,j] == -1 and adj[j,i] == 1`) are retained.
- **CAM-UV** — Causal Additive Models with Unobserved variables (Maeda & Shimizu, UAI 2021).
  Returns observed parents `P` **and** pairs sharing an unobserved confounder `U`.

### Metrics

Scores are thresholded at the **95th percentile of the fault-free score distribution**,
computed from normal data only and never seeing an attack label.

**AUC-ROC is the primary metric** — threshold-free and invariant to the operating point. F1 is
reported at the stated 95th-percentile operating point and should be compared only against
other rows in these tables.

---

## 2. Results

### 2.1 Robotics — PEPPER

| Model | Variant | F1 | AUC-ROC | AUC-PR |
| :--- | :--- | ---: | ---: | ---: |
| PCMCI | Linear / Ridge | 0.6395 | 0.7825 | 0.8810 |
| PCMCI | Polynomial degree 3 | 0.7812 | 0.8053 | 0.9090 |
| PCMCI | **RBF + Ridge** | 0.7938 | **0.8752** | **0.9419** |
| GES | Linear / Ridge | 0.5445 | 0.5906 | 0.7647 |
| GES | Polynomial degree 3 | 0.6172 | 0.6999 | 0.8325 |
| GES | RBF + Ridge | 0.6583 | 0.6516 | 0.8195 |
| CAM-UV | Linear / Ridge | 0.4747 | 0.6497 | 0.7989 |
| CAM-UV | Polynomial degree 3 | 0.5531 | 0.6888 | 0.8205 |
| CAM-UV | RBF + Ridge | 0.7941 | 0.7818 | 0.8909 |
| *Neural baseline* | *TCN autoencoder (tuned)* | *0.8053* | *0.8856* | *0.9164* |
| *Neural baseline* | *LSTM autoencoder (tuned)* | *0.7972* | *0.8804* | *0.9002* |
| *Neural baseline* | *Transformer autoencoder (tuned)* | *0.7140* | *0.8139* | *0.8731* |

### 2.2 Healthcare — PTB-XL ECG

| Model | Variant | F1 | AUC-ROC | AUC-PR |
| :--- | :--- | ---: | ---: | ---: |
| PCMCI | Linear / Ridge | 0.3015 | 0.6181 | 0.7360 |
| PCMCI | Polynomial degree 3 | 0.1147 | 0.5075 | 0.6198 |
| PCMCI | RBF + Ridge | 0.5206 | 0.7147 | 0.8218 |
| GES | **Linear / Ridge** | **0.5818** | **0.8114** | 0.8395 |
| GES | Polynomial degree 3 | 0.4050 | 0.7867 | 0.8027 |
| GES | RBF + Ridge | 0.5156 | 0.7620 | 0.8073 |
| CAM-UV | Linear / Ridge | 0.5194 | 0.7861 | 0.8143 |
| CAM-UV | Polynomial degree 3 | 0.4035 | 0.7691 | 0.7906 |
| CAM-UV | RBF + Ridge | 0.4568 | 0.7692 | 0.7994 |

**PCMCI is the weakest algorithm on ECG, and this is a corrected result.** Earlier drafts
reported PCMCI healthcare near AUC 0.80. That depended on `RIDGE_ALPHA = 1000.0` — one hundred
times what GES and CAM-UV used on identical data. With regularisation made uniform, PCMCI
falls to 0.6181 / 0.5075 / 0.7147. `RidgeCV` was tested as a principled label-blind
alternative and performs worse still (AUC 0.4856), because cross-validation optimises
reconstruction error rather than drift-based detection. No label-blind rule recovers the old
figure, which means it was tuned against the task.

### 2.3 Industrial — Tennessee Eastman

| Model | Variant | F1 | AUC-ROC | AUC-PR |
| :--- | :--- | ---: | ---: | ---: |
| PCMCI | Linear / Ridge | 0.8798 | 0.8860 | 0.9940 |
| PCMCI | **Polynomial degree 3** | 0.8823 | **0.9151** | 0.9956 |
| PCMCI | RBF + Ridge | 0.8476 | 0.8495 | 0.9920 |
| GES | Linear / Ridge | 0.8582 | 0.8407 | 0.9915 |
| GES | Polynomial degree 3 | 0.8556 | 0.8849 | 0.9939 |
| GES | **RBF + Ridge** | **0.8927** | 0.8994 | 0.9948 |
| CAM-UV | Linear / Ridge | 0.8643 | 0.8428 | 0.9917 |
| CAM-UV | Polynomial degree 3 | 0.8126 | 0.8212 | 0.9901 |
| CAM-UV | RBF + Ridge | 0.8839 | 0.8922 | 0.9944 |

All three algorithms perform comparably on TEP (AUC 0.82–0.92). This is the domain where
CAM-UV is competitive.

### 2.4 What wins where

| Domain | Best by AUC-ROC | Value |
| :--- | :--- | ---: |
| Robotics | PCMCI — RBF | 0.8752 |
| Healthcare | GES — Linear | 0.8114 |
| Industrial | PCMCI — Polynomial | 0.9151 |

**No algorithm sweeps.** PCMCI leads robotics and industrial, GES leads healthcare, CAM-UV
is within noise of the leaders on industrial. Earlier drafts showed PCMCI ahead everywhere;
that was an artefact of per-configuration tuning, now removed.

**These are the best *causal* results, which is not the same as the best results.** A
parameter-free control that only measures distance from the fault-free operating point scores
**0.9286 on industrial — above every causal configuration in the table above**, and the neural
architectures reach 0.9995 there. On healthcare the same control manages only 0.6136 against
the causal 0.8114. The comparison against non-causal detectors is §5, and it should be read
before any of these three rows is described as a win.

### 2.5 Statistical significance

Single-run results invite the question of which differences are real. Two methods are applied,
and they disagree in an instructive way.

`tools/significance.py` computes the analytical Hanley–McNeil interval, which is closed-form
but assumes independent samples. `tools/bootstrap_intervals.py` resamples the **real
per-sample scores** (persisted as `scores_<variant>.npz`) with a **moving-block** bootstrap,
which preserves the local temporal dependence that overlapping windows create.

**The block bootstrap is the one to trust, and it is far less flattering.**

| Domain | Best configuration | 95% CI (block bootstrap) | Runner-up | Separable? |
| :--- | :--- | :--- | :--- | :---: |
| Robotics | PCMCI — RBF (0.8752) | [0.7559, 0.9587] | PCMCI — Poly [0.6516, 0.9233] | **no** |
| Healthcare | GES — Linear (0.8114) | [0.7628, 0.8540] | GES — Poly [0.7314, 0.8379] | **no** |
| Industrial | PCMCI — Poly (0.9151) | [0.8880, 0.9394] | GES — RBF [0.8206, 0.9508] | **no** |

**No domain winner is statistically separable from its runner-up.** Under the analytical
interval, healthcare GES-linear appeared significantly ahead of all eight other
configurations (z = 2.23 to 26.67); under the block bootstrap it overlaps its own runner-up.
The difference is the independence assumption: adjacent evaluation windows overlap and their
scores are autocorrelated, so the effective sample size is well below n and the analytical
interval is too narrow.

Interval widths track sample size directly, and robotics is the weakest case:

| Domain | n | typical CI width |
| :--- | ---: | ---: |
| Robotics | 181–1,378 | 0.16 – 0.28 |
| Healthcare | 3,115–4,806 | 0.09 – 0.13 |
| Industrial (PCMCI) | 16,716 | 0.05 – 0.07 |

One further consequence: `PCMCI — Polynomial` on healthcare has a bootstrap interval of
**[0.4429, 0.5663]**, which **contains 0.5**. That configuration is formally indistinguishable
from chance, not merely weak.

**What can honestly be claimed.** The broad ordering is stable — PCMCI and GES lead their
respective domains, CAM-UV trails on robotics and healthcare, and everything performs
comparably on industrial. But **no specific ranking between adjacent configurations is
statistically established** on these sample sizes, and the results chapter should not present
one as though it were. Establishing separability would require either substantially more
evaluation data or repeated runs over resampled training splits, which is noted in the
limitations.

---

## 3. Explainability: the XAI course lab check

The reviewer's requirement: reproduce the XAI course lab result on robotics — **LED attacks
must attribute to the Tablet sensors, Joint attacks to joint temperatures** — and extend the
qualitative analysis to the other datasets.

### 3.1 Robotics

Top-ranked attributed variable per attack class:

| Attack | PCMCI (lin/poly/rbf) | GES (lin/poly/rbf) | CAM-UV (lin/poly/rbf) |
| :--- | :--- | :--- | :--- |
| **LedsControl** | TabletTouchX ×3 | TabletTouchX ×3 | TabletTouchX ×3 |
| **JointControl** | KneePicthTemp / WheelBStiff / KneePicthTemp | KneePicthTemp ×3 | KneePicthTemp ×3 |
| WheelsControl | WheelFRTemp / HeadPicthTemp / HeadPicthTemp | LElbowYawTemp / LShoulderRollTemp / LaserShovel03X | WheelFRTemp / WheelFRStiff / KneePicthTemp |

- **LED → Tablet: `TabletTouchX` is rank-1 in 9 of 9 configurations** — every algorithm, every
  regression variant, without exception.
- **Joint → joint temperature: `KneePicthTemp` is rank-1 in 8 of 9** — all three GES variants,
  all three CAM-UV variants, and PCMCI under linear and RBF. The single exception is PCMCI
  under polynomial, which returns `WheelBStiff`, a joint-stiffness channel, so it remains
  joint-related.

Both lab checks pass, and they pass under **name-agnostic** feature selection (§7.2).
`tools/check_lab_requirements.py` recomputes both counts from the generated attribution CSVs,
so a rerun that breaks either cannot pass unnoticed.

**The attribution formula itself differs by domain, deliberately.** Robotics ranks on an
unnormalised reduction of the per-target residual. The PCMCI folders use
`np.linalg.norm(norm_agg_attack[:, j])`, which is the XAI course lab's own formula, character
for character; that is required here, because the reviewer asked that PCMCI-linear on Pepper
reproduce the lab and a different ranking rule would not. The GES and CAM-UV robotics folders
take the root mean square of the same vector instead. RMS is the L2 norm over `sqrt(T)` for a
window of `T` steps, and `T` does not vary between the variables being ranked, so the two
orderings — and every top-3 read off them — are identical.

Healthcare and industrial instead divide that norm by each variable's own fault-free baseline
residual. Their variables span very different amplitudes — ECG limb versus precordial leads,
TEP flows versus temperatures — so an unnormalised ranking would surface whichever channel is
naturally largest rather than whichever was actually perturbed.

All nine robotics configurations use the first rule and all eighteen others use the second, so
the cross-algorithm attribution tables above compare like with like *within* each domain.
`tools/audit_pipeline.py` fails if that ever stops being true. Attribution rankings should not
be compared *across* domains, and no table in this document does so.

### 3.2 Industrial — coherence against the documented TEP fault table

Top-ranked variable, linear variant, against the cause named in the TEP specification:

| Fault | Documented cause | PCMCI | GES | CAM-UV |
| :---: | :--- | :--- | :--- | :--- |
| 4 | Reactor cooling water inlet temp, step | **XMV_10** ✓ | **XMV_10** ✓ | **XMV_10** ✓ |
| 5 | Condenser cooling water inlet temp, step | **XMV_11** ✓ | XMV_5 | XMEAS_20 |
| 7 | C header pressure loss | **XMV_4** ✓ | XMV_5 | **XMV_4** ✓ |
| 11 | Reactor cooling water, random variation | XMEAS_20 | **XMV_10** ✓ | **XMV_10** ✓ |
| 14 | Reactor cooling water valve sticking | XMEAS_9 | XMEAS_21 | XMEAS_21 |

Counting top-ranked attributions only: **PCMCI and CAM-UV each localise three faults to the
exact valve named in the specification, GES two.**

**Detection strength and root-cause specificity are separate properties, and the feature-scaling
correction (§7.1a) separated them sharply.** That fix cost CAM-UV industrial 0.05 AUC — and
simultaneously took its exact localisations from none to three. The configuration that detects
best here is not the one that explains best: PCMCI polynomial leads detection at AUC 0.9151
while CAM-UV linear, at 0.8428, matches PCMCI's localisation count. A results chapter that
ranked these methods on AUC alone would have reported the opposite conclusion about which one
identifies causes.

### 3.3 Healthcare — not verifiable against truth

PTB-XL publishes **no per-lead ground truth** for which lead each perturbation targets.
Healthcare attributions are reported as attack-discriminative but cannot be checked against a
documented cause. This is a dataset limitation, not a pipeline one.

---

## 4. Reproducing the XAI course lab

The reviewer asked that PCMCI with linear regression on Pepper reproduce the course lab, and
noted the explainability results looked different. It does reproduce it, and the discrepancy
has a single numerical cause with a one-line fix.

### The target, and who reaches it

| | WheelsControl | JointControl | LedsControl |
| :--- | :--- | :--- | :--- |
| Lab output plot (as it once ran) | `WheelFRTemp` | `KneePicthTemp` | `TabletTouchX`, `TabletTouchY` |
| **This pipeline** | **`WheelFRTemp`** | **`KneePicthTemp`** | **`TabletTouchX`, `TabletTouchY`** |
| **Lab script + Ridge** | **`WheelFRTemp`** | **`KneePicthTemp`** | **`TabletTouchX`, `TabletTouchY`** |
| Lab script as supplied | `LaserF05X` | `LaserF15X` | `LaserF15X` |

The configuration was never in question: same `subsample = 74`, same 244 variables, same
`tau_max = 2`, same `var_indices[:3]`, and the learned graphs are identical at **178,364
edges with the same support**.

### The cause: an ill-conditioned regression, not a modelling choice

Several laser targets are regressed on the **X and Y components of the same beam**, which are
near-collinear. The design matrix is then numerically singular at double precision:

| target | cond(X) | parents |
| :--- | ---: | :--- |
| `LaserF15X` | **3.7 × 10⁹** | LElbowYaw, `LaserR07X`, `LaserR07Y` |
| `LaserF05X` | **4.4 × 10⁹** | LShoulderPitch, `LaserR09X`, `LaserR09Y` |
| `TabletTouchX` | 3.2 × 10² | LElbowRoll, RElbowRoll, LaserR03X |
| `KneePicthTemp` | 3.8 × 10² | LShoulderRoll, LElbowRoll, LWristYaw |
| `WheelFRTemp` | 3.6 × 10² | LElbowRoll, LHand, RElbowRoll |

Seven orders of magnitude separate the laser targets from everything else.
`np.linalg.lstsq(..., rcond=None)` returns a minimum-norm solution for them, and the resulting
coefficient drift is around **10¹¹** against ~10⁴ for `TabletTouchX` and ~10² for
`WheelFRTemp`. Since attribution ranks by the L2 norm of that drift, the blow-up buries the
real signal.

It is not a scale effect: correlation between variable magnitude and reported drift is only
**r = 0.277**, and `WheelFRTemp` has the *largest* standard deviation of the five with the
*smallest* drift. Nor is it a warm-up effect: skipping the first 1 to 30 timesteps changes
nothing, because the conditioning problem is present at every step.

### The fix, and what it also corrects

Replacing the two `lstsq` calls with `Ridge(alpha=10.0)` — the only change in
`PCMCI/PCMCI_robotic/_lab_reference_fixed.py` — restores the published attributions exactly.

It also removes an artefact in the reported metrics. Unregularised, the 10¹¹ drift inflates
the detection threshold so far that nothing in the fault-free tail can exceed it:

| lab script | Precision | Recall | F1 | fp |
| :--- | ---: | ---: | ---: | ---: |
| as supplied (`lstsq`) | 1.0000 | 0.9204 | 0.9585 | **0** |
| with `Ridge` | 0.8667 | 0.9204 | 0.8927 | 16 |

**The perfect precision was the same numerical artefact**, not detection quality.

This pipeline is immune by construction: Ridge for the offline fit, and RLS initialised at
those coefficients with `P0 = I/alpha`, which is implicitly regularised. That is why it
reproduces the lab plot where the unregularised script cannot.

### Two further defects in the supplied script

**The normalisation crashes, and is otherwise inert.** `nonconst_data[:, j] /= (max - min) + min`
simplifies to `x /= max(x)`; `AccelerometerX` has `max == 0`, producing NaN that tigramite
rejects. Running with proper min-max normalisation gives **identical graphs** — same 178,364
edges, same support, downstream metrics bit-identical — because `ParCorr` is invariant to
affine rescaling. The line only ever causes the crash.

**Detection is scored asymmetrically.** `detect_anomalies` scans attacks from index 0 but
fault-free data only from `normal_data_len`. Making the scan symmetric drops the reported F1
from 0.9585 to 0.8812 independently of the conditioning issue.

Both scripts are in the repository — `_lab_reference.py` as supplied, `_lab_reference_fixed.py`
with the Ridge change — so the whole comparison reproduces end to end.

---

## 5. Neural baselines — LSTM, TCN and Transformer

All three architectures the reviewer named are implemented in
[`Baselines/neural_baselines.py`](Baselines/neural_baselines.py) as reconstruction
autoencoders sharing one protocol, so architecture is the only thing that varies: same
variables, same 70/30 fault-free split, same 95th-percentile threshold, training to
convergence with early stopping, three seeds each, and the same 12-configuration search
selected on **fault-free validation loss only** — never on an attack label.

| Architecture | F1 | AUC-ROC | AUC-PR | selected config |
| :--- | ---: | ---: | ---: | :--- |
| **TCN autoencoder** | **0.8053** ± 0.0139 | **0.8856** ± 0.0051 | **0.9164** | `hidden=64, layers=1, lr=0.01` |
| LSTM autoencoder | 0.7972 ± 0.0039 | 0.8804 ± 0.0032 | 0.9002 | `hidden=64, layers=1, lr=0.001` |
| Transformer autoencoder | 0.7140 ± 0.0195 | 0.8139 ± 0.0158 | 0.8731 | `hidden=64, layers=1, lr=0.01` |
| *PCMCI — RBF (best causal)* | *0.7938* | *0.8752* | *0.9419* | |

Tuning mattered: the untuned LSTM default (`hidden=16, layers=1, lr=0.01, seq=5`) scores
F1 0.7072, AUC-ROC 0.7600, AUC-PR 0.8566 — that row is retained in
`Baselines/outputs/metrics_summary.csv` as the pre-tuning reference. Window length dominated
the search: every `seq=10` configuration beat every `seq=5` one, so the original default was
handicapping the baseline structurally, not merely by training budget.

**The best neural architecture is not outperformed by any causal configuration, and the
comparison cannot be significance-tested:**

| Robotics, best of each family | AUC-ROC |
| :--- | ---: |
| TCN autoencoder | 0.8856 |
| PCMCI — RBF | 0.8752 |

`tools/delong_test.py` produces no causal-versus-neural row, and the reason is structural
rather than an omission: the autoencoders score overlapping windows rather than a
subsampled series, so no causal score file pairs with theirs. An earlier version of this
file reported z values and the verdict "TCN significantly better" for these pairs. That
was wrong — no paired test exists for them. The ordering above rests on point estimates
alone, and the neural score falls inside the causal configuration's block-bootstrap
interval of [0.7559, 0.9587].

So on Pepper a tuned TCN matches the two best causal configurations and significantly
outperforms the remaining seven. **The claim that causal methods outperform standard neural
detectors on this dataset is not supported.**

### The explainability comparison is more complicated than expected

| Attack | Causal models | LSTM | TCN | Transformer |
| :--- | :--- | :--- | :--- | :--- |
| **LedsControl** | `TabletTouchX` | **`LedFaceBL4`** | `AccelerometerX` | **`LedFaceBL4`** |
| **JointControl** | **`KneePicthTemp`** | `RShoulderPitch` | `RShoulderPitch` | `RShoulderPitch` |
| WheelsControl | `WheelFRTemp` | `WheelBSpeed` | `WheelBSpeed` | `WheelBSpeed` |

Two observations that cut in opposite directions, and both should be reported.

**Against the causal models:** on LedsControl the tuned LSTM and Transformer attribute to
`LedFaceBL4` and `LedFaceBR5` — *actual LED channels*, the subsystem the attack directly
targets. The causal models return `TabletTouchX`, a correlated subsystem. The XAI lab
expects the Tablet answer, on the reasoning that tablet interaction drives the LEDs, but the
neural attribution is arguably the more direct identification of what was attacked.

**For the causal models:** on JointControl every neural architecture returns
`RShoulderPitch`, a joint *position*, while the causal models return `KneePicthTemp`, a joint
*temperature*. The documented fault is thermal, so here the causal attribution identifies the
mechanism and the neural one identifies a kinematic correlate.

**The defensible claim is therefore narrow.** Post-hoc attribution on a tuned neural detector
is not inferior at naming relevant variables — on one attack class it is arguably better.
What the causal models provide that attribution cannot is a *directed graph with parent–child
structure*: an answer to "through what path", not only "which variable". That, rather than
detection accuracy or top-ranked relevance, is where the causal approach is distinct.

### All three domains, and a control that reframes the comparison

The reviewer asked for the neural comparison to extend beyond Pepper — *"at least transformer,
and possibly on all datasets"*.
[`Baselines/neural_baselines_all.py`](Baselines/neural_baselines_all.py) runs all three
architectures on all three domains, reading each domain's variables and resampling rate from
that domain's own PCMCI cache. Configuration is held fixed at `hidden=64, layers=1, lr=0.001,
seq=10` so that **domain** is the only axis; the tuned per-architecture numbers above remain
the robotics-specific comparison, which is why TCN and Transformer score slightly lower here.

Alongside them, [`Baselines/trivial_baseline.py`](Baselines/trivial_baseline.py) scores a
detector with **no parameters and no training** — mean `|z|` from the fault-free mean and
standard deviation, same 95th-percentile threshold. It cannot represent structure, dynamics or
interaction, only distance from the fault-free operating point. It is the floor any real
method has to clear.

| Domain | Trivial control | LSTM | TCN | Transformer | Best causal |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Robotics (Pepper) | 0.8139 | 0.8804 | 0.8809 | 0.7815 | **0.8752** |
| Healthcare (ECG) | 0.6136 | 0.6815 | 0.6624 | 0.6102 | **0.8114** |
| Industrial (TEP) | 0.9286 | 0.9669 | 0.9995 | **0.9996** | 0.9151 |

AUC-ROC; three seeds each; n = 155 / 4,793 / 16,590.

**The three domains give three different answers, and that is the substantive result.**

**Industrial — the causal approach loses to a detector with no parameters.** The trivial
control reaches **0.9286, above all nine causal configurations** (best 0.9151), and the TCN and
Transformer reach 0.9995. TEP faults are large sustained level shifts in raw sensor space, so
distance from the operating point is very nearly sufficient — while a coefficient-drift
detector is deliberately insensitive to exactly that, tracking whether *relationships* change
rather than whether *values* move. Two cautions on the numbers themselves: prevalence is
**95.2%** (twenty fault files against one fault-free run), so AUC-PR near 1.000 sits barely
above the 0.952 floor and carries little information, and F1 is inflated for the same reason.
AUC-ROC is unaffected and is the figure to read.

**Healthcare — this is where causal structure earns its keep.** The trivial control gets 0.6136
and the best neural architecture 0.6815, while the best causal configuration reaches 0.8114.
ECG anomalies perturb inter-lead *relationships* while each lead individually stays in range,
which is the regime a drift-based detector is built for and a distance-based one is blind to.

One qualification on that margin. GES runs at `subsample=10` and the neural baselines read
PCMCI's cache at `subsample=7`, so 0.8114 and 0.6815 are not on the same evaluation set
(limitation 5). Comparing only configurations at the *matched* rate, PCMCI-RBF at 0.7147
against LSTM at 0.6815 narrows the gap to **+0.033**. Both comparisons favour the causal
model and both clear the trivial control, but the honest headline is the matched-rate one.

**Robotics is intermediate** — the trivial control at 0.8139 is already within reach of both
the causal result (0.8752) and the neural one (0.8809).

**What this supports.** Causal anomaly detection is not the stronger method in general; on
industrial it is beaten by arithmetic. It is stronger where anomalies are **relational rather
than magnitude-based**, and healthcare demonstrates that case cleanly. That conditional claim
is supported by these measurements. An unconditional one is not.

---

## 6. CAM-UV: the confounder output works, and that is the problem

CAM-UV was chosen over PCMCI and GES for one reason: it returns not only observed parents
`P` but also `U`, the pairs sharing an unobserved common cause. Understanding what `U` does on
these datasets took two corrections, both recorded here because the first conclusion was
confidently wrong.

### An earlier draft reported that `U` never works. That was a bug in the test

The original `confounder_ablation.py` planted a hidden cause `u` behind `x1` and `x3`, swept
sample size, alpha and `num_explanatory_vals`, and found an empty `U` in all 24 combinations.
The conclusion drawn — that CAM-UV cannot recover a confounder it is handed — does not follow,
because that system had two independent defects:

- **Every relation was linear and every noise term Gaussian.** That is the classically
  non-identifiable regime for this family of methods; no amount of data resolves direction in it.
- **`x3 = 0.6·u + 0.1·e` made `x3` a near-perfect observation of `u`** — about 97% of its
  variance. With `u` effectively observed, a direct edge is the defensible answer and a latent
  confounder is not. The test could not have passed.

On a system CAM-UV is actually specified for — nonlinear additive relations, non-Gaussian
noise, and a latent `u` behind two variables that *both* have strong observed parents, so
neither proxies `u` — it recovers the planted pair:

Sweeping the same grid over both systems — 4 sample sizes × 3 alphas × 2 values of
`num_explanatory_vals`, each through **two independent implementations** — separates them
cleanly:

| System | `lingam` | `causallearn` |
| :--- | :---: | :---: |
| A · proxy / linear-Gaussian (the broken control) | 0/24 | 0/24 |
| B · latent / nonlinear (the specified regime) | **14/24** | **14/24** |

and recovery on system B follows sample size and significance level exactly as a statistical
test should:

| n | α = 0.01 | α = 0.05 | α = 0.10 |
| ---: | :---: | :---: | :---: |
| 200 | — | — | — |
| 500 | — | ✓ | ✓ |
| 1000 | — | ✓ | ✓ |
| 1500 | ✓ | ✓ | ✓ |

The two implementations — the authors' own `lingam.camuv`, which `camuv_engine.py` wraps, and
`causallearn`'s CAMUV — agree on **every one of the 96 runs**, which rules out the wrapper.
Recovery needs roughly 500 samples at α = 0.05, and a stricter α becomes affordable only as n
grows. Strong confounding fails separately: once `u` dominates, the pair is absorbed as a
direct edge during parent-finding and the confounder loop then skips it by construction, since
it only tests pairs with no edge between them.

`confounder_ablation.py` runs both systems, the broken one kept as a negative control.

### On the real datasets `U` is not empty — it is most of the graph

The claim that `U` came back empty on Pepper, ECG and TEP was simply false, and reading the
cached graphs back shows the opposite. `CAM-UV/confounder_structure.py` reports:

| Domain | vars | samples | samples/var | edges | `U` pairs | `P` edges | saturation | CAM-UV linear AUC |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Industrial (TEP) | 52 | 35 | **0.67** | 63 | 14 | 35 | 1.1% | 0.8428 |
| Healthcare (ECG) | 12 | 3,500 | 291.67 | 87 | 39 | 9 | 59.1% | 0.7861 |
| Robotics (Pepper) | 35 | 1,415 | 40.43 | 960 | 480 | **0** | **80.7%** | 0.6497 |

Saturation is the share of all `C(d,2)` variable pairs flagged as confounded.

**On robotics CAM-UV found no directed structure at all.** Zero observed-parent edges; the
adjacency matrix is exactly symmetric, 480 pairs × 2 directions. Every relationship among the
35 sensors was attributed to latent confounding, and with 80.7% of all possible pairs flagged
the result is a near-complete graph rather than a discovery.

**Healthcare saturates too, but there the flagged pairs are physiologically right.** Splitting
its 39 pairs by lead type separates two cases that a saturated graph would blur:

| Lead pair type | possible | flagged | share |
| :--- | ---: | ---: | ---: |
| Independent × derived (`I`/`II` against `III`, `aVR`, `aVL`, `aVF`) | 8 | **1** | 12.5% |
| Precordial × precordial (`V1`–`V6`) | 15 | **14** | 93.3% |

The derived limb leads are exact algebraic functions of `I` and `II` (§10), so they are
*deterministically* related, not latent-confounded — and CAM-UV declines to flag 7 of those 8.
The precordial leads share unobserved cardiac electrical sources that no observed lead
captures, and it flags 14 of 15. That is the distinction the method exists to make, and on
healthcare it makes it correctly. Saturation alone therefore does not imply the output is
meaningless; robotics is the case where nothing indicates it is meaningful.

**Saturation tracks detection quality monotonically across all three domains** — 1.1% → 0.8428,
59.1% → 0.7861, 80.7% → 0.6497. The mechanism is straightforward: the detector regresses each
variable on its parents, and a near-complete symmetric graph gives every target nearly every
other variable as a parent, so there is no sparsity for coefficient drift to be specific
about. Three points is a suggestive mechanism, not an established trend, and it is reported as
such.

**TEP's low saturation is not selectivity — it is absent statistical power, and that is now
measured rather than suspected.** Discovery there ran on **35 samples for 52 variables** — 0.67
per variable — because the 500-row fault-free file is cut to a 70% training split and then
subsampled by 10. CAM-UV flags a pair by *rejecting* independence with HSIC, and 35 samples
have little power to reject anything.

`CAM-UV/CAM_UV_industrial/tep_power_check.py` sweeps the discovery sample count over the same
350-row training window and settles it:

| samples | observed-parent edges | confounded pairs | graph density |
| ---: | ---: | ---: | ---: |
| 35 — *the evaluation protocol* | 35 | 14 | 2.4% |
| 70 | 35 | 28 | 3.4% |
| 175 | 59 | 38 | 5.1% |
| **350** | **115** | **90** | **11.1%** |

Ten times the samples yields **6.4× the confounded pairs and 4.7× the density**, rising
monotonically with no sign of a plateau. Had TEP's structure genuinely been sparse, the graph
would have stayed sparse as data grew. So the 1.1% pair saturation the evaluation protocol
sees is a property of the sample count, not of the Tennessee Eastman process, and it should
not be cited as evidence that CAM-UV is selective on industrial data.

This is a diagnostic: it does not change the evaluation protocol, which stays at
`subsample = 10` so that GES and CAM-UV remain exactly matched within the domain. It does
mean the industrial column of the table above is the one row whose *sparsity* carries no
information. GES industrial sees the identical 35 samples; PCMCI industrial runs at
`subsample = 1` and sees 350, so it is unaffected.

**The corrected reading.** CAM-UV's distinguishing output is not inert — where it fires at all
it is overactive. On Pepper it saturates at 80.7%, the graph collapses into near-completeness,
and detection degrades accordingly; on ECG it saturates but discriminates correctly. TEP is the
one domain where CAM-UV is competitive on detection: its RBF variant reaches 0.8922 and its
interval overlaps the domain best (§2.5). But its graph there is sparse only because the
protocol gives it 35 samples, so TEP supports a claim about CAM-UV's *detection* and no claim
at all about its *structure*.

That is a sharper and more useful result than either "the confounder claim does not hold" or
"CAM-UV performs worse". It also points at a concrete remedy — thresholding or ranking `U`
by confidence rather than integrating every flagged pair as an edge — which is left as future
work, and at a second one for the industrial setting: discovery there should run at the full
350 samples even if evaluation continues at `subsample = 10`, since nothing requires the two
to use the same rate.

---

## 7. Pipeline uniformity — how "nothing else" is enforced

### 7.1 Hyperparameters

Cross-algorithm mismatches: **0**, verified by an audit over every constant.

| Domain | DETECTION_THRESH | CAUSAL_STRENGTH | RLS_LAMBDA | RIDGE_ALPHA (lin/poly/rbf) |
| :--- | ---: | ---: | ---: | :--- |
| Robotics | 1.2 | 1.0 | 0.995 | 10 / 0.01 / 1.0 |
| Healthcare | 2.0 | 1.0 | 1.0 | 10 / 10 / 10 |
| Industrial | 3.0 | 0.0 | 1.0 | 10 / 10 / 10 |

Remaining variation is **per-dataset** (calibration) or **per-regression-variant** — both
permitted axes. `DETECTION_THRESHOLD_MULTIPLIER` is in any case mathematically inert under a
percentile threshold: normal and attack scores both scale by `1/DTM`, so the percentile scales
identically and the comparison is invariant.

Three discovery settings also vary, and they are listed here rather than left implicit because
each materially affects the learned graph:

| Setting | Where it applies | Value | Why |
| :--- | :--- | ---: | :--- |
| `maxP` | GES robotics only | 1 | Bounds the parent set. GES on 35 strongly coupled sensors is otherwise intractable; healthcare and industrial run unbounded. |
| collinearity jitter | GES robotics, all CAM-UV | 1e-5 | Pepper and TEP contain exactly collinear sensor pairs, which make the GES score matrix singular. Seeded at 42, so runs reproduce. |
| `TOP_VARIANCE_SENSORS` | GES and CAM-UV robotics | 35 | The variance cap, justified in §7.2. PCMCI needs no cap and uses all 244. |

Each is constant across the three regression variants within its dataset, so **regression type
remains the only axis varying inside a folder**. `tools/audit_pipeline.py` checks exactly this
and fails if any of them differs between variants of the same dataset.

### 7.1a Feature scaling, and what fixing it cost

A fourth setting used to vary and should not have. Four linear scripts — GES and CAM-UV, on
robotics and industrial — fitted Ridge directly on the raw design matrix, while the polynomial
and RBF scripts *in the same folders* fitted it on `StandardScaler` output, as did all nine
PCMCI scripts and all nine healthcare scripts. Ridge penalises coefficients, so on a raw design
matrix the penalty falls on each parent according to its units rather than its explanatory
value. That is not a difference in regression type, so it violated the constraint.

Scaling everywhere is the correct direction to unify: polynomial and RBF genuinely require it
(degree-3 expansion of unscaled values, and an RBF kernel width that is meaningless without it),
so the four linear scripts were the outliers. Correcting them costs accuracy in every case:

| Configuration | AUC-ROC before | after | change |
| :--- | ---: | ---: | ---: |
| GES — robotics, linear | 0.7663 | 0.5906 | **−0.1757** |
| CAM-UV — industrial, linear | 0.8927 | 0.8428 | −0.0499 |
| CAM-UV — robotics, linear | 0.6741 | 0.6497 | −0.0244 |
| GES — industrial, linear | 0.8593 | 0.8407 | −0.0186 |

The lower figures are the ones reported in §2. Keeping the higher ones would have meant
retaining a per-configuration advantage that no principle selects — the same objection that
removed `RIDGE_ALPHA = 1000.0` from PCMCI healthcare in §2.2, and it applies here even though
this time the correction runs against the tidier story.

It also removes a confound from the headline comparison. PCMCI scaled in all nine of its
scripts while GES and CAM-UV did not on two domains, so the previous cross-algorithm
comparison on robotics and industrial was partly a comparison of preprocessing. PCMCI's lead
on robotics is *larger* after the fix, not smaller.

### 7.1b One configuration was not tracking the same quantity

A second single-file defect, found the same way. `PCMCI_robotic/anomaly_detection_rbf.py`
stored `fine_coeffs[var] = model.coef_` and fed the RLS filter `X_rbf[t]` — no intercept in
either. That is internally consistent, so it neither crashed nor produced a dimension error,
but it made this **the only one of 27 configurations whose drift vector excluded the intercept
term**. "Coefficient drift" therefore meant something slightly different in that one file, and
that file happens to hold the best robotics result.

Restoring the intercept, so it matches the other eight RBF scripts and all eighteen linear and
polynomial ones:

| PCMCI — robotics, RBF | F1 | AUC-ROC | AUC-PR |
| :--- | ---: | ---: | ---: |
| before | 0.8804 | 0.8519 | 0.9390 |
| **after** | 0.7938 | **0.8752** | **0.9419** |

The primary metric improves and F1 falls, which is not a contradiction: F1 is read at the fixed
95th-percentile operating point, and a better-ranked score distribution can place that
percentile less favourably. The explainability result is unaffected — `TabletTouchX` still
leads LedsControl and `KneePicthTemp` still leads JointControl.

`tools/audit_pipeline.py` now checks that the baseline coefficient vector and the online design
matrix agree about the intercept in all 27, however each file spells the construction.

### 7.1b What the audit checks

Uniformity is a claim about 11,000 lines, so it is verified mechanically rather than asserted.
`tools/audit_pipeline.py` confirms all eight of:

| Check | Result |
| :--- | :--- |
| RLS update byte-identical after stripping comments | 1 implementation across 27 scripts |
| 95th-percentile fault-free threshold | 27/27 |
| No threshold fitted against evaluation labels | 27/27 |
| Attribution depth uniform (top-3) | 27/27 |
| Per-variant score files written for the bootstrap | 27/27 |
| Per-domain constants, no cross-algorithm mismatch | 4 constants |
| Cached graphs guarded against a subsample change | 18/18 GES and CAM-UV |
| Discovery knobs vary by dataset, never by variant | consistent in every folder |

`tools/validate_algorithms.py` additionally checks the mathematics against closed-form answers
rather than by inspection:

| Check | Measured |
| :--- | :--- |
| RLS at `lam=1` equals ordinary least squares | max coefficient difference **7.5 × 10⁻¹⁰** |
| RLS with forgetting tracks a step change in a coefficient | reaches 2.5004 against a true 2.5, while `lam=1` lags at 1.2522 |
| Drift responds to a broken relationship, not to rescaling | broken **4.1707**, rescaled-but-intact **0.0002**, unchanged 0.0008 |
| GES keeps only unambiguously directed CPDAG edges | rule present in 9/9, drops undirected pairs |

The third of these is the detection premise itself, and the separation is four orders of
magnitude: multiplying a variable by 5 while leaving its causal relationship intact produces
essentially no drift, whereas changing the relationship produces a large one. That is the
property the whole approach rests on, and it is measured rather than assumed.

### 7.2 Feature selection: criterion and why 35

GES and CAM-UV cannot search 247 sensors tractably, so robotics caps them at
`TOP_VARIANCE_SENSORS = 35`. The criterion is the **highest standard deviation on the
training split at full temporal resolution**. No sensor is selected by name.

**The value 35 is not arbitrary.** Three independent considerations agree on roughly it:

| Consideration | Value |
| :--- | :--- |
| Elbow of the scree curve of sorted standard deviations (max distance to chord) | **K = 32** |
| Total variance retained at K = 35 | **94.55%** (93.2% at K=30, 95.6% at K=40) |
| Tractability — a 160-variable GES run did not converge in 66 min CPU | K ≪ 160 |

The variance curve is flat in this region, so the exact value is not load-bearing: any K
between about 30 and 40 retains 93–96% of the variance and sits near the elbow. The cap is a
computational necessity; the ranking that applies it is purely statistical.

**Every sensor the qualitative analysis cites is far inside the cap** — `TabletTouchX` at
variance rank 2, `WheelBTemp` 3, `WheelFRTemp` 4, `KneePicthTemp` 6, `TabletTouchY` 7 of 249.
The attribution results are therefore discovered by the model, not admitted by the selection.

An earlier version force-included any sensor matching `("Tablet","Temp","Stiff","Wheel")`,
which would have made the LED→Tablet result true by construction. The root cause was a bug:
variance was measured *after* subsampling by 10, which aliases the transient tablet channels
and pushed them from rank 2 and 7 down to 145 and 142. Measuring at full temporal resolution
retains every cited sensor on merit, so the tag list is unnecessary and has been removed.

### 7.3 The decision threshold never sees a label

24 of 27 configurations previously selected their threshold by maximising F1 against the
evaluation labels (`precision_recall_curve` → `argmax`). That is test-set leakage, and it
inflated F1 wherever the models were weak. All 27 now use the 95th percentile of the
fault-free score distribution.

AUC-ROC was unaffected throughout — it is rank-based and threshold-independent. On healthcare,
five of six AUC values were bit-identical before and after, confirming the models'
discriminative ability never changed; only the reported operating point did.

### 7.4 Resampling rate is algorithm-coupled, by measurement

PCMCI derives its rate from an FFT/Nyquist analysis (74 on Pepper, 7 on ECG, 1 on TEP); GES and
CAM-UV use 10. This is a requirement of the discovery method, not tuning, and was verified
in both directions:

- Forcing GES healthcare to `subsample=1` collapses it from AUC 0.8114 to 0.6184 (linear) and
  0.7867 to 0.4992 (polynomial — random). Adjacent ECG samples become near-identical,
  one-step prediction turns trivial, and the RLS drift signal vanishes.
- Forcing PCMCI robotics to `subsample=10` costs AUC 0.7825 → 0.6813 **and drops
  `KneePicthTemp` out of the JointControl top-3 entirely**, failing the lab check.

PCMCI models time explicitly with lags and needs its derived rate; GES and CAM-UV do static
discovery with a lag-1 encoding and need approximately decorrelated samples.

### 7.5 Cache integrity

Graph discovery is cached. A cache built by older code silently produces metrics for a
pipeline that no longer exists — `PCMCI_healthcare` was found in exactly that state (stored
`subsample=1`; the current derivation yields 7). All nine caches have been regenerated from
scratch, and GES/CAM-UV now **raise** if a cached graph's `subsample` disagrees with the
code, naming the file to delete.

---

## 8. Reproducing

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r PCMCI/PCMCI_robotic/requirements.txt

cd PCMCI/PCMCI_robotic
python anomaly_detection.py              # linear
python anomaly_detection_polynomial.py   # polynomial
python anomaly_detection_rbf.py          # RBF
```

```bash
python Baselines/lstm_shap_baseline.py   # neural baseline, from repo root
python tools/collect_results.py          # regenerate the tables above
```

Each folder carries its own pinned `requirements.txt`. The first run of a folder learns and
caches the causal graph; later runs reuse it. Delete the `.npz` to force rediscovery.

Each run writes to its folder's `outputs/`:

| File | Contents |
| :--- | :--- |
| `metrics_summary.csv` | one row per variant — confusion matrix, F1, AUC-ROC, AUC-PR |
| `explainability_top_features.csv` | top-3 attributed variables per attack class |
| `*_top_anomalous_variables.png` | the explainability histogram per attack class |

---

## 9. Repository layout

```
PCMCI/{PCMCI_robotic,PCMCI_healthcare,PCMCI_industrial}/
GES/{GES_robotic,GES_healthcare,GES_industrial}/
CAM-UV/{CAM_UV_robotics,CAM_UV_healthcare,CAM_UV_industrial}/
    anomaly_detection*.py     one per regression variant
    reporting_helper.py       metrics, plots, CSV export
    requirements.txt          pinned to the environment that produced the results
    FINAL_RESULTS.md          per-folder analysis
    outputs/                  generated results
CAM-UV/camuv_engine.py        shared loader for the real lingam CAM-UV
Baselines/                    LSTM autoencoder + SHAP neural baseline
tools/collect_results.py      aggregates every metrics_summary.csv
```

---

## 10. Data provenance

| Domain | Source |
| :--- | :--- |
| Robotics | PEPPER humanoid telemetry, 247 non-constant sensors, 3 injected attack classes |
| Healthcare | PTB-XL 12-lead ECG. **All 12 leads are used.** They are not independent: Einthoven (`III = II − I`) and Goldberger (`aVR = −(I+II)/2`, `aVL = I − II/2`, `aVF = II − I/2`) relate the six limb leads algebraically. In this data those identities hold to a relative error of 6×10⁻⁴ to 2×10⁻³ — near-dependent but not exact, so the 12-lead matrix is full rank (12 of 12) and there is no numerical singularity. See limitation 11. |
| Industrial | Tennessee Eastman Process simulation, 52 variables, 20 documented fault classes |

---

## 11. Known limitations

1. **PCMCI on ECG is weak** (AUC 0.49–0.71). Its earlier stronger numbers depended on
   `RIDGE_ALPHA = 1000.0`, which no label-blind procedure reproduces.
2. **Healthcare attributions cannot be validated** — PTB-XL publishes no per-lead ground truth.
3. **GES and CAM-UV see 35 of 247 robotics sensors.** The cap is a computational necessity;
   selection within it is purely statistical.
4. **Two configurations learn more variables than they have samples.** PCMCI robotics fits 244
   variables from 192 samples (0.79 per variable); GES and CAM-UV industrial fit 52 from
   **35** (0.67 per variable), because the 500-row fault-free file is cut to a 70% training
   split and then subsampled by 10. Both graphs are underdetermined and detection works
   despite it, but the industrial case additionally makes its *sparsity* uninterpretable:
   CAM-UV flags a confounded pair by rejecting independence with HSIC, and 35 samples have
   little power to reject, so its 1.1% pair saturation is consistent with genuine sparsity and
   with no power to detect anything (§6). PCMCI industrial runs at `subsample = 1` and sees 350
   samples, so it is not affected.
5. **Evaluation-set sizes differ between PCMCI and GES/CAM-UV** because their resampling
   rates differ (§7.4). GES and CAM-UV are exactly matched within every domain.
6. **F1 is operating-point dependent.** Prefer AUC-ROC for cross-model comparison.
7. **`reporting_helper` zero-pads score arrays** while labelling those windows positive,
   injecting a small number of guaranteed false negatives. This applies identically to all 28
   configurations, so comparisons are unaffected, but absolute F1 is slightly pessimistic.
8. **The FFT resampling derivation is PCMCI-internal.** GES and CAM-UV use a fixed 10 rather
   than sharing that code, so a new dataset would not adapt automatically.
9. **The causal pipelines are not the strongest detectors on two of the three domains, and on
   one of them they lose to a method with no parameters** (§5). On Pepper the tuned TCN
   (AUC 0.8856) is not outperformed by any causal configuration. On industrial a trivial
   mean-`|z|` control reaches 0.9286 and the neural architectures 0.9995, against a causal best
   of 0.9151. Healthcare is the one domain where causal structure leads clearly. All three
   architectures now run on all three domains.
10. **Single run for the causal pipelines, and no ranking is statistically separable.** Only
    the neural baselines are averaged over seeds. A moving-block bootstrap on the real
    per-sample scores (§2.5) shows no domain winner separable from its runner-up, and the
    analytical Hanley–McNeil intervals are too narrow because they assume independent samples.
    Establishing separability would need more evaluation data or repeated runs over resampled
    training splits.
11. **Part of the healthcare causal graph is algebraic, not physiological — and removing it
    costs detection.** All 12 ECG leads are used, but four of them are exact functions of the
    other two: Einthoven (`III = II − I`) and Goldberger (`aVR = −(I+II)/2`, `aVL = I − II/2`,
    `aVF = II − I/2`). Measured against their algebraic forms in this data, those four leads
    correlate at **r = 0.99998 to 0.999995** — they carry no independent information.

    Causal discovery cannot distinguish a definition from a mechanism. Edges directly
    connecting `I`/`II` to a derived lead account for **8% (CAM-UV), 11% (PCMCI) and 18%
    (GES)** of all edges, with limb-limb edges at 17–35%; a bounded minority, since most
    structure involves the precordial leads V1–V6, which are not algebraically determined.

    Rerunning GES healthcare on the eight independent leads (I, II, V1–V6) gives a genuine
    trade-off rather than a clear improvement:

    | GES healthcare | 12 leads | 8 independent leads |
    | :--- | ---: | ---: |
    | F1 | 0.5818 | **0.6327** |
    | AUC-ROC | **0.8114** | 0.7667 |

    The redundant leads carry no independent *information* but do carry independent *noise
    realisations*, and a coefficient-drift detector gains detection surface from every extra
    monitored target — so they raise AUC while contributing arithmetic to the graph. The
    12-lead configuration is retained for detection results, and this is stated as a
    qualification on the causal-structure interpretation in §2.2 rather than resolved.
