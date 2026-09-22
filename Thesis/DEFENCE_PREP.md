# Defence preparation

Above five points a *controrelatore* is mandatory, and their job is to find the weakest
joint in the work. These are the sixteen questions most likely to be asked, in roughly the
order of how uncomfortable they are, each with the answer and the file that proves it.

Every number here traces to a CSV. None of it needs to be memorised — what matters is
knowing *which* measurement answers each question, because a candidate who says "that is in
the density table, and it goes the other way on industrial" has clearly done the work,
whether or not they recall the fourth decimal.

The general rule for this defence: **the results are mixed, and the mixedness is the
finding.** Do not defend a claim the data does not support. Every question below has an
honest answer that is stronger than a defensive one.

---

## 1. "PCMCI is not in Table 1 of causal-learn. Why is it in your study?"

Because every constraint-based method causal-learn ships assumes i.i.d. observations, and
all three of these domains are time series. Autocorrelation inflates apparent dependence, so
PC reports edges that are artefacts of persistence.

**And it was measured, not assumed.** Eight independent AR(1) processes, no cross-variable
causation, so every edge either algorithm reports is a false positive by construction:

| phi | PC | PCMCI (FDR) |
|---:|---:|---:|
| 0.00 | 0.0429 | 0.0500 |
| 0.95 | **0.4982** | 0.0536 |

At zero autocorrelation PC is calibrated almost perfectly against a nominal 0.05 — that is
the control showing the test is fair. At phi = 0.95 it links half of all unrelated pairs.

`tools/pc_vs_pcmci.csv`, `tools/pc_vs_pcmci_ablation.py`

> If pushed on CD-NOD: it targets distribution shift across regimes, not lagged causation.

---

## 2. "Your detector scores coefficient drift, not reconstruction error. Justify that."

Reconstruction error asks whether a value is unusual. Drift asks whether the *relationship*
between variables has changed. They come apart exactly where anomaly detection is hard:

| Scenario | Mean drift |
|---|---:|
| Relationship breaks | **4.1707** |
| Values rescaled, relationship intact | 0.0002 |
| Nothing changes | 0.0008 |

A reconstruction-error detector cannot separate rows one and two. This separates them by
four orders of magnitude. `tools/algorithm_validation.csv`

---

## 3. "Causal methods lose on two of your three domains. What is the contribution?"

Say it plainly: they do lose, and that is the result.

| Domain | Best causal | Neural (tuned) | Trivial |
|---|---:|---:|---:|
| Robotics | 0.8752 | **0.8856** | 0.8139 |
| Healthcare | **0.8114** | 0.7077 | 0.6136 |
| Industrial | 0.9151 | **0.9997** | 0.9286 |

The contribution is the comparison itself, run under conditions that make it mean something:
one detector, one threshold rule, one attribution rule across all 27 configurations, tuned
baselines on every domain, and a trivial control that several methods fail to beat. Most of
the published comparisons this thesis reviews do not clear that bar, which is why their
positive results are hard to interpret.

A study that only reports where causal discovery wins is the thing this thesis argues
against.

---

## 4. "A z-score with no learned parameters beats your causal pipeline on TEP. So why bother?"

Because that is the honest finding and it has a clear explanation: TEP faults are large and
abrupt, so they show up as amplitude deviations, and amplitude deviations are what a z-score
detects. Structure buys nothing when the signal is that obvious.

The right conclusion is a scope limit, not a rescue: **causal discovery earns its cost when
anomalies are structural rather than amplitude-based.** That is a more useful sentence for a
practitioner than a claimed universal win.

`Baselines/outputs/trivial_baseline.csv`

---

## 5. "The three algorithms do not score the same samples. Is your comparison valid?"

Not for a paired significance test, and the thesis says so rather than hiding it.

| Domain | PCMCI | GES / CAM-UV | Neural |
|---|---:|---:|---:|
| Robotics | 181 | 1378 | 155 |
| Healthcare | 4806 | 3115 | 4793 |
| Industrial | 16716 | 1323 | 16590 |

PCMCI selects its subsample rate adaptively; GES and TS-CAM-UV are fixed at 10. Of 108
within-domain comparisons, **54 are testable and 54 are not** — and every untestable one is
PCMCI against another algorithm, which shows the pattern is the subsample policy rather than
an accident.

What is testable was tested with DeLong. What is not is listed individually with its reason.
`tools/delong_tests.csv`, `tools/delong_not_pairable.csv`

The comparison is valid as a comparison of each algorithm *as intended to be applied*,
preprocessing included. It is not valid as an isolation of the discovery step, and the thesis
should claim only the first.

---

## 6. "Your healthcare graph links 63 of 66 pairs. Where is the structure?"

For PCMCI, there is almost none, and that is the explanation for its poor healthcare result
rather than an embarrassment. Density against performance:

| Folder | Density | Best AUC |
|---|---:|---:|
| GES_healthcare | **51.5%** | **0.8114** |
| CAM_UV_healthcare | 72.7% | 0.7861 |
| PCMCI_healthcare | **95.5%** | **0.7147** |

Perfectly monotone: sparser graph, better detection, across all three algorithms. That is
direct evidence for the structural-constraint mechanism — on this domain.

**Be careful not to overclaim.** On industrial the ordering reverses, the densest graph
winning. On robotics it is confounded by PCMCI seeing 244 variables against 35. So the
mechanism is supported on healthcare and only there. `tools/graph_density.csv`

---

## 7. "Removing the algebraically dependent ECG leads made GES worse. Doesn't that undermine you?"

It qualifies the result, and it is better to raise it than to be caught by it.

| | GES linear |
|---|---:|
| 12 leads | 0.8114 |
| 8 independent leads | **0.7418** |

Einthoven and Goldberger make four of the twelve leads exact linear functions of the other
eight, so part of what GES was recovering was algebra. The honest reading: **some of the
healthcare advantage was algebraic, and what remains is still real.** 0.7418 still clears the
tuned neural baseline at 0.7077 and the trivial control at 0.6136.

Note also that PCMCI polynomial moves the *other* way, 0.5075 to 0.6355 — the exact
dependencies were degrading its regressions. `*/[A-Z]*_healthcare/outputs_leads8/`

---

## 8. "Prove there is no label leakage."

Three independent mechanisms, all machine-checked:

- The threshold is the 95th percentile of held-out **fault-free** scores. No anomaly label is
  consulted. Audit check 2.
- Audit check 3 greps every script for `precision_recall_curve` and `argmax(f1)` — the two
  idioms that fit a threshold to test labels — and fails the build if either appears.
- Scaler statistics and baseline coefficients are fitted on the first 70% only,
  chronologically, before any anomaly occurs.

`tools/audit_pipeline.py`, 16/16 checks passing.

---

## 9. "TS-CAM-UV is not reproducible. How do you defend results built on it?"

By not pretending otherwise. The graph is cached and shipped, so every number is verifiable
from the artefact even though a fresh discovery run would produce a different graph. This is
also why `tools/measure_runtime.py` deliberately does *not* re-run discovery — doing so
would change the graph and with it every downstream result.

The limitation is real and belongs in the threats section: the TS-CAM-UV numbers are one
draw, not a distribution.

---

## 10. "PCMCI sees 244 variables on robotics; GES and CAM-UV see 35. Is that a fair comparison?"

No, and the thesis says so. The cap exists because GES and TS-CAM-UV do not complete on 244
variables in tractable time.

| Domain | PCMCI | GES | CAM-UV | Matched |
|---|---:|---:|---:|---|
| Robotics | 244 | 35 | 35 | **no** |
| Healthcare | 12 | 12 | 12 | yes |
| Industrial | 52 | 52 | 52 | yes |

So **RQ1's evidence rests on healthcare and industrial**, and robotics is reported as
descriptive rather than comparative. Audit check 14 reports this every run.

---

## 11. "Why the 95th percentile? That is arbitrary."

It is a choice, not a derivation, and its purpose is comparability: it fixes the false
positive rate at 5% for every configuration, so F1 differences reflect discriminative
ability rather than threshold placement. Any label-blind rule would serve; this one is
conventional and is applied identically to all 27 configurations and all baselines.

It is also why AUC-ROC is the primary metric and F1 secondary — a better-ranked score can
score worse on F1 under a fixed-percentile threshold.

---

## 12. "Your bootstrap intervals are very wide. Is anything actually significant?"

Some things, and the wide intervals are the honest ones. PCMCI robotics RBF:

| Method | 95% interval | Width |
|---|---|---:|
| Analytical Hanley–McNeil | [0.8258, 0.9246] | 0.099 |
| Moving-block bootstrap | **[0.7559, 0.9587]** | 0.203 |

The analytical interval assumes independent samples. These are time series. It is narrower on
**every** configuration — by 54% and 63% on the two worst — so reporting it would claim a
separability the data does not support.

For differences, DeLong is the right test and it is more sensitive than comparing intervals,
because paired scores are correlated. 54 comparisons were run; most within-domain variant
differences are significant at 0.05. `tools/bootstrap_intervals.csv`, `tools/delong_tests.csv`

---

## 13. "Ridge alpha varies across your regression variants. Is RQ2 confounded?"

Partly, on robotics only, and it is pinned rather than hidden. Audit check 11 verifies that
within every (domain, variant) cell the ridge alpha is **identical across all three
algorithms**. So RQ1 — which algorithm — is unconfounded. RQ2 — which regression variant —
is partly confounded on robotics, where alpha differs by variant (10.0 / 0.01 / 1.0).

---

## 14. "Twelve hyperparameter configurations is not much of a search."

Correct, and it is described as a grid rather than as exhaustive. What matters for the
argument is that the **same** grid was applied to every architecture on every domain, so the
baselines are not handicapped relative to each other or to the causal pipeline.

Worth volunteering: until recently only robotics was searched, and seven of the nine
architecture-domain cells moved when the search was extended. **No conclusion changed** —
healthcare 0.6815 to 0.7077, industrial 0.9996 to 0.9997 — but the earlier version compared
a tuned causal pipeline against an untuned neural one, which is the criticism this thesis
makes of the literature. It was found and fixed. Saying so is a strength.

---

## 15. "What would you do differently?"

Have an answer ready; not having one reads as not having thought about it.

- **Run every algorithm at a common scoring resolution from the start.** It is the single
  choice that most limits what can be concluded, and it made half the comparisons untestable.
- **Fix the variable set across algorithms**, accepting a smaller robotics problem, so RQ1
  had three domains of evidence rather than two.
- **Include a fourth domain where anomalies are structural rather than amplitude-based** —
  on the evidence here, that is the regime where causal discovery should win, and healthcare
  is currently the only case supporting it.
- **Pre-register the analysis.** Several of the corrections in the git history — the missing
  intercept, the untuned baselines, the attribution split — were found by auditing after the
  fact rather than prevented by design.

---

## 16. "You claim this reproduces the XAI laboratory. Against what?"

**Know the exact strength of this claim before you are asked, because it is the one thing in
the repository that cannot be checked from the repository.**

What is mechanically verified, by `tools/check_lab_requirements.py` on every build: the LED
attacks attribute to a Tablet sensor in **9 of 9** configurations, and the Joint attacks to a
joint temperature in **8 of 9**.

What is not: the requirement itself was relayed second-hand. The laboratory handout is not in
the repository. So the encoding — LED to Tablet, Joint to temperature — rests on a
description rather than on the source document.

Say the narrow version and it is airtight:

> The pipeline satisfies the explainability requirement of the XAI laboratory — LED attacks
> attribute to Tablet sensors, Joint attacks to joint temperatures — in 9 of 9 and 8 of 9
> configurations respectively.

Not "reproduces the laboratory", which claims more than you can show. The narrow version
states the requirement being tested, so anyone holding the handout can check the encoding
instead of taking it on trust. **If you still have the handout, read the predicates in
`LAB` in that file against it before submitting** — it is a ten-minute check that closes the
last gap.

---

## Two things to have at your fingertips

**The number that most surprised you, and why.** The trivial z-score baseline beating every
causal configuration on TEP, 0.9286 against 0.9151. It is the result that most changed how
the thesis reads, and volunteering it demonstrates you were testing rather than confirming.

**The correction you are proudest of.** The missing intercept in one of 27 configurations:
internally consistent, so it produced no error, but it measured a different quantity from the
other 26. Restoring it moved that configuration from 0.8519 to 0.8752. It was found by an
audit that checks all 27 rather than by anything going wrong.
