# Chapter 5 draft — verified line by line

Checked against the code, not from memory. Four errors, one of them fundamental. Everything
not listed as an error was verified correct, and there is a lot of it — the structure is
sound and most of the detail is right.

---

## Error 1 — §5.3 describes the wrong detector. This is the serious one.

The draft says:

> a symmetric reconstruction error metric was utilized across all configurations:
> $r_t = \lVert \hat{x}_t - x_t \rVert^2$

**The causal pipeline does not use reconstruction error.** From
`PCMCI/PCMCI_healthcare/anomaly_detection.py:307-310`:

```python
w_t, P_t, _ = _rls_update(x_t, y_t, w_t, P_t, lam=RLS_LAMBDA)
err[var][t, :] = w_t - fine_coeffs[var]
norm_agg[t, i] = np.linalg.norm(err[var][t, :])
```

The score is **coefficient drift** — how far the online regression coefficients have moved
from their fault-free baseline:

$$s_j(t) = \big\lVert \mathbf{w}_j(t) - \mathbf{w}_j^{(0)} \big\rVert_2$$

This is not a detail. It is the thesis's entire contribution. Reconstruction error asks *is
this value unusual*; drift asks *has the relationship between these variables changed*. The
measured difference, from `tools/algorithm_validation.csv`:

| Scenario | Mean drift |
|---|---:|
| Relationship breaks | **4.1707** |
| Values rescaled, relationship intact | 0.0002 |
| Nothing changes | 0.0008 |

A reconstruction-error detector cannot separate the first two rows. Writing $r_t =
\lVert\hat{x}_t - x_t\rVert^2$ in the methods chapter tells the examiner you built the
baseline, not the contribution — and the results chapter then makes no sense.

The formula as written *is* correct for the three neural autoencoders. It is wrong as a
description of all configurations. Two formulas are needed, not one.

Also missing: a 15-step warm-up (`if t < 15: err = 0`) before drift is scored.

---

## Error 2 — §5.1 says the pipeline uses 8 leads. It uses 12.

The draft says:

> Consequently, the dataset was strictly restricted to the 8 algebraically independent leads

`GES/GES_healthcare/anomaly_detection_ges.py:49`:

```python
ABLATION_8_LEADS = os.environ.get("ECG_INDEPENDENT_LEADS") == "1"
```

With the variable unset — which is how every headline result was produced — the pipeline
runs on **all twelve leads**. The eight-lead runs live in a separate `outputs_leads8/`
directory precisely so they cannot overwrite the headline numbers.

This matters because the two give different answers:

| | GES linear |
|---|---:|
| 12 leads (`outputs/`) | **0.8114** |
| 8 leads (`outputs_leads8/`) | **0.7418** |

Every AUC quoted in the results chapter is the twelve-lead number. A methods chapter saying
"strictly restricted to 8" contradicts the results chapter on its own terms, and the
contradiction is one directory listing away from being found.

**This is the third time this error has appeared** — it was in the Chapter 6 draft and in the
discussion prose too. The reasoning in the draft is right; the tense is wrong. The 12-lead
run is the study; the 8-lead run is a robustness check that *qualifies* it. And note the
direction: restricting to independent leads made GES **worse**, which means part of what it
was exploiting was Einthoven's algebra. That is a genuinely interesting finding and it is
lost entirely if the chapter claims the restriction was always in place.

---

## Error 3 — the subsampling rates are per-algorithm, not per-domain

The draft gives one rate per domain: $k=74$ for robotics, $k=10$ for industrial. Measured
from the cached graphs:

| Domain | PCMCI | GES | TS-CAM-UV |
|---|---:|---:|---:|
| Robotics | **74** | 10 | 10 |
| Healthcare | 7 | 10 | 10 |
| Industrial | **1** | 10 | 10 |

So $k=74$ is true for PCMCI robotics and false for the other two algorithms in that domain;
$k=10$ is true for GES and TS-CAM-UV industrial and false for PCMCI, which scores every
sample. Stating either as a property of the domain misdescribes six of the nine cells.

This is not a small bookkeeping point — it is the study's **primary threat to internal
validity**, already documented by audit check 13, and it has a measured cost: PCMCI robotics
forced to 10 falls 0.7825 → 0.6813; GES healthcare forced to 1 falls 0.8114 → 0.6184. It
deserves its own subsection in Chapter 5, not a silent averaging into one number.

It also has a consequence the draft should state: because the algorithms score different
samples, **no paired significance test can compare PCMCI against GES or TS-CAM-UV**. See
`tools/delong_not_pairable.csv` — 54 of 108 comparisons are untestable for exactly this
reason.

---

## Error 4 — §5.1 says 244 variables without saying who sees them

The draft describes the robotics dataset as 244 sensors. True of the raw recording and of
PCMCI. GES and TS-CAM-UV run on **35**, a cap declared in their robotics scripts for
tractability.

| Domain | PCMCI | GES | TS-CAM-UV | Matched? |
|---|---:|---:|---:|---|
| Robotics | **244** | **35** | **35** | no |
| Healthcare | 12 | 12 | 12 | yes |
| Industrial | 52 | 52 | 52 | yes |

RQ1's evidence therefore rests on healthcare and industrial, where the algorithms see the
same variables. Audit check 14 reports this. It needs to be in the chapter.

---

## Verified correct

Everything below was checked and is right. Worth knowing so you do not rewrite it.

- **244 robotics sensors, 52 TEP variables, 12 ECG leads** — all confirmed
- **The three Pepper attacks** — `WheelsControl`, `LedsControl`, `JointControl` ✓
- **Einthoven and Goldberger reasoning** — the algebra is right ($III = II - I$), and the
  motivation for the ablation is exactly correct. Only the tense is wrong.
- **"merely 35 samples for 52 variables"** — confirmed in
  `CAM-UV/CAM_UV_industrial/tep_power_check.csv`. The power sweep is a real strength: at
  subsample 10 there are 35 samples and density 0.0238; at subsample 1, 350 samples and
  density 0.1112. The sparsity is absent statistical power, not selectivity, and you have the
  measurement to prove it.
- **70/30 chronological split** — `TRAINING_FRAC = 0.7`, identical in all 27 scripts ✓
- **Z-score scaling fitted on the training split only** — `StandardScaler` fitted inside
  `fit_normal_coeffs` on the first 70% ✓. Your leakage argument is correct.
- **95th-percentile label-blind threshold, 5% FPR budget** — correct, and audit checks 2
  and 3 enforce both halves of it ✓
- **The HPO protocol, in full** — 12 configurations, $h \in \{32,64\}$, $l \in \{1,2,3\}$,
  $\alpha \in \{0.001,0.01\}$, selection on a disjoint 15% fault-free validation split,
  patience 15 ✓ every detail correct

One addition for §5.4: the search now runs on **all three domains**. Until recently only
robotics was searched, and seven of the nine architecture-domain cells moved when the others
were added. Say the grid ran everywhere — it now does — and the sentence about baselines not
being "structurally handicapped" becomes defensible rather than aspirational.

---

## What Chapter 5 is still missing

Three things belong here, and an examiner will look for them.

**A subsection on scoring resolution.** Error 3 above. It is the primary threat to internal
validity and it is currently absent.

**Graph density.** `tools/graph_density.csv` measures how much structure each graph actually
imposes, and it is the evidence for the mechanism the discussion wants to claim:

| Domain | Sparsest | | Densest | |
|---|---|---:|---|---:|
| Healthcare | GES 51.5% | **0.8114** | PCMCI 95.5% | **0.7147** |
| Industrial | GES 3.2% | 0.8994 | PCMCI 38.5% | **0.9151** |
| Robotics | GES 10.9% | 0.6999 | CAM-UV 80.7% | 0.7818 |

On healthcare the ordering is perfectly monotone — sparser graph, better detection, across
all three algorithms. On industrial it **reverses**. On robotics it is confounded by PCMCI
seeing 244 variables against 35. So "structural constraint prevented overfitting" is
supported on healthcare and only there, which is a far more defensible claim than the general
one, and you have the table to back it.

**The trivial control.** `Baselines/trivial_baseline.py` — mean absolute z-score, no learned
parameters, same threshold rule. It belongs in §5.4 beside the neural baselines, not least
because it beats every causal configuration on industrial (0.9286 against 0.9151).
