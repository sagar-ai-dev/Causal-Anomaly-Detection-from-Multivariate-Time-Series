# Chapter 2 scaffold

Not prose — a structure and a map from each of the 62 references to the claim it supports,
so writing this chapter is filling in a frame rather than facing a blank file.

The organising decision, made once and then everything follows: **Chapter 2 exists to set up
the gap that Chapter 5 closes.** That gap is not "nobody has applied causal discovery to
anomaly detection" — several people have, and they are cited below. It is that the
comparisons in the literature are not constructed so that their positive results can be
interpreted. Build every section toward that.

---

## 2.1 Causal discovery: what it is and what the families are

**Claim:** causal structure is recoverable from observational data under assumptions, and
the algorithms divide into families that make different assumptions.

| Reference | Carries |
|---|---|
| `pearl2009causality` | the framework; interventions vs observation |
| `spirtes2000causation` | constraint-based discovery, PC, the canonical text |
| `peters2017elements` | modern treatment, assumptions stated cleanly |
| `glymour2019review` | survey; use for the family taxonomy |
| `vowels2022dags` | survey; more recent, more critical |
| `zheng2024causallearn` | **Table 1 is your taxonomy** — cite it as the frame for §2.2 |

**Structure the families as causal-learn does**, because that is where your three algorithms
come from and it makes the selection argument in Chapter 4 follow naturally:
constraint-based, score-based, functional causal models, causal representation learning.

| Family | Cite | Yours |
|---|---|---|
| Constraint-based | `spirtes2000causation` | PCMCI |
| Score-based | `chickering2002ges`, `ramsey2017fges` | GES |
| Functional (FCM) | `shimizu2006lingam`, `hoyer2008anm`, `buhlmann2014cam`, `maeda2021camuv` | TS-CAM-UV |
| Continuous optimisation | `zheng2018notears` | not used — one sentence on why |

`Thesis/ALGORITHM_SELECTION.md` has the full argument and the density and import evidence.

---

## 2.2 Causal discovery for time series

**Claim:** i.i.d. methods fail on serially correlated data, and this motivated a separate
line of work.

| Reference | Carries |
|---|---|
| `granger1969causality` | the historical starting point; predictive not structural |
| `runge2019pcmci` | PCMCI, the MCI step, the autocorrelation problem |
| `runge2020pcmciplus` | contemporaneous links; why you did not need it |
| `runge2023causalinference` | recent survey of the area |
| `runge2023tigramite` | the implementation |

**This is where your PC-vs-PCMCI ablation belongs as a forward reference.** The literature
asserts that autocorrelation inflates false positives; you measured it — PC goes from 0.0429
to 0.4982 as phi goes from 0 to 0.95, PCMCI stays at 0.05. `tools/pc_vs_pcmci.csv`. Saying
"this is verified in §5.x" here is what makes the chapter yours rather than a summary.

---

## 2.3 Anomaly detection in multivariate time series

**Claim:** the field is dominated by reconstruction-based deep models.

| Reference | Carries |
|---|---|
| `chandola2009survey` | the canonical taxonomy |
| `blazquezgarcia2021review` | time-series specific review |
| `hochreiter1997lstm`, `bai2018tcn`, `vaswani2017attention` | your three architectures |
| `malhotra2016lstmad` | LSTM encoder-decoder for anomaly detection |
| `su2019omnianomaly` | stochastic recurrent, a strong published baseline |
| `audibert2020usad` | USAD |
| `hundman2018telemanom` | nonparametric thresholding — contrast with your 95th percentile |
| `deng2021gdn` | graph neural networks; structure without causality |

**`deng2021gdn` is worth a paragraph of its own.** It learns a graph and uses it for anomaly
detection without claiming the graph is causal. That is the nearest neighbour to your work
and the clearest way to say what causal discovery adds beyond structure.

---

## 2.4 Evaluation practice — where your contribution is positioned

**Claim:** published comparisons in this area are frequently constructed so that a positive
result cannot be interpreted. **This is the most important section in the chapter.**

| Reference | Carries |
|---|---|
| `kim2022rigorous` | point-adjust protocols inflate scores; rigour in TSAD |
| `wu2023flawed` | benchmarks are flawed and create illusory progress |
| `schmidl2022anomaly` | large-scale comparison; simple methods often win |
| `doshivelez2017rigorous` | rigour in interpretability evaluation |
| `demsar2006statistical` | how to compare methods across datasets |
| `davis2006prroc` | why AUC-PR matters under class imbalance |

Four specific failures to name, each of which you avoided and can prove:

1. **Thresholds fitted to test labels.** You use a label-blind 95th percentile; audit check 3
   greps for `precision_recall_curve` and `argmax(f1)` and fails the build.
2. **Tuned proposal against untuned baseline.** You searched the same 12-configuration grid
   on every architecture and every domain. Volunteer that this was *fixed* mid-project.
3. **No trivial control.** Yours is `Baselines/trivial_baseline.py`, and it **beats every
   causal configuration on industrial**, 0.9286 to 0.9151.
4. **Overlapping intervals reported as differences.** You use DeLong where samples pair, and
   say explicitly where they do not — 54 of 108 comparisons.

`schmidl2022anomaly` finding that simple methods often win is the perfect setup for your own
trivial-baseline result. Cite it before you report yours.

---

## 2.5 Causal methods for fault diagnosis and root-cause analysis

**Claim:** causal discovery has been applied to this problem, mostly in microservices, and
the results are promising but the evaluations have the weaknesses of §2.4.

| Reference | Carries |
|---|---|
| `ikram2022rcd` | RCD; hierarchical discovery for root cause |
| `li2022circa` | CIRCA; causal inference for online systems |
| `pham2024baro` | BARO; robust RCA |
| `pham2024howfar` | **"how far are we?" — a critical assessment, use it** |
| `chiang2001faultdetection` | classical industrial FDD |
| `yin2012comparison` | data-driven fault diagnosis comparison on TEP |

`yin2012comparison` is directly relevant — a comparison study on your industrial dataset.
Position your TEP result against it honestly: your causal configurations do **not** beat a
z-score there, and that is consistent with a literature in which simple methods do well on
TEP.

---

## 2.6 Explainability and attribution

**Claim:** post-hoc attribution is the standard, and it has known failure modes that
structural attribution avoids.

| Reference | Carries |
|---|---|
| `lundberg2017shap` | SHAP, your baseline attribution |
| `ribeiro2016lime` | LIME |
| `sundararajan2017integrated` | integrated gradients |
| `adebayo2018sanity` | **saliency maps fail sanity checks** — the key critical citation |
| `rudin2019stop` | argue for inherently interpretable models instead |
| `theissler2022xaitimeseries` | XAI for time series specifically |

`rudin2019stop` is the argument your method instantiates: the causal attribution is *not*
post-hoc, it is read from the model that made the decision. That is the theoretical framing
for your 9/9 versus 2/3 attribution result, and it is stronger than presenting the numbers
alone.

---

## 2.7 Supporting method references

Cite where used, not in a block.

| Reference | Where |
|---|---|
| `haykin2002adaptive` | RLS, §4 detector |
| `hoerl1970ridge` | ridge regression |
| `rahimi2007random` | random Fourier features, the RBF variant |
| `gretton2005hsic` | HSIC, the independence test CAM-UV uses |
| `hanley1982roc` | analytical AUC interval — and why you do not rely on it |
| `kunsch1989blockbootstrap`, `politis1994stationary`, `efron1994bootstrap` | the block bootstrap |
| `delong1988comparing` | **paired AUC comparison — the test you actually run** |

---

## 2.8 Datasets

| Reference | Where |
|---|---|
| `wagner2020ptbxl`, `goldberger2000physionet` | PTB-XL provenance |
| `strodthoff2021ptbxlbenchmark` | published PTB-XL benchmarks — position against these |
| `goldberger1942augmented` | **the augmented leads — your Einthoven/Goldberger argument** |
| `downs1993tep`, `rieth2017tepdata` | Tennessee Eastman |

Pepper has no citation because the recordings come from the XAI course laboratory. Say so
explicitly; an uncited dataset otherwise looks like an omission.

---

## The gap statement this chapter has to earn

By §2.7 the reader should accept all four of these, each supported above:

1. Causal discovery divides into families that make genuinely different assumptions, so
   comparing one algorithm to one baseline tells you little. *(§2.1)*
2. Time series break the assumptions of most of them, and only some algorithms address it.
   *(§2.2)*
3. Deep reconstruction models are the default, and they are strong. *(§2.3)*
4. Existing comparisons are constructed so that positive results cannot be interpreted.
   *(§2.4, §2.5)*

Then the gap is one sentence: **a comparison across algorithm families, on multiple domains,
under one detector and one label-blind threshold, against tuned baselines and a trivial
control.** That is what you built, and unusually, it is a gap you can claim without
overstating — because the answer it produced was mostly negative for causal discovery, which
is not what someone constructing a comparison to win would have reported.
