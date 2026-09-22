# Why these three algorithms

Zheng et al. (2024) is the paper the algorithm choice should be argued from, because its
Table 1 is a taxonomy, not just a feature list. It sorts causal discovery into four
families, and a comparison that draws one representative from each family says something
general; a comparison that draws three from the same family only ranks siblings.

## Table 1, and where this thesis sits

| Family | Methods in causal-learn 0.1.3.8 | This thesis |
|---|---|---|
| Constraint-based | PC, MV-PC, FCI, CD-NOD | **PCMCI** — see below |
| Score-based | **GES**, A*, Dynamic Programming, GRaSP | **GES** ✓ |
| Function-based (FCM) | ANM, PNL, LiNGAM, DirectLiNGAM, VAR-LiNGAM, RCD, **CAM-UV** | **CAM-UV** ✓ |
| Causal representation learning | GIN | not covered |

Three families out of four, one representative each. That is the design, and it is worth
stating in exactly those terms, because it explains why the three algorithms disagree: they
disagree by construction. GES optimises a score over Markov equivalence classes and returns
a CPDAG, so it leaves some edges undirected. CAM-UV assumes an additive functional form with
non-Gaussian noise and can therefore orient edges GES cannot. PCMCI conditions on the past
and returns lagged edges neither of the others can express.

## The one that is not in Table 1 — now measured, not argued

**PCMCI comes from tigramite, not causal-learn.** Say this plainly in Chapter 4, before an
examiner who knows the paper notices it. The good news is that it no longer has to be
defended on theory alone.

`tools/pc_vs_pcmci_ablation.py` builds a system where the correct answer is known and is
*no edges at all*: eight independent AR(1) processes,

    x_i(t) = phi * x_i(t-1) + e_i(t),   e_i independent across i

Every variable is autocorrelated; none causes any other. So every cross-variable edge either
algorithm reports is a false positive by construction, and the error rate can be read off
rather than inferred. Sweeping phi walks the data from i.i.d., where the PC assumption holds
exactly, to strongly persistent, where it fails. Both run at alpha = 0.05, the pipeline value.

| phi | PC | PCMCI (FDR) |
|---:|---:|---:|
| 0.00 | **0.0429** | 0.0500 |
| 0.30 | 0.0518 | 0.0589 |
| 0.50 | 0.1393 | 0.0589 |
| 0.70 | 0.2393 | 0.0554 |
| 0.90 | 0.4321 | 0.0500 |
| 0.95 | **0.4982** | 0.0536 |

20 seeds per row, 8 variables, 1000 samples, tau_max 3. Full data in `tools/pc_vs_pcmci.csv`.

Read the first row and the last. At phi = 0 PC is calibrated almost perfectly — 0.0429 against
a nominal 0.05 — which is the control that shows the test itself is sound and the implementation
is not being slandered. At phi = 0.95 the same algorithm on the same kind of data links **half
of all unrelated variable pairs**. PCMCI stays between 0.050 and 0.059 across the entire sweep.

That is the substitution argued from measurement: on data with the autocorrelation every domain
in this thesis has, and no causal structure whatsoever, PC's false positive rate inflates
roughly twelvefold and PCMCI's does not move.

**One caveat to state honestly, because it is a trap.** PCMCI tests every ordered pair at every
lag — 2(tau_max + 1) = 8 tests per pair at tau_max = 3 — so declaring a pair linked whenever any
of those fires accumulates their error rates to 1 − 0.95^8 = 0.34 before the data is even seen.
The uncorrected column in the CSV sits at 0.28–0.29 for exactly that reason, and reporting it
as PCMCI's error rate would be a counting mistake, not a finding. The table above is
FDR-corrected, which is what tigramite recommends and what makes the comparison against PC's
one-decision-per-pair output mean the same thing on both sides.

CD-NOD is the nearest alternative in Table 1, but it targets distribution shift across regimes,
not lagged causation, so it does not fill the constraint-based slot either.

## A second finding: the healthcare graph is nearly complete

The same ablation measured graph density on the real data, under the rule the pipeline actually
applies — significance **and** an effect-size floor:

    normal_matrix = val_matrix * (p_matrix < ALPHA)
                    * (abs(val_matrix) > CAUSAL_STRENGTH_MULTIPLIER * mean(abs(val_matrix)))

| Domain | Variables | Multiplier | p < alpha only | Pipeline rule |
|---|---:|---:|---:|---:|
| Healthcare | 12 | 1.0 | 66 / 66 | **63 / 66** |
| Robotics | 244 | 1.0 | 18958 / 29646 | 18958 / 29646 |
| Industrial | 52 | 0.0 | 511 / 1326 | 511 / 1326 |

Two things fall out, both worth a paragraph in the discussion.

**The effect-size filter is almost inert.** It removes three pairs on healthcare and none
anywhere else, and on industrial the multiplier is 0.0 so it cannot remove anything at all.
Graph density is set by significance, not by strength. The multiplier is now pinned by audit
check 16 so the three variants within a folder can never sparsify differently.

**The healthcare graph links 63 of 66 possible pairs — 95%.** Nearly every variable is a parent
of nearly every other, so the regressions have almost no structure left to exploit and the
detector is close to regressing each lead on all the others. This is a candidate explanation
for PCMCI trailing GES on healthcare, 0.6181 against 0.8114, and it fits the independent-lead
ablation: dropping the four algebraically dependent leads lifted the polynomial variant from
0.5075 to 0.6355. A saturated graph is not evidence of rich physiology; it is evidence the
significance threshold is not discriminating.

## What each algorithm actually imports

Verified from the source, because the citation has to match the code that ran.

| Algorithm | Import | Cite |
|---|---|---|
| PCMCI | `tigramite.pcmci`, `tigramite.independence_tests.parcorr` | `runge2019pcmci`, `runge2023tigramite` |
| GES | `causallearn.search.ScoreBased.GES` | `chickering2002ges`, `zheng2024causallearn` |
| CAM-UV | `lingam.camuv`, via `camuv_engine.py` | `maeda2021camuv`, `ikeuchi2023lingam` |
| CAM-UV cross-check | `causallearn.search.FCMBased.lingam.CAMUV` | `zheng2024causallearn` |

Two details worth a sentence each in Chapter 5.

**CAM-UV runs from `lingam`, not causal-learn.** `camuv_engine.py` loads `hsic`, `utils` and
`camuv` directly into a minimal namespace, because `lingam/__init__.py` eagerly imports every
algorithm the package ships and fails on unrelated missing dependencies. So the thesis must
cite Ikeuchi et al. (2023) for the implementation that produced the results, alongside Maeda
and Shimizu (2021) for the method.

**Both implementations are run in the confounder ablation.** `confounder_ablation.py` executes
every configuration through the `lingam` and causal-learn implementations of the same
algorithm, so a disagreement would indicate a wrapper bug rather than a property of the
method. This is a real methodological strength and it is currently invisible in the write-up.

## Citation corrections made

| Entry | Was | Now |
|---|---|---|
| `zheng2024causallearn` | volume only, flagged unverified | 25(60):1--8, 2024, with URL |
| `ikeuchi2023lingam` | author **Ide, Mauricio** | author **Ide, Mayumi** |
| `gretton2005hsic` | `@article` | `@inproceedings` — ALT is a conference |

Three `CHECK` notes remain, on fields no source to hand settles: the pages and issue of
`runge2023causalinference`, the citation tigramite itself requests, and the ALT page range.
