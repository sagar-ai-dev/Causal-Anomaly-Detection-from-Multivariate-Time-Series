# CAM-UV Causal Anomaly Detection — Industrial (Tennessee Eastman Process)

Authentic **CAM-UV** (Causal Additive Models with Unobserved variables; Maeda & Shimizu,
UAI 2021) applied to the TEP benchmark: a simulated chemical plant with 52 sensors governed
by deterministic control loops.

The engine is the paper authors' own reference implementation, shipped in the `lingam`
package and loaded through [`camuv_engine.py`](../camuv_engine.py). Nothing about the
algorithm is reimplemented here.

## Methodology

Normal operation is modelled as a causal graph, then process faults are detected as
*coefficient drift* in the causal regression equations, tracked online by Recursive Least
Squares. Fault-free and fault data are scored as fresh, equal-length runs over the same
post-onset window, so the comparison is not confounded by run length or by pre-fault samples.

## Install & Run

```powershell
python -m pip install -r requirements.txt
python anomaly_detection.py             # builds and caches the causal graph
python anomaly_detection_polynomial.py  # reuses the cache
python anomaly_detection_rbf.py         # reuses the cache
```

## Final Results

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| CAM-UV | RBF + Ridge | 1.000 | 0.764 | **0.866** | 0.775 | **0.891** | **0.994** | **Selected** |
| CAM-UV | Linear / Ridge | 1.000 | 0.720 | 0.837 | 0.733 | 0.886 | 0.994 | Comparison |
| CAM-UV | Polynomial degree 3 | 1.000 | 0.675 | 0.806 | 0.690 | 0.823 | 0.990 | Comparison |

RBF + Ridge is selected, leading on every metric.

**Precision is exactly 1.000 for all three variants — zero false positives across the entire
fault-free control run.** The pipeline is conservative rather than weak: recall of 0.67–0.76
is dominated by faults undetectable by construction, not by threshold miscalibration.

### Recovered structure

| Property | Value |
|---|---|
| Variables | 52 |
| Discovery samples | 35 (protocol-fixed) |
| Lag-1 edges | 63 |
| Graph density | **2.4%** |
| Confounded pairs | 14 of 1,326 (**1.1%**) |
| Monitored targets | 39 of 52 |
| Parents per target | mean 1.6, max 5 |

This is the opposite extreme from robotics (70.6% dense) and healthcare (65.9%). TEP's sensors
are coupled through explicit control loops rather than shared latent dynamics, so observed
parents genuinely explain most of the structure and CAM-UV's residual independence test is not
saturated.

### Sample-size caveat

The protocol fixes `subsample = 10`, leaving only 35 training rows for 52 variables at discovery
time. `subsample` cannot be changed independently — it also drives the evaluation window — so
this was tested with a discovery-only sweep (`tep_power_check.py`):

| Discovery samples | Parent edges | Confounded pairs | Density | Runtime |
|---:|---:|---:|---:|---:|
| 35 *(protocol)* | 35 | 14 | 2.4% | 16 s |
| 70 | 35 | 28 | 3.4% | 25 s |
| 175 | 59 | 38 | 5.1% | 85 s |
| 350 | 115 | 90 | 11.1% | 635 s |

**The 35-sample graph is genuinely power-limited** — 10× the data recovers 3.3× the parent
edges and 6.4× the confounded pairs — so the exact edge counts above should not be read as
TEP's complete causal structure. **The domain conclusion still holds:** even at 350 samples the
graph reaches only 11.1% density, categorically sparser than robotics (70.6%) or healthcare
(65.9%) at comparable sample sizes (both 400). TEP's sparsity is real, not an artifact.

## Independent validation: the known-hard TEP faults

| Fault | CAM-UV recall | Literature |
|---|---:|---|
| 3 | 0.000 | canonically near-undetectable |
| 9 | 0.111 | canonically near-undetectable |
| 15 | 0.111 | canonically near-undetectable |
| 1, 2, 5, 6, 7, 8, 12, 13, 14, 16, 17, 18 | 1.000 | readily detectable |

Faults 3, 9 and 15 produce no sustained shift in the measured variables. PCMCI and GES
reproduce this pattern independently, and CAM-UV does so here. Twelve of twenty faults are
detected at perfect recall.

**One CAM-UV-specific miss:** fault 4 (reactor cooling water inlet temperature step) has recall
0.000 despite *not* being canonically hard — PCMCI and GES both detect it. With 13 of 52
variables unmonitored under the sparse graph, fault 4's signature apparently lands among them.
This is the cost of the sparsity that otherwise makes CAM-UV workable here.

## Ablation: does the latent structure help?

Holding `alpha` at 0.05 and toggling only `CONFOUNDER_MODE`:

| | Config A `integrate` | Config B `observed_only` | Change |
|---|---:|---:|---:|
| Total lag-1 edges | 63 | 35 | **-44%** |
| Monitored targets | 39 of 52 | 30 of 52 | -23% |

| Variant | delta F1 | **delta AUC-ROC** |
|---|---:|---:|
| Linear / Ridge | +0.003 | **-0.003** |
| Polynomial degree 3 | -0.015 | **-0.023** |
| RBF + Ridge | -0.001 | **-0.039** |
| **Mean** | **-0.004** | **-0.022** |

**Removing 44% of the graph changes mean AUC-ROC by -0.022** — inside CAM-UV's ±0.05 run-to-run
noise band. Detection runs on the 35 observed-parent edges from TEP's control loops, not on
latent structure. Unlike healthcare (mixed-sign deltas = noise), all three signs are negative
here, and RBF is the only case in the thesis where withholding confounders broke perfect
precision (FP 0 → 9). Weak evidence of a small real effect, still inside the noise floor.

**This is why CAM-UV is competitive on TEP:** it performs well by behaving like an ordinary
causal discovery method, with its distinguishing feature contributing almost nothing. See
`FINAL_RESULTS.md` for full tables.

## Comparison with PCMCI and GES

Same protocol (simulation run 1, fault onset at sample 160, class-balanced thresholds), so this
is a fair comparison on rate-based metrics (the discovery engine is not the only difference — see `FINAL_RESULTS.md`):

| Algorithm | Selected variant | F1 | AUC-ROC |
|---|---|---:|---:|
| PCMCI | Polynomial degree 3 | **0.879** | **0.915** |
| GES | RBF + Ridge | 0.877 | 0.897 |
| CAM-UV | RBF + Ridge | 0.866 | 0.891 |

**All three are within 0.013 F1 and 0.024 AUC-ROC.** This is the only domain in the thesis
where CAM-UV is genuinely competitive — contrast robotics, where the same algorithm on the
same pipeline flagged 70.5% of pairs as confounded and became unusable. CAM-UV's viability is
domain-dependent, and the deciding factor is whether sensor coupling is explained by observed
variables or by shared latent dynamics.

## Data

All 23 TEP pickles in this folder were **0 bytes** before this rebuild. They were restored from
`GES/GES_industrial/TEP/`, which holds the `simulationRun == 1` subset the protocol actually
consumes (17.3 MB rather than 4.16 GB).

The `faultNumber` label-leak guard is retained and verified — TEP frames prepend three metadata
columns to the 52 sensors, and `faultNumber` is 0 in fault-free data and N in fault N data.

## Files in this folder

`tep_power_check.py` is a provenance utility, not part of the detection pipeline. No
`anomaly_detection` script imports it. It regenerates the discovery sample-size sweep quoted
above (35 -> 350 samples), and is kept so that table has a reproducible source. Running it
rebuilds the sweep; it does not touch the evaluation protocol or any shipped result.

## Provenance

This folder previously contained scripts that imported `tigramite` and ran **PCMCI** while
being presented as CAM-UV. Any industrial CAM-UV metrics published before this rebuild describe
PCMCI and must not be cited as CAM-UV results. Stale PCMCI-era figures dated 2026-08-19 have
been removed.

See [`FINAL_RESULTS.md`](FINAL_RESULTS.md) for the full analysis and protocol.
