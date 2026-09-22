# CAM-UV Causal Anomaly Detection — Robotics (Pepper)

Authentic **CAM-UV** (Causal Additive Models with Unobserved variables; Maeda & Shimizu,
UAI 2021) applied to the Pepper humanoid robot sensor stream.

The engine is the paper authors' own reference implementation, shipped in the `lingam`
package and loaded through [`camuv_engine.py`](../camuv_engine.py). Nothing about the
algorithm is reimplemented here.

## Methodology

Normal operation is modelled as a causal graph, then attacks are detected as *coefficient
drift* in the causal regression equations, tracked online by Recursive Least Squares. Both
classes are scored as fresh, equal-length windows, so the comparison is not confounded by
run length.

CAM-UV differs from PCMCI and GES in returning **two** objects: `P` (observed parents) and
`U` (pairs sharing an *unobserved* confounder). How `U` is used is the central design
decision, exposed as `CONFOUNDER_MODE`; this folder ships `integrate`, which adds each
confounded pair as a symmetric pair of lag-1 edges.

## Configuration

| Constant | Value |
| :--- | ---: |
| `TOP_VARIANCE_SENSORS` | 35 |
| `CAMUV_ALPHA` | 0.05 |
| `CAMUV_NUM_EXPLANATORY` | 2 |
| `CONFOUNDER_MODE` | `integrate` |
| `RLS_LAMBDA` | 0.995 |
| `DETECTION_THRESHOLD_MULTIPLIER` | 1.2 |

Sensor selection is by **highest standard deviation on the training split at full temporal
resolution**, with no sensor selected by name. Root README section 7.2 justifies the cap at
35: the scree-curve elbow falls at K = 32, K = 35 retains 94.55% of total variance, and a
160-variable run did not converge in 66 minutes of CPU.

## Install and run

```powershell
python -m pip install -r requirements.txt
python anomaly_detection.py             # builds and caches the causal graph
python anomaly_detection_polynomial.py  # reuses the cache
python anomaly_detection_rbf.py         # reuses the cache
```

## Results

Threshold is the 95th percentile of the fault-free score distribution, computed from normal
data only and never seeing an attack label.

| Variant | Precision | Recall | F1 | AUC-ROC | AUC-PR |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Linear / Ridge | 0.9051 | 0.3217 | 0.4747 | 0.6497 | 0.7989 |
| Polynomial degree 3 | 0.9213 | 0.3952 | 0.5531 | 0.6888 | 0.8205 |
| **RBF + Ridge** | 0.9528 | 0.6807 | **0.7941** | **0.7818** | **0.8909** |

**This is the weakest of the three algorithms on this domain.** PCMCI reaches AUC-ROC 0.8752
and GES 0.5906 on the same data. Precision is high across all three variants and recall is
low, which is the signature of a detector that fires rarely but is usually right when it does.

## The graph is entirely latent-confounder structure

| | |
| :--- | ---: |
| Variables | 35 |
| Lag-1 edges | 960 |
| Confounded pairs in `U` | 480 |
| **Observed-parent edges from `P`** | **0** |
| Pair saturation, share of all C(35,2) pairs flagged | **80.7%** |
| Graph density | 0.807 |

**CAM-UV recovered no directed structure on Pepper at all.** The adjacency matrix is exactly
symmetric — 480 pairs times 2 directions accounts for all 960 edges — so every relationship
among the 35 sensors was attributed to latent confounding and none to an observed parent. With
80.7% of all possible pairs flagged, the result is a near-complete graph rather than a
discovery, and the detector downstream inherits no useful sparsity: each target is regressed on
nearly every other variable, leaving coefficient drift nothing specific to be about.

This is the mechanism behind the weak detection above, and it is the robotics end of a pattern
that holds across all three domains — see [`confounder_structure.py`](../confounder_structure.py)
and root README section 6. On TEP only 1.1% of pairs are flagged and CAM-UV is competitive
(its RBF variant reaches 0.8922); on ECG 59.1% are flagged but the *choice* of pairs is physiologically correct.
Pepper is the case where nothing indicates the output is meaningful.

CAM-UV decides confounding by HSIC residual independence. Pepper's sensors are strongly coupled
through shared mechanical dynamics, and genuine mechanical collinearity is not distinguishable
from a latent common cause to such a test.

## Explainability

The XAI course lab check is **LED attacks must attribute to the Tablet sensors**, and it passes
in all three variants:

| Attack | Linear / Ridge | Polynomial degree 3 | RBF + Ridge |
| :--- | :--- | :--- | :--- |
| **LedsControl** | **TabletTouchX** | **TabletTouchX** | **TabletTouchX** |
| JointControl | HeadPicthTemp | **KneePicthTemp** | **KneePicthTemp** |
| WheelsControl | WheelFRTemp | WheelFRStiff | KneePicthTemp |

`TabletTouchX` is rank-1 on LedsControl in 3 of 3 variants, with `TabletTouchY` rank-2 or
rank-3 in all three. The joint-temperature check passes under polynomial and RBF and misses
under linear, which returns a different temperature channel.

These attributions are reached without any name-based selection: `TabletTouchX` is variance
rank 2 of 249 and `TabletTouchY` rank 7, so both enter the 35-sensor cap on merit.

## Reproducibility caveat

**CAM-UV is not exactly reproducible run to run.** Repeated runs with identical alpha, data and
`np.random.seed(42)` differ, because the variance originates inside CAM-UV's GAM fitting and
HSIC sampling. PCMCI and GES both reproduce exactly. Treat these figures as single-run
measurements rather than constants; root README section 2.5 reports block-bootstrap intervals
on the real per-sample scores.

## Output naming

```
linear_ridge_top_anomalous_variables.png
polynomial_degree_3_top_anomalous_variables.png
rbf_+_ridge_top_anomalous_variables.png
explainability_top_features.csv
metrics_summary.csv
scores_<variant>.npz
```

Switching `CONFOUNDER_MODE` to `observed_only` writes the figure and attribution names under a
`configB_` prefix, so an ablation run cannot overwrite the shipped outputs. Graph caches are
tagged by mode and alpha (`pepper_normal_camuv_{mode}_a{alpha}.npz`), so changing either cannot
silently reuse a stale graph.

## Superseded results

An earlier version of this file reported Configuration A at F1 0.8331 with AUC-ROC 0.7048, and
a graph of 2,417 edges, 1,207 confounded pairs and 70.6% density over **59 variables**,
alongside a Configuration A/B comparison and an alpha sweep. Those figures came from a
superseded pipeline and cannot be produced from this folder: the variable count is now capped
at 35, for which 1,207 confounded pairs is arithmetically impossible, since C(35,2) = 595.

That version also selected sensors by force-including any name matching `Tablet`, `Temp`,
`Stiff` or `Wheel`, which would have made the LED-to-Tablet result true by construction. The
force-include existed to work around a bug: variance was measured *after* subsampling by 10,
which aliases the transient tablet channels and pushed them to ranks 145 and 142. Measuring at
full temporal resolution puts them at ranks 2 and 7, so the tag list was unnecessary and has
been removed. The current attributions are discovered, not admitted.

See [`FINAL_RESULTS.md`](FINAL_RESULTS.md) for the full analysis.
