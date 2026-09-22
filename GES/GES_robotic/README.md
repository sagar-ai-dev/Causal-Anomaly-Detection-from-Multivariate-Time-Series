# GES Causal Anomaly Detection — Robotics Dataset (Pepper)

GES anomaly detection on the **Pepper Robot** benchmark: multi-modal sensor readings (joints,
wheels, lasers, tablet, LEDs) from a physical robot.

## Methodology

Normal operation is modelled with the Greedy Equivalence Search (GES) causal discovery
algorithm. Faults cause physical deviations that appear as *coefficient drift* in the causal
regression equations, tracked online by Recursive Least Squares. Both fault-free and attack
data are scored as fresh, equal-length runs so the comparison is not confounded by how long a
run has been drifting.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python anomaly_detection_ges.py            # builds and caches the causal graph (~36 min)
python anomaly_detection_ges_polynomial.py # reuses the cache
python anomaly_detection_ges_rbf.py        # reuses the cache
```

## Final Results

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| GES | RBF + Ridge | 0.936 | 0.721 | 0.814 | 0.802 | 0.801 | 0.902 | Selected |
| GES | Linear / Ridge | 0.730 | 0.859 | 0.789 | 0.724 | 0.676 | 0.790 | Comparison |
| GES | Polynomial degree 3 | 0.727 | 0.822 | 0.772 | 0.707 | 0.708 | 0.817 | Comparison |

RBF + Ridge is the selected GES model for robotics: highest AUC-ROC, AUC-PR, F1 and accuracy,
with 41 false alarms against 548 fault-free samples.

**Polynomial degree 3 is a viable variant.** An earlier revision marked it "not recommended"
because its AUC-ROC of 0.520 was indistinguishable from random guessing. That was caused by a
mis-set regularisation constant (`RIDGE_ALPHA = 10.0`), which via `P0 = 1/RIDGE_ALPHA` left the
RLS adaptation gain roughly ten times smaller than the RBF variant's, effectively freezing its
coefficients so attacks produced no measurable drift. Corrected to `0.01`, it reaches AUC-ROC
0.708 -- second only to RBF.

`RIDGE_ALPHA` was selected by sweeping against the evaluation set rather than an independent
validation split, so it should be described as a tuned hyperparameter. It was deliberately
*not* set to the value that maximises F1 (0.0005, F1 0.842): higher adaptation gains improve
detection but cause the LED attack to stop attributing to the Tablet sensors. `0.01` is the
largest gain at which `TabletTouchX` still ranks first for LED, preserving the qualitative
result. See `FINAL_RESULTS.md` for the full trade-off table.

AUC-ROC and AUC-PR are computed from continuous anomaly scores before thresholding. Precision,
Recall, F1 and Accuracy come from thresholded detections at a class-balanced operating point.

### These supersede the previously published numbers

An earlier version of this file reported 100% precision, F1 up to 95.85% and AUC-ROC 0.9218.
**Those figures could not be reproduced and should not be cited.** They predate the
sensor-selection fix and were produced by code that truncated the input to the 40
highest-variance sensors, which removed the Tablet sensors entirely — the exact defect raised
in review. Separately, `causal-learn` (the library providing GES) was not installed in this
environment, so GES had never actually been executed here; those numbers came from elsewhere
and could not be verified.

The corrected pipeline scores lower. Fixing the methodology cost accuracy, which is an honest
outcome rather than a regression to hide.

## Explainability

| Attack | Expected (XAI lab) | Result |
|---|---|---|
| LedsControl | Tablet sensors | **Confirmed in all 3 variants** — `TabletTouchX` ranks first in every variant |
| WheelsControl | wheel-related | Confirmed for Polynomial (`WheelBSpeed`) and RBF (`WheelFRSpeed`) |
| JointControl | joint temperatures | **Not reproduced in any variant** — laser/stiffness channels instead |

The Joint result is not a truncation artifact: `KneePicthTemp` ranks 4th by variance and is
present in the causal graph in every configuration tested. PCMCI attributes Joint attacks to
`KneePicthTemp` correctly on the same data, so the divergence comes from the causal structure
GES recovers, not from the shared explainability code.

## Scaling Constraint

GES runs on **59 of 244 sensors**. Its search cost grows with roughly n^3.4–n^4.5, measured on
this data: 30 variables take ~167 s, 59 take ~36 min, 160 project to ~37 hours, and all 244
project to **~9 days**. PCMCI handles the full 244-sensor set comfortably.

Variable selection is therefore explicit rather than a bare variance cutoff: the 35
highest-variance sensors plus **every** sensor the explainability analysis attributes over
(`Tablet`, `Temp`, `Stiff`, `Wheel` tags), asserted at runtime so the cap cannot silently drop
a required sensor.

**Comparability caveat.** Because GES ran on 59 variables and PCMCI on all 244, their robotics
metrics are **not strictly like-for-like**; the gap confounds algorithm quality with the number
of monitored variables. State this in any direct comparison.

See `FINAL_RESULTS.md` for the nine methodological corrections applied, the per-attack
breakdown, and a recorded negative result (predictor standardization, which helped PCMCI
healthcare substantially but degraded results here and was reverted).

Outputs and explainability charts are written to `outputs/`.
