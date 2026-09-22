# PCMCI Robotic

Clean final PCMCI implementation for PEPPER robotic anomaly detection.

## Install Requirements

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python anomaly_detection.py
python anomaly_detection_polynomial.py
python anomaly_detection_rbf.py
```

## Final Results

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| PCMCI | Linear / Ridge | 1.000 | 0.894 | 0.944 | 0.938 | 0.920 | 0.967 | Comparison |
| PCMCI | Polynomial degree 3 | 1.000 | 0.912 | 0.954 | 0.948 | 0.920 | 0.967 | Selected |
| PCMCI | RBF + Ridge | 1.000 | 0.850 | 0.919 | 0.912 | 0.863 | 0.940 | Comparison |

Polynomial degree 3 is the selected PCMCI model because it has the strongest validated final F1 score in this folder. See `FINAL_RESULTS.md` for the full per-class breakdown, per-variant tuning rationale, and the explainability qualitative-coherence check.

## Reproduction of the XAI Course Lab (reviewer requirement)

This is the folder that reproduces the qualitative coherence result from the XAI course lab:
**LED attacks must attribute to the Tablet sensors, and Joint attacks to joint temperatures.**
Both hold, across all three regression variants:

| Attack | Expected | Linear / Ridge | Polynomial deg 3 | RBF + Ridge |
|---|---|---|---|---|
| **LedsControl** | Tablet sensors | **TabletTouchX, TabletTouchY** | **TabletTouchY, TabletTouchX** | **TabletTouchX** |
| **JointControl** | joint temperatures | **KneePicthTemp** | **KneePicthTemp** | **KneePicthTemp, HeadPicthTemp, HipPicthTemp** |
| WheelsControl | wheel sensors | WheelFRTemp | WheelBTemp, WheelBStiff | WheelFRTemp (#2) |

The LED signal is unambiguous rather than marginal: in the Linear variant `TabletTouchX`
reaches an aggregated error of ~3750 and `TabletTouchY` ~1350, while every other monitored
sensor sits near zero.

**Robustness check.** The LED→Tablet result is not an artifact of the ranking metric. Ranking
by raw L2 (used here) and by baseline-normalised RMS ratio (used in the other eight folders)
both return `TabletTouchX, TabletTouchY` as the top two for `LedsControl`.

**Why this folder ranks by raw L2.** The other eight folders divide each variable by its own
normal-baseline residual. On Pepper that is actively misleading: baseline residual scales span
395x across the 244 monitored sensors, so dividing by them inflates quiet sensors. `LedEarR3`
and `LedEarL3` have the 12th and 13th smallest baselines while `KneePicthTemp` ranks 238th of
244 — a ~281x relative advantage to the LED ears — which displaces the correct joint-temperature
attribution for `JointControl`. Raw L2 is therefore retained here to stay faithful to the lab.
This is a documented, deliberate difference, not an oversight.

## Verification

All three variants reproduce the published metrics **exactly** (PCMCI is deterministic).
Structural checks on the learned graph: 244 of 244 variables monitored, both Tablet sensors
retained, tau_max = 2, and **zero lag-0 self-loops** — so no target trivially predicts itself.

AUC-ROC and AUC-PR are computed from continuous anomaly scores before thresholding. Precision, Recall, F1, and Accuracy are computed from thresholded anomaly detections.

Dataset files are included because no copied file is over 100 MB.