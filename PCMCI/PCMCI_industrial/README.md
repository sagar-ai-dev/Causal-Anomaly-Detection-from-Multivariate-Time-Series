# PCMCI Industrial

Clean final PCMCI implementation for TEP industrial anomaly detection.

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
| PCMCI | Polynomial degree 3 | 1.000 | 0.785 | 0.879 | 0.795 | 0.915 | 0.996 | Selected |
| PCMCI | Linear / OLS | 0.997 | 0.750 | 0.856 | 0.760 | 0.873 | 0.993 | Comparison |
| PCMCI | RBF + Ridge | 1.000 | 0.733 | 0.846 | 0.746 | 0.850 | 0.992 | Comparison |

Polynomial degree 3 is the selected PCMCI model for the industrial domain (highest AUC-ROC,
AUC-PR, F1 and accuracy). All three variants reach genuine operating points with near-zero
false alarms (0-32 false positives out of 796 fault-free samples).

Recall of 0.73-0.78 is expected rather than a weakness: TEP faults 3, 9 and 15 are documented
in the fault-detection literature as near-undetectable from these measurements, and this
pipeline independently reproduces that pattern (recall 0.00-0.20 on those three, 0.96-1.00 on
the readily-detectable faults). Most of the false negatives come from that hard trio.

Evaluation uses simulation run 1 throughout, with the held-out fault-free testing run as the
control group, and scores both classes from sample 160 onward because TEP does not inject the
fault until then. Thresholds are chosen with class-balanced weighting; see `FINAL_RESULTS.md`.

**These results supersede everything produced before this revision.** The previous pipeline was
feeding `faultNumber` -- the ground-truth label -- into the model as an input feature, and
reading every other sensor from a wrong, 3-column-shifted index. See `FINAL_RESULTS.md` for
that and seven other corrections.

AUC-ROC and AUC-PR are computed from continuous anomaly scores before thresholding. Precision, Recall, F1, and Accuracy are computed from thresholded anomaly detections.

Large dataset files are tracked with Git LFS where needed.
