# PCMCI Healthcare

Clean final PCMCI implementation for ECG healthcare anomaly detection.

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
| PCMCI | RBF + Ridge | 0.745 | 0.885 | 0.809 | 0.761 | 0.817 | 0.846 | Selected |
| PCMCI | Linear / Ridge | 0.733 | 0.854 | 0.789 | 0.738 | 0.806 | 0.863 | Comparison |
| PCMCI | Polynomial degree 3 | 0.652 | 0.882 | 0.750 | 0.663 | 0.726 | 0.763 | Comparison |

RBF + Ridge is the selected PCMCI model for the healthcare domain, on the strength of the highest
AUC-ROC, F1 and accuracy. Linear / Ridge is close behind on AUC-ROC (0.806 vs 0.817) and leads on
AUC-PR (0.863 vs 0.846), so it is a defensible alternative selection.

All three variants reach a genuine operating point (true negatives 8721 / 5523 / 8887), so the
F1 values above reflect real detection behaviour.

AUC-ROC and AUC-PR are computed from continuous anomaly scores before thresholding. Precision,
Recall, F1, and Accuracy are computed from thresholded anomaly detections.

These numbers supersede a previously reported F1 of ~0.998 / AUC ~0.997, which was traced to two
stacked bugs (inverted ISNR scoring and an RLS drift confound) that cancelled each other out.
They also supersede an intermediate revision in which Linear and Polynomial scored 0.609 and 0.506
AUC-ROC; those figures reflected numerical-conditioning defects in those two implementations
(unstandardized predictors, and a discarded regression intercept) rather than genuine limits of
the linear or polynomial basis. See `FINAL_RESULTS.md` for all five methodological corrections,
per-variant tuning, recorded negative results, and the revised scientific discussion.

Dataset files are included because no copied file is over 100 MB.

## Data Provenance

The three pickle files in `Healthcare dataset/` are **derived data**, not raw recordings:

| File | Contents |
|---|---|
| `healthcare_normal_train.pkl` | 50,000 x 12 fault-free 12-lead ECG samples (offline fit + baseline) |
| `healthcare_normal_full.pkl` | 20,000 x 12 fault-free samples |
| `healthcare_attacks.pkl` | dict of 20 synthetic attack segments, 1,000 x 12 each |
| `healthcare_normal.npz` | cached PCMCI causal graph (regenerated automatically if deleted) |

They were extracted from the **PTB-XL** database, a large publicly available electrocardiography
dataset published on PhysioNet:

> Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W.,
> Schaeffter, T. *PTB-XL, a large publicly available electrocardiography dataset.*
> Scientific Data 7, 154 (2020). https://doi.org/10.1038/s41597-020-0495-6
>
> Available at PhysioNet: https://physionet.org/content/ptb-xl/

The raw PTB-XL source files (`records100/`, `records500/`, `ptbxl_database.csv`,
`scp_statements.csv`) are **not** included in this repository. They total roughly 3 GB, and the
pipeline never reads them directly -- it consumes only the derived pickles above. They were
removed to keep the repository lightweight and can be re-downloaded from the PhysioNet link
above if the derived files ever need to be regenerated.

The 12 columns correspond to the standard 12-lead ECG montage:
`I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6`.

**Note on the attack labels.** The 20 attack segments are identified only as `Attack_1` ...
`Attack_20`; there is no published per-lead ground truth for which lead each one perturbs. The
explainability attributions in `outputs/` are therefore reported as attack-discriminative but
cannot be independently verified against a known target, unlike the industrial (TEP) domain.
