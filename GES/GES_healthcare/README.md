# GES Healthcare — Anomaly Detection on PTB-XL ECG (12-Lead)

Applies the **Greedy Equivalence Search (GES)** causal discovery algorithm to the **PTB-XL**
12-lead ECG dataset. The pipeline learns causal parent–child relationships between ECG leads
from fault-free data, then flags deviations via coefficient-drift scoring tracked online with
Recursive Least Squares.

## Dataset

| Property | Value |
|---|---|
| Source | PTB-XL (PhysioNet) — see Data Provenance below |
| Signal | 12-lead ECG: I, II, III, AVR, AVL, AVF, V1–V6 |
| Fault-free training | 50,000 × 12 (`healthcare_normal_train.pkl`) |
| Fault-free full | 20,000 × 12 (`healthcare_normal_full.pkl`) |
| Attacks | 20 segments × 1,000 × 12 (`healthcare_attacks.pkl`, a dict) |

## Install & Run

```powershell
python -m pip install -r requirements.txt
python anomaly_detection_ges.py
python anomaly_detection_ges_polynomial.py
python anomaly_detection_ges_rbf.py
```

## Final Results

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| GES | Linear / Ridge | 0.718 | 0.826 | 0.768 | 0.735 | 0.811 | 0.840 | Selected |
| GES | Polynomial degree 3 | 0.760 | 0.748 | 0.754 | 0.740 | 0.787 | 0.803 | Comparison |
| GES | RBF + Ridge | 0.600 | 0.942 | 0.733 | 0.635 | 0.762 | 0.807 | Comparison |

Linear / Ridge is selected, leading on every metric except recall.

All three variants select their own operating point (thresholds 0.0128 / 0.0222 / 0.0283) and
produce distinct confusion matrices. An earlier revision reported identical matrices for
Polynomial and RBF: that was an artifact of scoring the RLS warm-up window, where the residual
is pinned to zero by construction, which collapsed the threshold to 0.0. Excluding the warm-up
resolved it and lifted every variant substantially (Linear AUC-ROC 0.697 → 0.811).

**Comparison with PCMCI on byte-identical data** (both read the same restored pickles): GES
leads on Linear (AUC 0.811 vs 0.806) and Polynomial (0.787 vs 0.726); PCMCI leads on RBF
(0.817 vs 0.762). The two algorithms are broadly comparable on ECG — unlike robotics, where
PCMCI is well ahead.

AUC-ROC and AUC-PR come from continuous scores before thresholding; Precision, Recall, F1 and
Accuracy from thresholded detections at a class-balanced operating point.

### These supersede all previously published numbers

An earlier version of this file was marked "COMPLETE & LOCKED" and reported 100% precision with
AUC-ROC 0.9980. **Those figures could not have been produced from this folder:** all three data
pickles were 0 bytes, so the pipeline had never been runnable here. The scripts additionally
pointed `NORMAL_FILE` at a nonexistent CSV and mis-handled the attack dictionary, so they would
have raised an exception even with data present. Do not cite the old numbers.

## Explainability

Attribution previously ranked leads by raw residual L2 norm, so **V2** — by far the
highest-amplitude lead (std 295 vs AVL's 97) — dominated nearly every attack irrespective of
which lead was perturbed. After normalizing by each lead's own fault-free baseline residual as
a length-independent RMS ratio, **V2 no longer ranks first in any of the 60 attributions**
(20 attacks × 3 variants):

| Variant | Distinct top-ranked leads |
|---|---|
| Linear / Ridge | I, II, III, AVL, AVF, V3, V4, V6 (8) |
| Polynomial degree 3 | I, II, III, AVL, AVF, V1, V3, V4, V6 (9) |
| RBF + Ridge | I, III, AVL, AVF, V3, V4, V6 (7) |

**Caveat.** PTB-XL attacks are labelled only `Attack_1`…`Attack_20` with no published per-lead
ground truth, so these attributions are attack-discriminative but cannot be verified against a
known target — unlike the industrial (TEP) domain, where the fault table permits a real
coherence check.

## Data Provenance

The pickles are **derived data** extracted from the PTB-XL database:

> Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W.,
> Schaeffter, T. *PTB-XL, a large publicly available electrocardiography dataset.*
> Scientific Data 7, 154 (2020). https://doi.org/10.1038/s41597-020-0495-6
> PhysioNet: https://physionet.org/content/ptb-xl/

They were restored from `PCMCI/PCMCI_healthcare/Healthcare dataset/`, which holds the same
extracts. GES and PCMCI healthcare therefore consume byte-identical inputs, making the
cross-algorithm comparison in this domain legitimate.

See `FINAL_RESULTS.md` for the ten methodological corrections applied, including strict DAG
enforcement (34 of 72 raw adjacency entries retained as genuinely directed) and a note on why
predictor standardization was tested rather than assumed.
