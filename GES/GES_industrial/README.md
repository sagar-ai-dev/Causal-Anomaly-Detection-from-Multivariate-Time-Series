# GES Causal Anomaly Detection — Industrial Dataset (TEP)

GES anomaly detection on the **Tennessee Eastman Process (TEP)** benchmark: a simulated
chemical plant with 52 sensors governed by deterministic control loops.

## Methodology

Normal operation is modelled with the Greedy Equivalence Search (GES) causal discovery
algorithm. Process faults cause deviations that appear as *coefficient drift* in the causal
regression equations, tracked online by Recursive Least Squares. Fault-free and fault data are
scored as fresh, equal-length runs over the same post-onset window, so the comparison is not
confounded by run length or by pre-fault samples.

## Install & Run

```powershell
python -m pip install -r requirements.txt
python anomaly_detection_ges.py            # builds and caches the causal graph (~10 min)
python anomaly_detection_ges_polynomial.py # reuses the cache
python anomaly_detection_ges_rbf.py        # reuses the cache
```

## Final Results

| Algorithm | Variant | Precision | Recall | F1 | Accuracy | AUC-ROC | AUC-PR | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| GES | RBF + Ridge | 0.999 | 0.782 | 0.877 | 0.791 | 0.897 | 0.995 | Selected |
| GES | Linear / Ridge | 0.998 | 0.744 | 0.853 | 0.755 | 0.860 | 0.993 | Comparison |
| GES | Polynomial degree 3 | 1.000 | 0.726 | 0.841 | 0.739 | 0.891 | 0.994 | Comparison |

RBF + Ridge is selected, leading on AUC-ROC, AUC-PR, F1 and accuracy.

All three variants are **conservative rather than weak**: precision is 0.998–1.000, i.e. between
zero and two false alarms across the entire fault-free control. Recall of 0.73–0.78 is dominated
by three faults that are undetectable by construction.

### Independent validation

Per-fault recall reproduces a documented property of the benchmark without being tuned to:

| Fault | Recall | Literature |
|---|---|---|
| 3 | 0.032 | canonically near-undetectable |
| 9 | 0.159 | canonically near-undetectable |
| 15 | 0.079 | canonically near-undetectable |
| 1, 5, 6 | 1.000 | readily detectable |

Faults 3, 9 and 15 produce no sustained shift in the measured variables. Both GES and PCMCI
reproduce this pattern independently.

### These supersede all previously published numbers

An earlier version of this file reported F1 up to 99.50% with AUC-ROC 0.9576–0.9965. **Those
figures could not have been produced from this folder:** all 23 TEP data pickles were 0 bytes,
and `NORMAL_FILE` pointed at a CSV that does not exist. The pipeline additionally leaked
`faultNumber` — the ground-truth label — into the model as an input feature, which would inflate
any result it did produce. Do not cite the old numbers.

## Comparison with PCMCI

Both algorithms use the same protocol (run 1, fault onset at sample 160, class-balanced
thresholds), so this is a fair comparison:

| Variant | GES AUC-ROC | PCMCI AUC-ROC |
|---|---|---|
| Linear | 0.860 | 0.873 |
| Polynomial degree 3 | 0.891 | **0.915** |
| RBF + Ridge | **0.897** | 0.850 |

Closely matched — PCMCI leads on Polynomial, GES on RBF, Linear within noise. Unlike robotics,
where GES runs on a reduced sensor set and cannot be compared like-for-like, this domain
supports a direct comparison and neither algorithm dominates.

## Data

All 23 TEP pickles were empty and were reconstructed from `PCMCI/PCMCI_industrial/TEP/`. Only
the `simulationRun == 1` subset the protocol actually reads was extracted — **17.3 MB versus
4.16 GB, about 241× smaller** — which is functionally identical, since each raw fault file holds
500 concatenated runs and the protocol uses 960 of 480,000 rows (0.20%).

`extract_tep_faults.py` in `PCMCI/PCMCI_industrial/TEP/` documents how the original pickles were
derived from the `.RData` sources.

## Explainability

Per-fault attributions are in `outputs/explainability_top_features.csv`.

**Verified coherence.** Checked by hand against the TEP fault table: GES localises the two
valve-related reactor cooling faults precisely (fault 11 and fault 14 both attribute to
`XMV_10`, the reactor cooling water valve — for fault 14 that is the exact component that
sticks), but misses the two temperature-step faults (4 and 5) that PCMCI attributes correctly.

**Attribution is more concentrated than PCMCI's:** 6 distinct top-ranked sensors versus 10, with
`XMEAS_13`/`XMV_6`/`XMEAS_27` topping 16 of 20 faults. GES therefore detects about as well as
PCMCI on TEP while localising faults noticeably worse — detection strength and root-cause
specificity are separate properties.

See `FINAL_RESULTS.md` for the eight methodological corrections applied.
