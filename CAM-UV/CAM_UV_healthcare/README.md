# CAM-UV Causal Anomaly Detection — Healthcare (PTB-XL ECG)

Authentic **CAM-UV** (Causal Additive Models with Unobserved variables; Maeda & Shimizu,
UAI 2021) applied to 12-lead PTB-XL ECG data.

The engine is the paper authors' own reference implementation, shipped in the `lingam`
package and loaded through [`camuv_engine.py`](../camuv_engine.py). Nothing about the
algorithm is reimplemented here.

## Methodology

Normal ECG behaviour is modelled as a causal graph, then anomalies are detected as
*coefficient drift* in the causal regression equations, tracked online by Recursive Least
Squares. Both classes are scored as fresh, equal-length windows.

CAM-UV returns **two** objects: `P` (observed parents) and `U` (pairs sharing an *unobserved*
confounder). `U` is mapped to symmetric lag-1 edges (`CONFOUNDER_MODE = "integrate"`), never
contemporaneously — a contemporaneous symmetric pair would make two leads mutual same-time
predictors, which is a mutual-dependency cycle.

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
| CAM-UV | Linear / Ridge | 0.689 | 0.825 | **0.751** | 0.708 | **0.786** | **0.814** | **Selected** |
| CAM-UV | Polynomial degree 3 | 0.711 | 0.776 | 0.742 | 0.712 | 0.769 | 0.791 | Comparison |
| CAM-UV | RBF + Ridge | 0.655 | 0.854 | 0.741 | 0.682 | 0.769 | 0.799 | Comparison |

Linear / Ridge is selected, leading on F1, AUC-ROC and AUC-PR. The three variants are within
0.009 F1 of each other, indicating the limiting factor is the recovered causal structure
rather than the regression basis.

Unlike robotics, **all three variants run successfully** at the default `alpha = 0.05`: with
only 12 variables, a dense graph stays computationally tractable.

### Recovered structure

| Property | Value |
|---|---|
| Variables | 12 (all leads) |
| Lag-1 edges | 87 |
| Graph density | 65.9% |
| Confounded pairs | 39 of 66 (59.1%) |
| Monitored targets | **12 of 12** |
| Parents per target | mean 7.2, max 9 |

## Independent validation: the confounder/parent split is physiologically correct

The 12-lead ECG is an unusually good test of CAM-UV, because the right answer is known in
advance from electrocardiography — and this check was not tuned for.

- **Limb leads** III, aVR, aVL, aVF are *exact linear functions* of I and II (Einthoven's
  triangle, Goldberger equations). An observed variable fully explains them, so CAM-UV should
  call them **observed parents**.
- **Precordial leads** V1–V6 are six chest electrodes projecting the same cardiac dipole
  vector from different positions — correlated, but none an algebraic function of another. No
  observed lead explains them, so CAM-UV should call them **latent-confounded**.

| Structure | Expected | Recovered |
|---|---|---|
| Algebraically-determined limb pairs flagged confounded | few | **1 of 8** |
| Precordial V–V pairs flagged confounded | most | **14 of 15** |

The recovered parent edges make it explicit — every derived limb lead resolves to lead I:

```
 II <- I, AVF      AVR <- I      AVL <- I      AVF <- I      III <- I, AVF
```

**The latent variable CAM-UV posits is real and nameable: the cardiac dipole vector.** All 12
leads are projections of it, it is never measured directly, and it is exactly the kind of
unobserved common cause the algorithm was designed to find.

## Ablation: does the latent structure actually help?

Section 3 shows CAM-UV's confounder output is *correct*. It does not show it is *useful*. An
ablation holding `alpha` fixed at 0.05 and toggling only `CONFOUNDER_MODE` tests the second
claim directly.

| | Config A `integrate` | Config B `observed_only` | Change |
|---|---:|---:|---:|
| Total lag-1 edges | 87 | 9 | **-90%** |
| Monitored leads | 12 of 12 | 6 of 12 | -50% |

| Variant | delta F1 | **delta AUC-ROC** |
|---|---:|---:|
| Linear / Ridge | -0.023 | **-0.063** |
| Polynomial degree 3 | +0.003 | **+0.019** |
| RBF + Ridge | -0.007 | **-0.0002** |
| **Mean** | **-0.009** | **-0.015** |

**Deleting 90% of the graph changes mean AUC-ROC by -0.015** — inside CAM-UV's own ±0.05
run-to-run noise band. One variant improved; RBF was unchanged to four decimals. The 39
confounded pairs correctly capture the physiological V1–V6 cardiac dipole, but contribute
essentially nothing to detection, which runs on the 9 observed limb-lead algebra edges.

CAM-UV's distinguishing output is here **correct and useless at the same time**. See
`FINAL_RESULTS.md` §4 for the full tables and the caveat on Config B Linear's degenerate F1.

## Comparison with PCMCI and GES

Same protocol, same data, same evaluation — only the discovery engine differs:

| Algorithm | Selected variant | F1 | AUC-ROC |
|---|---|---:|---:|
| PCMCI | RBF + Ridge | **0.809** | **0.817** |
| GES | Linear / Ridge | 0.768 | 0.811 |
| CAM-UV | Linear / Ridge | 0.751 | 0.786 |

CAM-UV detects least well of the three despite being demonstrably right about the latent
structure. By correctly attributing the precordial leads to a latent common cause it declines
to place observed-parent edges between them; in `integrate` mode they re-enter as lag-1 edges,
giving a 65.9%-dense near-vector-autoregression where each lead is predicted from ~7 others.
That redundancy damps the residual anomaly detection depends on. **Being right about causal
structure and being good at detection are separate properties.**

## Provenance

This folder previously contained scripts that imported `tigramite` and ran **PCMCI** while
being presented as CAM-UV. Any healthcare CAM-UV metrics published before this rebuild
describe PCMCI and must not be cited as CAM-UV results.

See [`FINAL_RESULTS.md`](FINAL_RESULTS.md) for the full analysis and protocol.
