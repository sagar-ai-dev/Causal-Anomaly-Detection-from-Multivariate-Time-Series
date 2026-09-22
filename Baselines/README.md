# Neural Baselines — LSTM, TCN and Transformer

Deep-learning comparison for the causal anomaly-detection thesis, across all three domains.
It answers one question: **do the causal pipelines beat properly tuned neural anomaly
detectors, and is their explanation better?** On two of the three domains they do not.

## What is implemented

| Component | Status |
| :--- | :--- |
| LSTM autoencoder | implemented — `neural_baselines.py` |
| **TCN autoencoder** (dilated causal convolutions) | implemented — `neural_baselines.py` |
| **Transformer autoencoder** (multi-head self-attention) | implemented — `neural_baselines.py` |
| SHAP post-hoc XAI, all three | implemented — `GradientExplainer` |
| Architecture search, all three | implemented — same budget each |

`lstm_shap_baseline.py` remains as the standalone single-architecture script;
`architecture_search.py` is the LSTM-only search that established the tuning protocol. Both
are superseded by `neural_baselines.py` for the three-way comparison.

## Results

Three seeds each, trained to convergence with early stopping, hyperparameters selected on
**fault-free validation loss only** — never on an attack label.

| Architecture | F1 | AUC-ROC | AUC-PR | selected config |
| :--- | ---: | ---: | ---: | :--- |
| **TCN autoencoder** | **0.8053** ± 0.0139 | **0.8856** ± 0.0051 | **0.9164** | `hidden=64, layers=1, lr=0.01` |
| LSTM autoencoder | 0.7972 ± 0.0039 | 0.8804 ± 0.0032 | 0.9002 | `hidden=64, layers=1, lr=0.001` |
| Transformer autoencoder | 0.7140 ± 0.0195 | 0.8139 ± 0.0158 | 0.8731 | `hidden=64, layers=1, lr=0.01` |

The Transformer is clearly weakest. With 159 training windows over 244 features it is the
most data-hungry of the three and the least suited to this regime; the result should be read
as "a Transformer is a poor fit for a dataset this small", not as a general statement.

### Tuning was necessary, not cosmetic

An untuned LSTM (`hidden=16, layers=1, lr=0.01, seq=5`) scores **AUC 0.7600**; tuned it
reaches **0.8804**. Window length dominated the search — every `seq=10` configuration beat
every `seq=5` one — so the original default was handicapping the baseline structurally rather
than by training budget. A comparison against that default would not have been evidence of
anything.

## Comparison against the causal pipelines (Robotics / Pepper)

| Model | F1 | AUC-ROC |
| :--- | ---: | ---: |
| **TCN autoencoder** | 0.8053 | **0.8856** |
| **LSTM autoencoder** | 0.7972 | 0.8804 |
| PCMCI — RBF + Ridge | 0.7938 | **0.8752** |
| Transformer autoencoder | 0.7140 | 0.8139 |
| PCMCI — Polynomial degree 3 | 0.7812 | 0.8053 |
| PCMCI — Linear / Ridge | 0.6395 | 0.7825 |
| CAM-UV — RBF + Ridge | 0.7941 | 0.7818 |
| GES — Linear / Ridge | 0.5445 | 0.5906 |
| GES — Polynomial degree 3 | 0.6172 | 0.6999 |
| CAM-UV — Polynomial degree 3 | 0.5531 | 0.6888 |
| CAM-UV — Linear / Ridge | 0.4747 | 0.6497 |
| GES — RBF + Ridge | 0.6583 | 0.6516 |

Testing the best neural architecture against each causal configuration under the analytical
Hanley–McNeil interval, which this study rejects as too narrow (see
`tools/significance.py`):

| TCN (AUC 0.8856) vs | z | Verdict |
| :--- | ---: | :--- |
| PCMCI — RBF | 0.88 | not distinguishable |
| PCMCI — Polynomial | 1.95 | not distinguishable |
| PCMCI — Linear, all GES, all CAM-UV | 2.42 – 7.75 | apparently separated |

**That verdict does not survive the uncertainty analysis.** The analytical interval assumes
independent observations. Adjacent evaluation windows overlap and their scores are
autocorrelated, so the effective sample size is well below n and the interval is too narrow
— the same reason the thesis reports moving-block bootstrap intervals instead. Under the
bootstrap the TCN score falls inside PCMCI—RBF's interval of [0.7559, 0.9587]. A paired
test is not available for this comparison at all: the autoencoders score overlapping
windows rather than a subsampled series, so no causal score file pairs with theirs. The
supportable statement is that a tuned TCN and the best causal configuration are not
separable on this dataset, in either direction.

## Explainability — the comparison cuts both ways

Top attributed variable per attack class:

| Attack | Causal models | LSTM | TCN | Transformer |
| :--- | :--- | :--- | :--- | :--- |
| **LedsControl** | `TabletTouchX` | **`LedFaceBL4`** | `AccelerometerX` | **`LedFaceBL4`** |
| **JointControl** | **`KneePicthTemp`** | `RShoulderPitch` | `RShoulderPitch` | `RShoulderPitch` |
| WheelsControl | `WheelFRTemp` | `WheelBSpeed` | `WheelBSpeed` | `WheelBSpeed` |

**Against the causal models.** On LedsControl the tuned LSTM and Transformer attribute to
`LedFaceBL4` and `LedFaceBR5` — *actual LED channels*, the subsystem the attack directly
targets. The causal models return `TabletTouchX`, a correlated subsystem. The XAI lab expects
the Tablet answer on the reasoning that tablet interaction drives the LEDs, but the neural
attribution is arguably the more direct identification of what was attacked.

**For the causal models.** On JointControl every neural architecture returns `RShoulderPitch`,
a joint *position*, while the causal models return `KneePicthTemp`, a joint *temperature*.
The documented fault is thermal, so the causal attribution identifies the mechanism where the
neural one identifies a kinematic correlate.

**What survives is narrow.** Post-hoc attribution on a tuned neural detector is not inferior
at naming relevant variables — on one attack class it is arguably better. What the causal
models provide that attribution cannot is a **directed graph with parent–child structure**:
an answer to *through what path*, not only *which variable*. That is where the causal approach
is distinct, and it is a claim about the kind of explanation rather than its accuracy.

SHAP attribution runs for all three architectures via `GradientExplainer`, which supports
convolutional and attention operations where `DeepExplainer`'s layer rules do not. On
WheelsControl all three SHAP results agree with the residual-magnitude ranking, which is a
useful consistency check on the attribution machinery.

## All three domains, against a no-learning control

`neural_baselines_all.py` runs the three architectures on all three domains, reading each
domain's variables and resampling rate from that domain's own PCMCI cache. `trivial_baseline.py`
adds a control with **no parameters at all** — mean `|z|` from the fault-free mean and standard
deviation, same 95th-percentile threshold. It is the floor any real method has to clear.

| Domain | Trivial control | LSTM | TCN | Transformer | Best causal |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Robotics (Pepper) | 0.8139 | 0.8804 | 0.8809 | 0.7815 | **0.8752** |
| Healthcare (ECG) | 0.6136 | 0.6815 | 0.6624 | 0.6102 | **0.8114** |
| Industrial (TEP) | 0.9286 | 0.9669 | 0.9995 | **0.9996** | 0.9151 |

AUC-ROC; three seeds each; n = 155 / 4,793 / 16,590.

**On industrial the causal pipelines lose to arithmetic.** The parameter-free control reaches
0.9286, above all nine causal configurations, and the TCN and Transformer reach 0.9995. TEP
faults are large sustained level shifts in raw sensor space, and a coefficient-drift detector
is deliberately insensitive to exactly that.

**On healthcare the ordering reverses.** The control manages 0.6136 and the best neural
architecture 0.6815, against a causal 0.8114 — ECG anomalies perturb inter-lead *relationships*
while each lead individually stays in range. Restricting to configurations at the matched
resampling rate (PCMCI-RBF 0.7147 vs LSTM 0.6815) narrows that to +0.033, which is the honest
headline; GES runs at `subsample=10` and the neural baselines at PCMCI's 7.

The defensible claim is conditional: causal detection is stronger where anomalies are
**relational rather than magnitude-based**, not in general.

## Why the comparison is fair

- **Same variables and rate** — `subsample` (74) and `nonconst` (244 sensors) read from
  `PCMCI/PCMCI_robotic/pepper_csv/pepper_normal.npz`.
- **Same fault-free split** — `TRAINING_FRAC = 0.7`; the normal class is scored on the
  held-out 30%, never on training data. The scaler is fitted on the training portion only.
- **Same threshold rule** — 95th percentile of the fault-free score distribution, identical to
  all 27 causal configurations, never seeing an attack label.
- **Same training procedure** — to convergence, early stopping on a *random* validation split
  (a trailing block is a different operating regime and makes early stopping fire at epoch 1).
- **Same search protocol** — 12 configurations per architecture, selected on validation loss.
- **Three seeds** each; the spread is reported.

## Known limitations

- **Small training set.** At PCMCI's `subsample=74`, the 70% fault-free split yields ~159
  training windows for a 244-dimensional input. All three architectures are in a data-poor
  regime, and the Transformer suffers most.
- **The cross-domain run holds configuration fixed** at `hidden=64, layers=1, lr=0.001,
  seq=10` so that domain is the only axis. It is therefore not a per-domain tuned result, and
  the robotics figures in it sit slightly below the searched ones above.
- **Industrial prevalence is 95.2%** — twenty fault files against one fault-free run — so F1
  and AUC-PR are inflated there and only AUC-ROC should be read across domains.
- **Evaluation set is 155 samples** against PCMCI robotics' 181, because `seq_length=10`
  windows the series differently. Attack counts are comparable but not identical.
- **`reporting_helper` zero-pads** score arrays while labelling those windows positive, which
  costs a few false negatives. It applies identically to all configurations.

## Reproducing

```bash
pip install -r Baselines/requirements.txt
python Baselines/neural_baselines.py            # robotics, searched then evaluated
python Baselines/neural_baselines.py --quick    # skip the search

python Baselines/neural_baselines_all.py        # all three domains, fixed config
python Baselines/neural_baselines_all.py --domain healthcare --arch Transformer
python Baselines/trivial_baseline.py            # the no-learning control
```

Run from the repository root. Requires `PCMCI/PCMCI_robotic/pepper_csv/pepper_normal.npz`;
the script exits with a message if that cache is absent, since it cannot otherwise match the
variable selection.
