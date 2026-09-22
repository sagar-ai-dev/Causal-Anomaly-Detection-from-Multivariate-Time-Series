"""
Neural baselines on all three domains: LSTM, TCN and Transformer.

The reviewer asked for the neural comparison to extend beyond Pepper -- "at least transformer,
and possibly on all datasets" -- so this runs the architectures on robotics, healthcare and
industrial under the same protocol the causal pipelines use for each domain.

Per-domain settings are read from that domain's own PCMCI cache, so the baseline sees exactly
the variables and resampling rate the causal pipelines see:

    robotics    pepper_normal.npz        244 vars, subsample 74   3 attack files
    healthcare  healthcare_normal.npz     12 vars, subsample  7   attack dict
    industrial  tep_normal.npz            52 vars, subsample  1   20 fault files, onset 160

Shared across every architecture and domain: the 70/30 fault-free split with the scaler fitted
on the training portion only, training to convergence with early stopping on a random
validation split, the 95th-percentile fault-free threshold, and reconstruction MSE at the
final timestep as the score. Only the architecture and the domain vary.

Usage:
    python Baselines/neural_baselines_all.py                     # all domains, all architectures
    python Baselines/neural_baselines_all.py --domain healthcare
    python Baselines/neural_baselines_all.py --arch Transformer
"""

import csv
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath("PCMCI/PCMCI_robotic"))

from neural_baselines import (  # noqa: E402
    LSTMAutoencoder, TCNAutoencoder, TransformerAutoencoder,
    sequences, recon_error,
)

ARCHITECTURES = {
    "LSTM Autoencoder": LSTMAutoencoder,
    "TCN Autoencoder": TCNAutoencoder,
    "Transformer Autoencoder": TransformerAutoencoder,
}

TRAINING_FRAC = 0.7
VAL_FRAC = 0.15
THRESHOLD_PERCENTILE = 95.0
MAX_EPOCHS = 300
PATIENCE = 15
SEEDS = [42, 43, 44]
SEQ_LENGTH = 10
CONFIG = dict(hidden_dim=64, num_layers=1, lr=0.001)

# Per-architecture, per-domain hyperparameter search, enabled with --search.
#
# Without it this script trains one fixed configuration everywhere, which made the
# cross-domain comparison unfair in a direction that flattered the causal pipelines: the
# robotics neural baselines had been searched over twelve configurations each while
# healthcare and industrial had not been searched at all. Comparing a tuned causal pipeline
# against an untuned neural one is precisely the criticism this thesis levels at the
# published literature, so it cannot stand in the thesis's own evidence.
#
# The grid matches Baselines/neural_baselines.py exactly -- twelve configurations per
# architecture -- and selection is on fault-free validation loss, so no attack label is
# consulted at any point.
SEARCH = {
    "hidden_dim": [32, 64],
    "num_layers": [1, 2, 3],
    "lr": [0.001, 0.01],
}


# ------------------------------------------------------------------ per-domain data loading

def load_robotics():
    P = "PCMCI/PCMCI_robotic/pepper_csv/"
    z = np.load(os.path.join(P, "pepper_normal.npz"), allow_pickle=True)
    sub, nc = int(z["subsample"]), z["nonconst"]
    names = z["var"][nc].tolist()
    v = pd.read_csv(os.path.join(P, "normal.csv")).drop(columns=["timestamp"]).values
    train = np.nan_to_num(v[:int(TRAINING_FRAC * len(v))][::sub, nc])
    heldout = np.nan_to_num(v[::sub, nc])[len(train):]
    attacks = {}
    for a in ("WheelsControl.csv", "JointControl.csv", "LedsControl.csv"):
        d = pd.read_csv(os.path.join(P, a)).drop(columns=["timestamp"]).values
        attacks[a] = np.nan_to_num(d[::sub, nc])
    return train, heldout, attacks, names, "Robotics (Pepper)"


def load_healthcare():
    P = "PCMCI/PCMCI_healthcare/Healthcare dataset/"
    z = np.load(os.path.join(P, "healthcare_normal.npz"), allow_pickle=True)
    sub, nc = int(z["subsample"]), z["nonconst"]
    names = [str(x) for x in np.asarray(z["var"])[nc]]
    norm = pd.read_pickle(os.path.join(P, "healthcare_normal_train.pkl"))
    v = norm.values
    train = np.nan_to_num(v[:int(TRAINING_FRAC * len(v))][::sub, nc])
    heldout = np.nan_to_num(v[::sub, nc])[len(train):]
    ad = pd.read_pickle(os.path.join(P, "healthcare_attacks.pkl"))
    frames = ad if isinstance(ad, dict) else {"healthcare_attacks": ad}
    attacks = {k: np.nan_to_num(fr.values[::sub, nc]) for k, fr in frames.items()}
    return train, heldout, attacks, names, "Healthcare (ECG)"


def load_industrial():
    P = "PCMCI/PCMCI_industrial/TEP/"
    z = np.load(os.path.join(P, "tep_normal.npz"), allow_pickle=True)
    sub, nc = int(z["subsample"]), z["nonconst"]
    names = [str(x) for x in np.asarray(z["var"])[nc]]
    tr = pd.read_pickle(os.path.join(P, "TEP_FaultFree_Training_run1.pkl"))
    train = np.nan_to_num(tr.values[::sub, nc])
    te = pd.read_pickle(os.path.join(P, "TEP_FaultFree_Testing_run1.pkl"))
    onset = 160 // max(sub, 1)
    heldout = np.nan_to_num(te.values[::sub, nc])[onset:]
    attacks = {}
    for i in range(1, 21):
        fp = os.path.join(P, f"TEP_Faulty_Testing_fault{i}.pkl")
        if not os.path.exists(fp):
            continue
        d = pd.read_pickle(fp)
        attacks[f"Fault_{i}"] = np.nan_to_num(d.values[::sub, nc])[onset:]
    return train, heldout, attacks, names, "Industrial (TEP)"


LOADERS = {"robotics": load_robotics, "healthcare": load_healthcare, "industrial": load_industrial}


# ------------------------------------------------------------------------- shared protocol

def train_model(cls, x_tr, x_val, input_dim, seed, **kw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = cls(input_dim, **kw)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(m.parameters(), lr=kw.get("lr", 0.001))
    dl = DataLoader(TensorDataset(x_tr), batch_size=128, shuffle=True)
    best, state, stale, best_ep = float("inf"), None, 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        m.train()
        for (b,) in dl:
            opt.zero_grad(); crit(m(b), b).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = float(crit(m(x_val), x_val))
        if v < best - 1e-6:
            best, best_ep, stale = v, ep, 0
            state = {k: t.clone() for k, t in m.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if state:
        m.load_state_dict(state)
    m.eval()
    return m, best, best_ep


def run(domain, arch_filter=None):
    train_raw, heldout_raw, attacks_raw, names, label = LOADERS[domain]()
    scaler = StandardScaler().fit(train_raw)
    x_all = sequences(scaler.transform(train_raw), SEQ_LENGTH)
    if len(x_all) < 20:
        print(f"  {label}: too few training windows ({len(x_all)}), skipping")
        return []
    n_val = max(1, int(VAL_FRAC * len(x_all)))
    perm = np.random.default_rng(0).permutation(len(x_all))
    x_tr = torch.tensor(x_all[perm[n_val:]], dtype=torch.float32)
    x_val = torch.tensor(x_all[perm[:n_val]], dtype=torch.float32)
    x_ho = torch.tensor(sequences(scaler.transform(heldout_raw), SEQ_LENGTH), dtype=torch.float32)

    print(f"\n{'='*84}\n {label}   {train_raw.shape[1]} vars   "
          f"{len(x_tr)} train / {len(x_val)} val / {len(x_ho)} held-out windows   "
          f"{len(attacks_raw)} attack classes\n{'='*84}")

    out = []
    for aname, cls in ARCHITECTURES.items():
        if arch_filter and arch_filter.lower() not in aname.lower():
            continue
        runs, first = [], None

        # Select the configuration on fault-free validation loss only.
        cfg = dict(CONFIG)
        if SEARCH_ENABLED:
            import itertools as _it
            trials = []
            for hd, nl, lr in _it.product(*SEARCH.values()):
                try:
                    _, v, _ = train_model(cls, x_tr, x_val, train_raw.shape[1], SEEDS[0],
                                          hidden_dim=hd, num_layers=nl, lr=lr)
                except Exception:
                    continue
                trials.append((v, dict(hidden_dim=hd, num_layers=nl, lr=lr)))
            if trials:
                cfg = min(trials, key=lambda t: t[0])[1]
                print(f"  {aname:26} searched {len(trials)}/12 -> {cfg}")

        for seed in SEEDS:
            try:
                m, _, ep = train_model(cls, x_tr, x_val, train_raw.shape[1], seed, **cfg)
            except Exception as exc:
                print(f"  {aname:26} FAILED {type(exc).__name__}: {exc}")
                runs = []; break
            ns, _ = recon_error(m, x_ho)
            thr = float(np.percentile(ns, THRESHOLD_PERCENTILE))
            yt, ys = [np.zeros(len(ns), int)], [ns]
            per_file = {}
            for k, raw in attacks_raw.items():
                w = sequences(scaler.transform(raw), SEQ_LENGTH)
                if len(w) == 0:
                    continue
                s, resid = recon_error(m, torch.tensor(w, dtype=torch.float32))
                per_file[k] = (s, resid)
                yt.append(np.ones(len(s), int)); ys.append(s)
            yt, ys = np.concatenate(yt), np.concatenate(ys)
            runs.append(dict(f1=f1_score(yt, (ys > thr).astype(int)),
                             auc=roc_auc_score(yt, ys),
                             aucpr=average_precision_score(yt, ys), n=len(yt), ep=ep))
            if first is None:
                first = per_file
        if not runs:
            continue
        f1 = np.array([r["f1"] for r in runs]); auc = np.array([r["auc"] for r in runs])
        apr = np.mean([r["aucpr"] for r in runs])
        print(f"  {aname:26} F1 {f1.mean():.4f} +/- {f1.std():.4f}   "
              f"AUC {auc.mean():.4f} +/- {auc.std():.4f}   AUC-PR {apr:.4f}   "
              f"n={runs[0]['n']}   epochs~{int(np.mean([r['ep'] for r in runs]))}")
        for k, (s, resid) in list(first.items())[:3]:
            mag = np.linalg.norm(resid, axis=0)
            top = np.argsort(mag)[::-1][:3]
            print(f"      ATTR {k}: " + ", ".join(names[t] for t in top))
        out.append(dict(Dataset=label, Model="Neural Baseline", Variant=aname,
                        F1=f"{f1.mean():.4f}", F1_std=f"{f1.std():.4f}",
                        AUC_ROC=f"{auc.mean():.4f}", AUC_std=f"{auc.std():.4f}",
                        AUC_PR=f"{apr:.4f}", n=runs[0]["n"]))
    return out


SEARCH_ENABLED = False


def main():
    global SEARCH_ENABLED
    SEARCH_ENABLED = "--search" in sys.argv
    domain = None
    if "--domain" in sys.argv:
        domain = sys.argv[sys.argv.index("--domain") + 1]
    arch = None
    if "--arch" in sys.argv:
        arch = sys.argv[sys.argv.index("--arch") + 1]

    rows = []
    for d in ([domain] if domain else ["robotics", "healthcare", "industrial"]):
        try:
            rows += run(d, arch)
        except Exception as exc:
            print(f"\n  {d}: FAILED {type(exc).__name__}: {exc}")

    if not rows:
        return
    name = "neural_all_domains_tuned.csv" if SEARCH_ENABLED else "neural_all_domains.csv"
    out = os.path.join(_HERE, "outputs", name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out} ({len(rows)} rows)")
    print(f"\n{'Domain':20} {'Architecture':26} {'F1':>8} {'AUC-ROC':>9} {'AUC-PR':>8} {'n':>7}")
    print("-" * 82)
    for r in rows:
        print(f"{r['Dataset']:20} {r['Variant']:26} {r['F1']:>8} {r['AUC_ROC']:>9} "
              f"{r['AUC_PR']:>8} {r['n']:>7}")


if __name__ == "__main__":
    main()
