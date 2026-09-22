"""
Architecture search for the LSTM autoencoder baseline.

The thesis claims the causal pipelines outperform a neural detector. That claim is only
worth anything if the neural detector was given a fair chance, so this searches over depth,
width, learning rate and window length rather than accepting the defaults.

Selection is on a held-out fault-free VALIDATION split only -- never on attack labels and
never on the evaluation data. The winning configuration is then reported once on the
evaluation set, exactly as lstm_shap_baseline.py does. That keeps the comparison honest:
tuning the baseline against its own test set would be the same defect removed from the
causal pipelines.

Usage:  python Baselines/architecture_search.py        (from the repository root)
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

sys.path.append(os.path.abspath("PCMCI/PCMCI_robotic"))

PREFIX = "PCMCI/PCMCI_robotic/pepper_csv/"
NORMAL_FILE = "normal.csv"
ATTACK_FILES = ["WheelsControl.csv", "JointControl.csv", "LedsControl.csv"]

TRAINING_FRAC = 0.7
VAL_FRAC = 0.15
THRESHOLD_PERCENTILE = 95.0
MAX_EPOCHS = 300
PATIENCE = 15
SEED = 42

# Search space. Kept deliberately modest -- the point is to show the default was not a
# straw man, not to squeeze the last decimal.
GRID = {
    "hidden_dim": [8, 16, 32, 64],
    "num_layers": [1, 2],
    "lr": [0.001, 0.01],
    "seq_length": [5, 10, 20],
}


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(h)
        return out


def sequences(a, L):
    if len(a) <= L:
        return np.empty((0, L, a.shape[1]), dtype=a.dtype)
    return np.stack([a[i:i + L] for i in range(len(a) - L)])


def mse(model, x):
    with torch.no_grad():
        out = model(x)
        return torch.mean((out[:, -1, :] - x[:, -1, :]) ** 2, dim=1).numpy()


def train(x_tr, x_val, input_dim, hidden_dim, num_layers, lr, seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = LSTMAutoencoder(input_dim, hidden_dim, num_layers)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    dl = DataLoader(TensorDataset(x_tr), batch_size=128, shuffle=True)
    best, best_state, stale, best_ep = float("inf"), None, 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        m.train()
        for (b,) in dl:
            opt.zero_grad(); crit(m(b), b).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            v = float(crit(m(x_val), x_val))
        if v < best - 1e-6:
            best, best_ep, stale = v, ep, 0
            best_state = {k: t.clone() for k, t in m.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state:
        m.load_state_dict(best_state)
    m.eval()
    return m, best, best_ep


def main():
    cache = os.path.join(PREFIX, "pepper_normal.npz")
    if not os.path.exists(cache):
        print("PCMCI cache missing."); return
    f = np.load(cache, allow_pickle=True)
    sub, nonconst = int(f["subsample"]), f["nonconst"]

    df = pd.read_csv(os.path.join(PREFIX, NORMAL_FILE), delimiter=",")
    vals = df.drop(columns=["timestamp"]).values
    train_raw = np.nan_to_num(vals[:int(TRAINING_FRAC * len(vals))][::sub, nonconst])
    full_raw = np.nan_to_num(vals[::sub, nonconst])
    heldout_raw = full_raw[len(train_raw):]
    scaler = StandardScaler().fit(train_raw)

    attacks_raw = {}
    for a in ATTACK_FILES:
        d = pd.read_csv(os.path.join(PREFIX, a), delimiter=",")
        attacks_raw[a] = np.nan_to_num(d.drop(columns=["timestamp"]).values[::sub, nonconst])

    combos = list(itertools.product(*GRID.values()))
    print(f"Searching {len(combos)} configurations. Selection on fault-free validation loss "
          f"only -- attack labels are never used for selection.\n")
    print(f"{'hidden':>7} {'layers':>7} {'lr':>7} {'seq':>5} {'val_loss':>10} {'epoch':>6}")
    print("-" * 50)

    results = []
    for hidden_dim, num_layers, lr, L in combos:
        x_all = sequences(scaler.transform(train_raw), L)
        if len(x_all) < 20:
            continue
        n_val = max(1, int(VAL_FRAC * len(x_all)))
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(x_all))
        x_tr = torch.tensor(x_all[perm[n_val:]], dtype=torch.float32)
        x_va = torch.tensor(x_all[perm[:n_val]], dtype=torch.float32)
        try:
            m, vloss, ep = train(x_tr, x_va, len(nonconst), hidden_dim, num_layers, lr)
        except Exception as e:
            print(f"{hidden_dim:7d} {num_layers:7d} {lr:7.3f} {L:5d}  FAILED {type(e).__name__}")
            continue
        print(f"{hidden_dim:7d} {num_layers:7d} {lr:7.3f} {L:5d} {vloss:10.5f} {ep:6d}")
        results.append(dict(hidden_dim=hidden_dim, num_layers=num_layers, lr=lr,
                            seq_length=L, val_loss=vloss, epoch=ep))

    if not results:
        print("no configuration trained"); return
    best = min(results, key=lambda r: r["val_loss"])
    print(f"\nSelected on validation loss: hidden={best['hidden_dim']} "
          f"layers={best['num_layers']} lr={best['lr']} seq={best['seq_length']} "
          f"(val {best['val_loss']:.5f})")

    # Report the winner ONCE on the evaluation set.
    L = best["seq_length"]
    x_all = sequences(scaler.transform(train_raw), L)
    n_val = max(1, int(VAL_FRAC * len(x_all)))
    perm = np.random.default_rng(0).permutation(len(x_all))
    x_tr = torch.tensor(x_all[perm[n_val:]], dtype=torch.float32)
    x_va = torch.tensor(x_all[perm[:n_val]], dtype=torch.float32)
    m, _, _ = train(x_tr, x_va, len(nonconst), best["hidden_dim"], best["num_layers"], best["lr"])

    ns = mse(m, torch.tensor(sequences(scaler.transform(heldout_raw), L), dtype=torch.float32))
    thr = float(np.percentile(ns, THRESHOLD_PERCENTILE))
    yt, ys = [np.zeros(len(ns), int)], [ns]
    for a, raw in attacks_raw.items():
        w = sequences(scaler.transform(raw), L)
        if len(w) == 0:
            continue
        s = mse(m, torch.tensor(w, dtype=torch.float32))
        yt.append(np.ones(len(s), int)); ys.append(s)
    yt = np.concatenate(yt); ys = np.concatenate(ys)
    yp = (ys > thr).astype(int)
    print(f"\nTUNED BASELINE  F1={f1_score(yt, yp):.4f}  AUC={roc_auc_score(yt, ys):.4f}  "
          f"threshold={thr:.4f}  n={len(yt)}")
    print("(default hidden=16 layers=1 lr=0.01 seq=5 gave F1 0.7072 / AUC 0.7600 "
          "under the same protocol)")


if __name__ == "__main__":
    main()
