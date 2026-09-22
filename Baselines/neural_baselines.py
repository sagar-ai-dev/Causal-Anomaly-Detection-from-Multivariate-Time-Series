"""
Neural baselines for the causal anomaly-detection thesis: LSTM, TCN and Transformer.

The reviewer asked for a comparison against "standard neural networks (autoencoder lstm, tcn,
transformer) and shap / post-hoc xai". All three are implemented here as reconstruction
autoencoders sharing one protocol, so the only thing that varies between them is the
architecture:

  * same variables and resampling rate, read from PCMCI's cached graph
  * same 70/30 fault-free split, scaler fitted on the training portion only
  * same decision rule -- the 95th percentile of the fault-free score distribution, which
    never sees an attack label
  * same training procedure -- to convergence, early stopping on a random validation split
  * same scoring -- reconstruction MSE at the final timestep of each window
  * same architecture search protocol, selected on fault-free validation loss only

That last point matters. A comparison against an untuned neural baseline is not evidence, so
each architecture gets the same search budget over its own hyperparameters, and selection
never touches an attack label.

Usage:
    python Baselines/neural_baselines.py               # all three, searched then evaluated
    python Baselines/neural_baselines.py --quick       # skip the search, use known-good configs
"""

import itertools
import math
import os
import io
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import csv
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("PCMCI/PCMCI_robotic"))

PREFIX = "PCMCI/PCMCI_robotic/pepper_csv/"
NORMAL_FILE = "normal.csv"
ATTACK_FILES = ["WheelsControl.csv", "JointControl.csv", "LedsControl.csv"]

TRAINING_FRAC = 0.7
VAL_FRAC = 0.15
THRESHOLD_PERCENTILE = 95.0
MAX_EPOCHS = 300
PATIENCE = 15
SEEDS = [42, 43, 44]
SHAP_SEED = 42          # seeds GradientExplainer's background sampling; see explain_shap


# ----------------------------------------------------------------------------- architectures

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, **_):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(h)
        return out


class _CausalConv1d(nn.Module):
    """Dilated causal convolution: no timestep sees its own future."""

    def __init__(self, cin, cout, kernel, dilation):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, kernel, dilation=dilation)

    def forward(self, x):
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TCNAutoencoder(nn.Module):
    """
    Temporal Convolutional Network autoencoder.

    Stacked dilated causal convolutions with residual connections give an exponentially
    growing receptive field without recurrence. The encoder compresses to `hidden_dim`
    channels; the decoder mirrors it back to the input dimension.
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=3, kernel=3, **_):
        super().__init__()
        enc, dec = [], []
        cin = input_dim
        for i in range(num_layers):
            enc.append(_CausalConv1d(cin, hidden_dim, kernel, dilation=2 ** i))
            enc.append(nn.ReLU())
            cin = hidden_dim
        for i in reversed(range(num_layers)):
            cout = hidden_dim if i > 0 else input_dim
            dec.append(_CausalConv1d(cin, cout, kernel, dilation=2 ** i))
            if i > 0:
                dec.append(nn.ReLU())
            cin = cout
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = x.transpose(1, 2)          # (B, T, F) -> (B, F, T) for Conv1d
        z = self.decoder(self.encoder(z))
        return z.transpose(1, 2)


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].size(1)])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerAutoencoder(nn.Module):
    """
    Transformer-encoder autoencoder.

    The input is projected to `d_model`, given sinusoidal positional encoding, passed through
    multi-head self-attention blocks, then projected back. `nhead` is chosen to divide
    `d_model`, which torch requires.
    """

    def __init__(self, input_dim, hidden_dim=64, num_layers=2, nhead=4, **_):
        super().__init__()
        d_model = hidden_dim
        while d_model % nhead != 0 and nhead > 1:
            nhead -= 1
        self.inp = nn.Linear(input_dim, d_model)
        self.pos = _PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=2 * d_model,
                                           batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, input_dim)

    def forward(self, x):
        return self.out(self.enc(self.pos(self.inp(x))))


ARCHITECTURES = {
    "LSTM Autoencoder": LSTMAutoencoder,
    "TCN Autoencoder": TCNAutoencoder,
    "Transformer Autoencoder": TransformerAutoencoder,
}

# Same search budget for every architecture -- 12 configurations each.
SEARCH = {
    "hidden_dim": [32, 64],
    "num_layers": [1, 2, 3],
    "lr": [0.001, 0.01],
}
SEQ_LENGTHS = [10]          # fixed: the LSTM search showed seq=10 dominates seq=5 and 20
QUICK = {"hidden_dim": 64, "num_layers": 2, "lr": 0.001}


# ----------------------------------------------------------------------------- shared protocol

def sequences(a, L):
    if len(a) <= L:
        return np.empty((0, L, a.shape[1]), dtype=a.dtype)
    return np.stack([a[i:i + L] for i in range(len(a) - L)])


def recon_error(model, x):
    with torch.no_grad():
        out = model(x)
        err = out[:, -1, :] - x[:, -1, :]
        return torch.mean(err ** 2, dim=1).numpy(), np.abs(err.numpy())


def train(cls, x_tr, x_val, input_dim, seed, **kw):
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
            opt.zero_grad()
            crit(m(b), b).backward()
            opt.step()
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


def load_data():
    cache = os.path.join(PREFIX, "pepper_normal.npz")
    if not os.path.exists(cache):
        raise SystemExit("PCMCI cache missing -- run PCMCI_robotic first.")
    f = np.load(cache, allow_pickle=True)
    sub, nonconst = int(f["subsample"]), f["nonconst"]
    var_names = f["var"][nonconst].tolist()
    df = pd.read_csv(os.path.join(PREFIX, NORMAL_FILE), delimiter=",")
    vals = df.drop(columns=["timestamp"]).values
    train_raw = np.nan_to_num(vals[:int(TRAINING_FRAC * len(vals))][::sub, nonconst])
    full_raw = np.nan_to_num(vals[::sub, nonconst])
    heldout_raw = full_raw[len(train_raw):]
    scaler = StandardScaler().fit(train_raw)
    attacks = {}
    for a in ATTACK_FILES:
        d = pd.read_csv(os.path.join(PREFIX, a), delimiter=",")
        attacks[a] = np.nan_to_num(d.drop(columns=["timestamp"]).values[::sub, nonconst])
    return scaler, train_raw, heldout_raw, attacks, len(nonconst), var_names, sub


def split(scaler, train_raw, L):
    x_all = sequences(scaler.transform(train_raw), L)
    n_val = max(1, int(VAL_FRAC * len(x_all)))
    perm = np.random.default_rng(0).permutation(len(x_all))
    return (torch.tensor(x_all[perm[n_val:]], dtype=torch.float32),
            torch.tensor(x_all[perm[:n_val]], dtype=torch.float32))


def evaluate(model, scaler, heldout_raw, attacks, L):
    ns, _ = recon_error(model, torch.tensor(sequences(scaler.transform(heldout_raw), L),
                                            dtype=torch.float32))
    thr = float(np.percentile(ns, THRESHOLD_PERCENTILE))
    yt, ys = [np.zeros(len(ns), int)], [ns]
    per_file = {}
    for a, raw in attacks.items():
        w = sequences(scaler.transform(raw), L)
        if len(w) == 0:
            continue
        s, resid = recon_error(model, torch.tensor(w, dtype=torch.float32))
        per_file[a] = (s, resid)
        yt.append(np.ones(len(s), int))
        ys.append(s)
    yt, ys = np.concatenate(yt), np.concatenate(ys)
    yp = (ys > thr).astype(int)
    return dict(threshold=thr, n=len(yt),
                f1=f1_score(yt, yp), auc=roc_auc_score(yt, ys),
                aucpr=average_precision_score(yt, ys)), per_file



def shap_attribution(model, x_background, x_samples, var_names, label,
                     outdir="Baselines/outputs",
                     dataset="Robotics (Pepper)", arch=None, fault=None):
    """
    Post-hoc SHAP attribution over the reconstruction error.

    GradientExplainer is used rather than DeepExplainer because it supports convolutional and
    attention architectures, where DeepExplainer's layer rules do not cover every op. The
    model is wrapped so its output is the scalar reconstruction error, which is the quantity
    the detector actually thresholds.
    """
    class _ErrWrap(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            return torch.mean((out[:, -1, :] - x[:, -1, :]) ** 2, dim=1).unsqueeze(1)

    wrapped = _ErrWrap(model)
    try:
        # GradientExplainer samples the background distribution, and it draws those samples
        # from whatever RNG state training happened to leave behind. Without this the same
        # trained model attributes differently on every run: a rerun that changed nothing
        # moved the LSTM's top LedsControl variable from LedEarR3 to LedFaceBR5 and the
        # TCN's second JointControl variable from SonarB to LShoulderPitch. Seeding here
        # makes the explanation a function of the model rather than of call order.
        #
        # It does not make SHAP stable, it makes it repeatable. How much the explanation
        # actually moves when only the explainer seed changes is measured separately, in
        # tools/shap_stability.py, because that instability is a property of post-hoc
        # attribution worth reporting rather than hiding.
        torch.manual_seed(SHAP_SEED)
        np.random.seed(SHAP_SEED)
        expl = shap.GradientExplainer(wrapped, x_background)
        vals = expl.shap_values(x_samples)
        if isinstance(vals, list):
            vals = vals[0]
        vals = np.asarray(vals)
        while vals.ndim > 3:
            vals = vals[..., 0]
        agg = np.abs(vals).sum(axis=1)                       # sum over the time axis
        order = np.argsort(agg.mean(axis=0))[::-1][:3]
        top = [var_names[i] for i in order]

        # Full SHAP ranking, not only the top three names. The comparison the supervisor
        # asked for -- causal attribution against the neural baselines on the same
        # relevance metric -- needs every variable's magnitude, and agg was discarded
        # here apart from the summary plot. Schema matches attribution_full.csv so one
        # analysis reads both. Rows for a given architecture and fault are replaced on
        # a re-run, never appended twice.
        _mean = agg.mean(axis=0)
        # Split on the last underscore only when the caller has not said which
        # half is which. Robotics fault names carry no underscore so the split
        # was unambiguous; Attack_1 and Fault_1 are not, and splitting those
        # silently produced an architecture called "LSTM Autoencoder_Attack".
        if arch is not None and fault is not None:
            _arch, _fault = arch, fault
        else:
            _arch, _, _fault = label.rpartition("_")
        os.makedirs(outdir, exist_ok=True)
        _fp = os.path.join(outdir, "shap_full.csv")
        _hdr = ["Dataset", "Model Variant", "Fault Name", "Rank", "Variable", "Score"]
        _kept = []
        if os.path.exists(_fp):
            with io.open(_fp, "r", newline="", encoding="utf-8") as _f:
                _rows = list(csv.reader(_f))
            _kept = [r for r in _rows[1:]
                     if len(r) > 2 and not (r[0] == dataset and
                                           r[1] == _arch and r[2] == _fault)]
        with io.open(_fp, "w", newline="", encoding="utf-8") as _f:
            _w = csv.writer(_f)
            _w.writerow(_hdr)
            _w.writerows(_kept)
            for _r, _i in enumerate(np.argsort(_mean)[::-1], start=1):
                _w.writerow([dataset, _arch, _fault,
                             _r, var_names[_i], "%.6f" % _mean[_i]])
        os.makedirs(outdir, exist_ok=True)
        safe = label.lower().replace(" ", "_").replace("/", "_")
        plt.figure()
        shap.summary_plot(agg, x_samples[:, -1, :].numpy(), feature_names=var_names, show=False)
        plt.savefig(os.path.join(outdir, f"shap_{safe}.png"), dpi=300, bbox_inches="tight")
        plt.close()
        return top
    except Exception as exc:
        return [f"SHAP unavailable ({type(exc).__name__})"]


def main():
    quick = "--quick" in sys.argv
    scaler, train_raw, heldout_raw, attacks, input_dim, var_names, sub = load_data()
    print(f"Variables {input_dim} (from PCMCI cache, subsample={sub}); "
          f"fault-free {len(train_raw)} train / {len(heldout_raw)} held out\n")

    summary = []
    shap_rows = []
    for name, cls in ARCHITECTURES.items():
        print("=" * 78)
        print(f" {name}")
        print("=" * 78)
        L = SEQ_LENGTHS[0]
        x_tr, x_val = split(scaler, train_raw, L)

        if quick:
            best_kw = dict(QUICK)
        else:
            print(f"{'hidden':>7} {'layers':>7} {'lr':>7} {'val_loss':>10} {'epoch':>6}")
            print("-" * 42)
            results = []
            for hd, nl, lr in itertools.product(*SEARCH.values()):
                try:
                    _, v, ep = train(cls, x_tr, x_val, input_dim, SEEDS[0],
                                     hidden_dim=hd, num_layers=nl, lr=lr)
                except Exception as exc:
                    print(f"{hd:7d} {nl:7d} {lr:7.3f}   FAILED {type(exc).__name__}")
                    continue
                print(f"{hd:7d} {nl:7d} {lr:7.3f} {v:10.5f} {ep:6d}")
                results.append((v, dict(hidden_dim=hd, num_layers=nl, lr=lr)))
            if not results:
                print("  no configuration trained\n")
                continue
            best_kw = min(results, key=lambda r: r[0])[1]
            print(f"\n  selected: {best_kw}")

        runs = []
        for seed in SEEDS:
            m, v, ep = train(cls, x_tr, x_val, input_dim, seed, **best_kw)
            met, per_file = evaluate(m, scaler, heldout_raw, attacks, L)
            runs.append(met)
            if seed == SEEDS[0]:
                first_model, first_per_file = m, per_file
        f1s = np.array([r["f1"] for r in runs])
        aucs = np.array([r["auc"] for r in runs])
        print(f"  {len(SEEDS)} seeds: F1 {f1s.mean():.4f} +/- {f1s.std():.4f}   "
              f"AUC {aucs.mean():.4f} +/- {aucs.std():.4f}   n={runs[0]['n']}")
        print(f"  RESULT {name}: F1={f1s.mean():.4f} AUC={aucs.mean():.4f} "
              f"AUCPR={np.mean([r['aucpr'] for r in runs]):.4f}")

        # top attributed variables per attack, by aggregated residual
        for a, (sc, resid) in first_per_file.items():
            mag = np.linalg.norm(resid, axis=0)
            top = np.argsort(mag)[::-1][:3]
            print(f"  ATTR {a}: " + ", ".join(var_names[t] for t in top))

        # Post-hoc SHAP on the anomalous windows of EVERY attack file.
        #
        # This previously ran on ATTACK_FILES[0] alone and only printed its result, which
        # made the comparison the reviewer asked for -- causal attribution against post-hoc
        # XAI -- impossible to report: one architecture, one fault, nothing persisted,
        # against nine causal configurations across three faults. SHAP now covers the same
        # three faults for every architecture and the rows are written to CSV alongside the
        # residual-based attributions, so the two families are compared over the same grid.
        for tgt in ATTACK_FILES:
            if tgt not in first_per_file:
                continue
            sc, _ = first_per_file[tgt]
            w = torch.tensor(sequences(scaler.transform(attacks[tgt]), L), dtype=torch.float32)
            idx = np.where(sc > runs[0]["threshold"])[0][:30]
            if len(idx) == 0:
                continue
            shap_top = shap_attribution(first_model, x_tr[:50], w[idx], var_names,
                                        f"{name}_{tgt.replace('.csv', '')}")
            print(f"  SHAP {tgt}: " + ", ".join(shap_top))
            shap_rows.append({
                "Dataset": "Robotics (Pepper)",
                "Model Variant": name,
                "Attribution": "SHAP (post-hoc)",
                "Fault Name": tgt,
                "Top Variable 1": shap_top[0] if len(shap_top) > 0 else "",
                "Top Variable 2": shap_top[1] if len(shap_top) > 1 else "",
                "Top Variable 3": shap_top[2] if len(shap_top) > 2 else "",
            })
        summary.append((name, f1s.mean(), aucs.mean(), np.mean([r["aucpr"] for r in runs]),
                        runs[0]["n"], best_kw))
        print()

    if shap_rows:
        import csv as _csv
        out = "Baselines/outputs/shap_attribution.csv"
        with io.open(out, "w", encoding="utf-8", newline="") as f:
            w_ = _csv.DictWriter(f, fieldnames=list(shap_rows[0].keys()))
            w_.writeheader()
            w_.writerows(shap_rows)
        print(f"wrote {out} ({len(shap_rows)} rows)")
        print()

    print("=" * 78)
    print(f"{'Architecture':26} {'F1':>8} {'AUC-ROC':>9} {'AUC-PR':>8} {'n':>6}  config")
    print("-" * 78)
    for name, f1, auc, apr, n, kw in summary:
        print(f"{name:26} {f1:8.4f} {auc:9.4f} {apr:8.4f} {n:6d}  {kw}")


if __name__ == "__main__":
    main()
