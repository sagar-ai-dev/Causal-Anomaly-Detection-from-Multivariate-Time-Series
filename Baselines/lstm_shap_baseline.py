"""
LSTM autoencoder + SHAP neural baseline on the Pepper robotics dataset.

Evaluation protocol is deliberately identical to the 27 causal configurations:
  * same variables and resampling rate (read from PCMCI's cached graph),
  * same 70/30 fault-free train/evaluation split (TRAINING_FRAC),
  * same decision-threshold rule -- the 95th percentile of the FAULT-FREE score
    distribution, which never sees an attack label,
  * same metric code (reporting_helper.generate_plots_and_reports).

Two defects in the original version of this script are fixed here, because together they
made the comparison against the causal pipelines meaningless:

  1. It trained for a fixed 5 epochs. The model had not converged -- loss was still
     falling and AUC still climbing at 60 epochs -- so the baseline was being
     handicapped relative to the tuned causal pipelines. Training now runs to
     convergence with a validation split and early stopping.

  2. It trained on ALL fault-free data and then scored the normal class on that same
     data, while the causal pipelines score a held-out 30%. That is a train/test leak
     on the negative class, flattering the baseline. The split is now honoured, and the
     scaler is fitted on the training portion only.

Reported metrics are the mean over SEEDS independent runs; a single-seed neural result is
not stable at these margins. The seed-0 model is the one carried into SHAP and into the
plots, so the artefacts correspond to a real run rather than an average.
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

# Import the same reporting helper the causal pipelines use, so metrics are computed by
# identical code rather than a parallel implementation that could drift.
sys.path.append(os.path.abspath("PCMCI/PCMCI_robotic"))
import reporting_helper

PREFIX = "PCMCI/PCMCI_robotic/pepper_csv/"
NORMAL_FILE = "normal.csv"
ATTACK_FILES = ["WheelsControl.csv", "JointControl.csv", "LedsControl.csv"]

TRAINING_FRAC = 0.7          # matches the causal pipelines
VAL_FRAC = 0.15              # of the training portion, held out for early stopping
SEQ_LENGTH = 5
HIDDEN_DIM = 16
BATCH_SIZE = 128
LEARNING_RATE = 0.01
MAX_EPOCHS = 300
PATIENCE = 15                # epochs without validation improvement before stopping
THRESHOLD_PERCENTILE = 95.0  # identical rule to all 27 causal configurations
SEEDS = [42, 43, 44, 45, 46]


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        h = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(h)
        return out


def create_sequences(data, seq_length=SEQ_LENGTH):
    if len(data) <= seq_length:
        return np.empty((0, seq_length, data.shape[1]), dtype=data.dtype)
    return np.stack([data[i:i + seq_length] for i in range(len(data) - seq_length)])


def reconstruction_mse(model, x):
    """Per-window reconstruction error at the final timestep."""
    with torch.no_grad():
        out = model(x)
        err = out[:, -1, :] - x[:, -1, :]
        return torch.mean(err ** 2, dim=1).numpy(), np.abs(err.numpy())


def train_to_convergence(x_train, x_val, input_dim, seed, verbose=False):
    """Train until validation loss stops improving. Returns the best-validation model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LSTMAutoencoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(TensorDataset(x_train), batch_size=BATCH_SIZE, shuffle=True)

    best_val, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val), x_val))

        if val_loss < best_val - 1e-6:
            best_val, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    if verbose:
        print(f"  seed {seed}: best epoch {best_epoch}/{epoch}, val loss {best_val:.5f}")
    return model, best_epoch, epoch, best_val


def main():
    print("\n========== NEURAL BASELINE: LSTM AUTOENCODER + SHAP ==========")

    causal_path = os.path.join(PREFIX, "pepper_normal.npz")
    if not os.path.exists(causal_path):
        print("PCMCI cache missing; cannot match variable selection. Run PCMCI_robotic first.")
        return

    cache = np.load(causal_path, allow_pickle=True)
    subsample, nonconst = int(cache["subsample"]), cache["nonconst"]
    var_names = cache["var"][nonconst].tolist()
    print(f"Variables: {len(nonconst)} (from PCMCI cache), subsample={subsample}")

    normal_df = pd.read_csv(os.path.join(PREFIX, NORMAL_FILE), delimiter=",")
    values = normal_df.drop(columns=["timestamp"]).values

    # Identical split to the causal pipelines: truncate to TRAINING_FRAC first, then subsample.
    train_raw = np.nan_to_num(values[:int(TRAINING_FRAC * len(values))][::subsample, nonconst])
    full_raw = np.nan_to_num(values[::subsample, nonconst])
    heldout_raw = full_raw[len(train_raw):]
    print(f"Fault-free split: {len(train_raw)} training rows, {len(heldout_raw)} held-out rows")

    if len(heldout_raw) <= SEQ_LENGTH:
        print("Held-out fault-free split too short to score. Aborting.")
        return

    # Scaler is fitted on the training portion ONLY -- fitting on all fault-free data would
    # leak held-out statistics into the normal class.
    scaler = StandardScaler().fit(train_raw)

    x_all = create_sequences(scaler.transform(train_raw))
    # Validation windows are drawn at RANDOM, not as a trailing contiguous block. A trailing
    # block is a different operating regime of the robot, so validation loss does not track
    # training loss and early stopping fires on the first epoch regardless of learning.
    n_val = max(1, int(VAL_FRAC * len(x_all)))
    _rng = np.random.default_rng(0)
    _perm = _rng.permutation(len(x_all))
    x_tr = torch.tensor(x_all[_perm[n_val:]], dtype=torch.float32)
    x_val = torch.tensor(x_all[_perm[:n_val]], dtype=torch.float32)
    print(f"Training windows: {len(x_tr)}, validation windows: {len(x_val)}, "
          f"input dim: {len(nonconst)}")

    x_heldout = torch.tensor(create_sequences(scaler.transform(heldout_raw)), dtype=torch.float32)

    attack_windows, attack_rows = {}, {}
    for name in ATTACK_FILES:
        df = pd.read_csv(os.path.join(PREFIX, name), delimiter=",")
        raw = np.nan_to_num(df.drop(columns=["timestamp"]).values[::subsample, nonconst])
        attack_rows[name] = raw.shape[0]
        attack_windows[name] = torch.tensor(
            create_sequences(scaler.transform(raw)), dtype=torch.float32)

    print(f"\nTraining to convergence (max {MAX_EPOCHS} epochs, patience {PATIENCE}), "
          f"{len(SEEDS)} seeds...")
    runs, first = [], None
    for seed in SEEDS:
        model, best_epoch, last_epoch, val_loss = train_to_convergence(
            x_tr, x_val, len(nonconst), seed, verbose=True)

        normal_scores, _ = reconstruction_mse(model, x_heldout)
        threshold = float(np.percentile(normal_scores, THRESHOLD_PERCENTILE))

        y_true = [np.zeros(len(normal_scores), dtype=int)]
        y_score = [normal_scores]
        per_file_scores, per_file_resid = {}, {}
        for name, window in attack_windows.items():
            if len(window) == 0:
                continue
            mse, resid = reconstruction_mse(model, window)
            per_file_scores[name] = mse
            per_file_resid[name] = resid
            y_true.append(np.ones(len(mse), dtype=int))
            y_score.append(mse)

        y_true = np.concatenate(y_true)
        y_score = np.concatenate(y_score)
        y_pred = (y_score > threshold).astype(int)
        runs.append({
            "seed": seed, "best_epoch": best_epoch, "epochs_run": last_epoch,
            "val_loss": val_loss, "threshold": threshold,
            "f1": f1_score(y_true, y_pred), "auc": roc_auc_score(y_true, y_score),
        })
        if first is None:
            first = (model, normal_scores, threshold, per_file_scores, per_file_resid)

    print("\n--- PER-SEED RESULTS ---")
    print(f"{'seed':>6} {'best_ep':>8} {'ran':>5} {'val_loss':>10} {'F1':>8} {'AUC':>8}")
    for r in runs:
        print(f"{r['seed']:6d} {r['best_epoch']:8d} {r['epochs_run']:5d} "
              f"{r['val_loss']:10.5f} {r['f1']:8.4f} {r['auc']:8.4f}")
    f1s = np.array([r["f1"] for r in runs])
    aucs = np.array([r["auc"] for r in runs])
    eps = np.array([r["best_epoch"] for r in runs])
    print(f"\n  F1  mean {f1s.mean():.4f} +/- {f1s.std():.4f}   "
          f"AUC mean {aucs.mean():.4f} +/- {aucs.std():.4f}")
    print(f"  converged at epoch {eps.mean():.1f} +/- {eps.std():.1f} "
          f"(the original script stopped at a fixed 5)")

    # Artefacts come from the first seed so they correspond to one real run, not an average.
    model, normal_scores, threshold, per_file_scores, per_file_resid = first
    print(f"\nArtefacts from seed {SEEDS[0]}; threshold {threshold:.4f}")

    pad = SEQ_LENGTH
    attack_scores_by_file, attack_labels_by_file, target_wise_residuals = {}, {}, {}
    for name, mse in per_file_scores.items():
        attack_scores_by_file[name] = np.concatenate([np.zeros(pad), mse])
        attack_labels_by_file[name] = np.ones(len(mse) + pad)
        target_wise_residuals[name] = np.vstack(
            [np.zeros((pad, len(nonconst))), per_file_resid[name]])

    reporting_helper.generate_plots_and_reports(
        output_dir="Baselines/outputs",
        model_variant="LSTM Autoencoder",
        model_name="Neural Baseline",
        normal_scores=np.concatenate([np.zeros(pad), normal_scores]),
        attack_scores_by_file=attack_scores_by_file,
        threshold=threshold,
        var_names=var_names,
        target_indices=list(range(len(nonconst))),
        directed_edges=[],
        target_wise_residuals_by_file=target_wise_residuals,
        attack_labels_by_file=attack_labels_by_file,
        dataset_name="Robotics (Pepper)",
    )

    print("\n--- SHAP POST-HOC XAI ---")

    class MSEWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.model = inner

        def forward(self, x):
            out = self.model(x)
            return torch.mean((out[:, -1, :] - x[:, -1, :]) ** 2, dim=1).unsqueeze(1)

    mse_model = MSEWrapper(model)
    explainer = shap.DeepExplainer(mse_model, x_tr[:100])

    target = ATTACK_FILES[0]
    windows = attack_windows[target]
    scores = per_file_scores[target]
    anomalous = np.where(scores > threshold)[0]
    if len(anomalous) == 0:
        print(f"No windows in {target} exceed the threshold; skipping SHAP.")
        return

    samples = windows[anomalous[:50]]
    # check_additivity is disabled because DeepExplainer approximates gradients through the
    # LSTM cell, so the additivity assertion fails on an otherwise correct explanation.
    shap_values = explainer.shap_values(samples, check_additivity=False)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap.summary_plot(np.abs(shap_values).sum(axis=1), samples[:, -1, :].numpy(),
                      feature_names=var_names, show=False)
    plt.savefig("Baselines/outputs/shap_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary over {len(samples)} anomalous windows of {target} saved to: "
          "Baselines/outputs/shap_feature_importance.png")


if __name__ == "__main__":
    main()
