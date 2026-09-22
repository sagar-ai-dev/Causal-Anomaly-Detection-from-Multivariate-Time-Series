"""
Professor Meli's XAI-course lab script, run verbatim as a reference implementation.

Only two changes from the version supplied:
  * the cached model filename is pepper_normal_LAB.npz, so this does not read or overwrite
    the thesis pipeline's own cache;
  * the plot is written to a scratch filename and plt.show() is already guarded.

Nothing about the method is altered. This exists to establish what the lab actually
produces on the Pepper data, so the thesis pipeline can be compared against it.
"""

import os
import numpy as np
import pandas as pd
import warnings
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from scipy.stats import ConstantInputWarning
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


warnings.filterwarnings('ignore', category=ConstantInputWarning)

ALPHA = 0.05
TRAINING_FRAC = 0.7
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.path.join(BASE_DIR, "pepper_csv")
TASK = "pepper"
MAX_FREQ_COMPONENTS = 5

DETECTION_THRESHOLD_MULTIPLIER = 0.5
CAUSAL_STRENGTH_MULTIPLIER = 1.0
CONSECUTIVE_K = 1


def read_data(path: str, task: str) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter="," if task == "pepper" else ";")
    if task == "pepper":
        df["Timestamp"] = df["timestamp"]
        df.set_index("Timestamp", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
    else:
        df["Timestamp"] = pd.to_datetime(df[" Timestamp"].str.strip(),
                                         format="%d/%m/%Y %I:%M:%S %p")
        df.set_index("Timestamp", inplace=True)
        df.drop(columns=[" Timestamp"], inplace=True)
    return df


def learn_causal_model(normal_csv_path: str, save_path: str):
    print("Learning causal model...")

    df = read_data(normal_csv_path, TASK)
    normal_data = pp.DataFrame(np.nan_to_num(df.values))
    normal_data.values[0] = normal_data.values[0][:int(TRAINING_FRAC * np.shape(normal_data.values[0])[0]), :]

    frequencies = []
    for index in range(np.shape(normal_data.values[0])[1]):
        if any([el for el in normal_data.values[0][:, index] if int(el) != el]):
            w = np.fft.fft(normal_data.values[0][:, index])
            freqs = np.fft.fftfreq(len(w))
            mods = abs(w)
            max_indices = np.argsort(mods)[::-1][:MAX_FREQ_COMPONENTS]
            main_freq = []
            for i in max_indices:
                freq = freqs[i]
                main_freq.append(freq)
            frequencies += main_freq

    sorted_freq = np.sort([el for el in frequencies if el > 0])[::-1]
    for freq in sorted_freq:
        if len([fr for fr in sorted_freq if fr < freq]) / len(sorted_freq) < 0.95:
            max_freq = freq
            sorted_freq = [s for s in sorted_freq if s <= max_freq]
            break

    subsample = max(1, int(np.floor(1 / 10 / max_freq)))
    normal_data.values[0] = normal_data.values[0][::max(1, subsample), :]
    nonconst = [idx for idx in range(np.shape(normal_data.values[0])[1])
                if np.std(normal_data.values[0][:, idx]) > 0.01 * np.mean(normal_data.values[0][:, idx])]
    nonconst_data = normal_data.values[0][:, nonconst]
    for j in range(np.shape(nonconst_data)[1]):
        # MINIMAL GUARD ONLY. The lab line is
        #     x /= (max(x) - min(x)) + min(x)
        # which simplifies to x /= max(x). AccelerometerX has max == 0 on this data, so the
        # division produces NaN and tigramite refuses the frame -- the script cannot run as
        # supplied. Skipping the division for a zero divisor is the smallest change that lets
        # it proceed; nothing else about the method is altered. (29 further columns have a
        # negative max, where the division silently flips their sign; that is left as-is so
        # this stays a faithful reference rather than a correction.)
        _den = (np.max(nonconst_data[:, j]) - np.min(nonconst_data[:, j])) + np.min(nonconst_data[:, j])
        if _den != 0:
            nonconst_data[:, j] /= _den
    print("nonconst_data shape:", np.shape(nonconst_data))

    tau_max = int(np.floor(max_freq / np.mean(np.unique(sorted_freq))))
    print("tau_max:", tau_max, " subsample:", subsample, " n_vars:", len(nonconst))

    dataframe = pp.DataFrame(nonconst_data)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=ALPHA)

    np.savez(save_path,
             val_matrix=results["val_matrix"],
             p_matrix=results["p_matrix"],
             var=df.columns,
             subsample=subsample,
             nonconst=nonconst)
    print(f"Saved causal model to {save_path}")
    return results, subsample, nonconst, tau_max


def fit_normal_coeffs(normal_data: np.ndarray, causal_matrix: np.ndarray):
    indices = np.array(np.where(causal_matrix != 0))
    fine_coeffs = {}

    for var in np.unique(indices[1, :]):
        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        max_delay = var_indices[-1][2]
        var_indices = var_indices[:3]

        stack = [normal_data[max_delay - el[2]: len(normal_data) - el[2], el[0]]
                 for el in var_indices]
        X = np.column_stack(stack)
        y = normal_data[max_delay:, var]

        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
        fine_coeffs[var] = coeffs

    return fine_coeffs, indices


def compute_online_errors(data: np.ndarray, fine_coeffs: dict,
                          causal_matrix: np.ndarray, indices: np.ndarray):
    max_time = data.shape[0] - causal_matrix.shape[2]
    err = {}
    norm_agg = np.zeros((max_time, len(np.unique(indices[1, :]))))

    for t in range(max_time):
        for i, var in enumerate(np.unique(indices[1, :])):
            var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
            var_indices.sort(key=lambda x: x[2])
            max_delay = var_indices[-1][2]
            var_indices = var_indices[:3]

            end_idx = t + causal_matrix.shape[2]
            stack = [data[max_delay - el[2]: end_idx - el[2], el[0]]
                     for el in var_indices]
            X = np.column_stack(stack)
            y = data[max_delay:end_idx, var]

            X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
            coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept, y, rcond=None)

            if var not in err:
                err[var] = np.zeros((max_time, len(coeffs)))
            err[var][t, :] = coeffs - fine_coeffs[var]
            norm_agg[t, i] = np.linalg.norm(err[var][t, :])

    return err, norm_agg


def detect_anomalies(err_normal, err_attack, normal_data_len, normal):
    indices_error = []
    for var in err_attack.keys():
        for j in range(err_attack[var].shape[1]):
            thresh = DETECTION_THRESHOLD_MULTIPLIER * np.linalg.norm(err_normal[var][:normal_data_len, j])
            if not normal:
                indices_error += list(np.where(abs(err_attack[var][:, j]) > thresh)[0])
            else:
                indices_error += list(np.where(abs(err_attack[var][normal_data_len:, j]) > thresh)[0])
    return len(np.unique(indices_error))


def score_anomaly_timeline(err_normal, err_attack, normal_data_len, normal):
    if normal:
        start = normal_data_len
        max_time = min((err_attack[var].shape[0] - start for var in err_attack.keys()), default=0)
    else:
        start = 0
        max_time = min((err_attack[var].shape[0] for var in err_attack.keys()), default=0)

    scores = np.zeros(max_time, dtype=float)
    for var in err_attack.keys():
        for j in range(err_attack[var].shape[1]):
            thresh = DETECTION_THRESHOLD_MULTIPLIER * np.linalg.norm(err_normal[var][:normal_data_len, j])
            thresh = max(float(thresh), np.finfo(float).eps)
            values = np.abs(err_attack[var][start:start + max_time, j]) / thresh
            scores = np.maximum(scores, values)

    predictions = (scores > 1.0).astype(int)
    return predictions, scores


def compute_extended_metrics(normal_predictions, normal_scores, attack_predictions, attack_scores):
    y_true = np.concatenate([
        np.zeros(len(normal_predictions), dtype=int),
        np.ones(sum(len(pred) for pred in attack_predictions), dtype=int),
    ])
    y_pred = np.concatenate([normal_predictions] + attack_predictions)
    y_score = np.concatenate([normal_scores] + attack_scores)

    accuracy = accuracy_score(y_true, y_pred)
    try:
        auc_roc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc_roc = np.nan
    try:
        auc_pr = average_precision_score(y_true, y_score)
    except ValueError:
        auc_pr = np.nan
    return accuracy, auc_roc, auc_pr


def main():
    print(f"\n========== LAB REFERENCE: {TASK.upper()} ==========")
    causal_path = os.path.join(PREFIX, f"{TASK}_normal_LAB.npz")

    if not os.path.exists(causal_path):
        learn_causal_model(os.path.join(PREFIX, "normal.csv"), causal_path)
    else:
        print("Lab causal model found, loading...")

    f = np.load(causal_path, allow_pickle=True)
    val_matrix, p_matrix = f["val_matrix"], f["p_matrix"]
    subsample, nonconst = int(f["subsample"]), f["nonconst"]

    normal_matrix = val_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > CAUSAL_STRENGTH_MULTIPLIER * np.mean(abs(val_matrix)))

    normal_df = read_data(os.path.join(PREFIX, "normal.csv"), TASK)
    normal_data = np.nan_to_num(normal_df.values[:int(TRAINING_FRAC * len(normal_df))][::subsample, nonconst])
    normal_data_full = np.nan_to_num(normal_df.values[::subsample, nonconst])

    fine_coeffs, indices = fit_normal_coeffs(normal_data, normal_matrix)
    print(f"monitored targets: {len(np.unique(indices[1, :]))}, "
          f"normal_data rows: {len(normal_data)}, full: {len(normal_data_full)}")

    err_normal, norm_agg_normal = compute_online_errors(normal_data_full, fine_coeffs, normal_matrix, indices)
    _ = norm_agg_normal

    attack_paths = [
        os.path.join(PREFIX, "WheelsControl.csv"),
        os.path.join(PREFIX, "JointControl.csv"),
        os.path.join(PREFIX, "LedsControl.csv")
    ]
    attack_dfs = [read_data(p, TASK) for p in attack_paths]

    tpos, fpos, fneg = [], [], []
    attack_predictions, attack_scores = [], []

    fpos.append(detect_anomalies(err_normal, err_normal, len(normal_data), normal=True))
    normal_predictions, normal_scores = score_anomaly_timeline(
        err_normal, err_normal, len(normal_data), normal=True)

    norm_agg_attacks = []
    attack_names = []

    for path, df_attack in zip(attack_paths, attack_dfs):
        attack_name = os.path.basename(path)
        attack_names.append(attack_name)
        print(f"\n--- Analyzing anomaly: {attack_name} ---")

        attack_data = np.nan_to_num(df_attack.values[::subsample, nonconst])
        err_attack, norm_agg_attack = compute_online_errors(attack_data, fine_coeffs, normal_matrix, indices)
        norm_agg_attacks.append(norm_agg_attack)

        tp_count = detect_anomalies(err_normal, err_attack, len(normal_data), normal=False)
        tpos.append(tp_count)
        fneg.append(attack_data.shape[0] - tp_count)
        attack_prediction, attack_score = score_anomaly_timeline(
            err_normal, err_attack, len(normal_data), normal=False)
        if len(attack_prediction) < attack_data.shape[0]:
            pad_len = attack_data.shape[0] - len(attack_prediction)
            attack_prediction = np.concatenate([np.zeros(pad_len, dtype=int), attack_prediction])
            attack_score = np.concatenate([np.zeros(pad_len, dtype=float), attack_score])
        attack_predictions.append(attack_prediction)
        attack_scores.append(attack_score)

        dep_vals = {f["var"][nonconst][var]: np.linalg.norm(norm_agg_attack[:, i])
                    for i, var in enumerate(np.unique(indices[1, :]))}
        dep_vals_sorted = dict(sorted(dep_vals.items(), key=lambda x: x[1], reverse=True))
        top3 = list(dep_vals_sorted.items())[:3]
        print("  LAB_ATTR " + attack_name + ": " + ", ".join(f"{k}" for k, _ in top3))

    precision = np.sum(tpos) / (np.sum(tpos) + np.sum(fpos))
    recall = np.sum(tpos) / (np.sum(tpos) + np.sum(fneg))
    f1 = 2 * np.sum(tpos) / (2 * np.sum(tpos) + np.sum(fpos) + np.sum(fneg))
    accuracy, auc_roc, auc_pr = compute_extended_metrics(
        normal_predictions, normal_scores, attack_predictions, attack_scores)

    print("\n========== LAB METRICS ==========")
    print(f"LAB Precision: {precision:.4f}")
    print(f"LAB Recall:    {recall:.4f}")
    print(f"LAB F1 Score:  {f1:.4f}")
    print(f"LAB Accuracy:  {accuracy:.4f}")
    print(f"LAB AUC-ROC:   {auc_roc:.4f}")
    print(f"LAB AUC-PR:    {auc_pr:.4f}")
    print(f"LAB tp={np.sum(tpos)} fp={np.sum(fpos)} fn={np.sum(fneg)}")
    print("=================================\n")


if __name__ == "__main__":
    main()
