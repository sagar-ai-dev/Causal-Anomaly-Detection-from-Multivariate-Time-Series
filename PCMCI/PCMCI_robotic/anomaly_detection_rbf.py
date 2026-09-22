"""
==============================================================
 Causal Discovery & Anomaly Detection (RBF-Feature Tigramite)
==============================================================

Workflow:
1. Data Loading
2. Causal Model Learning (offline PCMCI)
3. Offline RBF-Feature Coefficient Estimation
4. Online Monitoring (moving-window coefficient updates)
5. Anomaly Detection
6. Metrics Computation

Author: (Sagar)
==============================================================
"""

# ================== IMPORTS ==================
import os
import numpy as np
import pandas as pd
import warnings
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from scipy.stats import ConstantInputWarning

from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore", category=ConstantInputWarning)

# ================== GLOBAL CONFIG ==================
ALPHA = 0.05
TRAINING_FRAC = 0.7
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.path.join(BASE_DIR, "pepper_csv")
TASK = "pepper"
MAX_FREQ_COMPONENTS = 5

# RBF feature regression config
RBF_GAMMA = 0.001
RBF_N_COMPONENTS = 50
RBF_RANDOM_STATE = 42
RIDGE_ALPHA = 1.0
RLS_LAMBDA = 0.995

# Anomaly detection tuning config
DETECTION_THRESHOLD_MULTIPLIER = 1.2
CAUSAL_STRENGTH_MULTIPLIER = 1.0


def _rls_update(x, y, w_prev, P_prev, lam=0.995):
    """
    Recursive Least Squares update with forgetting factor `lam`.

    Identical in all 27 configurations. The numerical safeguards are not optional:

      * Inputs and outputs are sanitised. One NaN entering the state poisons every
        subsequent update for that target, silently.
      * `denom` is checked for finiteness as well as magnitude. A non-finite denominator
        yields a zero or NaN gain rather than raising, so the filter would keep running on
        corrupted state.
      * P is re-symmetrised each step. Floating-point error in
        `P <- (P - g (Px)^T) / lam` accumulates asymmetry over thousands of iterations; the
        covariance then stops being positive-definite, the gain degrades, and the filter can
        diverge. This is a standard RLS failure mode and the guard is cheap.
    """
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = float(np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0))
    w_prev = np.nan_to_num(np.asarray(w_prev, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    P_prev = np.nan_to_num(np.asarray(P_prev, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    y_pred = float(x @ w_prev)
    e_t = float(y - y_pred)
    Px = P_prev @ x
    denom = float(lam + x @ Px)
    if (not np.isfinite(denom)) or abs(denom) < 1e-12:
        return w_prev, P_prev, e_t
    g_t = Px / denom
    w_new = w_prev + g_t * e_t
    P_new = (P_prev - np.outer(g_t, Px)) / float(lam)

    # RLS symmetry guard -- see docstring.
    P_new = (P_new + P_new.T) / 2.0

    w_new = np.nan_to_num(w_new, nan=0.0, posinf=0.0, neginf=0.0)
    P_new = np.nan_to_num(P_new, nan=0.0, posinf=0.0, neginf=0.0)
    return w_new, P_new, e_t


# ================================================================
#                         DATA LOADING
# ================================================================
def read_data(path: str, task: str) -> pd.DataFrame:
    """
    Load CSV data, handle timestamp indexing.

    Visual:
        CSV -> DataFrame indexed by Timestamp
    """
    df = pd.read_csv(path, delimiter="," if task == "pepper" else ";")
    if task == "pepper":
        df["Timestamp"] = df["timestamp"]
        df.set_index("Timestamp", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
    else:
        df["Timestamp"] = pd.to_datetime(
            df[" Timestamp"].str.strip(), format="%d/%m/%Y %I:%M:%S %p"
        )
        df.set_index("Timestamp", inplace=True)
        df.drop(columns=[" Timestamp"], inplace=True)
    return df


# ================================================================
#                     LEARN CAUSAL MODEL
# ================================================================
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
            main_freq = [freqs[i] for i in max_indices]
            frequencies += main_freq

    sorted_freq = np.sort([el for el in frequencies if el > 0])[::-1]
    for freq in sorted_freq:
        if len([fr for fr in sorted_freq if fr < freq]) / len(sorted_freq) < 0.95:
            max_freq = freq
            sorted_freq = [s for s in sorted_freq if s <= max_freq]
            break

    subsample = max(1, int(np.floor(1 / 10 / max_freq)))
    normal_data.values[0] = normal_data.values[0][::max(1, subsample), :]
    nonconst = [idx for idx in range(np.shape(normal_data.values[0])[1]) if np.std(normal_data.values[0][:, idx]) > 0.01 * np.mean(normal_data.values[0][:, idx])]
    nonconst_data = normal_data.values[0][:, nonconst]
    for j in range(np.shape(nonconst_data)[1]):
        # Min-max normalisation, matching PCMCI_healthcare and PCMCI_industrial. The earlier
        # form here was `x /= (max - min) + min`, which simplifies to `x /= max` and is not a
        # min-max scaling at all. ParCorr computes partial correlations and is invariant to
        # affine rescaling, so the learned graph is unchanged either way -- this is a
        # correctness and uniformity fix, not a result-changing one. See README section 4.
        col = nonconst_data[:, j]
        col_min = np.min(col)
        col_max = np.max(col)
        scale = col_max - col_min
        if scale <= 1e-12:
            nonconst_data[:, j] = 0.0
        else:
            nonconst_data[:, j] = (col - col_min) / scale
    
    tau_max = int(np.floor(max_freq / np.mean(np.unique(sorted_freq))))
    dataframe = pp.DataFrame(nonconst_data)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=ALPHA)

    np.savez(save_path, val_matrix=results["val_matrix"], p_matrix=results["p_matrix"], var=df.columns, subsample=subsample, nonconst=nonconst)
    return results, subsample, nonconst, tau_max

# ================================================================
#               OFFLINE RBF COEFFICIENT FITTING
# ================================================================
def fit_normal_rbf_coeffs(
    normal_data: np.ndarray,
    causal_matrix: np.ndarray,
    gamma: float = RBF_GAMMA,
    n_components: int = RBF_N_COMPONENTS,
    random_state: int = RBF_RANDOM_STATE,
    alpha: float = RIDGE_ALPHA,
):
    """
    Compute offline (baseline) coefficients using fixed RBF features.
    """
    indices = np.array(np.where(causal_matrix != 0))
    fine_coeffs = {}
    scalers = {}
    rbf_maps = {}

    for var in np.unique(indices[1, :]):
        var_indices = [
            indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var
        ]
        var_indices.sort(key=lambda x: x[2])
        max_delay = var_indices[-1][2]
        # At most three parents per target, taken in lag order, so the design matrix is
        # never wider than three columns plus the intercept however dense the graph. The
        # same truncation runs in fit_normal_coeffs and compute_online_errors; if the two
        # disagreed the online coefficients would not correspond to the offline baseline
        # they are differenced against.
        var_indices = var_indices[:3]

        stack = [
            normal_data[max_delay - el[2] : len(normal_data) - el[2], el[0]]
            for el in var_indices
        ]
        X = np.column_stack(stack)
        y = normal_data[max_delay:, var]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rbf_map = RBFSampler(
            gamma=gamma, n_components=n_components, random_state=random_state
        )
        X_rbf = rbf_map.fit_transform(X_scaled)

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_rbf, y)

        # Intercept first, matching the other eight RBF scripts and every linear and
        # polynomial script. Dropping it here made this the only configuration of 27 whose
        # drift vector excluded the intercept term, so "coefficient drift" meant something
        # slightly different in this one file.
        fine_coeffs[var] = np.concatenate([[float(model.intercept_)], np.ravel(model.coef_)])
        scalers[var] = scaler
        rbf_maps[var] = rbf_map

    return fine_coeffs, indices, scalers, rbf_maps


# ================================================================
#           ONLINE RBF COEFFICIENTS & ERROR COMPUTATION
# ================================================================
def compute_online_rbf_errors(
    data: np.ndarray,
    fine_coeffs: dict,
    causal_matrix: np.ndarray,
    indices: np.ndarray,
    scalers: dict,
    rbf_maps: dict,
    alpha: float = RIDGE_ALPHA,
    rls_lambda: float = RLS_LAMBDA,
):
    """
    Recompute coefficients online sample-by-sample via RLS in fixed RBF feature space.
    """
    max_time = data.shape[0] - causal_matrix.shape[2]
    unique_vars = np.unique(indices[1, :])
    err = {}
    norm_agg = np.zeros((max_time, len(unique_vars)))

    w_state = {}
    P_state = {}
    var_prep = {}

    for var in unique_vars:
        w0 = fine_coeffs[var].copy()
        P0 = np.eye(len(w0)) * (1.0 / alpha)
        w_state[var] = w0
        P_state[var] = P0

        var_indices = [
            indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var
        ]
        var_indices.sort(key=lambda x: x[2])
        max_delay = var_indices[-1][2]
        # At most three parents per target, taken in lag order, so the design matrix is
        # never wider than three columns plus the intercept however dense the graph. The
        # same truncation runs in fit_normal_coeffs and compute_online_errors; if the two
        # disagreed the online coefficients would not correspond to the offline baseline
        # they are differenced against.
        var_indices = var_indices[:3]

        stack = [
            data[max_delay - el[2] : max_time + max_delay - el[2], el[0]] for el in var_indices
        ]
        X = np.column_stack(stack)
        y = data[max_delay : max_time + max_delay, var]

        X_scaled = scalers[var].transform(X.astype(float))
        X_rbf = rbf_maps[var].transform(X_scaled)
        var_prep[var] = (X_rbf, y)
        err[var] = np.zeros((max_time, len(w0)))

    for t in range(max_time):
        for i, var in enumerate(unique_vars):
            X_rbf, y = var_prep[var]
            x_vec = np.concatenate([[1.0], X_rbf[t]])
            y_t = float(y[t])

            w_prev = w_state[var]
            P_prev = P_state[var]

            w_new, P_new, _ = _rls_update(x_vec, y_t, w_prev, P_prev, rls_lambda)
            w_state[var] = w_new
            P_state[var] = P_new

            err[var][t, :] = w_new - fine_coeffs[var]
            norm_agg[t, i] = np.linalg.norm(err[var][t, :])

    return err, norm_agg


# ================================================================
#                        ANOMALY DETECTION
# ================================================================
def compute_windowed_errors(data, window_len, fine_coeffs, causal_matrix, indices, *extra):
    """Fresh windowed runs so fault-free and attack data accumulate equal RLS drift."""
    n_windows = max(1, data.shape[0] // window_len)
    err_parts, agg_parts = {}, []
    for w in range(n_windows):
        chunk = data[w * window_len:(w + 1) * window_len]
        if chunk.shape[0] <= causal_matrix.shape[2] + 1:
            continue
        e, a = compute_online_rbf_errors(chunk, fine_coeffs, causal_matrix, indices, *extra)
        agg_parts.append(a)
        for var, arr in e.items():
            err_parts.setdefault(var, []).append(arr)
    err = {v: np.concatenate(q, axis=0) for v, q in err_parts.items()}
    agg = np.concatenate(agg_parts, axis=0) if agg_parts else np.zeros((0, len(np.unique(indices[1, :]))))
    return err, agg

def score_anomaly_timeline(err_baseline, err_eval, baseline_len):
    """
    Symmetric scoring, identical to the other 26 configurations: err_eval (a fresh online run)
    is scored against per-variable thresholds from err_baseline (a fresh windowed fault-free
    run of the same length). Both classes accumulate the same amount of RLS drift.
    """
    max_time = min((err_eval[var].shape[0] for var in err_eval.keys()), default=0)

    scores = np.zeros(max_time, dtype=float)
    for var in err_eval.keys():
        for j in range(err_eval[var].shape[1]):
            thresh = DETECTION_THRESHOLD_MULTIPLIER * np.linalg.norm(err_baseline[var][:baseline_len, j])
            thresh = max(float(thresh), np.finfo(float).eps)
            values = np.abs(err_eval[var][:max_time, j]) / thresh
            scores = np.maximum(scores, values)

    return (scores > 1.0).astype(int), scores


# ================================================================
#                         MAIN PIPELINE
# ================================================================
def main():
    print(f"\n========== TASK: {TASK.upper()} ==========")
    print(f"RBF gamma: {RBF_GAMMA}")
    print(f"RBF components: {RBF_N_COMPONENTS}")
    print(f"Ridge alpha: {RIDGE_ALPHA}")

    causal_path = os.path.join(PREFIX, f"{TASK}_normal.npz")
    print(f"Causal model path: {causal_path}")

    if not os.path.exists(causal_path):
        learn_causal_model(os.path.join(PREFIX, "normal.csv"), causal_path)
    else:
        print("Causal model found, loading...")

    f = np.load(causal_path, allow_pickle=True)
    val_matrix, p_matrix = f["val_matrix"], f["p_matrix"]
    subsample, nonconst = int(f["subsample"]), f["nonconst"]

    normal_matrix = val_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > CAUSAL_STRENGTH_MULTIPLIER * np.mean(abs(val_matrix)))

    normal_df = read_data(os.path.join(PREFIX, "normal.csv"), TASK)
    normal_data = np.nan_to_num(
        normal_df.values[: int(TRAINING_FRAC * len(normal_df))][::subsample, nonconst]
    )
    normal_data_full = np.nan_to_num(normal_df.values[::subsample, nonconst])

    fine_coeffs, indices, scalers, rbf_maps = fit_normal_rbf_coeffs(
        normal_data, normal_matrix
    )

    err_normal, norm_agg_normal = compute_online_rbf_errors(
        normal_data_full, fine_coeffs, normal_matrix, indices, scalers, rbf_maps
    )
    _ = norm_agg_normal

    attack_paths = [
        os.path.join(PREFIX, "WheelsControl.csv"),
        os.path.join(PREFIX, "JointControl.csv"),
        os.path.join(PREFIX, "LedsControl.csv"),
    ]
    attack_dfs = [read_data(p, TASK) for p in attack_paths]

    attack_predictions, attack_scores = [], []

    attack_data_list = [np.nan_to_num(d.values[::subsample, nonconst]) for d in attack_dfs]
    window_len = int(np.mean([len(a) for a in attack_data_list]))
    err_baseline, _ = compute_windowed_errors(normal_data, window_len, fine_coeffs, normal_matrix, indices, scalers, rbf_maps)
    baseline_len = next(iter(err_baseline.values())).shape[0]
    held_out = normal_data_full[len(normal_data):]
    err_eval, _ = compute_windowed_errors(held_out, window_len, fine_coeffs, normal_matrix, indices, scalers, rbf_maps)
    normal_predictions, normal_scores = score_anomaly_timeline(err_baseline, err_eval, baseline_len)

    norm_agg_attacks = []
    attack_names = []

    for path, df_attack in zip(attack_paths, attack_dfs):
        attack_name = os.path.basename(path)
        attack_names.append(attack_name)
        print(f"\n--- Analyzing anomaly: {attack_name} ---")

        attack_data = np.nan_to_num(df_attack.values[::subsample, nonconst])
        err_attack, norm_agg_attack = compute_online_rbf_errors(
            attack_data, fine_coeffs, normal_matrix, indices, scalers, rbf_maps
        )
        norm_agg_attacks.append(norm_agg_attack)

        attack_prediction, attack_score = score_anomaly_timeline(err_baseline, err_attack, baseline_len)
        if len(attack_prediction) < attack_data.shape[0]:
            pad_len = attack_data.shape[0] - len(attack_prediction)
            attack_prediction = np.concatenate([np.zeros(pad_len, dtype=int), attack_prediction])
            attack_score = np.concatenate([np.zeros(pad_len, dtype=float), attack_score])
        attack_predictions.append(attack_prediction)
        attack_scores.append(attack_score)

    import reporting_helper

    attack_scores_dict = dict(zip(attack_names, attack_scores))
    attack_labels_dict = {}
    target_wise_residuals_by_file = dict(zip(attack_names, norm_agg_attacks))

    for name, data_matrix in zip(attack_names, [read_data(p, TASK).values[::subsample, nonconst] for p in attack_paths]):
        attack_labels_dict[name] = np.ones(data_matrix.shape[0])

    output_dir = os.path.join(BASE_DIR, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Decision threshold is derived from FAULT-FREE data only: the THRESHOLD_PERCENTILE-th
    # percentile of the normal-score distribution. It never sees an attack label, so the
    # reported F1 is an achievable operating point, not the maximum attainable F1 obtained
    # by searching thresholds against ground truth. Identical rule for all 27 configurations.
    THRESHOLD_PERCENTILE = 95.0
    best_threshold = float(np.percentile(np.nan_to_num(normal_scores), THRESHOLD_PERCENTILE))
    print(f"Fault-free {THRESHOLD_PERCENTILE:.0f}th-percentile threshold: {best_threshold:.4f}")

    reporting_helper.generate_plots_and_reports(
        output_dir=output_dir,
        model_variant="RBF + Ridge",
        model_name="PCMCI",
        normal_scores=normal_scores,
        attack_scores_by_file=attack_scores_dict,
        threshold=best_threshold,
        var_names=f["var"][nonconst].tolist(),
        target_indices=np.unique(indices[1, :]).tolist(),
        directed_edges=[], 
        target_wise_residuals_by_file=target_wise_residuals_by_file,
        attack_labels_by_file=attack_labels_dict,
        dataset_name="Robotics (Pepper)"
    )


# ================================================================
#                          RUN SCRIPT
# ================================================================
if __name__ == "__main__":
    main()
