"""
==============================================================
 Causal Discovery & Anomaly Detection (Linear Least-Squares Tigramite - Healthcare / ECG)
==============================================================

Workflow:
1. Data Loading
2. Causal Model Learning (offline PCMCI)
3. Offline Linear Least-Squares Coefficient Estimation
4. Online Monitoring (moving-window linear coefficient updates)
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
from sklearn.linear_model import Ridge


warnings.filterwarnings('ignore', category=ConstantInputWarning)

# ================== GLOBAL CONFIG ==================
ALPHA = 0.05
TRAINING_FRAC = 0.7
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.path.join(BASE_DIR, "Healthcare dataset")
TASK = "healthcare"
MAX_FREQ_COMPONENTS = 5

NORMAL_FILE = "healthcare_normal_train.pkl"
ATTACK_FILE = "healthcare_attacks.pkl"

# Anomaly detection tuning config
DETECTION_THRESHOLD_MULTIPLIER = 2.0
ANOMALY_SCORE_THRESHOLD = 1.0
CAUSAL_STRENGTH_MULTIPLIER = 1.0
# Regularisation strength is a property of the regression type and must be the
# same for all three discovery algorithms on a given dataset. GES and CAM-UV use
# 10.0 throughout healthcare; PCMCI previously used 1000.0 (linear, polynomial)
# and 1.0 (RBF), i.e. up to 1000x the regularisation on identical data.
RIDGE_ALPHA = 10.0
RLS_LAMBDA = 1.0

ECG_LEADS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ================================================================
#                         DATA LOADING
# ================================================================

# ============================================================================
#            ECG INDEPENDENT-LEAD ABLATION  (off unless requested)
# ============================================================================
# Einthoven's and Goldberger's relations make four of the twelve standard leads
# exact linear functions of the other eight:
#
#     III = II - I,  aVR = -(I + II)/2,  aVL = I - II/2,  aVF = II - I/2
#
# A causal graph over all twelve therefore recovers algebra alongside
# physiology. Setting ECG_INDEPENDENT_LEADS=1 restricts the pipeline to the
# eight independent leads so the two can be told apart. Cache and output paths
# are suffixed, so an ablation run cannot overwrite the headline results.
INDEPENDENT_LEADS = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
ABLATION_8_LEADS = os.environ.get("ECG_INDEPENDENT_LEADS") == "1"
ABL = "_leads8" if ABLATION_8_LEADS else ""


def restrict_leads(df):
    """Keep only the eight independent leads, when the ablation is enabled."""
    if not ABLATION_8_LEADS or not hasattr(df, "columns"):
        return df
    keep = [c for c in INDEPENDENT_LEADS if c in df.columns]
    return df[keep] if len(keep) == len(INDEPENDENT_LEADS) else df


def _read_data_unrestricted(path: str, task: str):
    """
    Load CSV or PKL data for Healthcare ECG 12-lead time series.
    """
    if path.endswith(".pkl"):
        res = pd.read_pickle(path)
        if isinstance(res, pd.DataFrame):
            available_leads = [c for c in ECG_LEADS if c in res.columns]
            return res[available_leads] if available_leads else res
        return res
    else:
        df = pd.read_csv(path, delimiter="," if task == "pepper" else ";")
        available_leads = [c for c in ECG_LEADS if c in df.columns]
        if available_leads:
            df = df[available_leads]
        if len(df.columns) == 12:
            df.columns = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    return df


def read_data(path, task):
    return restrict_leads(_read_data_unrestricted(path, task))


# ================================================================
#                     LEARN CAUSAL MODEL
# ================================================================
def learn_causal_model(normal_csv_path: str, save_path: str):
    """
    Learn the causal graph using PCMCI.
    Tau_max is automatically computed from the dominant frequency.
    """
    print("Learning causal model...")

    df = read_data(normal_csv_path, TASK)
    normal_data = pp.DataFrame(np.nan_to_num(df.values))
    # restrict to training_frac
    normal_data.values[0] = normal_data.values[0][:int(TRAINING_FRAC * np.shape(normal_data.values[0])[0]), :]

    frequencies = []
    for index in range(np.shape(normal_data.values[0])[1]):
        series = normal_data.values[0][:, index]
        if np.std(series) <= 0:
            continue
        centered = series - np.mean(series)
        if np.allclose(centered, 0):
            continue
        w = np.fft.fft(centered)
        freqs = np.fft.fftfreq(len(w))
        mods = abs(w)
        max_indices = np.argsort(mods)[::-1][:MAX_FREQ_COMPONENTS]
        main_freq = []
        for i in max_indices:
            freq = freqs[i]
            if freq > 0:
                main_freq.append(freq)
        frequencies += main_freq

    sorted_freq = np.sort([el for el in frequencies if el > 0])[::-1]
    if len(sorted_freq) == 0:
        sorted_freq = np.array([0.1])
    for freq in sorted_freq:
        if len([fr for fr in sorted_freq if fr < freq]) / len(sorted_freq) < 0.95:
            max_freq = freq
            sorted_freq = [s for s in sorted_freq if s <= max_freq]
            break
    else:
        max_freq = float(sorted_freq[0])

    subsample = max(1, int(np.floor(1 / 10 / max_freq)))
    normal_data.values[0] = normal_data.values[0][::max(1, subsample), :]
    nonconst = [idx for idx in range(np.shape(normal_data.values[0])[1]) if np.std(normal_data.values[0][:, idx]) > 1e-8]
    nonconst_data = normal_data.values[0][:, nonconst]
    for j in range(np.shape(nonconst_data)[1]):
        col = nonconst_data[:, j]
        col_min = np.min(col)
        col_max = np.max(col)
        scale = col_max - col_min
        if scale <= 1e-12:
            nonconst_data[:, j] = 0.0
        else:
            nonconst_data[:, j] = (col - col_min) / scale
    print(np.shape(nonconst_data))

    # Evaluate links
    tau_max = int(np.floor(max_freq / np.mean(np.unique(sorted_freq))))
    print(tau_max)

    # Run PCMCI
    dataframe = pp.DataFrame(nonconst_data)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=ALPHA)

    # Save model for reuse
    np.savez(save_path,
             val_matrix=results["val_matrix"],
             p_matrix=results["p_matrix"],
             var=df.columns,
             subsample=subsample,
             nonconst=nonconst)
    print(f"Saved causal model to {save_path}")
    return results, subsample, nonconst, tau_max


# ================================================================
#       OFFLINE LINEAR LEAST-SQUARES COEFFICIENT FITTING
# ================================================================
def fit_normal_coeffs(normal_data: np.ndarray, causal_matrix: np.ndarray):
    """
    Compute offline (baseline) linear coefficients for each variable using least squares.

    Predictors are standardized per variable (StandardScaler) before fitting. The raw ECG
    leads have standard deviations of roughly 100-300 and differ substantially between
    leads; feeding them unscaled into the RLS covariance update leaves it badly
    conditioned, which suppresses the coefficient drift this detector relies on.
    Standardizing raised this variant's AUC-ROC from 0.609 to 0.809 (see FINAL_RESULTS.md).
    The fitted scalers are returned so online monitoring applies the identical transform.
    """
    indices = np.array(np.where(causal_matrix != 0))
    fine_coeffs = {}
    scale_maps = {}

    for var in np.unique(indices[1, :]):
        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        # At most three parents per target, taken in lag order, so the design matrix is
        # never wider than three columns plus the intercept however dense the graph. The
        # same truncation runs in fit_normal_coeffs and compute_online_errors; if the two
        # disagreed the online coefficients would not correspond to the offline baseline
        # they are differenced against.
        var_indices = var_indices[:3]
        max_delay = var_indices[-1][2]

        stack = [normal_data[max_delay - el[2]: len(normal_data) - el[2], el[0]]
                 for el in var_indices]
        X = np.column_stack(stack)
        y = normal_data[max_delay:, var]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scale_maps[var] = scaler

        # Ridge-regularized baseline (matching the Polynomial and RBF variants, which both
        # use Ridge). The previous unregularized lstsq fit produced a less stable baseline
        # on collinear ECG leads: AUC-ROC 0.739 with lstsq vs 0.809 with Ridge(alpha=1000).
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        model.fit(X_scaled, y)
        fine_coeffs[var] = np.concatenate([[float(model.intercept_)], np.ravel(model.coef_)])

    return fine_coeffs, indices, scale_maps


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


def compute_online_errors(data: np.ndarray, fine_coeffs: dict,
                           causal_matrix: np.ndarray, indices: np.ndarray,
                           scale_maps: dict):
    """
    Recompute linear coefficients online sample-by-sample via RLS.

    Applies the same per-variable StandardScaler fitted offline, so the online
    design matrix matches the basis the baseline coefficients were fitted in.
    """
    max_time = data.shape[0] - causal_matrix.shape[2]
    err = {}
    norm_agg = np.zeros((max_time, len(np.unique(indices[1, :]))))

    for i, var in enumerate(np.unique(indices[1, :])):
        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        # At most three parents per target, taken in lag order, so the design matrix is
        # never wider than three columns plus the intercept however dense the graph. The
        # same truncation runs in fit_normal_coeffs and compute_online_errors; if the two
        # disagreed the online coefficients would not correspond to the offline baseline
        # they are differenced against.
        var_indices = var_indices[:3]

        stack = [data[var_indices[-1][2] - el[2]: data.shape[0] - el[2], el[0]] for el in var_indices]
        X_raw = np.column_stack(stack)
        X_scaled = scale_maps[var].transform(X_raw)
        X_full = np.column_stack([np.ones(len(X_scaled)), X_scaled])
        y_full = data[var_indices[-1][2]:, var]

        num_features = X_full.shape[1]
        w_t = fine_coeffs[var].copy()
        P_t = np.eye(num_features) * (1.0 / RIDGE_ALPHA)
        err[var] = np.zeros((max_time, num_features))

        for t in range(max_time):
            if t < 15:
                err[var][t, :] = 0
                continue
            x_t = X_full[t]
            y_t = y_full[t]
            w_t, P_t, _ = _rls_update(x_t, y_t, w_t, P_t, lam=RLS_LAMBDA)
            err[var][t, :] = w_t - fine_coeffs[var]
            norm_agg[t, i] = np.linalg.norm(err[var][t, :])

    return err, norm_agg


def compute_windowed_errors(data: np.ndarray, window_len: int, fine_coeffs: dict,
                             causal_matrix: np.ndarray, indices: np.ndarray,
                             scale_maps: dict):
    """
    Run compute_online_errors over successive, non-overlapping windows of `data`,
    each reinitialized fresh from fine_coeffs -- matching exactly how each attack
    file is scored -- instead of one long continuous run that lets the online
    coefficients drift arbitrarily far from the offline baseline over many
    thousands of steps regardless of whether anything anomalous is happening.
    Returns err (per var, concatenated across windows) and norm_agg concatenated
    the same way. Any leftover rows shorter than window_len are dropped.
    """
    n_windows = data.shape[0] // window_len
    err_parts = {}
    norm_agg_parts = []
    for w in range(n_windows):
        chunk = data[w * window_len: (w + 1) * window_len]
        err_w, norm_agg_w = compute_online_errors(chunk, fine_coeffs, causal_matrix, indices, scale_maps)
        norm_agg_parts.append(norm_agg_w)
        for var, arr in err_w.items():
            err_parts.setdefault(var, []).append(arr)
    err = {var: np.concatenate(parts, axis=0) for var, parts in err_parts.items()}
    num_vars = len(np.unique(indices[1, :]))
    norm_agg = np.concatenate(norm_agg_parts, axis=0) if norm_agg_parts else np.zeros((0, num_vars))
    return err, norm_agg


def score_anomaly_timeline(err_baseline, err_eval, baseline_len):
    """
    Score err_eval (a fresh, single online-monitoring run -- an attack file or a
    windowed normal-evaluation chunk) against a per-variable threshold derived
    from err_baseline (a fresh, windowed normal-baseline run of the same kind).
    """
    max_time = min((err_eval[var].shape[0] for var in err_eval.keys()), default=0)

    scores = np.zeros(max_time, dtype=float)
    for var in err_eval.keys():
        for j in range(err_eval[var].shape[1]):
            thresh = DETECTION_THRESHOLD_MULTIPLIER * np.linalg.norm(err_baseline[var][:baseline_len, j])
            thresh = max(float(thresh), np.finfo(float).eps)

            values = np.abs(err_eval[var][:max_time, j]) / thresh
            scores = np.maximum(scores, values)

    predictions = (scores > ANOMALY_SCORE_THRESHOLD).astype(int)
    return predictions, scores




# ================================================================
#                         MAIN PIPELINE
# ================================================================
def main():
    print(f"\n========== TASK: {TASK.upper()} ==========")
        
    causal_path = os.path.join(PREFIX, f"{TASK}_normal{ABL}.npz")
    print(f"Causal model path: {causal_path}")

    # 1) Learn or load causal model
    if not os.path.exists(causal_path):
        learn_causal_model(os.path.join(PREFIX, NORMAL_FILE), causal_path)
    else:
        print("Causal model found, loading...")

    f = np.load(causal_path, allow_pickle=True)
    val_matrix, p_matrix = f["val_matrix"], f["p_matrix"]
    subsample, nonconst = int(f["subsample"]), f["nonconst"]

    normal_matrix = val_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > CAUSAL_STRENGTH_MULTIPLIER * np.mean(abs(val_matrix)))

    # 2) Load normal data
    normal_df = read_data(os.path.join(PREFIX, NORMAL_FILE), TASK)
    normal_data = np.nan_to_num(normal_df.values[:int(TRAINING_FRAC * len(normal_df))][::subsample, nonconst])
    normal_data_full = np.nan_to_num(normal_df.values[::subsample, nonconst])

    # 3) Offline linear least-squares coefficients
    fine_coeffs, indices, scale_maps = fit_normal_coeffs(normal_data, normal_matrix)

    # 4) Load attacks first so we know the window length to evaluate "normal"
    # data with: each attack gets its own fresh, short online-monitoring run,
    # so the normal baseline/held-out data must be evaluated the same way
    # (fresh, attack-length windows) rather than one long continuous run --
    # otherwise online coefficients drift further from the offline baseline
    # the longer the run goes, confounding "anomalous" with "far into a long run".
    attack_data_dict = pd.read_pickle(os.path.join(PREFIX, ATTACK_FILE))
    if isinstance(attack_data_dict, dict):
        attack_names = list(attack_data_dict.keys())
        attack_dfs = [restrict_leads(attack_data_dict[k][[c for c in ECG_LEADS if c in attack_data_dict[k].columns]]) for k in attack_names]
    else:
        attack_paths = [os.path.join(PREFIX, fname) for fname in [ATTACK_FILE]]
        attack_dfs = [read_data(p, TASK) for p in attack_paths]
        attack_names = [os.path.basename(p) for p in attack_paths]

    attack_data_list = [np.nan_to_num(df.values[::subsample, nonconst]) for df in attack_dfs]
    window_len = int(np.mean([len(a) for a in attack_data_list]))
    print(f"Normal-evaluation window length (matches mean attack length): {window_len}")

    # 5) Windowed normal baseline (for per-variable thresholds) and windowed
    # held-out normal evaluation (the "control group"), each made of fresh,
    # window_len-long RLS runs -- exactly like each attack gets.
    err_normal_baseline, norm_agg_baseline = compute_windowed_errors(
        normal_data, window_len, fine_coeffs, normal_matrix, indices, scale_maps
    )
    baseline_len = next(iter(err_normal_baseline.values())).shape[0]

    # Per-variable normal-baseline residual scale, for normalizing explainability
    # rankings so naturally-high-amplitude leads (e.g. V2) don't dominate regardless
    # of which variable was actually attacked.
    normal_residual_scale = np.array([
        np.sqrt(np.mean(norm_agg_baseline[:, j] ** 2))
        for j in range(norm_agg_baseline.shape[1])
    ])

    held_out_data = normal_data_full[len(normal_data):]
    err_normal_eval, norm_agg_eval = compute_windowed_errors(
        held_out_data, window_len, fine_coeffs, normal_matrix, indices, scale_maps
    )

    attack_scores = []

    _, normal_scores = score_anomaly_timeline(
        err_normal_baseline, err_normal_eval, baseline_len
    )

    norm_agg_attacks = []

    for attack_name, attack_data in zip(attack_names, attack_data_list):
        print(f"\n--- Analyzing anomaly: {attack_name} ---")

        err_attack, norm_agg_attack = compute_online_errors(attack_data, fine_coeffs, normal_matrix, indices, scale_maps)
        norm_agg_attacks.append(norm_agg_attack)
        _, attack_score = score_anomaly_timeline(
            err_normal_baseline, err_attack, baseline_len
        )
        # score_anomaly_timeline returns predictions and scores of equal length, so the
        # score alone determines the padding. Windows added here are labelled positive,
        # which costs a few guaranteed false negatives -- identically for all 27 runs.
        if len(attack_score) < attack_data.shape[0]:
            pad_len = attack_data.shape[0] - len(attack_score)
            attack_score = np.concatenate([np.zeros(pad_len, dtype=float), attack_score])
        attack_scores.append(attack_score)

    # 6) Metrics & Reporting via Shared Helper
    import reporting_helper

    attack_scores_dict = dict(zip(attack_names, attack_scores))
    attack_labels_dict = {}
    target_wise_residuals_by_file = dict(zip(attack_names, norm_agg_attacks))

    for name, attack_data in zip(attack_names, attack_data_list):
        attack_labels_dict[name] = np.ones(attack_data.shape[0])

    output_dir = os.path.join(BASE_DIR, "outputs" + ABL)
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
        model_variant="Linear / Ridge",
        model_name="PCMCI",
        normal_scores=normal_scores,
        attack_scores_by_file=attack_scores_dict,
        threshold=best_threshold,
        var_names=f["var"][nonconst].tolist(),
        target_indices=np.unique(indices[1, :]).tolist(),
        directed_edges=[], 
        target_wise_residuals_by_file=target_wise_residuals_by_file,
        attack_labels_by_file=attack_labels_dict,
        dataset_name="Healthcare (ECG)",
        normal_residual_scale=normal_residual_scale
    )


if __name__ == "__main__":
    main()
