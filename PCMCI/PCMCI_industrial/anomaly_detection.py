from sklearn.preprocessing import StandardScaler
"""
==============================================================
 Causal Discovery & Anomaly Detection (Linear Least-Squares Tigramite - Industrial / TEP)
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
from sklearn.linear_model import Ridge
import warnings
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from scipy.stats import ConstantInputWarning


warnings.filterwarnings('ignore', category=ConstantInputWarning)

# ================== GLOBAL CONFIG ==================
ALPHA = 0.05
TRAINING_FRAC = 0.7
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.path.join(BASE_DIR, "TEP")
TASK = "tep"
MAX_FREQ_COMPONENTS = 5

NORMAL_FILE = "TEP_FaultFree_Training_run1.pkl"
NORMAL_TEST_FILE = "TEP_FaultFree_Testing_run1.pkl"

# TEP metadata columns that are not sensors. faultNumber in particular encodes the ground
# truth (0 = fault-free, N = fault N) and must never reach the model as a feature.
TEP_META_COLUMNS = ["faultNumber", "simulationRun", "sample"]

# Fault-free files are pre-restricted to a single simulation run; faulty files contain all
# 500 runs, so they are filtered to this run to keep the comparison like-for-like.
SIMULATION_RUN = 1

# In the TEP testing scenario each 960-sample run is fault-free until the fault is injected
# 8 simulated hours in (sample 160, at the 3-minute sampling interval). Samples before this
# index are genuinely normal and must not be labelled as anomalies. Verified empirically:
# fault 6's pre-160 statistics match the fault-free run (mean 0.249 vs 0.250) and diverge
# sharply afterwards.
FAULT_ONSET_SAMPLE = 160


# Anomaly detection tuning config
DETECTION_THRESHOLD_MULTIPLIER = 3.0
ANOMALY_SCORE_THRESHOLD = 1.0
CAUSAL_STRENGTH_MULTIPLIER = 0.0
RIDGE_ALPHA = 10.0
RLS_LAMBDA = 1.0


# ================================================================
#                         DATA LOADING
# ================================================================
def read_data(path: str, task: str, run: int = SIMULATION_RUN):
    """
    Load a TEP pickle as a pure sensor matrix.

    The raw TEP frames carry three metadata columns (faultNumber, simulationRun, sample)
    ahead of the 52 sensors. These MUST be dropped: faultNumber is 0 in fault-free data and
    N in fault N data, so leaving it in feeds the ground-truth label straight into the model
    as an input feature. Fault files also contain all 500 simulation runs concatenated, while
    the fault-free files are already restricted to a single run, so faulty data is filtered to
    the same run to keep the comparison like-for-like.
    """
    import pandas as pd
    df = pd.read_pickle(path)

    if "simulationRun" in df.columns:
        df = df[df["simulationRun"] == run]
    if "sample" in df.columns:
        df = df.sort_values("sample")

    df = df.drop(columns=[c for c in TEP_META_COLUMNS if c in df.columns])

    if len(df.columns) == 52:
        df.columns = [f"XMEAS_{i}" for i in range(1, 42)] + [f"XMV_{i}" for i in range(1, 12)]
    return df.reset_index(drop=True)


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
    """
    indices = np.array(np.where(causal_matrix != 0))
    fine_coeffs = {}

    for var in np.unique(indices[1, :]):
        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        max_delay = var_indices[-1][2]
        # Top-3 parent-lag links, matching the XAI course lab (`var_indices[:3]`) and
        # the other 26 configurations. The number of causal parents fed to the
        # regression is part of the shared pipeline, not a per-folder choice.
        # At most three parents per target, taken in lag order, so the design matrix is
        # never wider than three columns plus the intercept however dense the graph. The
        # same truncation runs in fit_normal_coeffs and compute_online_errors; if the two
        # disagreed the online coefficients would not correspond to the offline baseline
        # they are differenced against.
        var_indices = var_indices[:3]

        stack = [normal_data[max_delay - el[2]: len(normal_data) - el[2], el[0]]
                 for el in var_indices]
        X = np.column_stack(stack)
        y = normal_data[max_delay:, var]

        # Ridge, not OLS. GES and CAM-UV both fit Ridge(alpha=RIDGE_ALPHA) for the linear
        # variant on this dataset; fitting unregularised OLS here meant the three algorithms
        # were not running the same regression, so the industrial linear comparison was not
        # like-for-like. Coefficient layout ([intercept, ...]) is unchanged.
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        model.fit(X, y)
        fine_coeffs[var] = np.concatenate([[float(model.intercept_)], np.ravel(model.coef_)])

    return fine_coeffs, indices


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
                           causal_matrix: np.ndarray, indices: np.ndarray):
    """
    Recompute linear coefficients online sample-by-sample via RLS.
    """
    max_time = data.shape[0] - causal_matrix.shape[2]
    err = {}
    norm_agg = np.zeros((max_time, len(np.unique(indices[1, :]))))

    for i, var in enumerate(np.unique(indices[1, :])):
        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        max_delay = var_indices[-1][2]
        # Top-3 parent-lag links, matching the XAI course lab (`var_indices[:3]`) and the
        # other 26 configurations. max_delay is taken from the full sorted list before the
        # cap, exactly as fit_normal_coeffs does, so the two stay dimensionally consistent.
        # At most three parents per target, taken in lag order, so the design matrix is
        # never wider than three columns plus the intercept however dense the graph. The
        # same truncation runs in fit_normal_coeffs and compute_online_errors; if the two
        # disagreed the online coefficients would not correspond to the offline baseline
        # they are differenced against.
        var_indices = var_indices[:3]

        stack = [data[max_delay - el[2]: data.shape[0] - el[2], el[0]] for el in var_indices]
        X_full = np.column_stack([np.ones(len(stack[0])), np.column_stack(stack)])
        y_full = data[max_delay:, var]

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


def score_anomaly_timeline(err_baseline, err_eval, baseline_len):
    """
    Score err_eval (a fresh online-monitoring run) against per-variable thresholds derived
    from err_baseline (the fault-free training run). Standard math: |drift| / threshold.
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
    print(f"\n========== TASK: {TASK.upper()} (LINEAR) ==========")

    causal_path = os.path.join(PREFIX, f"{TASK}_normal.npz")
    print(f"Causal model path: {causal_path}")

    if not os.path.exists(causal_path):
        learn_causal_model(os.path.join(PREFIX, NORMAL_FILE), causal_path)
    else:
        print("Causal model found, loading...")

    f = np.load(causal_path, allow_pickle=True)
    val_matrix, p_matrix = f["val_matrix"], f["p_matrix"]
    subsample, nonconst = int(f["subsample"]), f["nonconst"]

    normal_matrix = val_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > CAUSAL_STRENGTH_MULTIPLIER * np.mean(abs(val_matrix)))

    # ---- Offline baseline: fault-free TRAINING run -------------------------------
    normal_df = read_data(os.path.join(PREFIX, NORMAL_FILE), TASK)
    normal_train = np.nan_to_num(normal_df.values[::subsample, nonconst])

    scaler = StandardScaler()
    normal_train = scaler.fit_transform(normal_train)

    fine_coeffs, indices = fit_normal_coeffs(normal_train, normal_matrix)
    err_baseline, norm_agg_baseline = compute_online_errors(normal_train, fine_coeffs, normal_matrix, indices)
    baseline_len = next(iter(err_baseline.values())).shape[0]

    # Per-variable baseline residual scale: normalizes the explainability ranking so that
    # sensors with naturally large variance (e.g. XMEAS_19) cannot dominate every fault.
    normal_residual_scale = np.array([
        np.sqrt(np.mean(norm_agg_baseline[:, j] ** 2)) for j in range(norm_agg_baseline.shape[1])
    ])

    # ---- Evaluation window -------------------------------------------------------
    # TEP injects each fault at sample FAULT_ONSET_SAMPLE; earlier samples are genuinely
    # fault-free. Both the normal control and every fault run are therefore scored over the
    # identical post-onset index window, so the two classes see the same amount of RLS
    # adaptation and every scored attack sample is truly anomalous.
    onset_idx = FAULT_ONSET_SAMPLE // max(subsample, 1)
    print(f"Fault onset sample {FAULT_ONSET_SAMPLE} -> scoring from index {onset_idx} (subsample={subsample})")

    # ---- Held-out fault-free TESTING run = normal control group ------------------
    normal_test_df = read_data(os.path.join(PREFIX, NORMAL_TEST_FILE), TASK)
    normal_test = scaler.transform(np.nan_to_num(normal_test_df.values[::subsample, nonconst]))
    err_normal_eval, _ = compute_online_errors(normal_test, fine_coeffs, normal_matrix, indices)
    _, normal_scores_full = score_anomaly_timeline(err_baseline, err_normal_eval, baseline_len)
    normal_scores = normal_scores_full[onset_idx:]

    # ---- Faults ------------------------------------------------------------------
    attack_paths = [os.path.join(PREFIX, f"TEP_Faulty_Testing_fault{i}.pkl") for i in range(1, 21)]
    attack_names = [f"Fault_{i}" for i in range(1, 21)]

    attack_scores, norm_agg_attacks = [], []

    for attack_name, path in zip(attack_names, attack_paths):
        print(f"\n--- Analyzing fault: {attack_name} ---")
        df_attack = read_data(path, TASK)
        attack_data = scaler.transform(np.nan_to_num(df_attack.values[::subsample, nonconst]))

        err_attack, norm_agg_attack = compute_online_errors(attack_data, fine_coeffs, normal_matrix, indices)
        norm_agg_attacks.append(norm_agg_attack[onset_idx:])

        _, attack_score_full = score_anomaly_timeline(err_baseline, err_attack, baseline_len)
        attack_score = attack_score_full[onset_idx:]
        attack_scores.append(attack_score)

    # ---- Reporting ---------------------------------------------------------------
    import reporting_helper

    attack_scores_dict = dict(zip(attack_names, attack_scores))
    target_wise_residuals_by_file = dict(zip(attack_names, norm_agg_attacks))
    attack_labels_dict = {name: np.ones(len(s)) for name, s in zip(attack_names, attack_scores)}

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
        dataset_name="Industrial (TEP)",
        normal_residual_scale=normal_residual_scale
    )


if __name__ == "__main__":
    main()
