"""
==============================================================
 Causal Discovery & Anomaly Detection (TS-CAM-UV Polynomial - Healthcare)
==============================================================
Author: (Sagar)
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
from scipy.stats import ConstantInputWarning
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore', category=ConstantInputWarning)

# ================== TS-CAM-UV ENGINE ==================
# Authentic CAM-UV (Maeda & Shimizu, UAI 2021) via the paper authors' own `lingam`
# implementation, loaded through this repository's engine shim. This is a HARD dependency: if
# the engine cannot be loaded the run aborts immediately, rather than printing a warning and
# continuing to a later NameError.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import camuv_engine  # noqa: E402


# ================== GLOBAL CONFIG ==================
ALPHA = 0.05
TRAINING_FRAC = 0.7
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.path.join(BASE_DIR, "Healthcare dataset")
TASK = "healthcare"

NORMAL_FILE = "healthcare_normal_train.pkl"
ATTACK_FILE = "healthcare_attacks.pkl"
POLY_DEGREE = 3

# Anomaly detection tuning config
DETECTION_THRESHOLD_MULTIPLIER = 2.0
CAUSAL_STRENGTH_MULTIPLIER = 1.0
SUBSAMPLE_EXPECTED = 10
RIDGE_ALPHA = 10.0

# ================== TS-CAM-UV CONFIG ==================
# CAM-UV returns (P, U): observed parents and pairs sharing an unobserved common cause.
# U is what distinguishes CAM-UV from PCMCI and GES, so it is used rather than discarded.
#
#   "integrate"     -- U pairs become symmetric lag-1 edges alongside observed parents.
#   "observed_only" -- U is excluded from the regressors and passed to explainability only.
#
# Mapping is strictly AT LAG 1, never contemporaneous: a contemporaneous symmetric pair would
# make two variables mutual same-time predictors, which is a mutual-dependency cycle. At lag 1
# the relation is an ordinary vector-autoregression, acyclic in time and numerically stable.
CONFOUNDER_MODE = "integrate"
CAMUV_ALPHA = 0.05
CAMUV_MAX_SAMPLES = 400
CAMUV_NUM_EXPLANATORY = 2

RLS_LAMBDA = 1.0


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
    Load a healthcare ECG pickle/CSV file and enforce the 12 standard leads.
    """
    if str(path).endswith(".pkl"):
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path, delimiter=",")

    if "label" in df.columns:
        df = df.drop(columns=["label"])

    # Force column names to be the 12 medical leads to fix numeric X-axis plots
    if len(df.columns) == 12:
        df.columns = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    return df

def learn_causal_model(normal_csv_path: str, save_path: str):
    print("Learning causal model using authentic TS-CAM-UV...")

    df = read_data(normal_csv_path, TASK)
    raw_data = np.nan_to_num(df.values)
    raw_data = raw_data[:int(TRAINING_FRAC * len(raw_data)), :]

    # Resampling rate. This is a property of the causal discovery algorithm, not a tuned
    # parameter: PCMCI estimates lagged conditional independencies (tau_max > 1) and uses
    # whatever rate its FFT/Nyquist analysis derives (1 here, 74 on Pepper). GES and CAM-UV
    # do static discovery with a lag-1 encoding and require approximately decorrelated
    # samples. Measured: forcing subsample=1 here collapses GES healthcare from AUC 0.8114
    # to 0.6184 (linear) and 0.7867 to 0.4992 (polynomial, i.e. random), because adjacent
    # samples become near-identical, one-step prediction turns trivial and the RLS drift
    # signal the detector depends on vanishes. The converse also holds: PCMCI on Pepper at
    # subsample=10 loses AUC 0.7825 -> 0.6813 and drops KneePicthTemp out of the
    # JointControl top-3 entirely, failing the XAI lab check.
    subsample = SUBSAMPLE_EXPECTED
    raw_data = raw_data[::subsample, :]

    nonconst = [idx for idx in range(raw_data.shape[1]) if np.std(raw_data[:, idx]) > 1e-8]
    nonconst_data = raw_data[:, nonconst]

    nonconst_data = nonconst_data[:min(CAMUV_MAX_SAMPLES, nonconst_data.shape[0]), :]

    np.random.seed(42)
    nonconst_data = nonconst_data + np.random.normal(0, 1e-5, nonconst_data.shape)

    for j in range(np.shape(nonconst_data)[1]):
        rng = np.max(nonconst_data[:, j]) - np.min(nonconst_data[:, j])
        if rng > 1e-8:
            nonconst_data[:, j] = (nonconst_data[:, j] - np.min(nonconst_data[:, j])) / rng
        else:
            nonconst_data[:, j] = 0.0

    num_explanatory = min(CAMUV_NUM_EXPLANATORY, max(1, nonconst_data.shape[1] - 1))
    print(f"Running authentic CAM-UV on {nonconst_data.shape[1]} variables x "
          f"{nonconst_data.shape[0]} samples (alpha={CAMUV_ALPHA}, mode={CONFOUNDER_MODE}, "
          f"num_explanatory_vals={num_explanatory})...")
    P, U = camuv_engine.execute(nonconst_data, alpha=CAMUV_ALPHA,
                                num_explanatory_vals=num_explanatory)

    num_features = nonconst_data.shape[1]
    val_matrix = np.zeros((num_features, num_features, 2))
    p_matrix = np.ones((num_features, num_features, 2))

    # --- Observed parents: lag-1 directed edges -------------------------------------------
    n_parent_edges = 0
    for child, parents in enumerate(P):
        for parent in parents:
            if parent < num_features:
                val_matrix[parent, child, 1] = 1.0
                p_matrix[parent, child, 1] = 0.001
                n_parent_edges += 1

    # --- Unobserved confounders (see CONFOUNDER_MODE above) -------------------------------
    n_conf_edges = 0
    for pair in (U if CONFOUNDER_MODE == "integrate" else []):
        pair = sorted(pair)
        if len(pair) != 2:
            continue
        i, j = pair
        if i < num_features and j < num_features:
            for a, b in ((i, j), (j, i)):
                if val_matrix[a, b, 1] == 0.0:
                    val_matrix[a, b, 1] = 1.0
                    p_matrix[a, b, 1] = 0.001
                    n_conf_edges += 1

    print(f"CAM-UV structure [{CONFOUNDER_MODE}]: {n_parent_edges} observed-parent edges, "
          f"{len(U)} unobserved-confounder pairs contributing {n_conf_edges} lag-1 edges")

    conf_arr = (np.array(sorted([sorted(p) for p in U]), dtype=int)
                if U else np.zeros((0, 2), dtype=int))

    np.savez(save_path, val_matrix=val_matrix, p_matrix=p_matrix, var=df.columns, subsample=subsample, nonconst=nonconst, confounded_pairs=conf_arr)
    print(f"Saved causal model to {save_path}")
    return {"val_matrix": val_matrix, "p_matrix": p_matrix}, subsample, nonconst, 1

def fit_normal_coeffs(normal_data: np.ndarray, causal_matrix: np.ndarray, degree=POLY_DEGREE, alpha=RIDGE_ALPHA):
    indices = np.array(np.where(causal_matrix != 0))
    fine_coeffs = {}
    poly_maps = {}
    for var in np.unique(indices[1, :]):
        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])

        # RAM Protection: Truncate causal parents to Top 3
        # At most three parents per target, taken in lag order, so the design matrix is
        # never wider than three columns plus the intercept however dense the graph. The
        # same truncation runs in fit_normal_coeffs and compute_online_errors; if the two
        # disagreed the online coefficients would not correspond to the offline baseline
        # they are differenced against.
        var_indices = var_indices[:3]
        max_delay = var_indices[-1][2]

        stack = [normal_data[max_delay - el[2]: len(normal_data) - el[2], el[0]] for el in var_indices]
        X = np.column_stack(stack)
        y = normal_data[max_delay:, var]

        # Mathematical Bounds: StandardScaler + Ridge
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X_scaled)

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_poly, y)
        fine_coeffs[var] = np.concatenate([[float(model.intercept_)], np.ravel(model.coef_)])
        poly_maps[var] = (scaler, poly)
    return fine_coeffs, indices, poly_maps

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


def read_data(path, task):
    return restrict_leads(_read_data_unrestricted(path, task))


def compute_online_errors(data: np.ndarray, fine_coeffs: dict, causal_matrix: np.ndarray, indices: np.ndarray, poly_maps: dict, alpha=RIDGE_ALPHA, rls_lambda=RLS_LAMBDA):
    max_time = data.shape[0] - causal_matrix.shape[2]
    unique_vars = np.unique(indices[1, :])
    err = {}
    norm_agg = np.zeros((max_time, len(unique_vars)))

    w_state, P_state, var_prep = {}, {}, {}

    for var in unique_vars:
        scaler, poly = poly_maps[var]
        w0 = fine_coeffs[var].copy()
        P0 = np.eye(len(w0)) * (1.0 / alpha)
        w_state[var] = w0
        P_state[var] = P0

        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        # At most three parents per target, taken in lag order, so the design matrix is
        # never wider than three columns plus the intercept however dense the graph. The
        # same truncation runs in fit_normal_coeffs and compute_online_errors; if the two
        # disagreed the online coefficients would not correspond to the offline baseline
        # they are differenced against.
        var_indices = var_indices[:3]
        max_delay = var_indices[-1][2]

        stack = [data[max_delay - el[2]: max_time + max_delay - el[2], el[0]] for el in var_indices]
        X = np.column_stack(stack)
        X_scaled = scaler.transform(X)
        X_poly = poly.transform(X_scaled)
        y = data[max_delay: max_time + max_delay, var]
        var_prep[var] = (X_poly, y)
        err[var] = np.zeros((max_time, len(w0)))

    for t in range(max_time):
        for i, var in enumerate(unique_vars):
            if t < 15: # Startup guard
                err[var][t, :] = 0
                continue
            X_des, y = var_prep[var]
            x_vec = np.concatenate([[1.0], X_des[t]])
            y_t = float(y[t])

            w_t, P_t, _ = _rls_update(x_vec, y_t, w_state[var], P_state[var], rls_lambda)
            w_state[var], P_state[var] = w_t, P_t

            err[var][t, :] = w_t - fine_coeffs[var]
            norm_agg[t, i] = np.linalg.norm(err[var][t, :])

    return err, norm_agg

def compute_windowed_errors(data, window_len, fine_coeffs, causal_matrix, indices, *extra):
    """
    Evaluate `data` as successive fresh windows, each reinitialised from fine_coeffs, which is
    exactly how every attack segment is scored.

    Previously the fault-free control was scored as the tail of one continuous 50,000-step RLS
    run while each attack was a fresh ~1,000-step run. RLS coefficients drift steadily away
    from the fixed offline baseline the longer a run continues, so the control accumulated
    larger residuals than the attacks for reasons unrelated to anomalousness. Scoring both
    classes as fresh runs of equal length removes that confound.
    """
    n_windows = max(1, data.shape[0] // window_len)
    err_parts, agg_parts = {}, []
    for w in range(n_windows):
        chunk = data[w * window_len:(w + 1) * window_len]
        if chunk.shape[0] <= causal_matrix.shape[2] + 1:
            continue
        e, a = compute_online_errors(chunk, fine_coeffs, causal_matrix, indices, *extra)
        agg_parts.append(a)
        for var, arr in e.items():
            err_parts.setdefault(var, []).append(arr)
    err = {v: np.concatenate(p, axis=0) for v, p in err_parts.items()}
    agg = np.concatenate(agg_parts, axis=0) if agg_parts else np.zeros((0, len(np.unique(indices[1, :]))))
    return err, agg


# RLS warm-up length. compute_online_errors forces the residual to exactly zero for this many
# steps at the start of every run, so those samples carry no information.
STARTUP_GUARD = 15


def score_fresh(err_baseline, err_eval, baseline_len):
    """
    Score one fresh online run against thresholds from the fault-free baseline.

    The first STARTUP_GUARD samples of every run are dropped rather than scored. The online
    error is pinned to exactly zero during RLS warm-up, so scoring them injects a block of
    artificial zeros into whichever class the run belongs to. With equal-length windows that
    affected about 15% of both classes, and the F1-optimal threshold consequently collapsed to
    0.0 -- flagging every non-zero sample and producing identical confusion matrices for models
    whose scores actually differ. Excluding the warm-up removes the artifact from both classes
    symmetrically.
    """
    max_time = min((err_eval[v].shape[0] for v in err_eval), default=0)
    scores = np.zeros(max_time, dtype=float)
    for var in err_eval:
        for j in range(err_eval[var].shape[1]):
            thresh = DETECTION_THRESHOLD_MULTIPLIER * np.linalg.norm(err_baseline[var][:baseline_len, j])
            thresh = max(float(thresh), np.finfo(float).eps)
            scores = np.maximum(scores, np.abs(err_eval[var][:max_time, j]) / thresh)
    if max_time > STARTUP_GUARD:
        scores = scores[STARTUP_GUARD:]
    return (scores > 1.0).astype(int), scores



def main():
    print(f"\n========== TASK: {TASK.upper()} (GES POLYNOMIAL) ==========")
    causal_path = os.path.join(PREFIX, f"{TASK}_normal_camuv_{CONFOUNDER_MODE}_a{CAMUV_ALPHA}{ABL}.npz")

    if not os.path.exists(causal_path):
        learn_causal_model(os.path.join(PREFIX, NORMAL_FILE), causal_path)

    f = np.load(causal_path, allow_pickle=True)
    val_matrix, p_matrix = f["val_matrix"], f["p_matrix"]
    subsample, nonconst = int(f["subsample"]), f["nonconst"]
    if subsample != SUBSAMPLE_EXPECTED:
        raise RuntimeError(
            f"Cached graph was built at subsample={subsample} but this code expects "
            f"{SUBSAMPLE_EXPECTED}. The cache is stale -- delete {causal_path} and re-run. "
            f"Silently reusing it would report metrics for a pipeline that no longer exists.")

    normal_matrix = val_matrix * (p_matrix < ALPHA) * (abs(val_matrix) > CAUSAL_STRENGTH_MULTIPLIER * np.mean(abs(val_matrix)))

    normal_df = read_data(os.path.join(PREFIX, NORMAL_FILE), TASK)
    normal_data = np.nan_to_num(normal_df.values[:int(TRAINING_FRAC * len(normal_df))][::subsample, nonconst])
    normal_data_full = np.nan_to_num(normal_df.values[::subsample, nonconst])

    fine_coeffs, indices, poly_maps = fit_normal_coeffs(normal_data, normal_matrix)
    err_normal, norm_agg_normal = compute_online_errors(normal_data_full, fine_coeffs, normal_matrix, indices, poly_maps)

    # healthcare_attacks.pkl is a dict of 20 separate attack segments, not one frame.
    attack_dict = pd.read_pickle(os.path.join(PREFIX, ATTACK_FILE))
    if isinstance(attack_dict, dict):
        attack_names = list(attack_dict.keys())
        attack_frames = [restrict_leads(attack_dict[k]) for k in attack_names]
    else:
        attack_names, attack_frames = ["healthcare_attacks"], [attack_dict]

    attack_data_list = []
    for fr in attack_frames:
        fr = fr.drop(columns=["label"]) if "label" in fr.columns else fr
        attack_data_list.append(np.nan_to_num(fr.values[::subsample, nonconst]))

    window_len = int(np.mean([len(a) for a in attack_data_list]))
    print(f"Fresh-window evaluation length (mean attack length): {window_len}")

    # Fault-free baseline and held-out control, both scored as fresh equal-length windows.
    err_baseline, norm_agg_baseline = compute_windowed_errors(
        normal_data, window_len, fine_coeffs, normal_matrix, indices, poly_maps)
    baseline_len = next(iter(err_baseline.values())).shape[0]

    normal_residual_scale = np.array([
        np.sqrt(np.mean(norm_agg_baseline[:, j] ** 2)) for j in range(norm_agg_baseline.shape[1])
    ])

    held_out = normal_data_full[len(normal_data):]
    err_eval, _ = compute_windowed_errors(
        held_out, window_len, fine_coeffs, normal_matrix, indices, poly_maps)
    normal_predictions, normal_scores = score_fresh(err_baseline, err_eval, baseline_len)

    attack_predictions, attack_scores, norm_agg_attacks = [], [], []
    for attack_name, attack_data in zip(attack_names, attack_data_list):
        print(f"\n--- Analyzing anomaly: {attack_name} ---")
        err_attack, norm_agg_attack = compute_online_errors(
            attack_data, fine_coeffs, normal_matrix, indices, poly_maps)
        norm_agg_attacks.append(norm_agg_attack)
        ap, asc = score_fresh(err_baseline, err_attack, baseline_len)
        attack_predictions.append(ap)
        attack_scores.append(asc)

    import reporting_helper
    # Ensure correct physical labels are passed
    true_target_indices = [nonconst[var] for var in np.unique(indices[1, :])]

    attack_scores_dict = dict(zip(attack_names, attack_scores))
    attack_labels_dict = {name: np.ones(len(sc)) for name, sc in zip(attack_names, attack_scores)}
    target_wise_residuals_by_file = dict(zip(attack_names, norm_agg_attacks))

    # Decision threshold is derived from FAULT-FREE data only: the THRESHOLD_PERCENTILE-th
    # percentile of the normal-score distribution. It never sees an attack label, so the
    # reported F1 is an achievable operating point, not the maximum attainable F1 obtained
    # by searching thresholds against ground truth. Identical rule for all 27 configurations.
    THRESHOLD_PERCENTILE = 95.0
    best_threshold = float(np.percentile(np.nan_to_num(normal_scores), THRESHOLD_PERCENTILE))
    print(f"Fault-free {THRESHOLD_PERCENTILE:.0f}th-percentile threshold: {best_threshold:.4f}")

    output_dir = os.path.join(BASE_DIR, "outputs" + ABL)

    reporting_helper.generate_plots_and_reports(
        output_dir=output_dir,
        model_variant="Polynomial degree 3",
        model_name="TS-CAM-UV",
        normal_scores=normal_scores,
        attack_scores_by_file=attack_scores_dict,
        threshold=best_threshold,
        var_names=normal_df.columns[nonconst].tolist(),
        target_indices=true_target_indices,
        directed_edges=[],
        target_wise_residuals_by_file=target_wise_residuals_by_file,
        attack_labels_by_file=attack_labels_dict,
        dataset_name="Healthcare (ECG)",
        output_prefix=("" if CONFOUNDER_MODE == "integrate" else "configB_"),
        normal_residual_scale=normal_residual_scale
    )

if __name__ == "__main__":
    main()
