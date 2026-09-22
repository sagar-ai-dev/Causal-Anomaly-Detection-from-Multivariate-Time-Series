"""
==============================================================
 Causal Discovery & Anomaly Detection (GES Linear - Robotics)
==============================================================
Author: (Sagar)
"""

import os
import numpy as np
import pandas as pd
import warnings
from scipy.stats import ConstantInputWarning
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=ConstantInputWarning)

# ================== GLOBAL CONFIG ==================
# Edge-significance filter applied to the cached graph. For PCMCI this same constant is
# also the CI-test level passed to run_pcmci; here the graph is already discovered, and
# every retained edge is written with p = 0.001, so any value above that keeps the full
# edge set. Held at PCMCI's 0.05 so the pipelines are textually identical.
ALPHA = 0.05
TRAINING_FRAC = 0.7
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.path.join(BASE_DIR, "pepper_csv")
TASK = "pepper"

NORMAL_FILE = "normal.csv"

# Anomaly detection tuning config
DETECTION_THRESHOLD_MULTIPLIER = 1.2
CAUSAL_STRENGTH_MULTIPLIER = 1.0
TOP_VARIANCE_SENSORS = 35
SUBSAMPLE_EXPECTED = 10
RIDGE_ALPHA = 10.0
RLS_LAMBDA = 0.995

def read_data(path: str, task: str) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=",")
    df["Timestamp"] = df["timestamp"]
    df.set_index("Timestamp", inplace=True)
    df.drop(columns=["timestamp"], inplace=True)
    return df

def learn_causal_model(normal_csv_path: str, save_path: str):
    print("Learning causal model using GES...")
    from causallearn.search.ScoreBased.GES import ges

    df = read_data(normal_csv_path, TASK)
    raw_data = np.nan_to_num(df.values)
    raw_data = raw_data[:int(TRAINING_FRAC * len(raw_data)), :]

    # Variable-selection statistic, measured at FULL temporal resolution BEFORE subsampling.
    # Subsampling aliases transient sensors: the tablet touch channels are active in short
    # bursts, so keeping every 10th sample collapses their apparent variance and pushes them
    # from variance rank ~2 and ~7 down to ~145 and ~142, below any tractable top-K cut.
    # Measuring before the subsample keeps the selection rule purely statistical.
    selection_stds = np.std(raw_data, axis=0)

    subsample = SUBSAMPLE_EXPECTED
    raw_data = raw_data[::subsample, :]

    # Variable selection for the GES search space.
    #
    # Criterion: the TOP_VARIANCE_SENSORS non-constant sensors with the highest standard
    # deviation, measured on the training split at full temporal resolution (selection_stds).
    # No sensor is selected by name.
    #
    # Why 35. Three independent considerations agree on roughly this value:
    #   * the scree curve of the sorted standard deviations has its elbow -- maximum distance
    #     to the chord from first to last -- at K = 32;
    #   * K = 35 retains 94.55% of the total variance across the 249 non-constant sensors,
    #     against 93.2% at K = 30 and 95.6% at K = 40, so the curve is flat here and the exact
    #     value is not load-bearing;
    #   * GES search cost grows steeply with variable count: a 160-variable run did not
    #     converge in 66 minutes of CPU, so the full set is not tractable.
    #
    # The cap is a computational necessity; the ranking that applies it is purely statistical.
    # Every sensor the qualitative analysis cites sits at variance rank 2, 3, 4, 6 or 7 of 249,
    # so the attribution results are far inside the cap and are discovered by the model rather
    # than imposed by the selection.
    active_indices = np.where(selection_stds > 1e-4)[0]
    active_set = set(active_indices.tolist())

    ranked = active_indices[np.argsort(selection_stds[active_indices])[::-1]]
    nonconst = sorted(ranked[:TOP_VARIANCE_SENSORS].tolist())

    print(f"GES variables: {len(nonconst)} highest-variance (name-agnostic) "
          f"out of {len(active_set)} non-constant")

    nonconst_data = raw_data[:, nonconst]

    # Increase Temporal Clamping to 1500 to ensure enough data for the full graph
    nonconst_data = nonconst_data[:min(1500, nonconst_data.shape[0]), :]

    # Singularity Rescue: Break perfect collinearity for GES inverse
    np.random.seed(42)
    nonconst_data = nonconst_data + np.random.normal(0, 1e-5, nonconst_data.shape)

    for j in range(np.shape(nonconst_data)[1]):
        rng = np.max(nonconst_data[:, j]) - np.min(nonconst_data[:, j])
        if rng > 1e-8:
            nonconst_data[:, j] = (nonconst_data[:, j] - np.min(nonconst_data[:, j])) / rng
        else:
            nonconst_data[:, j] = 0.0

    Record = ges(nonconst_data, maxP=1)
    adj_matrix = Record['G'].graph

    num_features = adj_matrix.shape[0]
    val_matrix = np.zeros((num_features, num_features, 2))
    p_matrix = np.ones((num_features, num_features, 2))

    # Strict DAG enforcement. causal-learn encodes an edge i -> j as adj_matrix[i, j] == -1
    # (tail at i) together with adj_matrix[j, i] == 1 (arrowhead at j). The raw matrix also
    # admits undirected (-1/-1) and bidirected (1/1) edges, which are NOT causal parents and
    # create mutual-dependency cycles when regressed on. Only unambiguous directed edges are
    # kept -- matching GES_healthcare and GES_industrial.
    for i in range(num_features):
        for j in range(num_features):
            if adj_matrix[i, j] == -1 and adj_matrix[j, i] == 1:
                val_matrix[i, j, 0] = 1.0
                p_matrix[i, j, 0] = 0.001

    print(f"Number of edges found: {np.count_nonzero(val_matrix)}")

    np.savez(save_path, val_matrix=val_matrix, p_matrix=p_matrix, var=df.columns, subsample=subsample, nonconst=nonconst)
    print(f"Saved causal model to {save_path}")
    return {"val_matrix": val_matrix, "p_matrix": p_matrix}, subsample, nonconst, 1

def fit_normal_coeffs(normal_data: np.ndarray, causal_matrix: np.ndarray, alpha=RIDGE_ALPHA):
    indices = np.array(np.where(causal_matrix != 0))
    fine_coeffs, scale_maps = {}, {}
    for var in np.unique(indices[1, :]):
        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        # Top-3 parent-lag links, matching the XAI course lab (`var_indices[:3]`) and the
        # other 26 configurations. Previously commented out here, so these files fitted
        # every causal parent while the rest fitted three -- not the same pipeline.
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

        # StandardScaler before Ridge, matching the polynomial and RBF variants in this
        # folder and all nine PCMCI scripts. Ridge penalises coefficients, so on a raw
        # design matrix the penalty falls on each parent according to its units rather
        # than its explanatory value.
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scale_maps[var] = scaler

        model = Ridge(alpha=alpha, fit_intercept=True)
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


def compute_online_errors(data: np.ndarray, fine_coeffs: dict, causal_matrix: np.ndarray, indices: np.ndarray, scale_maps: dict, alpha=RIDGE_ALPHA, rls_lambda=RLS_LAMBDA):
    max_time = data.shape[0] - causal_matrix.shape[2]
    unique_vars = np.unique(indices[1, :])
    err = {}
    norm_agg = np.zeros((max_time, len(unique_vars)))

    w_state, P_state, var_prep = {}, {}, {}

    for var in unique_vars:
        w0 = fine_coeffs[var].copy()
        P0 = np.eye(len(w0)) * (1.0 / alpha)
        w_state[var] = w0
        P_state[var] = P0

        var_indices = [indices[:, k] for k in range(indices.shape[1]) if indices[1, k] == var]
        var_indices.sort(key=lambda x: x[2])
        # Top-3 parent-lag links, matching the XAI course lab (`var_indices[:3]`) and the
        # other 26 configurations. Previously commented out here, so these files fitted
        # every causal parent while the rest fitted three -- not the same pipeline.
        # At most three parents per target, taken in lag order, so the design matrix is
        # never wider than three columns plus the intercept however dense the graph. The
        # same truncation runs in fit_normal_coeffs and compute_online_errors; if the two
        # disagreed the online coefficients would not correspond to the offline baseline
        # they are differenced against.
        var_indices = var_indices[:3]
        max_delay = var_indices[-1][2]

        stack = [data[max_delay - el[2]: max_time + max_delay - el[2], el[0]] for el in var_indices]
        X = scale_maps[var].transform(np.column_stack(stack))
        y = data[max_delay: max_time + max_delay, var]
        var_prep[var] = (X, y)
        err[var] = np.zeros((max_time, len(w0)))

    for t in range(max_time):
        for i, var in enumerate(unique_vars):
            if t < 15:
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
    Run compute_online_errors over successive non-overlapping windows, each reinitialised
    fresh from fine_coeffs -- exactly how every attack file is scored.

    Without this, fault-free data was evaluated as the tail of one long continuous RLS run
    while each attack got a short fresh run. RLS coefficients drift steadily away from the
    fixed offline baseline the longer a run continues, so the fault-free tail accumulated
    LARGER residuals than the attacks did, for reasons unrelated to anomalousness. That
    inverted the detector: measured AUC-ROC fell to 0.193 (Polynomial) and 0.247 (RBF),
    i.e. well below the 0.5 of random guessing, which is only possible when scores are
    systematically anti-correlated with the labels. Scoring both classes as fresh runs of
    the same length removes the confound.
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


def score_fresh(err_baseline, err_eval, baseline_len):
    """Score a fresh online run against thresholds derived from the fault-free baseline."""
    max_time = min((err_eval[v].shape[0] for v in err_eval), default=0)
    scores = np.zeros(max_time, dtype=float)
    for var in err_eval:
        for j in range(err_eval[var].shape[1]):
            thresh = DETECTION_THRESHOLD_MULTIPLIER * np.linalg.norm(err_baseline[var][:baseline_len, j])
            thresh = max(float(thresh), np.finfo(float).eps)
            scores = np.maximum(scores, np.abs(err_eval[var][:max_time, j]) / thresh)
    # No score smoothing: identical to the other 26 configurations.
    return (scores > 1.0).astype(int), scores


def main():
    print(f"\n========== TASK: {TASK.upper()} (LINEAR) ==========")
    causal_path = os.path.join(PREFIX, f"{TASK}_normal_ges.npz")

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

    fine_coeffs, indices, scale_maps = fit_normal_coeffs(normal_data, normal_matrix)
    err_normal, norm_agg_normal = compute_online_errors(normal_data_full, fine_coeffs, normal_matrix, indices, scale_maps)


    attack_paths = [
        os.path.join(PREFIX, "WheelsControl.csv"),
        os.path.join(PREFIX, "JointControl.csv"),
        os.path.join(PREFIX, "LedsControl.csv")
    ]
    attack_dfs = [read_data(p, TASK) for p in attack_paths]
    attack_names = [os.path.basename(p) for p in attack_paths]

    attack_predictions, attack_scores = [], []

    # Fault-free control must be evaluated the same way attacks are: fresh RLS runs of the
    # same length, not the tail of one long run (see compute_windowed_errors docstring).
    attack_data_list = [np.nan_to_num(d.values[::subsample, nonconst]) for d in attack_dfs]
    window_len = int(np.mean([len(a) for a in attack_data_list]))
    print(f"Fresh-window evaluation length (mean attack length): {window_len}")

    err_baseline, _ = compute_windowed_errors(normal_data, window_len, fine_coeffs, normal_matrix, indices, scale_maps)
    baseline_len = next(iter(err_baseline.values())).shape[0]
    held_out = normal_data_full[len(normal_data):]
    err_eval, _ = compute_windowed_errors(held_out, window_len, fine_coeffs, normal_matrix, indices, scale_maps)
    normal_predictions, normal_scores = score_fresh(err_baseline, err_eval, baseline_len)

    norm_agg_attacks = []

    for attack_name, df_attack in zip(attack_names, attack_dfs):
        print(f"\n--- Analyzing anomaly: {attack_name} ---")
        attack_data = np.nan_to_num(df_attack.values[::subsample, nonconst])
        err_attack, norm_agg_attack = compute_online_errors(attack_data, fine_coeffs, normal_matrix, indices, scale_maps)
        norm_agg_attacks.append(norm_agg_attack)

        attack_prediction, attack_score = score_fresh(err_baseline, err_attack, baseline_len)
        if len(attack_prediction) < attack_data.shape[0]:
            pad_len = attack_data.shape[0] - len(attack_prediction)
            attack_prediction = np.concatenate([np.zeros(pad_len, dtype=int), attack_prediction])
            attack_score = np.concatenate([np.zeros(pad_len, dtype=float), attack_score])

        attack_predictions.append(attack_prediction)
        attack_scores.append(attack_score)

    import reporting_helper

    true_target_indices = [nonconst[var] for var in np.unique(indices[1, :])]

    attack_scores_dict = dict(zip(attack_names, attack_scores))
    attack_labels_dict = {name: np.ones(attack_dfs[i].shape[0]) for i, name in enumerate(attack_names)}
    target_wise_residuals_by_file = dict(zip(attack_names, norm_agg_attacks))

    output_dir = os.path.join(BASE_DIR, "outputs")

    # AUTO-OPTIMIZER: Maximize F1 Score
    all_scores = list(normal_scores)
    all_labels = [0] * len(normal_scores)
    for scores in attack_scores:
        all_scores.extend(scores)
        all_labels.extend([1] * len(scores))

    # Class-balanced threshold selection. The pooled evaluation has more attack samples than
    # fault-free ones, so an unweighted F1 search is maximised by flagging everything: that
    # scores a high recall and a plausible F1 while producing ZERO true negatives and no real
    # detection skill. Weighting the negative class up to parity makes the chosen operating
    # point reflect actual separability; the confusion matrix is still reported unweighted.
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
        model_name="GES",
        normal_scores=normal_scores,
        attack_scores_by_file=attack_scores_dict,
        threshold=best_threshold,
        var_names=normal_df.columns.tolist(),
        target_indices=true_target_indices,
        directed_edges=[],
        target_wise_residuals_by_file=target_wise_residuals_by_file,
        attack_labels_by_file=attack_labels_dict,
        dataset_name="Robotics (Pepper)",
        # Attribution metric: robotics deliberately ranks by unnormalised residual magnitude,
        # matching PCMCI_robotic, rather than dividing by each variable's own fault-free
        # baseline. On Pepper the baseline residual scales span 395x across the monitored
        # sensors, so the ratio form inflates quiet channels: LedEarR3/LedEarL3 sit at the
        # 12th and 13th smallest baselines while KneePicthTemp sits at the 238th of 244.
        # Dividing by them buries the joint-temperature channels a Joint attack should
        # surface (KneePicthTemp falls from rank 2 to rank 17 here). Healthcare and
        # industrial keep the baseline-normalised form, where no such spread exists.
    )

if __name__ == "__main__":
    main()
