"""
==============================================================
 Reporting Helper Module (All Domains - TS-CAM-UV)
==============================================================

Provides helper utilities for per-class metrics reporting and 
professor-approved Linear Red Staircase Bar Charts (Feature Importance)
for time-series features. All metrics are exhaustively calculated.
"""

import math
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score

# Tracks which model variants have had stale CSV rows purged in this process.
_CSV_PURGED_VARIANTS = set()
_CSV_PURGED_FULL = set()

def plot_feature_importance_subplots(
    norm_agg_list,
    indices,
    nonconst,
    var_names,
    attack_names,
    output_filename="top_anomalous_variables.png",
    dataset_name="Dataset",
    model_variant="Model",
    normal_residual_scale=None,
):
    """
    Professor-approved Linear Red Staircase Bar Charts (Feature Importance).
    REMOVED log-scale to ensure true proportional representation.

    normal_residual_scale, when provided, is a per-variable baseline residual
    norm (same variable ordering as norm_agg_attack's columns) computed from
    normal (non-attack) data. Ranking divides each variable's attack-time
    residual by this baseline so variables with naturally larger raw scale
    (e.g. high-amplitude ECG leads) don't dominate the ranking regardless of
    which variable was actually attacked.
    """
    n_attacks = len(norm_agg_list)
    if n_attacks == 0:
        return

    # Create a dynamic grid (max 4 columns wide)
    cols = min(4, n_attacks)
    rows = math.ceil(n_attacks / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)
    plt.suptitle("Top Anomalous Variables per Attack", fontsize=18, fontweight='bold')

    for i, norm_agg_attack in enumerate(norm_agg_list):
        r = i // cols
        c = i % cols
        ax = axes[r, c]

        # Calculate L2 Norm directly for ALL provided features, normalized per
        # variable by its own normal-baseline residual scale.
        num_vars = norm_agg_attack.shape[1]
        dep_vals = {}
        for j in range(num_vars):
            if indices is not None and j < len(indices) and indices[j] < len(var_names):
                v_name = var_names[indices[j]]
            else:
                v_name = var_names[j] if j < len(var_names) else f"Var_{j}"
            # RMS, not raw L2: an L2 norm scales with sqrt(sequence length), so dividing a
            # short attack run's L2 by a long baseline run's L2 deflates every variable by a
            # constant sqrt(n_attack/n_baseline) factor (0.169 for the healthcare split),
            # pinning most bars to that floor and making the chart unreadable. RMS is
            # length-independent, so 1.0 means "same drift as normal" and >1 means elevated.
            raw_norm = float(np.sqrt(np.mean(norm_agg_attack[:, j] ** 2)))
            if normal_residual_scale is not None:
                scale = max(float(normal_residual_scale[j]), 1e-8)
                dep_vals[v_name] = raw_norm / scale
            else:
                dep_vals[v_name] = raw_norm

        # Sort variables from highest to lowest error
        dep_sorted = dict(sorted(dep_vals.items(), key=lambda x: x[1], reverse=True))

        # Limit to top 15 for readability
        top_n = min(15, len(dep_sorted))
        top_items = list(dep_sorted.items())[:top_n]
        if not top_items:
            continue
        top_vars, top_vals = zip(*top_items)

        csv_path = os.path.join(os.path.dirname(output_filename), "explainability_top_features.csv")
        _hdr = ['Dataset', 'Model Variant', 'Fault Name',
                'Top Variable 1', 'Top Variable 2', 'Top Variable 3']

        # Re-running a variant must REPLACE its rows, not append duplicates. This block
        # previously opened the file in 'a' mode unconditionally, so a second run of the same
        # script silently doubled that variant's rows. Any statistic computed downstream from
        # the CSV (attribution concentration, distinct-sensor counts) was then measured over
        # duplicated data. On the first row of each variant in this process, existing rows for
        # that variant are dropped and the file rewritten; subsequent rows append normally.
        if model_variant not in _CSV_PURGED_VARIANTS:
            _kept = []
            if os.path.isfile(csv_path):
                with open(csv_path, 'r', newline='') as _f:
                    _rows = list(csv.reader(_f))
                _kept = [r for r in _rows[1:] if len(r) > 1 and r[1] != model_variant]
            with open(csv_path, 'w', newline='') as _f:
                _w = csv.writer(_f)
                _w.writerow(_hdr)
                _w.writerows(_kept)
            _CSV_PURGED_VARIANTS.add(model_variant)

        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            tv = list(top_vars)
            while len(tv) < 3:
                tv.append("N/A")
            writer.writerow([dataset_name, model_variant, attack_names[i], tv[0], tv[1], tv[2]])
        # Full attribution ranking -- every variable and its score, not only the top three.
        # A top-k analysis, a relevance rate over that top set, and a consistency comparison
        # across methods all need the whole ranking; the top-3 export above cannot support
        # any of them. That export is deliberately left untouched, so every table already
        # built from it is unchanged and the uniformity audit still sees one export depth.
        # Derived from csv_path, not rebuilt: only one of the three helper variants
        # defines _OUTPUT_PREFIX, and rebuilding the path assumed all three did.
        _full_path = csv_path.replace("explainability_top_features",
                                      "attribution_full")
        if model_variant not in _CSV_PURGED_FULL:
            _fkept = []
            if os.path.isfile(_full_path):
                with open(_full_path, 'r', newline='') as _ff:
                    _frows = list(csv.reader(_ff))
                _fkept = [r for r in _frows[1:] if len(r) > 1 and r[1] != model_variant]
            with open(_full_path, 'w', newline='') as _ff:
                _fw = csv.writer(_ff)
                _fw.writerow(['Dataset', 'Model Variant', 'Fault Name',
                              'Rank', 'Variable', 'Score'])
                _fw.writerows(_fkept)
            _CSV_PURGED_FULL.add(model_variant)
        with open(_full_path, 'a', newline='') as _ff:
            _fw = csv.writer(_ff)
            for _rank, (_vname, _vscore) in enumerate(dep_sorted.items(), start=1):
                _fw.writerow([dataset_name, model_variant, attack_names[i],
                              _rank, _vname, "%.6f" % _vscore])


        # Plot the clean, linear bar chart
        ax.bar(top_vars, top_vals, color='salmon', edgecolor='white', linewidth=1)
        
        # STRICTLY LINEAR SCALE (Log scale is removed)
        
        ax.set_xticks(range(len(top_vars)))
        ax.set_xticklabels(top_vars, rotation=45, ha='right', fontsize=9)
        ylabel = "Drift vs Normal Baseline (RMS ratio)" if normal_residual_scale is not None else "Aggregated Error (L2 Norm)"
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(attack_names[i], fontsize=12)

    # Clean up empty subplots
    for i in range(n_attacks, rows * cols):
        fig.delaxes(axes[i // cols, i % cols])

    # Reserve a constant ~0.6in strip for the suptitle regardless of grid height. A fixed
    # fractional margin (top=0.92) collides with the subplot titles on short one-row grids,
    # where 8% of a 5in figure is not enough room for the figure title.
    fig_height = 5 * rows
    top_rect = max(0.70, 1.0 - (0.6 / fig_height))
    plt.tight_layout(rect=[0, 0, 1, top_rect])
    if os.path.dirname(output_filename):
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close()


def generate_plots_and_reports(
    output_dir: str,
    model_variant: str,
    model_name: str,
    normal_scores: np.ndarray,
    attack_scores_by_file: dict,
    threshold: float,
    var_names: list,
    target_indices: list,
    directed_edges: list,
    target_wise_residuals_by_file: dict,
    attack_labels_by_file: dict,
    dataset_name: str = "Dataset",
    normal_residual_scale=None
):
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "="*80)
    print(f" THESIS RESULTS EXPORT: {dataset_name} — {model_name} ({model_variant})")
    print(f" Threshold Applied: {threshold:.4f}")
    print("="*80)

    # Calculate Normal Baseline Stats (TN and FP)
    norm_pred = (normal_scores > threshold).astype(int)
    fp_normal = int(np.sum(norm_pred))
    tn_normal = len(norm_pred) - fp_normal
    
    all_y_true = list(np.zeros(len(normal_scores), dtype=int))
    all_y_pred = list(norm_pred)
    all_y_score = list(normal_scores)

    total_tp, total_fp, total_fn, total_tn = 0, fp_normal, 0, tn_normal

    print("\n--- PER-CLASS METRICS (INDIVIDUAL FAULTS) ---")
    for file_name, attack_scores in attack_scores_by_file.items():
        attack_pred = (attack_scores > threshold).astype(int)
        attack_true = np.ones(len(attack_scores), dtype=int)

        tp = int(np.sum(attack_pred))
        fn = len(attack_pred) - tp
        fp = fp_normal
        tn = tn_normal

        total_tp += tp
        total_fn += fn

        # Class-level metrics calculation
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"[{file_name}]")
        print(f"  Confusion Matrix : TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"  Accuracy         : {acc:.4f}")
        print(f"  Precision        : {precision:.4f}")
        print(f"  Recall           : {recall:.4f}")
        print(f"  F1 Score         : {f1:.4f}\n")

        all_y_true.extend(attack_true)
        all_y_pred.extend(attack_pred)
        all_y_score.extend(attack_scores)

    # Overall Global Metrics Calculation
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_score = np.array(all_y_score)

    acc_global = accuracy_score(all_y_true, all_y_pred)
    prec_global = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec_global = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_global = 2 * prec_global * rec_global / (prec_global + rec_global) if (prec_global + rec_global) > 0 else 0.0

    try:
        auc_roc = roc_auc_score(all_y_true, all_y_score)
    except ValueError:
        auc_roc = np.nan
    try:
        auc_pr = average_precision_score(all_y_true, all_y_score)
    except ValueError:
        auc_pr = np.nan

    print('-' * 80)
    print(' *** EXHAUSTIVE GLOBAL AGGREGATE METRICS (FOR THESIS TABLE) ***')
    print('-' * 80)
    print(f" Total True Positives (TP)  : {total_tp}")
    print(f" Total False Positives (FP) : {total_fp}")
    print(f" Total True Negatives (TN)  : {total_tn}")
    print(f" Total False Negatives (FN) : {total_fn}")
    print("-" * 80)
    print(f" Overall Accuracy           : {acc_global:.4f}")
    print(f" Overall Precision          : {prec_global:.4f}")
    print(f" Overall Recall             : {rec_global:.4f}")
    print(f" Overall F1 Score           : {f1_global:.4f}")
    print(f" Overall AUC-ROC            : {auc_roc:.4f}")
    print(f" Overall AUC-PR             : {auc_pr:.4f}")
    print("=" * 80 + "\n")


    # Persist the per-sample scores and labels. Without them no honest resampling interval
    # can be computed after the fact -- only a parametric approximation, which assumes the
    # very thing an interval is meant to test.
    # One file per variant -- a shared filename would let each variant overwrite the last,
    # leaving only the final one and silently losing two thirds of the scores.
    _safe_variant = model_variant.lower().replace(" ", "_").replace("/", "").replace("+", "")
    _sf = os.path.join(output_dir, f"scores_{_safe_variant}.npz")
    np.savez_compressed(
        _sf,
        model_variant=model_variant,
        dataset=dataset_name,
        normal_scores=np.asarray(normal_scores, dtype=float),
        **{f"attack__{k}": np.asarray(v, dtype=float)
           for k, v in attack_scores_by_file.items()},
    )

    # Persist thesis metrics so tables are generated from data rather than transcribed by
    # hand. Re-running a variant replaces its own row, so the file always reflects the
    # current code. This is the single source of truth for README/FINAL_RESULTS tables.
    _mcsv = os.path.join(output_dir, "metrics_summary.csv")
    _mhdr = ["Dataset", "Model", "Model Variant", "Threshold", "TP", "FP", "TN", "FN",
             "Accuracy", "Precision", "Recall", "F1", "AUC_ROC", "AUC_PR"]
    _mrow = [dataset_name, model_name, model_variant, f"{threshold:.6f}",
             total_tp, total_fp, total_tn, total_fn,
             f"{acc_global:.4f}", f"{prec_global:.4f}", f"{rec_global:.4f}",
             f"{f1_global:.4f}", f"{auc_roc:.4f}", f"{auc_pr:.4f}"]
    _mkeep = []
    if os.path.exists(_mcsv):
        with open(_mcsv, "r", newline="") as _mf:
            _mall = [r for r in csv.reader(_mf) if r]
        _mkeep = [r for r in _mall[1:]
                  if not (len(r) > 2 and r[0] == dataset_name and r[2] == model_variant)]
    with open(_mcsv, "w", newline="") as _mf:
        _mw = csv.writer(_mf)
        _mw.writerow(_mhdr)
        _mw.writerows(_mkeep)
        _mw.writerow(_mrow)
    print(f" Metrics row written to: {_mcsv}")

    # Plotting Output
    norm_agg_list = list(target_wise_residuals_by_file.values())
    attack_names = list(target_wise_residuals_by_file.keys())
    safe_variant_name = model_variant.lower().replace(" / ", "_").replace(" ", "_")
    output_plot_path = os.path.join(output_dir, f"{safe_variant_name}_top_anomalous_variables.png")

    plot_feature_importance_subplots(
        norm_agg_list=norm_agg_list,
        indices=target_indices,
        nonconst=list(range(len(var_names))),
        var_names=var_names,
        attack_names=attack_names,
        output_filename=output_plot_path,
        dataset_name=dataset_name,
        model_variant=model_variant,
        normal_residual_scale=normal_residual_scale
    )
    print(f"Saved Linear Staircase Visualizations to: {output_plot_path}")
