"""
Which lag actually reaches the regression, before and after the three-parent cap?

Chapter 3 describes the parent set as drawn "at previous timesteps", and until this
measurement existed the pipeline's own lag-0 (contemporaneous) edges were invisible in the
prose. This tool reproduces the pipeline's sparsification and its per-target parent list --
the same rule and the same sort as `parent_cap_bite.py` -- and additionally tabulates, per
domain and algorithm, what fraction of the surviving edges are lag 0.

PCMCI's `val_matrix` has one slice per lag in `0 .. tau_max`; `run_pcmci` is called with the
tigramite default `tau_min=0`, so slice 0 is a genuine contemporaneous test, not padding.
GES and CAM-UV write into a fixed slice chosen by the script that produced the cache (index 1
for every folder except GES_robotic, which writes index 0); this tool reads that back from
the data rather than assuming it, so a future change to any wrapper is caught automatically
rather than silently going stale here.

Usage:  python tools/parent_cap_lag.py
"""

import csv
import io
import os
import re
import sys

import numpy as np

CAP = 3

CONFIGS = [
    ("Healthcare", "PCMCI", "PCMCI/PCMCI_healthcare/Healthcare dataset/healthcare_normal.npz",
     "PCMCI/PCMCI_healthcare"),
    ("Healthcare", "GES", "GES/GES_healthcare/Healthcare dataset/healthcare_normal_ges.npz",
     "GES/GES_healthcare"),
    ("Healthcare", "CAM-UV",
     "CAM-UV/CAM_UV_healthcare/Healthcare dataset/healthcare_normal_camuv_integrate_a0.05.npz",
     "CAM-UV/CAM_UV_healthcare"),
    ("Robotics", "PCMCI", "PCMCI/PCMCI_robotic/pepper_csv/pepper_normal.npz",
     "PCMCI/PCMCI_robotic"),
    ("Robotics", "GES", "GES/GES_robotic/pepper_csv/pepper_normal_ges.npz", "GES/GES_robotic"),
    ("Robotics", "CAM-UV",
     "CAM-UV/CAM_UV_robotics/pepper_csv/pepper_normal_camuv_integrate_a0.05.npz",
     "CAM-UV/CAM_UV_robotics"),
    ("Industrial", "PCMCI", "PCMCI/PCMCI_industrial/TEP/tep_normal.npz", "PCMCI/PCMCI_industrial"),
    ("Industrial", "GES", "GES/GES_industrial/TEP/tep_normal.npz", "GES/GES_industrial"),
    ("Industrial", "CAM-UV", "CAM-UV/CAM_UV_industrial/TEP/tep_normal_camuv_integrate_a0.05.npz",
     "CAM-UV/CAM_UV_industrial"),
]


def constant(folder, name, default):
    """Read a module-level numeric constant out of the folder's anomaly_detection.py."""
    for fn in sorted(os.listdir(folder)):
        if not fn.startswith("anomaly_detection") or not fn.endswith(".py"):
            continue
        src = io.open(os.path.join(folder, fn), encoding="utf-8").read()
        m = re.search(r"^%s\s*=\s*([0-9.eE+-]+)" % name, src, re.M)
        if m:
            return float(m.group(1))
    return default


def edge_list(path, alpha, csm):
    """Reproduce the pipeline's sparsification: every surviving (source, target, lag) edge."""
    z = np.load(path, allow_pickle=True)
    val, p = z["val_matrix"], z["p_matrix"]
    mat = val * (p < alpha) * (np.abs(val) > csm * np.mean(np.abs(val)))
    idx = np.array(np.where(mat != 0))
    return idx, mat.shape[2]


def per_target(idx):
    """Group edges by target, sorted by lag -- the pipeline's own sort in fit_normal_coeffs."""
    out = {}
    if idx.size == 0:
        return out
    for var in np.unique(idx[1, :]):
        cols = [idx[:, k] for k in range(idx.shape[1]) if idx[1, k] == var]
        cols.sort(key=lambda x: x[2])
        out[int(var)] = cols
    return out


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.chdir(root)

    print("Lag-0 (contemporaneous) share of each graph, before and after the three-parent cap\n")
    print("  %-11s %-8s %8s %10s %10s %9s %9s"
          % ("domain", "algo", "edges", "lag0 pre", "kept", "lag0 kept", "kept %"))
    print("  " + "-" * 72)

    rows = []
    for domain, algo, path, folder in CONFIGS:
        if not os.path.exists(path):
            print("  %-11s %-8s cache missing; skipped" % (domain, algo))
            continue
        alpha = constant(folder, "ALPHA", 0.05)
        csm = constant(folder, "CAUSAL_STRENGTH_MULTIPLIER", 1.0)
        idx, n_lags = edge_list(path, alpha, csm)
        if idx.size == 0:
            print("  %-11s %-8s no surviving edges" % (domain, algo))
            continue

        lags = idx[2]
        edges = lags.shape[0]
        lag0_pre = int((lags == 0).sum())

        groups = per_target(idx)
        kept_total, kept_lag0 = 0, 0
        for cols in groups.values():
            kept = cols[:CAP]
            kept_total += len(kept)
            kept_lag0 += sum(1 for el in kept if el[2] == 0)

        pct_kept_lag0 = 100.0 * kept_lag0 / kept_total if kept_total else 0.0
        print("  %-11s %-8s %8d %9.1f%% %10d %9d %8.1f%%"
              % (domain, algo, edges, 100.0 * lag0_pre / edges, kept_total, kept_lag0,
                 pct_kept_lag0))
        rows.append({
            "Domain": domain, "Algorithm": algo, "Lag slots (0..max)": n_lags - 1,
            "Edges after sparsification": edges,
            "Lag-0 edges pre-cap": lag0_pre,
            "Percent lag-0 pre-cap": "%.1f" % (100.0 * lag0_pre / edges),
            "Kept parent-slots": kept_total,
            "Kept parent-slots at lag 0": kept_lag0,
            "Percent of kept slots at lag 0": "%.1f" % pct_kept_lag0,
        })

    if not rows:
        raise SystemExit("no caches found; run the pipelines first")

    out = "tools/parent_cap_lag.csv"
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n  wrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
