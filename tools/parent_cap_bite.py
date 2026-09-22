"""
How much of each discovered graph survives the three-parent cap?

`fit_normal_coeffs` does not regress a target on every parent the graph gives it. It sorts
that target's incoming edges by lag and keeps the first three:

    var_indices.sort(key=lambda x: x[2])
    var_indices = var_indices[:3]

So the design matrix is at most three columns plus an intercept, in all 27 configurations,
no matter how dense the graph is. That matters for the density argument. A graph linking 63
of 66 lead pairs and a graph linking 34 of 66 both feed three-column regressions; density
cannot act by making the regression larger, because it never does.

What density can still do, and what this tool measures, is decide *which* three. A target
with three candidates has its parents chosen by the graph. A target with eleven has three of
them chosen by lag order, with ties broken by the order `np.where` happened to emit -- which
is variable index, not causal strength. So on a dense graph the parent set is substantially
determined by an arbitrary tiebreak rather than by the discovery step.

The quantity to report is therefore not density alone but how often the cap binds, and how
many edges it drops. Both are computed here exactly as the pipeline computes them: the same
sparsification rule, the same sort, the same truncation.

Usage:  python tools/parent_cap_bite.py
"""

import csv
import io
import os
import re
import sys

import numpy as np

CAP = 3

# domain -> algorithm -> (cache, folder holding the constants)
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


def parent_sets(path, alpha, csm):
    """Reproduce the pipeline's sparsification and its per-target parent list, pre-cap."""
    z = np.load(path, allow_pickle=True)
    val, p = z["val_matrix"], z["p_matrix"]
    mat = val * (p < alpha) * (np.abs(val) > csm * np.mean(np.abs(val)))
    idx = np.array(np.where(mat != 0))
    if idx.size == 0:
        return {}
    out = {}
    for var in np.unique(idx[1, :]):
        cols = [idx[:, k] for k in range(idx.shape[1]) if idx[1, k] == var]
        cols.sort(key=lambda x: x[2])          # the pipeline's sort: by lag
        out[int(var)] = cols
    return out


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.chdir(root)

    print("Bite of the three-parent cap in fit_normal_coeffs\n")
    print("  Every regression uses at most %d parents plus an intercept. Density therefore" % CAP)
    print("  cannot act by enlarging the regression; it acts by deciding which %d survive.\n" % CAP)
    print("  %-11s %-8s %8s %8s %9s %9s %9s"
          % ("domain", "algo", "targets", "edges", "capped", "% capped", "% dropped"))
    print("  " + "-" * 70)

    rows = []
    for domain, algo, path, folder in CONFIGS:
        if not os.path.exists(path):
            print("  %-11s %-8s cache missing; skipped" % (domain, algo))
            continue
        alpha = constant(folder, "ALPHA", 0.05)
        csm = constant(folder, "CAUSAL_STRENGTH_MULTIPLIER", 1.0)
        ps = parent_sets(path, alpha, csm)
        if not ps:
            print("  %-11s %-8s no surviving edges" % (domain, algo))
            continue

        targets = len(ps)
        edges = sum(len(v) for v in ps.values())
        capped = sum(1 for v in ps.values() if len(v) > CAP)
        used = sum(min(len(v), CAP) for v in ps.values())
        dropped = edges - used

        print("  %-11s %-8s %8d %8d %9d %8.1f%% %8.1f%%"
              % (domain, algo, targets, edges, capped,
                 100.0 * capped / targets, 100.0 * dropped / edges))
        rows.append({"Domain": domain, "Algorithm": algo,
                     "Modelled targets": targets,
                     "Edges after sparsification": edges,
                     "Targets hitting the cap": capped,
                     "Percent of targets capped": "%.1f" % (100.0 * capped / targets),
                     "Edges used": used,
                     "Edges discarded by the cap": dropped,
                     "Percent of edges discarded": "%.1f" % (100.0 * dropped / edges),
                     "Mean parents before cap": "%.2f" % (edges / targets),
                     "Mean parents after cap": "%.2f" % (used / targets)})

    if not rows:
        raise SystemExit("no caches found; run the pipelines first")

    out = "tools/parent_cap_bite.csv"
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n  wrote %s" % out)

    print("\n  Reading, per domain:")
    for domain in ["Healthcare", "Robotics", "Industrial"]:
        sub = [r for r in rows if r["Domain"] == domain]
        if not sub:
            continue
        worst = max(sub, key=lambda r: float(r["Percent of edges discarded"]))
        best = min(sub, key=lambda r: float(r["Percent of edges discarded"]))
        print("    %-11s cap discards %s%% of %s's edges and %s%% of %s's."
              % (domain, worst["Percent of edges discarded"], worst["Algorithm"],
                 best["Percent of edges discarded"], best["Algorithm"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
