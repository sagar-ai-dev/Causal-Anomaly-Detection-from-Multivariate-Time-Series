"""
On what terms is each neural baseline compared against the causal pipeline?

The baselines were written to share the causal protocol, and the thesis leans on that: the
healthcare column is the one place a causal configuration beats the tuned neural networks,
and the claim is only worth anything if the two sides were given comparable problems.

They almost were. `Baselines/neural_baselines_all.py` takes its variable set and its
subsampling rate from PCMCI's cached graph in every domain -- `load_robotics`,
`load_healthcare` and `load_industrial` all read PCMCI's npz. That matches the best causal
configuration on robotics and on industrial, where PCMCI is itself the strongest. It does
not match on healthcare, where the strongest causal result is GES's at a coarser rate.

So on healthcare the baselines are scored over more samples than the configuration they are
being compared against. The variable set is identical, so the difference is resolution
alone, and its direction runs against the thesis: the baselines get more data, not less.
That is worth measuring rather than asserting, because a reader's first suspicion on seeing
a causal method beat three neural networks is that the neural networks were handicapped.

This script reports, per domain, the resolution each side actually ran at and how many
fault-free samples each therefore saw.

Usage:  python tools/baseline_parity.py
"""

import csv
import io
import os
import sys

import numpy as np
import pandas as pd

# domain -> (PCMCI cache the baselines read, fault-free table, loader for its row count)
DOMAINS = {
    "Robotics": {
        "cache": "PCMCI/PCMCI_robotic/pepper_csv/pepper_normal.npz",
        "normal": ("csv", "PCMCI/PCMCI_robotic/pepper_csv/normal.csv"),
        "folders": {"PCMCI": "PCMCI/PCMCI_robotic", "GES": "GES/GES_robotic",
                    "CAM-UV": "CAM-UV/CAM_UV_robotics"},
        "caches": {"PCMCI": "PCMCI/PCMCI_robotic/pepper_csv/pepper_normal.npz",
                   "GES": "GES/GES_robotic/pepper_csv/pepper_normal_ges.npz",
                   "CAM-UV": "CAM-UV/CAM_UV_robotics/pepper_csv/"
                             "pepper_normal_camuv_integrate_a0.05.npz"},
    },
    "Healthcare": {
        "cache": "PCMCI/PCMCI_healthcare/Healthcare dataset/healthcare_normal.npz",
        "normal": ("pkl", "PCMCI/PCMCI_healthcare/Healthcare dataset/"
                          "healthcare_normal_train.pkl"),
        "folders": {"PCMCI": "PCMCI/PCMCI_healthcare", "GES": "GES/GES_healthcare",
                    "CAM-UV": "CAM-UV/CAM_UV_healthcare"},
        "caches": {"PCMCI": "PCMCI/PCMCI_healthcare/Healthcare dataset/healthcare_normal.npz",
                   "GES": "GES/GES_healthcare/Healthcare dataset/healthcare_normal_ges.npz",
                   "CAM-UV": "CAM-UV/CAM_UV_healthcare/Healthcare dataset/"
                             "healthcare_normal_camuv_integrate_a0.05.npz"},
    },
    "Industrial": {
        "cache": "PCMCI/PCMCI_industrial/TEP/tep_normal.npz",
        "normal": ("pkl", "PCMCI/PCMCI_industrial/TEP/TEP_FaultFree_Training_run1.pkl"),
        "folders": {"PCMCI": "PCMCI/PCMCI_industrial", "GES": "GES/GES_industrial",
                    "CAM-UV": "CAM-UV/CAM_UV_industrial"},
        "caches": {"PCMCI": "PCMCI/PCMCI_industrial/TEP/tep_normal.npz",
                   "GES": "GES/GES_industrial/TEP/tep_normal.npz",
                   "CAM-UV": "CAM-UV/CAM_UV_industrial/TEP/tep_normal_camuv_integrate_a0.05.npz"},
    },
}


def rows_of(kind, path):
    if not os.path.exists(path):
        return None
    if kind == "csv":
        return len(pd.read_csv(path))
    return len(pd.read_pickle(path))


def best_causal(folder):
    """(variant, AUC) of the strongest regression variant in a folder."""
    p = os.path.join(folder, "outputs", "metrics_summary.csv")
    if not os.path.exists(p):
        return None, None
    best_v, best_a = None, -1.0
    for r in csv.DictReader(io.open(p, encoding="utf-8")):
        try:
            a = float(r["AUC_ROC"])
        except (KeyError, ValueError):
            continue
        if a > best_a:
            best_v, best_a = r.get("Model Variant", "?"), a
    return best_v, (best_a if best_a >= 0 else None)


def cache_facts(path):
    if not os.path.exists(path):
        return None, None
    z = np.load(path, allow_pickle=True)
    return int(z["subsample"]), int(len(z["nonconst"]))


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.chdir(root)

    print("Terms of comparison between the neural baselines and the causal pipelines\n")
    print("  The baselines read PCMCI's cache in every domain, so they inherit PCMCI's")
    print("  variable set and scoring resolution wherever they are compared.\n")

    rows = []
    for dom, cfg in DOMAINS.items():
        b_sub, b_vars = cache_facts(cfg["cache"])
        if b_sub is None:
            print("  %-11s PCMCI cache missing; skipped" % dom)
            continue
        n_rows = rows_of(*cfg["normal"])

        # strongest causal configuration in this domain, across the three algorithms
        champ, champ_auc, champ_algo = None, -1.0, None
        for algo, folder in cfg["folders"].items():
            v, a = best_causal(folder)
            if a is not None and a > champ_auc:
                champ, champ_auc, champ_algo = v, a, algo
        if champ_algo is None:
            print("  %-11s no causal metrics found; skipped" % dom)
            continue

        c_sub, c_vars = cache_facts(cfg["caches"][champ_algo])
        matched = (c_sub == b_sub) and (c_vars == b_vars)

        b_samples = (n_rows + b_sub - 1) // b_sub if n_rows else None
        c_samples = (n_rows + c_sub - 1) // c_sub if n_rows else None

        print("  %s" % dom)
        print("    baseline        PCMCI protocol   subsample %-3d vars %-4d samples %s"
              % (b_sub, b_vars, b_samples if b_samples else "?"))
        print("    best causal     %-6s %-9s subsample %-3d vars %-4d samples %s  AUC %.4f"
              % (champ_algo, champ[:9] if champ else "?", c_sub, c_vars,
                 c_samples if c_samples else "?", champ_auc))
        print("    -> %s\n" % ("matched" if matched
                               else "NOT matched: baseline sees %s samples, causal %s"
                               % (b_samples, c_samples)))

        rows.append({
            "Domain": dom,
            "Baseline subsample": b_sub,
            "Baseline variables": b_vars,
            "Baseline fault-free samples": b_samples if b_samples else "",
            "Best causal algorithm": champ_algo,
            "Best causal variant": champ,
            "Best causal AUC": "%.4f" % champ_auc,
            "Causal subsample": c_sub,
            "Causal variables": c_vars,
            "Causal fault-free samples": c_samples if c_samples else "",
            "Resolution matched": "yes" if matched else "no",
        })

    if not rows:
        raise SystemExit("no domains resolved; run the pipelines first")

    out = "tools/baseline_parity.csv"
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  wrote %s\n" % out)

    unmatched = [r for r in rows if r["Resolution matched"] == "no"]
    if not unmatched:
        print("  Every domain compares at a single resolution.")
        return 0
    for r in unmatched:
        b, c = r["Baseline fault-free samples"], r["Causal fault-free samples"]
        direction = ("the baseline, which sees more data" if b and c and b > c
                     else "the causal configuration, which sees more data")
        print("  %s is not resolution-matched. The asymmetry favours %s."
              % (r["Domain"], direction))
    print("\n  This is a declared limitation, not a defect: it is disclosed in the results")
    print("  chapter and in the caption of the baselines table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
