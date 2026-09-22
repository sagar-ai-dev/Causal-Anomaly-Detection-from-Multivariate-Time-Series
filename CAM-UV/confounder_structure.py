"""
How much of each TS-CAM-UV graph is latent-confounder structure rather than observed parents?

CAM-UV returns observed parents `P` and pairs `U` sharing an unobserved common cause, and the
pipelines run with CONFOUNDER_MODE="integrate", which adds each confounded pair as a pair of
lag-1 edges. This reads the three cached graphs back and reports what the resulting structure
is actually made of.

The question matters because `U` is the whole reason CAM-UV was chosen over PCMCI and GES. An
earlier draft reported that `U` came back empty on all three datasets. That was wrong -- it is
not empty anywhere, and on robotics it is the entire graph.

Saturation is the figure to watch: the share of all C(d,2) variable pairs flagged as
confounded. A method that declares most pairs confounded has not discovered structure, it has
returned a near-complete graph, and the detector downstream inherits no useful sparsity.

Read it alongside samples-per-variable, reported in the same table. CAM-UV flags a pair by
rejecting independence of the two residuals with HSIC, so a run with too few samples to reject
anything produces a sparse graph for a reason that has nothing to do with the data being
sparse. TEP is exactly that case -- 35 samples for 52 variables -- so its 1.1% saturation
cannot be read as selectivity.

Usage:  python CAM-UV/confounder_structure.py      (from the repository root)
"""

import csv
import glob
import os

import numpy as np
import pandas as pd

# Source of the fault-free training split each domain runs discovery on.
SOURCE = {
    "robotics": ("CAM-UV/CAM_UV_robotics/pepper_csv/", "normal.csv"),
    "healthcare": ("CAM-UV/CAM_UV_healthcare/Healthcare dataset/", "healthcare_normal_train.pkl"),
    "industrial": ("CAM-UV/CAM_UV_industrial/TEP/", "TEP_FaultFree_Training_run1.pkl"),
}
TRAINING_FRAC = 0.7


def discovery_samples(domain, subsample):
    """How many rows CAM-UV actually saw, after the training split and subsampling."""
    pre, fn = SOURCE[domain]
    fp = os.path.join(pre, fn)
    if not os.path.exists(fp):
        return None
    raw = pd.read_csv(fp) if fn.endswith(".csv") else pd.read_pickle(fp)
    return len(range(0, int(TRAINING_FRAC * len(raw)), max(subsample, 1)))

# AUC-ROC of the linear variant, from each folder's metrics_summary.csv.
AUC_COL, VARIANT = "AUC_ROC", "Linear / Ridge"


def linear_auc(domain):
    p = os.path.join("CAM-UV", f"CAM_UV_{domain}", "outputs", "metrics_summary.csv")
    if not os.path.exists(p):
        return None
    for row in csv.DictReader(open(p, encoding="utf-8")):
        if row.get("Model Variant") == VARIANT:
            return float(row[AUC_COL])
    return None


def main():
    rows = []
    for p in sorted(glob.glob("CAM-UV/**/*_camuv_*.npz", recursive=True)):
        z = np.load(p, allow_pickle=True)
        V = np.asarray(z["val_matrix"])[:, :, 1]
        cp = np.asarray(z["confounded_pairs"])
        d = V.shape[0]
        edges = int((V != 0).sum())
        conf_edges = sum(int(V[i, j] != 0) + int(V[j, i] != 0) for i, j in cp)
        possible = d * (d - 1) // 2
        domain = p.replace("\\", "/").split("/")[1].replace("CAM_UV_", "")
        n = discovery_samples(domain, int(z["subsample"]))
        rows.append(dict(domain=domain, d=d, n=n, spv=(n / d if n else None),
                         edges=edges, pairs=len(cp),
                         conf_edges=conf_edges,
                         conf_share=conf_edges / max(edges, 1),
                         saturation=len(cp) / max(possible, 1),
                         density=edges / max(d * d - d, 1),
                         auc=linear_auc(domain)))

    print(f"{'domain':12} {'vars':>5} {'n':>6} {'n/var':>7} {'edges':>6} {'conf':>5} "
          f"{'from conf':>10} {'pair':>10} {'CAM-UV':>8}")
    print(f"{'':12} {'':>5} {'':>6} {'':>7} {'':>6} {'pairs':>5} "
          f"{'edges':>10} {'saturation':>10} {'lin AUC':>8}")
    print("-" * 82)
    for r in sorted(rows, key=lambda x: -x["saturation"]):
        auc = f"{r['auc']:.4f}" if r["auc"] is not None else "n/a"
        spv = f"{r['spv']:.2f}" if r["spv"] else "?"
        print(f"{r['domain']:12} {r['d']:5d} {r['n'] or 0:6d} {spv:>7} {r['edges']:6d} "
              f"{r['pairs']:5d} {r['conf_share']:9.1%} {r['saturation']:10.1%} {auc:>8}")
    thin = [r for r in rows if r["spv"] and r["spv"] < 1.0]
    for r in thin:
        print()
        print(f"  WARNING {r['domain']}: {r['n']} samples for {r['d']} variables "
              f"({r['spv']:.2f}/var). Discovery is underdetermined and its low")
        print("          saturation cannot be distinguished from insufficient power "
              "to reject independence.")

    out = "CAM-UV/confounder_structure.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"\nwrote {out}")

    ordered = sorted([r for r in rows if r["auc"] is not None], key=lambda x: x["saturation"])
    if len(ordered) == 3:
        print("\nordered by pair saturation:")
        for r in ordered:
            print(f"  {r['domain']:12} saturation {r['saturation']:6.1%}   "
                  f"CAM-UV linear AUC {r['auc']:.4f}")
        mono = all(ordered[i]["auc"] >= ordered[i + 1]["auc"] for i in range(2))
        print(f"\n  AUC falls monotonically as saturation rises: {mono}")
        print("  (three points -- suggestive of a mechanism, not an established trend)")


if __name__ == "__main__":
    main()
