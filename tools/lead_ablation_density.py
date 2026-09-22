"""
Compare graph density before and after the eight-lead restriction, without the variable-count
confound.

The obvious comparison -- density of the twelve-lead graph against density of the eight-lead
graph -- does not work. Density is the share of variable pairs linked, and dropping four
variables drops the denominator from C(12,2) = 66 to C(8,2) = 28. Two of the three algorithms
then register as denser while finding strictly fewer edges, which is arithmetic rather than
structure.

Restricting the twelve-lead graph to the same eight leads fixes it. Both numbers are then out
of 28, over the same variables, and the only thing that differs is whether the four
algebraically dependent leads were present during discovery.

What the comparison is for: Section on graph density argues that a sparser graph detects
better on healthcare. The eight-lead ablation offers a within-algorithm test of that, free of
the cross-algorithm confounds -- different sample counts, different variable sets -- that the
main density argument has to concede.

The answer separates the algorithms rather than confirming a single story, which is why it is
worth persisting rather than quoting from a one-off calculation. GES and CAM-UV both get
denser and both get worse, consistent with the mechanism. PCMCI links exactly the same 26 of
28 pairs under both conditions and improves regardless, which rules density out as the
explanation for its gain and leaves the multicollinearity account standing.

Usage:  python tools/lead_ablation_density.py
"""

import csv
import io
import os
import sys

import numpy as np

ALPHA = 0.05
MULT = 1.0
INDEPENDENT = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]

# algorithm -> (cache pattern with a %s slot for the ablation suffix, 12-lead AUCs, 8-lead AUCs)
CONFIGS = [
    ("PCMCI", "PCMCI/PCMCI_healthcare/Healthcare dataset/healthcare_normal%s.npz"),
    ("GES", "GES/GES_healthcare/Healthcare dataset/healthcare_normal_ges%s.npz"),
    ("CAM-UV",
     "CAM-UV/CAM_UV_healthcare/Healthcare dataset/healthcare_normal_camuv_integrate_a0.05%s.npz"),
]

FOLDERS = {"PCMCI": "PCMCI/PCMCI_healthcare", "GES": "GES/GES_healthcare",
           "CAM-UV": "CAM-UV/CAM_UV_healthcare"}
VARIANTS = ["Linear / Ridge", "Polynomial degree 3", "RBF + Ridge"]


def linked_pairs(path, restrict_to=None):
    """Pairs carrying a surviving edge at any lag, optionally over a subset of variables."""
    z = np.load(path, allow_pickle=True)
    p, v = z["p_matrix"], z["val_matrix"]
    names = [str(x) for x in np.array(z["var"])[z["nonconst"]]]
    keep = (p < ALPHA) & (np.abs(v) > MULT * np.mean(np.abs(v)))
    idx = ([i for i, n in enumerate(names) if n in restrict_to] if restrict_to
           else list(range(len(names))))
    n = sum(1 for a in range(len(idx)) for b in range(a + 1, len(idx))
            if keep[idx[a], idx[b], :].any() or keep[idx[b], idx[a], :].any())
    return n, len(idx) * (len(idx) - 1) // 2


def mean_auc(folder, suffix):
    d = os.path.join(folder, "outputs" + suffix, "metrics_summary.csv")
    if not os.path.exists(d):
        return None
    rows = {r["Model Variant"]: r for r in csv.DictReader(io.open(d, encoding="utf-8"))}
    vals = [float(rows[v]["AUC_ROC"]) for v in VARIANTS if v in rows]
    return sum(vals) / len(vals) if vals else None


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.chdir(root)

    print("Graph density across the eight-lead restriction, over a common 28 pairs\n")
    print("  The twelve-lead graph is restricted to the same eight leads, so the denominator")
    print("  is identical on both sides and only the discovery input differs.\n")
    print("  %-11s %22s %22s %11s" % ("algorithm", "12-lead, 8-lead subset",
                                      "8-lead graph", "mean dAUC"))
    print("  " + "-" * 72)

    rows = []
    for name, pattern in CONFIGS:
        full, abl = pattern % "", pattern % "_leads8"
        if not (os.path.exists(full) and os.path.exists(abl)):
            print("  %-11s cache missing; skipped" % name)
            continue
        n12, t12 = linked_pairs(full, INDEPENDENT)
        n8, t8 = linked_pairs(abl)
        a12 = mean_auc(FOLDERS[name], "")
        a8 = mean_auc(FOLDERS[name], "_leads8")
        d = (a8 - a12) if (a12 is not None and a8 is not None) else float("nan")
        print("  %-11s %13d/%d = %5.1f%% %13d/%d = %5.1f%% %+11.4f"
              % (name, n12, t12, 100.0 * n12 / t12, n8, t8, 100.0 * n8 / t8, d))
        rows.append({"Algorithm": name,
                     "Pairs": t12,
                     "Linked 12-lead graph over 8 leads": n12,
                     "Density 12-lead subset": "%.4f" % (n12 / t12),
                     "Linked 8-lead graph": n8,
                     "Density 8-lead": "%.4f" % (n8 / t8),
                     "Mean AUC 12-lead": "%.4f" % a12 if a12 is not None else "",
                     "Mean AUC 8-lead": "%.4f" % a8 if a8 is not None else "",
                     "Mean AUC delta": "%.4f" % d})

    if not rows:
        raise SystemExit("no caches found; run the healthcare pipelines and the ablation first")

    out = "tools/lead_ablation_density.csv"
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n  wrote %s" % out)

    print("\n  Reading:")
    for r in rows:
        d12 = float(r["Density 12-lead subset"])
        d8 = float(r["Density 8-lead"])
        da = float(r["Mean AUC delta"])
        if abs(d8 - d12) < 1e-9:
            verdict = ("density identical, so it cannot explain the AUC change of %+.4f" % da)
        elif (d8 > d12) == (da < 0):
            verdict = "density and performance move oppositely, consistent with the mechanism"
        else:
            verdict = "density and performance move together, against the mechanism"
        print("    %-11s %s" % (r["Algorithm"], verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
