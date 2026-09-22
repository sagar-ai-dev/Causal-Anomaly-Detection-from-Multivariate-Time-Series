"""
How much structure does each discovered graph actually impose?

A causal detector is supposed to gain its advantage from structure: each variable is
regressed on its parents rather than on everything, and that restriction is what stops the
model fitting noise. The argument only works if the graph is actually sparse. A graph that
links every pair imposes no restriction at all, and a detector built on it is regressing
each variable on all the others under a different name.

So the claim "structural constraint prevented overfitting" is measurable, and this measures
it for all nine algorithm-domain pairs rather than asserting it for one.

All three algorithms cache val_matrix and p_matrix in the same layout and sparsify with the
same rule, so the comparison is like for like:

    kept = (p_matrix < ALPHA) & (abs(val_matrix) > CAUSAL_STRENGTH_MULTIPLIER * mean(abs(val_matrix)))

ALPHA is 0.05 everywhere. The multiplier is 1.0 in every folder except PCMCI industrial,
where it is 0.0 and the second condition is therefore always true.

Density is reported over unordered variable pairs, counting a pair as linked when any lag
carries a surviving edge. That is the right denominator for the question being asked --
whether variable i is constrained to ignore variable j -- and it is comparable across
algorithms even though PCMCI searches four lags where GES and CAM-UV search two.

Usage:  python tools/graph_density.py
"""

import csv
import glob
import io
import re
import sys

import numpy as np

ALPHA = 0.05


def multiplier_for(folder):
    """Read the folder's own sparsification constant rather than assuming it."""
    for p in glob.glob("*/%s/anomaly_detection*.py" % folder):
        m = re.search(r"^CAUSAL_STRENGTH_MULTIPLIER\s*=\s*([0-9.]+)",
                      io.open(p, encoding="utf-8").read(), re.M)
        if m:
            return float(m.group(1))
    return None


def cached_graph(folder):
    for pat in ("*/%s/**/*.npz" % folder, "*/%s/*.npz" % folder):
        for p in sorted(glob.glob(pat, recursive=True)):
            if "scores_" in p:
                continue
            try:
                z = np.load(p, allow_pickle=True)
            except Exception:
                continue
            if "p_matrix" in z and "val_matrix" in z:
                return p, z
    return None, None


def pair_density(p_matrix, val_matrix, mult):
    """(linked pairs, total pairs, linked under significance alone)."""
    d = p_matrix.shape[0]
    sig = p_matrix < ALPHA
    keep = sig & (np.abs(val_matrix) > mult * np.mean(np.abs(val_matrix)))
    linked = both = 0
    for i in range(d):
        for j in range(i + 1, d):
            if sig[i, j, :].any() or sig[j, i, :].any():
                linked += 1
            if keep[i, j, :].any() or keep[j, i, :].any():
                both += 1
    return both, d * (d - 1) // 2, linked


def main():
    folders = sorted({p.replace("\\", "/").split("/")[1]
                      for p in glob.glob("*/*/anomaly_detection*.py")})
    rows = []
    print("Graph density, all nine algorithm-domain pairs\n")
    print("  edge kept when  p < %.2f  AND  |val| > multiplier * mean|val|\n" % ALPHA)
    print("  %-22s %5s %5s %6s %11s %11s %9s"
          % ("folder", "vars", "lags", "mult", "pairs", "linked", "density"))
    print("  " + "-" * 76)

    for f in folders:
        path, z = cached_graph(f)
        if z is None:
            print("  %-22s no cached graph" % f)
            continue
        mult = multiplier_for(f)
        if mult is None:
            print("  %-22s no multiplier found" % f)
            continue
        p_m, v_m = z["p_matrix"], z["val_matrix"]
        kept, total, sig_only = pair_density(p_m, v_m, mult)
        dens = kept / total if total else float("nan")
        print("  %-22s %5d %5d %6.1f %11d %11d %8.1f%%"
              % (f, p_m.shape[0], p_m.shape[2], mult, total, kept, 100 * dens))
        rows.append({"Folder": f, "Cache": path.replace("\\", "/"),
                     "Variables": p_m.shape[0], "Lags": p_m.shape[2],
                     "Multiplier": mult, "Pairs": total,
                     "Linked significance only": sig_only,
                     "Linked pipeline rule": kept, "Density": "%.4f" % dens})

    if not rows:
        raise SystemExit("no cached graphs found")

    out = "tools/graph_density.csv"
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n  wrote %s" % out)

    print("\n  The effect-size floor barely matters. Significance alone against the full rule:")
    for r in rows:
        a, b = r["Linked significance only"], r["Linked pipeline rule"]
        if a != b:
            print("    %-22s %d -> %d pairs" % (r["Folder"], a, b))
    if all(r["Linked significance only"] == r["Linked pipeline rule"] for r in rows):
        print("    no folder changes at all")

    print("\n  Read this against the detection results before claiming structure did the work.")
    print("  A graph near 100%% density constrains nothing: every variable keeps every other")
    print("  as a candidate parent, and the regression is not meaningfully restricted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
