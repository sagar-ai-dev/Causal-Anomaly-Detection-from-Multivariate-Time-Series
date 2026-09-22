"""
DeLong's test for two AUCs measured on the same samples.

A bootstrap interval says how precisely one AUC is known. It cannot say whether two AUCs
differ, and reading two overlapping intervals as "no difference" is wrong in a specific
direction: paired scores are correlated, the difference has smaller variance than either
AUC alone, and overlapping intervals routinely hide a real and significant gap. DeLong
(1988) works on the difference directly, using the covariance between the two score
vectors, so it answers the question the thesis actually asks.

The catch, and it is the finding rather than a footnote: DeLong requires both AUCs to be
measured on *the same samples*, and in this study most pairs are not.

    domain        PCMCI      GES / CAM-UV     neural
    robotics        181                 1378        155
    healthcare     4806                 3115       4793
    industrial    16716                 1323      16590

The causal counts differ because the scoring resolution does -- PCMCI selects its subsample
rate adaptively from the frequency content of the signal, while GES and CAM-UV are fixed
at 10 -- so PCMCI simply does not score the same time points as the other two. The neural
counts differ again because the autoencoders consume overlapping windows of length 10 rather
than a subsampled series. No test can pair score vectors that index different samples, and
resampling one onto the other would invent data.

So this reports exactly three things, and refuses to guess at the rest:

  pairable        every comparison where the two configurations score identical samples,
                  tested and reported with a p-value. This covers all of RQ2 -- the three
                  regression variants within one algorithm and domain always share a sample
                  set -- and every GES against CAM-UV comparison, which share both a
                  subsample rate of 10 and an attack structure.

  not pairable    every comparison that cannot be tested, named individually with the
                  reason, so the gap is visible rather than silently absent.

  what it costs   the headline RQ3 claim -- best causal against best neural -- is among the
                  untestable ones. It is a comparison of 3115 samples against 4793. That is
                  a real limitation of the study design and belongs in the threats section,
                  not hidden behind a word like "significantly".

Usage:  python tools/delong_test.py
"""

import csv
import glob
import io
import itertools
import sys

import numpy as np

ALPHA = 0.05


def compute_midrank(x):
    """Midranks, which is what makes DeLong exact in the presence of ties."""
    J = np.argsort(x)
    Z = x[J]
    n = len(x)
    T = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[J] = T
    return out


def delong_cov(scores, y):
    """AUCs and their covariance matrix, following Sun and Xu (2014) for the fast form.

    scores is (k, n) for k predictors on the same n samples; y is the shared label vector.
    """
    order = np.argsort(-y, kind="mergesort")          # positives first, order preserved
    s = scores[:, order]
    m = int(np.sum(y == 1))
    n = s.shape[1] - m
    k = s.shape[0]

    pos, neg = s[:, :m], s[:, m:]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = compute_midrank(pos[r])
        ty[r] = compute_midrank(neg[r])
        tz[r] = compute_midrank(s[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01) if k > 1 else np.array([[np.var(v01, ddof=1)]])
    sy = np.cov(v10) if k > 1 else np.array([[np.var(v10, ddof=1)]])
    return aucs, np.atleast_2d(sx) / m + np.atleast_2d(sy) / n


def delong_p(y, s1, s2):
    """Two-sided p-value for AUC(s1) == AUC(s2) on shared samples."""
    from scipy.stats import norm
    aucs, cov = delong_cov(np.vstack([s1, s2]), y)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), float("nan"), 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(z), float(2 * norm.sf(abs(z)))


def load_config(path):
    """(key, labels, scores) for one configuration; key identifies the sample set."""
    z = np.load(path, allow_pickle=True)
    normal = np.asarray(z["normal_scores"], dtype=float)
    keys = sorted(k for k in z.files if k.startswith("attack__"))
    y = [np.zeros(len(normal), dtype=int)]
    s = [normal]
    shape = [("normal", len(normal))]
    for k in keys:
        a = np.asarray(z[k], dtype=float)
        y.append(np.ones(len(a), dtype=int))
        s.append(a)
        shape.append((k, len(a)))
    folder = path.replace("\\", "/").split("/")[1]
    return {
        "path": path, "folder": folder,
        "dataset": str(z["dataset"]), "variant": str(z["model_variant"]),
        "y": np.concatenate(y), "s": np.concatenate(s),
        "shape": tuple(shape),                 # identical shape => identical sample set
    }


def subsample_of(folder):
    for p in glob.glob("*/%s/**/*.npz" % folder, recursive=True) + glob.glob("*/%s/*.npz" % folder):
        if "scores_" in p:
            continue
        try:
            z = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if "subsample" in z:
            return int(z["subsample"])
    return None


def main():
    paths = sorted(glob.glob("*/*/outputs/scores_*.npz"))
    if not paths:
        raise SystemExit("No scores found. Run the pipelines first.")
    cfgs = [load_config(p) for p in paths]

    print("DeLong test for paired AUCs, alpha = %.2f\n" % ALPHA)
    print("Two configurations can be tested only when they score identical samples.")
    print("Sample sets, by folder:\n")
    print("  %-22s %-20s %8s %10s" % ("folder", "dataset", "n", "subsample"))
    print("  " + "-" * 64)
    seen = {}
    for c in cfgs:
        if c["folder"] in seen:
            continue
        seen[c["folder"]] = True
        print("  %-22s %-20s %8d %10s"
              % (c["folder"], c["dataset"], len(c["y"]), subsample_of(c["folder"])))

    tested, skipped = [], []
    for a, b in itertools.combinations(cfgs, 2):
        if a["dataset"] != b["dataset"]:
            continue
        label = ("%s %s  vs  %s %s"
                 % (a["folder"], a["variant"], b["folder"], b["variant"]))
        if a["shape"] != b["shape"]:
            skipped.append({
                "Dataset": a["dataset"], "Comparison": label,
                "Reason": "different sample sets (n=%d vs n=%d; subsample %s vs %s)"
                          % (len(a["y"]), len(b["y"]),
                             subsample_of(a["folder"]), subsample_of(b["folder"])),
            })
            continue
        assert np.array_equal(a["y"], b["y"]), "same shape but different labels: " + label
        auc1, auc2, z, p = delong_p(a["y"], a["s"], b["s"])
        tested.append({
            "Dataset": a["dataset"], "A": "%s %s" % (a["folder"], a["variant"]),
            "B": "%s %s" % (b["folder"], b["variant"]),
            "AUC_A": "%.4f" % auc1, "AUC_B": "%.4f" % auc2,
            "Difference": "%.4f" % (auc1 - auc2),
            "z": "%.4f" % z if z == z else "",
            "p": "%.6f" % p, "n": len(a["y"]),
            "Significant": "yes" if p < ALPHA else "no",
        })

    print("\n\n%d comparisons tested, %d not pairable\n" % (len(tested), len(skipped)))
    for d in sorted({t["Dataset"] for t in tested}):
        rows = [t for t in tested if t["Dataset"] == d]
        print("  " + d)
        print("  %-34s %-34s %8s %10s %5s" % ("A", "B", "diff", "p", "sig"))
        print("  " + "-" * 96)
        for r in sorted(rows, key=lambda x: float(x["p"])):
            print("  %-34s %-34s %8s %10s %5s"
                  % (r["A"][:34], r["B"][:34], r["Difference"], r["p"], r["Significant"]))
        print()

    with io.open("tools/delong_tests.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tested[0].keys()))
        w.writeheader()
        w.writerows(tested)
    with io.open("tools/delong_not_pairable.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Dataset", "Comparison", "Reason"])
        w.writeheader()
        w.writerows(skipped)
    print("wrote tools/delong_tests.csv (%d) and tools/delong_not_pairable.csv (%d)"
          % (len(tested), len(skipped)))

    reasons = {}
    for s_ in skipped:
        reasons[s_["Reason"]] = reasons.get(s_["Reason"], 0) + 1
    print("\nWhy the rest could not be tested:")
    for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print("  %4d  %s" % (n, r))
    print("\nThe causal-versus-neural comparison is absent from both files: the autoencoders")
    print("score overlapping windows rather than a subsampled series, so no score file pairs")
    print("with theirs at all. That comparison cannot be significance-tested as the study")
    print("stands, and the thesis should say so rather than write 'significantly'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
