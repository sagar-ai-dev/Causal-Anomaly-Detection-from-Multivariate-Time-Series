"""
Block-bootstrap confidence intervals for the causal pipelines, from real per-sample scores.

The neural baselines are averaged over three seeds; the causal configurations are single-run.
Causal discovery is deterministic given the data, so their variance cannot come from
reseeding -- it has to come from resampling the evaluation set.

Why a *block* bootstrap. The naive bootstrap and the analytical Hanley-McNeil interval both
assume independent samples. These are time series: adjacent windows overlap and their scores
are strongly autocorrelated, so the effective sample size is well below n and both methods
report intervals that are too narrow. Resampling contiguous blocks preserves local dependence
and gives an interval that does not pretend the samples are independent.

Block length follows the usual n^(1/3) rule of thumb, floored at 2.

Each configuration draws from its own generator, seeded from its file path, rather than from
one shared stream. With a shared stream the bounds of every configuration shift whenever a
file is added, removed or renamed, because each draw advances the state for the next -- which
made the reported intervals unreproducible across runs. Per-path seeding makes each
configuration's interval depend only on its own scores.

Requires `scores_<variant>.npz` in each outputs/ directory, written by reporting_helper. Run the
pipelines first if they are absent.

Usage:  python tools/bootstrap_intervals.py [--draws 2000]
"""
import csv
import glob
import sys
import zlib

import numpy as np
from sklearn.metrics import roc_auc_score

DOMAINS = ["Robotics (Pepper)", "Healthcare (ECG)", "Industrial (TEP)"]


def load_scores(path):
    """Return (labels, scores) for one configuration, normal first then each attack file."""
    z = np.load(path, allow_pickle=True)
    normal = np.asarray(z["normal_scores"], dtype=float)
    y = [np.zeros(len(normal), dtype=int)]
    s = [normal]
    for k in z.files:
        if k.startswith("attack__"):
            a = np.asarray(z[k], dtype=float)
            y.append(np.ones(len(a), dtype=int))
            s.append(a)
    return np.concatenate(y), np.concatenate(s), str(z["model_variant"]), str(z["dataset"])


def block_bootstrap_auc(y, s, draws, rng):
    """Percentile interval for AUC-ROC under a moving-block bootstrap."""
    n = len(y)
    L = max(2, int(round(n ** (1 / 3))))
    n_blocks = int(np.ceil(n / L))
    starts_max = max(1, n - L + 1)
    out = []
    for _ in range(draws):
        starts = rng.integers(0, starts_max, n_blocks)
        idx = np.concatenate([np.arange(st, min(st + L, n)) for st in starts])[:n]
        yy, ss = y[idx], s[idx]
        if yy.min() == yy.max():
            continue
        out.append(roc_auc_score(yy, ss))
    return np.asarray(out), L


def hanley_mcneil_ci(y, auc):
    """The analytical 95% interval, for comparison against the block bootstrap.

    Hanley and McNeil (1982) give a closed-form standard error for the area under the ROC
    curve, treating the sample as independent draws:

        Q1 = A / (2 - A),   Q2 = 2A^2 / (1 + A)

        SE = sqrt( [ A(1-A) + (n_pos - 1)(Q1 - A^2) + (n_neg - 1)(Q2 - A^2) ]
                   / (n_pos * n_neg) )

    The independence assumption is the whole point of computing it here. These scores are a
    time series, consecutive samples are strongly dependent, and the effective sample size is
    therefore far below the nominal one. The analytical interval does not know that and comes
    out too narrow, which is the argument for reporting the bootstrap instead -- an argument
    the thesis should make with both numbers on the page rather than in prose alone.
    """
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc ** 2 / (1.0 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc ** 2)
           + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    se = float(np.sqrt(max(var, 0.0)))
    return max(0.0, auc - 1.96 * se), min(1.0, auc + 1.96 * se)


def main():
    draws = 2000
    if "--draws" in sys.argv:
        draws = int(sys.argv[sys.argv.index("--draws") + 1])
    paths = sorted(glob.glob("*/*/outputs/scores_*.npz"))
    if not paths:
        raise SystemExit("No scores.npz found. Run the pipelines first.")

    recs = []
    for p in paths:
        y, s, variant, dataset = load_scores(p)
        point = roc_auc_score(y, s)
        seed = zlib.crc32(p.replace("\\", "/").encode()) & 0xFFFFFFFF
        boot, L = block_bootstrap_auc(y, s, draws, np.random.default_rng(seed))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        hm_lo, hm_hi = hanley_mcneil_ci(y, point)
        recs.append(dict(Dataset=dataset, Model=p.replace("\\", "/").split("/")[1],
                         Variant=variant, AUC=point, lo=lo, hi=hi, n=len(y), block=L,
                         hm_lo=hm_lo, hm_hi=hm_hi))

    print(f"Moving-block bootstrap on real per-sample scores, {draws} draws.\n")
    print(f"{'Domain':18} {'Folder':20} {'Variant':21} {'AUC':>7}  {'95% CI':>18} {'width':>7} {'blk':>4}")
    print("-" * 100)
    for d in DOMAINS:
        for r in sorted([x for x in recs if x["Dataset"] == d], key=lambda x: -x["AUC"]):
            print(f"{d:18} {r['Model']:20} {r['Variant']:21} {r['AUC']:7.4f}  "
                  f"[{r['lo']:.4f}, {r['hi']:.4f}] {r['hi']-r['lo']:7.4f} {r['block']:4d}")
        print()

    with open("tools/bootstrap_intervals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Folder", "Variant", "AUC", "CI_low", "CI_high", "n", "block",
                    "HM_low", "HM_high"])
        for r in recs:
            w.writerow([r["Dataset"], r["Model"], r["Variant"], f"{r['AUC']:.4f}",
                        f"{r['lo']:.4f}", f"{r['hi']:.4f}", r["n"], r["block"],
                        f"{r['hm_lo']:.4f}", f"{r['hm_hi']:.4f}"])
    print(f"wrote tools/bootstrap_intervals.csv ({len(recs)} configurations)")

    print("\n=== analytical interval against the bootstrap ===")
    print("The Hanley-McNeil interval assumes independent samples. These are time series.\n")
    print(f"{'Folder':20} {'Variant':21} {'bootstrap':>18} {'Hanley-McNeil':>18} {'narrower by':>12}")
    print("-" * 92)
    for r in sorted(recs, key=lambda x: -((x["hi"] - x["lo"]) - (x["hm_hi"] - x["hm_lo"]))):
        bw, hw = r["hi"] - r["lo"], r["hm_hi"] - r["hm_lo"]
        if bw <= 0:
            continue
        print(f"{r['Model']:20} {r['Variant']:21} "
              f"[{r['lo']:.4f}, {r['hi']:.4f}] [{r['hm_lo']:.4f}, {r['hm_hi']:.4f}] "
              f"{100.0 * (1 - hw / bw):11.1f}%")
    print()

    print("\n=== is the best configuration in each domain separable? ===")
    for d in DOMAINS:
        rs = sorted([x for x in recs if x["Dataset"] == d], key=lambda x: -x["AUC"])
        if len(rs) < 2:
            continue
        best, second = rs[0], rs[1]
        overlap = best["lo"] <= second["hi"]
        print(f"  {d:18} {best['Model']}/{best['Variant'][:14]} [{best['lo']:.4f}, {best['hi']:.4f}] "
              f"vs {second['Model']}/{second['Variant'][:14]} [{second['lo']:.4f}, {second['hi']:.4f}]"
              f"  -> {'intervals OVERLAP, not separable' if overlap else 'separable'}")


if __name__ == "__main__":
    main()
