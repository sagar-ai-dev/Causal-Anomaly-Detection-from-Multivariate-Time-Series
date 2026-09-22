"""
Analytical 95% confidence intervals on AUC-ROC, and pairwise significance.

Uses the Hanley & McNeil (1982) standard error, which needs only the AUC and the class
counts -- so it runs from the committed metrics_summary.csv files without re-running any
experiment.

IMPORTANT CAVEAT. Hanley-McNeil assumes independent samples. These are time series and
adjacent samples are autocorrelated, so the effective sample size is smaller than n and the
true intervals are WIDER than reported here. Every "not significant" below is therefore
safe; a "significant" should be read as borderline unless the margin is large.

Usage:  python tools/significance.py
"""
import csv, glob, math

DOM = ["Robotics (Pepper)", "Healthcare (ECG)", "Industrial (TEP)"]


def hanley_se(auc, npos, nneg):
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    v = (auc * (1 - auc) + (npos - 1) * (q1 - auc ** 2) + (nneg - 1) * (q2 - auc ** 2)) / (npos * nneg)
    return math.sqrt(max(v, 0.0))


def load():
    rows = []
    for p in sorted(glob.glob("*/*/outputs/metrics_summary.csv")):
        rows += list(csv.DictReader(open(p, newline="", encoding="utf-8")))
    for r in rows:
        r["npos"] = int(r["TP"]) + int(r["FN"])
        r["nneg"] = int(r["TN"]) + int(r["FP"])
        r["auc"] = float(r["AUC_ROC"])
        r["se"] = hanley_se(r["auc"], r["npos"], r["nneg"])
    return rows


def main():
    rows = load()
    print(f"{'Domain':18} {'Model':10} {'Variant':21} {'AUC':>7}  95% CI")
    print("-" * 72)
    for d in DOM:
        for r in sorted([x for x in rows if x["Dataset"] == d], key=lambda x: -x["auc"]):
            lo, hi = r["auc"] - 1.96 * r["se"], r["auc"] + 1.96 * r["se"]
            print(f"{d:18} {r['Model']:10} {r['Model Variant']:21} {r['auc']:7.4f}  [{lo:.4f}, {hi:.4f}]")
        print()

    print("=== pairwise: is the best in each domain separable from the rest? ===")
    for d in DOM:
        rs = sorted([x for x in rows if x["Dataset"] == d], key=lambda x: -x["auc"])
        best = rs[0]
        print(f"\n  {d} -- best: {best['Model']} {best['Model Variant']} ({best['auc']:.4f})")
        for other in rs[1:]:
            diff = best["auc"] - other["auc"]
            se = math.sqrt(best["se"] ** 2 + other["se"] ** 2)
            z = diff / se if se else 0.0
            mark = "significant    " if abs(z) > 1.96 else "NOT significant"
            print(f"    vs {other['Model']:10} {other['Model Variant']:21} "
                  f"d={diff:+.4f} z={z:5.2f}  {mark}")


if __name__ == "__main__":
    main()
