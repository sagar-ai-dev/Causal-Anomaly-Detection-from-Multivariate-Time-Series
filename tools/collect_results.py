"""
Collect every metrics_summary.csv into one table.

Thesis tables are generated from this, never transcribed by hand: a number that appears in
README.md or a FINAL_RESULTS.md can be traced to the run that produced it. Re-run any
configuration and re-run this script to refresh.

Usage:  python tools/collect_results.py [--markdown]
"""
import csv, glob,  sys

DOMAIN_ORDER = ["Robotics (Pepper)", "Healthcare (ECG)", "Industrial (TEP)"]
MODEL_ORDER = ["PCMCI", "GES", "CAM-UV", "Neural Baseline"]
VARIANT_ORDER = ["Linear / Ridge", "Polynomial degree 3", "RBF + Ridge", "LSTM Autoencoder"]


def load():
    rows = []
    for path in sorted(glob.glob("*/*/outputs/metrics_summary.csv")) + \
                sorted(glob.glob("Baselines/outputs/metrics_summary.csv")):
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_source"] = path
                rows.append(r)
    return rows


def sort_key(r):
    d = DOMAIN_ORDER.index(r["Dataset"]) if r["Dataset"] in DOMAIN_ORDER else 99
    m = MODEL_ORDER.index(r["Model"]) if r["Model"] in MODEL_ORDER else 99
    v = VARIANT_ORDER.index(r["Model Variant"]) if r["Model Variant"] in VARIANT_ORDER else 99
    return (d, m, v)


def main():
    rows = sorted(load(), key=sort_key)
    md = "--markdown" in sys.argv

    if md:
        print("| Dataset | Model | Variant | TP | FP | TN | FN | Acc | Prec | Rec | F1 | AUC-ROC | AUC-PR |")
        print("| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in rows:
            print(f"| {r['Dataset']} | {r['Model']} | {r['Model Variant']} | "
                  f"{r['TP']} | {r['FP']} | {r['TN']} | {r['FN']} | {r['Accuracy']} | "
                  f"{r['Precision']} | {r['Recall']} | **{r['F1']}** | **{r['AUC_ROC']}** | {r['AUC_PR']} |")
    else:
        hdr = f"{'Dataset':18} {'Model':16} {'Variant':21} {'F1':>8} {'AUC-ROC':>8} {'AUC-PR':>8}  n"
        print(hdr); print("-" * len(hdr))
        for r in rows:
            n = sum(int(r[k]) for k in ("TP", "FP", "TN", "FN"))
            print(f"{r['Dataset']:18} {r['Model']:16} {r['Model Variant']:21} "
                  f"{r['F1']:>8} {r['AUC_ROC']:>8} {r['AUC_PR']:>8}  {n}")
        print(f"\n{len(rows)} configurations "
              f"({len([r for r in rows if r['Model'] != 'Neural Baseline'])} causal "
              f"+ {len([r for r in rows if r['Model'] == 'Neural Baseline'])} neural baseline)")

        missing = 28 - len(rows)
        if missing:
            print(f"WARNING: expected 28 rows (27 causal + 1 baseline), found {len(rows)}")


if __name__ == "__main__":
    main()
