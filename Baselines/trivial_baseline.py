"""
A no-learning control: how much of each domain is solved without any model at all?

Neither the causal pipelines nor the neural autoencoders answer the question "is a model
needed here". This scores every domain with a detector that has no parameters and no training
beyond a mean and a standard deviation:

    score(t) = mean_j | (x_j(t) - mu_j) / sigma_j |

with mu and sigma taken from the fault-free training split, thresholded at the same 95th
fault-free percentile everything else uses. It cannot represent structure, dynamics, or
interaction -- only distance from the fault-free operating point.

It is the floor any real method has to clear. Where it scores well the domain is separable in
raw value space and detection results there say little about the modelling; where it scores
poorly, the gap over it is what the modelling actually bought.

Usage:  python Baselines/trivial_baseline.py
"""
import csv
import importlib.util
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

sys.path.insert(0, "Baselines")
_spec = importlib.util.spec_from_file_location("nba", "Baselines/neural_baselines_all.py")
_m = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(_m)
except SystemExit:
    pass

THRESHOLD_PERCENTILE = 95.0


def score(domain):
    train, heldout, attacks, _names, label = _m.LOADERS[domain]()
    mu, sd = train.mean(0), train.std(0)
    sd[sd == 0] = 1.0
    f = lambda a: np.abs((a - mu) / sd).mean(1)
    ns = f(heldout)
    thr = float(np.percentile(ns, THRESHOLD_PERCENTILE))
    y, s = [np.zeros(len(ns), int)], [ns]
    for raw in attacks.values():
        a = f(raw)
        y.append(np.ones(len(a), int))
        s.append(a)
    y, s = np.concatenate(y), np.concatenate(s)
    return dict(Dataset=label, Model="Trivial control", Variant="mean |z| from fault-free",
                F1=f"{f1_score(y, (s > thr).astype(int)):.4f}",
                AUC_ROC=f"{roc_auc_score(y, s):.4f}",
                AUC_PR=f"{average_precision_score(y, s):.4f}",
                n=len(y), prevalence=f"{y.mean():.4f}")


def main():
    rows = [score(d) for d in ("robotics", "healthcare", "industrial")]
    out = os.path.join("Baselines", "outputs", "trivial_baseline.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{'Domain':20} {'F1':>8} {'AUC-ROC':>9} {'AUC-PR':>8} {'n':>7} {'prevalence':>11}")
    print("-" * 68)
    for r in rows:
        print(f"{r['Dataset']:20} {r['F1']:>8} {r['AUC_ROC']:>9} {r['AUC_PR']:>8} "
              f"{r['n']:>7} {r['prevalence']:>11}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
