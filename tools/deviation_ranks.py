"""
Which variables actually move most during each Pepper attack?

This exists to settle an argument the thesis makes about SHAP. All three neural
architectures attribute the JointControl fault to RShoulderPitch, a kinematic joint angle,
and none of them to a joint temperature, which is what the laboratory requirement asks for.
The explanation offered is that a reconstruction autoencoder minimises squared error, so an
attribution over it necessarily surfaces whatever deviates most -- and that this happens to
be the symptom rather than the cause.

That is a claim about the raw data, not about the models, and it is checkable without
either. This measures the standardised deviation of every variable during each attack,
against fault-free statistics, and ranks them. If RShoulderPitch is not near the top, the
explanation is wrong and the thesis needs a different one.

The statistic is deliberately the simplest thing that could answer the question:

    z_i = | mean_attack(x_i) - mu_i | / sigma_i

with mu and sigma taken from the fault-free recording. No model, no graph, no threshold --
so it cannot be accused of inheriting an assumption from the pipeline it is being used to
explain.

Usage:  python tools/deviation_ranks.py
"""

import csv
import io
import os
import sys

import numpy as np
import pandas as pd

PREFIX = "PCMCI/PCMCI_robotic/pepper_csv"
NORMAL = "normal.csv"
ATTACKS = ["WheelsControl.csv", "JointControl.csv", "LedsControl.csv"]
TOP_N = 5

# Variables the two laboratory requirements name, so their ranks can be reported
# alongside whatever actually came top.
def is_joint_temp(v):
    return v.endswith("Temp") and any(j in v for j in
                                      ("Knee", "Hip", "Shoulder", "Elbow", "Wrist", "Head"))


def is_tablet(v):
    return v.startswith("Tablet")


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.chdir(root)
    npath = os.path.join(PREFIX, NORMAL)
    if not os.path.exists(npath):
        raise SystemExit("raw Pepper data not present; this needs " + npath)

    normal = pd.read_csv(npath)
    if "timestamp" in normal.columns:
        normal = normal.drop(columns=["timestamp"])
    mu = normal.mean()
    sd = normal.std().replace(0, np.nan)

    rows = []
    print("Standardised deviation during each attack, against fault-free statistics\n")
    print("  z_i = |mean_attack(x_i) - mu_i| / sigma_i,  no model involved\n")

    for fname in ATTACKS:
        apath = os.path.join(PREFIX, fname)
        if not os.path.exists(apath):
            continue
        atk = pd.read_csv(apath)
        if "timestamp" in atk.columns:
            atk = atk.drop(columns=["timestamp"])
        cols = [c for c in normal.columns if c in atk.columns]
        z = ((atk[cols].mean() - mu[cols]) / sd[cols]).abs().dropna().sort_values(ascending=False)
        order = list(z.index)
        n = len(order)

        print("  %s   (%d variables)" % (fname, n))
        for v in order[:TOP_N]:
            print("    %2d. %-26s %6.2f" % (order.index(v) + 1, v, z[v]))

        # the requirement-relevant variable that ranks highest, for contrast
        want = is_joint_temp if "Joint" in fname else (is_tablet if "Leds" in fname else None)
        best_req, best_rank = None, None
        if want is not None:
            for v in order:
                if want(v):
                    best_req, best_rank = v, order.index(v) + 1
                    break
            if best_req:
                print("    best requirement-satisfying variable: %s at rank %d, z = %.2f"
                      % (best_req, best_rank, z[best_req]))
        print()

        for v in order[:TOP_N]:
            rows.append({"Fault": fname.replace(".csv", ""), "Variable": v,
                         "Rank": order.index(v) + 1, "Mean abs z": "%.4f" % z[v],
                         "Variables ranked": n,
                         "Satisfies requirement":
                             "yes" if (want and want(v)) else ("no" if want else "")})
        if best_req and best_rank > TOP_N:
            rows.append({"Fault": fname.replace(".csv", ""), "Variable": best_req,
                         "Rank": best_rank, "Mean abs z": "%.4f" % z[best_req],
                         "Variables ranked": n, "Satisfies requirement": "yes"})

    out = "tools/deviation_ranks.csv"
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  wrote %s (%d rows)" % (out, len(rows)))

    jc = [r for r in rows if r["Fault"] == "JointControl"]
    if jc:
        top = min(jc, key=lambda r: r["Rank"])
        req = [r for r in jc if r["Satisfies requirement"] == "yes"]
        print()
        print("  For JointControl, the claim the thesis rests on:")
        print("    largest deviation overall      %s, rank %s, z = %s"
              % (top["Variable"], top["Rank"], top["Mean abs z"]))
        if req:
            b = min(req, key=lambda r: r["Rank"])
            print("    highest joint temperature      %s, rank %s, z = %s"
                  % (b["Variable"], b["Rank"], b["Mean abs z"]))
            print()
            print("  A reconstruction objective surfaces the first. The causal attribution")
            print("  returns the second in 8 of 9 configurations, despite it ranking lower.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
