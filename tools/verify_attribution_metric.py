"""
Does the robotics attribution survive a change of ranking metric?

The robotics folders rank attributed variables by an unnormalised reduction of the
attack-window residual, for parity with the XAI laboratory: the L2 norm in the PCMCI
folders and its root mean square in the other six, which differ by a constant factor
across variables and therefore rank identically. The other six configurations divide that
norm by the variable's own fault-free baseline, which makes the score length-independent
and scale-relative. Those are different rules and they can disagree, so "the LED attack
attributes to the tablet sensors under either rule" is a claim about this data rather than a
property of the metric.

It matters because the alternative reading is uncomfortable. TabletTouchX sits at variance
rank two of 249 sensors, and a ranking by raw magnitude with no division by baseline scale
naturally favours variables that are large to begin with. If the tablet wins only under the
raw rule, the attribution result is partly an artefact of that rule. If it wins under both,
it is not, and the thesis can say so.

The README asserted the two agree. Nothing in the committed outputs demonstrated it, because
only the ranking actually used is written to explainability_top_features.csv. The detection
script now also persists the per-variable residual matrices, so this recomputes both
rankings from the same residuals and the claim rests on an artefact rather than a sentence.

Only the reduction differs:

    raw L2          A_i = || eps_:,i ||_2
    normalised RMS  A_i = RMS_attack(eps_:,i) / max(RMS_normal(eps_:,i), 1e-8)

Usage:  python tools/verify_attribution_metric.py
"""

import csv
import io
import os
import sys

import numpy as np

RESID = "PCMCI/PCMCI_robotic/outputs/residuals_linear_ridge.npz"
TOP_N = 5
EPS = 1e-8


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.chdir(root)
    if not os.path.exists(RESID):
        raise SystemExit("run PCMCI/PCMCI_robotic/anomaly_detection.py first to write " + RESID)

    z = np.load(RESID, allow_pickle=True)
    names = [str(v) for v in z["var_names"]]
    idx = np.asarray(z["target_indices"])
    normal = np.asarray(z["normal"], dtype=float)
    attacks = {k[len("attack__"):]: np.asarray(z[k], dtype=float)
               for k in z.files if k.startswith("attack__")}

    def label(j):
        return names[idx[j]] if j < len(idx) and idx[j] < len(names) else "Var_%d" % j

    den = np.sqrt(np.mean(normal ** 2, axis=0))

    print("Attribution ranking under two metrics, PCMCI robotics, linear/ridge\n")
    print("  %d modelled variables, fault-free residual matrix %s\n"
          % (normal.shape[1], normal.shape))

    rows, disagreements = [], 0
    for fault in sorted(attacks):
        a = attacks[fault]
        raw = np.linalg.norm(a, axis=0)
        rms = np.sqrt(np.mean(a ** 2, axis=0)) / np.maximum(den, EPS)

        r_ord = np.argsort(raw)[::-1][:TOP_N]
        n_ord = np.argsort(rms)[::-1][:TOP_N]
        r_top = [label(j) for j in r_ord]
        n_top = [label(j) for j in n_ord]
        same2 = set(r_top[:2]) == set(n_top[:2])
        if not same2:
            disagreements += 1

        print("  %s" % fault)
        print("    raw L2         : %s" % ", ".join(r_top))
        print("    normalised RMS : %s" % ", ".join(n_top))
        print("    top two agree  : %s" % ("yes" if same2 else "NO"))
        print()
        rows.append({"Fault": fault,
                     "Raw L2 top 1": r_top[0], "Raw L2 top 2": r_top[1],
                     "Normalised RMS top 1": n_top[0], "Normalised RMS top 2": n_top[1],
                     "Top two agree": "yes" if same2 else "no",
                     "Top three overlap": len(set(r_top[:3]) & set(n_top[:3]))})

    # The fault-free baseline scale of every variable, which is the denominator the
    # normalised rule divides by. It is written out because the argument for retaining the
    # raw norm on robotics turns on how far these spread: the ear LEDs sit near the bottom
    # and the joint temperatures near the top, so dividing hands the quiet channels an
    # advantage large enough to displace the correct attribution.
    b_order = np.argsort(den)
    with io.open("tools/baseline_scales.csv", "w", encoding="utf-8", newline="") as f:
        bw = csv.DictWriter(f, fieldnames=["Variable", "Baseline rank", "Variables ranked",
                                           "Baseline RMS"])
        bw.writeheader()
        for r_, j in enumerate(b_order, 1):
            bw.writerow({"Variable": label(j), "Baseline rank": r_,
                         "Variables ranked": len(den), "Baseline RMS": "%.4f" % den[j]})
    print("  wrote tools/baseline_scales.csv (%d variables)\n" % len(den))

    out = "tools/attribution_metric_check.csv"
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  wrote %s\n" % out)

    led = [r for r in rows if "Leds" in r["Fault"]]
    if led:
        r = led[0]
        both_tablet = (r["Raw L2 top 1"].startswith("Tablet")
                       and r["Normalised RMS top 1"].startswith("Tablet"))
        print("  The LED attack, which is the claim the thesis makes:")
        print("    raw L2 top 1         %s" % r["Raw L2 top 1"])
        print("    normalised RMS top 1 %s" % r["Normalised RMS top 1"])
        print("    both a Tablet sensor: %s" % ("YES" if both_tablet else "NO"))
        if not both_tablet:
            print()
            print("    The claim in the thesis does not hold as written. Either restrict it")
            print("    to the raw L2 rule the robotics folders actually use, or drop it.")
    if disagreements:
        print("\n  %d of %d faults disagree on the top two." % (disagreements, len(rows)))
    else:
        print("\n  All %d faults agree on the top two under both metrics." % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
