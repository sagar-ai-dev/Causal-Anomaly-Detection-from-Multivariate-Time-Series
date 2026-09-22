"""
Graph density restricted to the lags each algorithm's edges actually occupy.

Section 6.6 argues from healthcare density: GES is sparsest and scores best, PCMCI
is densest and scores worst. The obvious objection is that density counts a pair as
linked whenever any lag carries an edge, and PCMCI is allocated more lag slots than
the other two, so it has more chances to link the same pair.

The objection is real but the arithmetic in it was wrong. GES and CAM-UV are run on
the unlagged observation matrix and the pipeline assigns their edges a single lag
(Section 4.3), so although their cached tensors allocate two slots, only one is ever
populated. PCMCI populates every slot it is given. The matched control is therefore
to restrict PCMCI to one lag, not two.

This measures both: how many slots each graph allocates, which of them actually carry
a surviving edge, and the density over the first lag alone against the density over
all of them. The sparsification rule is imported from graph_density.py rather than
restated, so the two tools cannot drift apart.

Run from the repository root:  python tools/lag_depth_density.py
"""

import csv
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import graph_density as gd

FOLDERS = ("PCMCI_healthcare", "GES_healthcare", "CAM_UV_healthcare",
           "PCMCI_robotic", "GES_robotic", "CAM_UV_robotics",
           "PCMCI_industrial", "GES_industrial", "CAM_UV_industrial")

OUT = os.path.join("tools", "lag_depth_density.csv")


def linked_pairs(keep, depth):
    """Unordered pairs carrying a surviving edge within the first `depth` lags."""
    d = keep.shape[0]
    k = keep[:, :, :depth]
    n = 0
    for i in range(d):
        for j in range(i + 1, d):
            if k[i, j, :].any() or k[j, i, :].any():
                n += 1
    return n


def main():
    rows = []
    print("Density against the lags each graph actually uses")
    print()
    print("  %-20s %6s %10s %8s %10s %10s"
          % ("folder", "slots", "occupied", "pairs", "all lags", "tau=0"))
    print("  " + "-" * 70)

    for folder in FOLDERS:
        path, z = gd.cached_graph(folder)
        if z is None:
            print("  %-20s no cached graph" % folder)
            continue
        mult = gd.multiplier_for(folder)
        if mult is None:
            print("  %-20s no multiplier found" % folder)
            continue

        p_m, v_m = z["p_matrix"], z["val_matrix"]
        keep = (p_m < gd.ALPHA) & (np.abs(v_m) > mult * np.mean(np.abs(v_m)))
        slots = p_m.shape[2]
        occupied = [t for t in range(slots) if keep[:, :, t].any()]
        d = p_m.shape[0]
        pairs = d * (d - 1) // 2
        all_lags = linked_pairs(keep, slots)
        lag0 = linked_pairs(keep, 1)

        print("  %-20s %6d %10s %8d %9.1f%% %9.1f%%"
              % (folder, slots, ",".join(str(t) for t in occupied) or "none",
                 pairs, 100.0 * all_lags / pairs, 100.0 * lag0 / pairs))

        rows.append({"Folder": folder, "Variables": d,
                     "Lag slots allocated": slots,
                     "Lag slots occupied": len(occupied),
                     "Occupied lags": " ".join(str(t) for t in occupied),
                     "Pairs": pairs,
                     "Linked over all lags": all_lags,
                     "Density over all lags": "%.4f" % (all_lags / pairs),
                     "Linked at lag zero": lag0,
                     "Density at lag zero": "%.4f" % (lag0 / pairs)})

    if not rows:
        raise SystemExit("no cached graphs found -- run this from the repository root")

    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("  GES and CAM-UV occupy one lag apiece, so restricting PCMCI to a single")
    print("  lag is the matched control rather than restricting it to two.")
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
