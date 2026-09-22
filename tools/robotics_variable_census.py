"""
Reconcile the three sensor counts the robotics domain is described by.

The thesis quotes 244 variables in one place and 249 in another, and both are correct, which
is exactly why the difference has to be written down somewhere a reader can check. They
count different things:

  255   columns in the fault-free recording once the timestamp is dropped
  249   of those vary at all at the full sampling rate -- what tools/deviation_ranks.py
        ranks, because it works on the raw series with no model
  244   still vary after subsampling to the discovery resolution of every 74th sample --
        what the causal graphs are learned over, and what `nonconst` in the cache holds

So six sensors are flat throughout the recording, and five more become flat once the series
is thinned. Neither number is wrong; describing 244 as the raw variable count is.

The script also reports a property of the source data that shows up in the attribution
figures and would otherwise puzzle a reader: eighteen sensor names occur twice, carrying
different readings, so pandas disambiguates the second occurrence with a `.1` suffix. That
is why a bar in the robotics attribution plots is labelled `LaserF01X.1`. They are distinct
physical channels that happen to share a name in the laboratory's export, not a duplicated
column.

Usage:  python tools/robotics_variable_census.py
"""

import csv
import io
import os
import sys

import numpy as np
import pandas as pd

PREFIX = "PCMCI/PCMCI_robotic/pepper_csv"
NORMAL = "normal.csv"
CACHE = "pepper_normal.npz"


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.chdir(root)

    npath = os.path.join(PREFIX, NORMAL)
    cpath = os.path.join(PREFIX, CACHE)
    if not os.path.exists(npath):
        raise SystemExit("raw Pepper data not present; this needs " + npath)

    df = pd.read_csv(npath)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    total = df.shape[1]
    sd_full = df.std().fillna(0)
    nonconst_full = int((sd_full != 0).sum())

    subsample = None
    nonconst_cache = None
    if os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        subsample = int(z["subsample"])
        nonconst_cache = int(len(z["nonconst"]))

    rate = subsample if subsample else 74
    sd_sub = df.iloc[::rate].std().fillna(0)
    nonconst_sub = int((sd_sub != 0).sum())

    dup_names = sorted({c[:-2] for c in df.columns if c.endswith(".1") and c[:-2] in df.columns})
    dup_differ = 0
    for base in dup_names:
        a = np.nan_to_num(df[base].values, nan=0.0)
        b = np.nan_to_num(df[base + ".1"].values, nan=0.0)
        if not np.allclose(a, b):
            dup_differ += 1

    print("Robotics sensor census\n")
    print("  %-46s %d" % ("columns after dropping timestamp", total))
    print("  %-46s %d" % ("non-constant at the full sampling rate", nonconst_full))
    print("  %-46s %d" % ("constant throughout the recording", total - nonconst_full))
    print("  %-46s %d" % ("non-constant after subsampling to 1 in %d" % rate, nonconst_sub))
    print("  %-46s %d" % ("rendered constant by the subsampling", nonconst_full - nonconst_sub))
    if nonconst_cache is not None:
        agree = "agrees" if nonconst_cache == nonconst_sub else "DISAGREES"
        print("  %-46s %d  (%s with the cache)" % ("variables in the cached graph",
                                                   nonconst_cache, agree))
    print("\n  %-46s %d" % ("sensor names occurring twice", len(dup_names)))
    print("  %-46s %d" % ("of those whose two channels differ", dup_differ))
    if dup_names:
        print("\n  Duplicated names are distinct channels sharing a label in the export, not")
        print("  copies. pandas suffixes the second `.1`, which is why the attribution")
        print("  figures carry a bar labelled %s.1" % dup_names[0])

    rows = [{"Quantity": k, "Count": v} for k, v in [
        ("Columns after dropping timestamp", total),
        ("Non-constant at full rate", nonconst_full),
        ("Constant throughout recording", total - nonconst_full),
        ("Subsample rate", rate),
        ("Non-constant after subsampling", nonconst_sub),
        ("Rendered constant by subsampling", nonconst_full - nonconst_sub),
        ("Duplicated sensor names", len(dup_names)),
        ("Duplicated names with differing data", dup_differ),
    ]]
    out = "tools/robotics_variable_census.csv"
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Quantity", "Count"])
        w.writeheader()
        w.writerows(rows)
    print("\n  wrote %s" % out)

    if nonconst_cache is not None and nonconst_cache != nonconst_sub:
        print("\n  The cache does not match a fresh subsample; the census cannot be trusted")
        print("  until that is explained.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
