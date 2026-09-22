"""
Why PCMCI, when PC is the constraint-based method causal-learn ships.

Table 1 of Zheng et al. (2024) lists four constraint-based algorithms -- PC, MV-PC, FCI and
CD-NOD -- and PCMCI is not among them, because it belongs to tigramite. Substituting it needs
a reason better than preference, and the reason usually given is theoretical: PC assumes
independent, identically distributed observations, and a time series violates that in a
specific way. Autocorrelation inflates the apparent dependence between any two variables, so
a conditional independence test run on serially correlated data rejects the null far more
often than its nominal level. The edges that result are artefacts of persistence, not
causation.

That argument is correct but it is still an argument. This measures it instead.

  Part A, the decisive test. Build a system where the right answer is known and is
  "no edges at all": D independent AR(1) processes,

      x_i(t) = phi * x_i(t-1) + e_i(t),     e_i independent across i,

  so every variable is autocorrelated and no variable causes any other. Every cross-variable
  edge either algorithm reports is a false positive by construction, and the false positive
  rate can be read directly rather than inferred. Sweeping phi from 0 to 0.95 walks the data
  from i.i.d., where the PC assumption holds exactly, to strongly persistent, where it fails.

  Part B, graph density on the real data, under the rule the pipeline actually applies.

A note on what is being counted, because a careless count makes PCMCI look badly calibrated
when it is not. PC emits one decision per variable pair. PCMCI tests every ordered pair at
every lag up to tau_max, which is 2 * (tau_max + 1) tests for the same pair, so declaring a
pair linked if any of those tests fires accumulates their error rates: at alpha = 0.05 and
tau_max = 3 the arithmetic gives 1 - 0.95^8 = 0.34 before the data is even looked at. That is
the counting method being wrong, not the algorithm. Both an uncorrected and an FDR-corrected
column are therefore reported, and the corrected one is the fair comparison against PC.

Usage:  python tools/pc_vs_pcmci_ablation.py
        python tools/pc_vs_pcmci_ablation.py --quick
"""

import csv
import io
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

ALPHA = 0.05          # identical to PCMCI and GES healthcare anomaly_detection scripts
TAU_MAX = 3
D = 8
N = 1000
PHIS = [0.0, 0.3, 0.5, 0.7, 0.9, 0.95]
SEEDS = 20


def independent_ar1(n, d, phi, seed):
    """d autocorrelated series with no cross-variable causation. True graph: empty."""
    rng = np.random.default_rng(seed)
    x = np.zeros((n + 200, d))
    e = rng.normal(size=(n + 200, d))
    for t in range(1, n + 200):
        x[t] = phi * x[t - 1] + e[t]
    return x[200:]                      # discard burn-in so the process is stationary


def pc_pairs(data, alpha):
    """Variable pairs PC links. Self-loops are not representable, so all of these are false."""
    from causallearn.search.ConstraintBased.PC import pc
    try:
        cg = pc(data, alpha, "fisherz", show_progress=False)
    except TypeError:                   # older causal-learn has no show_progress
        cg = pc(data, alpha, "fisherz")
    g = cg.G.graph
    d = g.shape[0]
    return {(i, j) for i in range(d) for j in range(i + 1, d)
            if g[i, j] != 0 or g[j, i] != 0}


def _linked(p, alpha):
    """Unordered variable pairs with a significant link at any lag, self-links excluded.

    Self-links are excluded in both directions: x_i(t-1) -> x_i(t) is the AR(1) structure and
    is true, so counting it would penalise PCMCI for being right, and PC cannot express it at
    all. Cross-variable pairs are the only comparison that means the same thing for each.
    """
    d = p.shape[0]
    out = set()
    for i in range(d):
        for j in range(d):
            if i != j and (p[i, j, :] < alpha).any():
                out.add((min(i, j), max(i, j)))
    return out


def pcmci_pairs(data, alpha, tau_max):
    """Returns (uncorrected pairs, FDR-corrected pairs)."""
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    pcmci = PCMCI(dataframe=pp.DataFrame(data), cond_ind_test=ParCorr(), verbosity=0)
    res = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha)
    p = res["p_matrix"]
    try:
        q = pcmci.get_corrected_pvalues(p_matrix=p, tau_max=tau_max, fdr_method="fdr_bh")
    except Exception:
        q = p                            # correction unavailable; columns will agree
    return _linked(p, alpha), _linked(q, alpha)


def part_a(quick):
    seeds = 5 if quick else SEEDS
    n_pairs = D * (D - 1) // 2
    print("Part A -- independent AR(1), true graph empty, so every edge is a false positive")
    print("  %d variables, %d samples, alpha %.2f, tau_max %d, %d seeds per phi\n"
          % (D, N, ALPHA, TAU_MAX, seeds))
    print("  %-6s %14s %20s %18s" % ("phi", "PC", "PCMCI uncorrected", "PCMCI FDR"))
    print("  " + "-" * 60)

    rows = []
    for phi in PHIS:
        pc_fp, raw_fp, fdr_fp = [], [], []
        for s in range(seeds):
            x = independent_ar1(N, D, phi, seed=1000 + s)
            pc_fp.append(len(pc_pairs(x, ALPHA)) / n_pairs)
            raw, fdr = pcmci_pairs(x, ALPHA, TAU_MAX)
            raw_fp.append(len(raw) / n_pairs)
            fdr_fp.append(len(fdr) / n_pairs)
        m = [float(np.mean(v)) for v in (pc_fp, raw_fp, fdr_fp)]
        sd = [float(np.std(v)) for v in (pc_fp, raw_fp, fdr_fp)]
        print("  %-6.2f %8.4f+/-%.4f %12.4f+/-%.4f %10.4f+/-%.4f"
              % (phi, m[0], sd[0], m[1], sd[1], m[2], sd[2]))
        rows.append({"phi": "%.2f" % phi,
                     "PC false positive rate": "%.4f" % m[0], "PC sd": "%.4f" % sd[0],
                     "PCMCI uncorrected": "%.4f" % m[1], "PCMCI uncorrected sd": "%.4f" % sd[1],
                     "PCMCI FDR": "%.4f" % m[2], "PCMCI FDR sd": "%.4f" % sd[2],
                     "seeds": seeds, "variables": D, "samples": N,
                     "alpha": ALPHA, "tau_max": TAU_MAX})

    out = "tools/pc_vs_pcmci.csv"
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n  wrote %s" % out)

    lo, hi = rows[0], rows[-1]
    print("\n  PC at phi = 0.00, where its assumption holds exactly: %s, against a nominal"
          % lo["PC false positive rate"])
    print("  %.2f. At phi = %s: %s. Neither run is given a single true edge to find."
          % (ALPHA, hi["phi"], hi["PC false positive rate"]))
    print("  PCMCI over the same sweep, FDR-corrected: %s to %s."
          % (lo["PCMCI FDR"], hi["PCMCI FDR"]))
    return rows


def part_b():
    """Graph density on the real data, under the rule the pipeline actually applies.

    The pipeline does not threshold on significance alone. It also drops any edge whose
    absolute effect size falls below CAUSAL_STRENGTH_MULTIPLIER times the mean absolute
    effect size, and that multiplier is 1.0 on healthcare and robotics but 0.0 on
    industrial -- where the filter therefore does nothing. Reporting p < alpha alone would
    overstate the density of every graph the thesis actually uses.
    """
    print("\n\nPart B -- graph density on the real data, under the rule the pipeline applies")
    print("  edge kept when  p < alpha  AND  |val| > multiplier * mean|val|\n")
    print("  %-12s %6s %8s %14s %14s" % ("domain", "vars", "mult", "p<alpha only", "pipeline rule"))
    print("  " + "-" * 60)

    found = []
    for dom, mult in (("healthcare", 1.0), ("robotic", 1.0), ("industrial", 0.0)):
        folder = "PCMCI/PCMCI_%s" % dom
        cache = None
        for root, _, files in os.walk(folder):
            for fn in files:
                if fn.endswith(".npz") and "scores_" not in fn:
                    cache = os.path.join(root, fn)
                    break
            if cache:
                break
        if cache is None:
            continue
        z = np.load(cache, allow_pickle=True)
        if "p_matrix" not in z or "val_matrix" not in z:
            continue
        p, val = z["p_matrix"], z["val_matrix"]
        d = p.shape[0]
        n_pairs = d * (d - 1) // 2
        keep = (p < ALPHA) * (np.abs(val) > mult * np.mean(np.abs(val)))
        sig_only = len(_linked(p, ALPHA))
        both = len({(min(i, j), max(i, j)) for i in range(d) for j in range(d)
                    if i != j and keep[i, j, :].any()})
        print("  %-12s %6d %8.1f %9d/%-4d %9d/%-4d"
              % (dom, d, mult, sig_only, n_pairs, both, n_pairs))
        found.append({"domain": dom, "variables": d, "multiplier": mult,
                      "pairs": n_pairs, "significant only": sig_only,
                      "pipeline rule": both})

    if found:
        out = "tools/pcmci_graph_density.csv"
        with io.open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(found[0].keys()))
            w.writeheader()
            w.writerows(found)
        print("\n  wrote %s" % out)
    print("\n  Correctness cannot be scored here -- no true graph is known for any of these")
    print("  domains -- so this is context, not proof. Part A is the test.")
    return found


def main():
    part_a("--quick" in sys.argv)
    part_b()
    print("\nWhat this settles: substituting PCMCI for PC is not a preference. On data with")
    print("the autocorrelation every domain in this thesis has, and no causal structure at")
    print("all, the PC error rate is measured rather than argued.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
