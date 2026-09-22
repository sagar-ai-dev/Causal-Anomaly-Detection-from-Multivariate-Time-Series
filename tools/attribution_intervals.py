"""
Uncertainty on the attribution results, and a test that has a level.

Chapter 2 names "treating a numerical difference as a demonstrated one" as the
fourth failure of this field. Section 6.4 reported attribution relevance and
consistency as bare point estimates while the thesis relocated its contribution
from detection to attribution. This tool closes that gap on the data that already
existed: no pipeline is rerun and no new measurement is taken.

Two decisions carry the statistics, and both are conservative.

The resampling unit is the ANOMALY CLASS, not the table cell. Three regression
variants of one algorithm scored on the same fault are three correlated
observations of one event, and resampling them independently would treat that
correlation as evidence. Whole classes are drawn and every cell belonging to a
drawn class is carried with it. The cost is visible and should be: only 14 of the
20 healthcare classes localise to lead territory, the industrial specification
documents an actuator for 5 of its 20 faults, and the robot laboratory defines 3
attacks. Those are the sample sizes the attribution argument rests on.

Comparisons are PAIRED within class, because every method is scored on the same
classes and between-class variance -- some faults are easier to attribute than
others, for every method at once -- is the dominant term and cancels.

THE DECISION RULE IS AN EXACT PAIRED SIGN TEST, NOT THE BOOTSTRAP INTERVAL.
An earlier version read a verdict off whether the percentile interval excluded
zero, and justified it by asserting that an interval excludes zero only when every
class agrees in sign. That containment runs the other way: all-agree implies
excludes-zero, not the converse, so 2^(1-n) is a LOWER BOUND on that rule's
false-positive rate and not its level. The bound is enough to disqualify a panel
(at n=3 it is 0.25 and at n=5 it is 0.0625, both above 0.05) but it certifies
nothing where it is small, so it could not license the healthcare verdicts it was
being used for. The exact sign test has a level at every n, states the same
resolution limit as a true minimum rather than a bound, and handles ties
explicitly. The bootstrap interval is retained and reported, but as a descriptive
range rather than as a test.

Multiplicity is controlled. Section 4.6 fixes the top-three ranking as the depth
at which every attribution result in this thesis is stated, so the primary family
is the three comparisons at that depth within each domain, and Holm's step-down
correction is applied across them. The other depths are reported with exact
p-values and are marked exploratory rather than silently omitted: the depth a
result is read at must not be chosen after seeing it.

Degeneracy is flagged. Where a method's selection depth reaches or exceeds the
pool it ranks, it returns everything it has and is no longer ranking at all: its
relevance is forced to equal its chance rate. GES ranks nine of the twelve
healthcare leads, so at top-10 it returns all nine and scores exactly chance by
construction. A comparison against it at that depth measures the pool and not the
method, and is marked so.

Writes tools/attribution_intervals.csv. Seeded, so the figures are reproducible.

Usage:  python tools/attribution_intervals.py
"""
import collections
import csv
import io
import math
import random
import statistics

RESAMPLES = 20000
SEED = 20240917
ALGOS = ["PCMCI", "GES", "CAM-UV"]
PRIMARY_RULE = "top3"          # the depth Section 4.6 pre-specifies
ALPHA = 0.05


def percentile_ci(draws, alpha=0.05):
    d = sorted(draws)
    lo = d[int((alpha / 2) * len(d))]
    hi = d[int((1 - alpha / 2) * len(d)) - 1]
    return lo, hi


def sign_test(n_pos, n_neg):
    """Exact two-sided paired sign test.

    Ties are discarded, which is the standard treatment and the conservative one:
    a tied class carries no evidence either way and inflating n with ties would
    understate the p-value. The smallest value this can return on n effective
    observations is 2^(1-n), which is the resolution of the design -- 0.25 on
    three classes, 0.0625 on five, 1.2e-4 on fourteen. Unlike the bootstrap rule
    it replaced, that is the exact minimum and not a lower bound on an unknown
    level, so a panel whose minimum exceeds 0.05 provably cannot return a result
    at the 5% level however large the observed difference is.
    """
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = max(n_pos, n_neg)
    tail = sum(math.comb(n, i) for i in range(k, n + 1))
    return min(1.0, 2.0 * tail / 2.0 ** n)


def min_attainable_p(n_eff):
    """The smallest two-sided p the sign test can return on n effective classes."""
    return 1.0 if n_eff == 0 else min(1.0, 2.0 ** (1 - n_eff))


def holm(pvals):
    """Holm step-down adjusted p-values, returned in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def cluster_ci(by_class, rng):
    """Interval on a mean, resampling whole classes and keeping their cells together."""
    keys = sorted(by_class)
    n = len(keys)
    flat = [v for k in keys for v in by_class[k]]
    draws = []
    for _ in range(RESAMPLES):
        picked = [by_class[k] for k in rng.choices(keys, k=n)]
        draws.append(statistics.mean([v for cell in picked for v in cell]))
    return statistics.mean(flat), percentile_ci(draws), n


def paired(a_by_class, b_by_class, rng):
    """Paired difference over the classes both methods were scored on."""
    keys = sorted(set(a_by_class) & set(b_by_class))
    n = len(keys)
    diff = {k: statistics.mean(a_by_class[k]) - statistics.mean(b_by_class[k]) for k in keys}
    draws = [statistics.mean([diff[k] for k in rng.choices(keys, k=n)])
             for _ in range(RESAMPLES)]
    obs = statistics.mean(diff[k] for k in keys)
    lo, hi = percentile_ci(draws)
    eps = 1e-9
    n_pos = sum(1 for k in keys if diff[k] > eps)
    n_neg = sum(1 for k in keys if diff[k] < -eps)
    n_tie = n - n_pos - n_neg
    return obs, lo, hi, n, n_pos, n_neg, n_tie


def main():
    rng = random.Random(SEED)
    out = [["kind", "domain", "rule", "method", "against", "n_classes",
            "value", "ci_low", "ci_high", "separates", "n_pos", "n_neg", "n_tie",
            "p_exact", "p_holm", "min_p", "family", "degenerate"]]

    # ----------------------------------------------------------- relevance
    rel = collections.defaultdict(lambda: collections.defaultdict(list))
    # how many variables each method actually selects, against the pool it ranks:
    # equal means it returned everything and is no longer ranking.
    sel = {}
    for r in csv.DictReader(io.open("tools/relevance_by_class.csv", encoding="utf-8")):
        key = (r["domain"], r["rule"], r["algorithm"])
        rel[key][r["fault"]].append(float(r["relevance_pct"]))
        sel.setdefault(key, (float(r["n_selected"]), float(r["pool"])))

    def degenerate(key):
        n_selected, pool = sel.get(key, (0.0, 1e9))
        return n_selected >= pool

    for (dom, rule, method) in sorted(rel):
        mean, (lo, hi), n = cluster_ci(rel[(dom, rule, method)], rng)
        out.append(["relevance", dom, rule, method, "", n,
                    "%.1f" % mean, "%.1f" % lo, "%.1f" % hi, "", "", "", "",
                    "", "", "%.5f" % min_attainable_p(n), "",
                    "yes" if degenerate((dom, rule, method)) else "no"])

    # SHAP is the reference because it is the established post-hoc method and the
    # question Section 6.4 asks is whether structural attribution matches it.
    paired_rows = []
    for (dom, rule, method) in sorted(rel):
        if method != "SHAP":
            continue
        for algo in ALGOS:
            key = (dom, rule, algo)
            if key not in rel:
                continue
            obs, lo, hi, n, npos, nneg, ntie = paired(
                rel[(dom, rule, "SHAP")], rel[key], rng)
            paired_rows.append({
                "dom": dom, "rule": rule, "against": algo, "n": n,
                "obs": obs, "lo": lo, "hi": hi,
                "npos": npos, "nneg": nneg, "ntie": ntie,
                "p": sign_test(npos, nneg),
                "min_p": min_attainable_p(npos + nneg),
                "degen": degenerate(key) or degenerate((dom, rule, "SHAP")),
            })

    # Holm within each domain over the primary family only. The other depths are
    # exploratory and are reported uncorrected and labelled as such, which is the
    # honest treatment: correcting them would imply they were planned, and hiding
    # them would be the selection the thesis criticises.
    for dom in {r["dom"] for r in paired_rows}:
        fam = [r for r in paired_rows
               if r["dom"] == dom and r["rule"] == PRIMARY_RULE]
        for r, a in zip(fam, holm([r["p"] for r in fam])):
            r["p_holm"] = a

    for r in paired_rows:
        primary = r["rule"] == PRIMARY_RULE
        # A verdict is reported only where the design can reach 0.05 at all.
        if r["min_p"] > ALPHA:
            verdict = "n/a"
        elif primary:
            verdict = "yes" if r["p_holm"] <= ALPHA else "no"
        else:
            verdict = "yes" if r["p"] <= ALPHA else "no"
        out.append(["relevance_paired", r["dom"], r["rule"], "SHAP", r["against"],
                    r["n"], "%.1f" % r["obs"], "%.1f" % r["lo"], "%.1f" % r["hi"],
                    verdict, r["npos"], r["nneg"], r["ntie"],
                    "%.4f" % r["p"],
                    ("%.4f" % r["p_holm"]) if primary else "",
                    "%.5f" % r["min_p"],
                    "primary" if primary else "exploratory",
                    "yes" if r["degen"] else "no"])

    # --------------------------------------------------------- consistency
    con = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(io.open("tools/consistency_by_class.csv", encoding="utf-8")):
        con[(r["domain"], "top" + r["top_k"], r["family"])][r["fault"]].append(
            float(r["mean_jaccard"]))

    for (dom, rule, fam) in sorted(con):
        mean, (lo, hi), n = cluster_ci(con[(dom, rule, fam)], rng)
        out.append(["consistency", dom, rule, fam, "", n,
                    "%.3f" % mean, "%.3f" % lo, "%.3f" % hi, "", "", "", "",
                    "", "", "%.5f" % min_attainable_p(n), "", "no"])

    for (dom, rule, fam) in sorted(con):
        if fam != "neural":
            continue
        key = (dom, rule, "causal")
        if key not in con:
            continue
        obs, lo, hi, n, npos, nneg, ntie = paired(
            con[(dom, rule, "neural")], con[key], rng)
        p = sign_test(npos, nneg)
        mp = min_attainable_p(npos + nneg)
        verdict = "n/a" if mp > ALPHA else ("yes" if p <= ALPHA else "no")
        out.append(["consistency_paired", dom, rule, "neural", "causal", n,
                    "%.3f" % obs, "%.3f" % lo, "%.3f" % hi, verdict,
                    npos, nneg, ntie, "%.4f" % p, "", "%.5f" % mp, "", "no"])

    with io.open("tools/attribution_intervals.csv", "w", encoding="utf-8",
                 newline="") as f:
        csv.writer(f).writerows(out)
    print("  wrote tools/attribution_intervals.csv  (%d rows)" % (len(out) - 1))

    sep = [r for r in out[1:] if r[9] == "yes"]
    na = [r for r in out[1:] if r[9] == "n/a"]
    print("  paired comparisons: %d separate, %d carry no verdict (design cannot "
          "reach %.2f)" % (len(sep), len(na), ALPHA))
    for r in sep:
        print("     %-11s %-6s %-9s %s vs %-7s %+7s  p=%s  %s%s"
              % (r[1], r[2], r[16], r[3], r[4], r[6], r[13],
                 "Holm p=" + r[14] + "  " if r[14] else "",
                 "DEGENERATE" if r[17] == "yes" else ""))


if __name__ == "__main__":
    main()
