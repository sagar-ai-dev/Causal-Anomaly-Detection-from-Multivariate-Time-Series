"""
Check that the attribution claims are true, not merely that their numbers exist.

verify_docs.py and verify_integers.py confirm that every figure quoted in the
thesis appears in some CSV the pipelines wrote. That catches a stale number and
misses a false sentence: "post-hoc attribution is ahead in three of the four"
passes both gates whatever the four comparisons actually say, because 3 and 4
are in the pool. Several defects found late in this work were of exactly that
kind -- a count that no longer matched the table it described, a pair of numbers
attached to the wrong pair of methods, a verdict the design could not support.

Every assertion below is re-derived from tools/relevance_by_class.csv and
tools/attribution_intervals.csv and compared against what the prose says. A
failure here means a sentence needs changing, not a number.

Exit code is non-zero if any claim fails, so this can gate a commit.

Usage:  python tools/verify_claims.py
"""
import sys
# The gates check that a number appears in some CSV. They do not check that the
# sentence around it is true. These are the claims the recent edits rest on,
# each re-derived from the data rather than read back from the prose.
import csv, io, collections, statistics

R = list(csv.DictReader(io.open("tools/relevance_by_class.csv", encoding="utf-8")))
A = list(csv.DictReader(io.open("tools/attribution_intervals.csv", encoding="utf-8")))
ok = True
nchecks = 0


def check(label, got, want):
    global ok, nchecks
    nchecks += 1
    good = got == want
    ok = good and ok
    print("  [%s] %-58s got %-22s want %s"
          % ("PASS" if good else "FAIL", label, got, want))


# ---- pools, the claim rewritten this pass
pool = {}
for r in R:
    if r["rule"] == "top3":
        pool[(r["domain"], r["algorithm"])] = r["pool"]
check("CAM-UV pool on the robot", pool[("robotics", "CAM-UV")], "35")
check("CAM-UV pool on the plant", pool[("industrial", "CAM-UV")], "39")
check("GES pool on the robot", pool[("robotics", "GES")], "32")
check("GES pool on the plant", pool[("industrial", "GES")], "25")
check("GES pool on the electrocardiogram", pool[("healthcare", "GES")], "9")

# ---- the four like-for-like comparisons: post-hoc ahead 3, level 1, causal 0
rel = collections.defaultdict(list)
for r in R:
    if r["rule"] == "top3":
        rel[(r["domain"], r["algorithm"])].append(float(r["relevance_pct"]))
mean = {k: statistics.mean(v) for k, v in rel.items()}
pairs = [("robotics", "PCMCI"), ("healthcare", "PCMCI"), ("industrial", "PCMCI"),
         ("healthcare", "CAM-UV")]
ahead = level = behind = 0
for dom, alg in pairs:
    assert pool[(dom, alg)] == pool[(dom, "SHAP")], (dom, alg)
    d = round(mean[(dom, "SHAP")] - mean[(dom, alg)], 4)
    ahead += d > 0.05
    level += abs(d) <= 0.05
    behind += d < -0.05
check("like-for-like pairs where both rank an equal pool", len(pairs), 4)
check("  of those, post-hoc ahead", ahead, 3)
check("  of those, level", level, 1)
check("  of those, intrinsic ahead", behind, 0)

# ---- the paired grid
paired = [r for r in A if r["kind"] == "relevance_paired"]
top3 = [r for r in paired if r["rule"] == "top3"]
check("rows in the chapter table (top-3 only)", len(top3), 9)
check("  separating", len([r for r in top3 if r["separates"] == "yes"]), 1)
check("  carrying no verdict", len([r for r in top3 if r["separates"] == "n/a"]), 6)
check("  not separating", len([r for r in top3 if r["separates"] == "no"]), 2)
check("rows in the appendix table (all four depths)", len(paired), 36)
check("  separating across all depths",
      len([r for r in paired if r["separates"] == "yes"]), 3)
check("  flagged degenerate", len([r for r in paired if r["degenerate"] == "yes"]), 1)

hp = [r for r in top3 if r["against"] == "PCMCI" and r["domain"] == "healthcare"][0]
check("healthcare SHAP-PCMCI Holm p at top-3", hp["p_holm"], "0.0352")
check("  its sign split", "%s/%s/%s" % (hp["n_pos"], hp["n_neg"], hp["n_tie"]), "10/1/3")

# ---- the degenerate row
deg = [r for r in paired if r["degenerate"] == "yes"][0]
check("degenerate row is healthcare GES at top-10",
      "%s %s %s" % (deg["domain"], deg["against"], deg["rule"]),
      "healthcare GES top10")
g10 = [r for r in R if r["domain"] == "healthcare" and r["algorithm"] == "GES"
       and r["rule"] == "top10"]
check("  GES selects its whole pool there",
      "%g of %s" % (statistics.mean(float(r["n_selected"]) for r in g10), g10[0]["pool"]),
      "9 of 9")
check("  so its lift is 1.00 by construction",
      "%.2f" % (statistics.mean(float(r["relevance_pct"]) for r in g10)
                / statistics.mean(float(r["chance_pct"]) for r in g10)), "1.00")

# ---- consistency, the figures quoted in 6.4.1
cp = {(r["domain"], r["rule"]): r for r in A if r["kind"] == "consistency_paired"}
for rule, want in (("top3", "0.425"), ("top5", "0.402"), ("top10", "0.314")):
    check("industrial neural-causal margin, %s" % rule,
          cp[("industrial", rule)]["value"], want)
    check("  and it separates", cp[("industrial", rule)]["separates"], "yes")
check("healthcare closest consistency margin p", cp[("healthcare", "top10")]["p_exact"],
      "0.1153")
check("  and it does not separate", cp[("healthcare", "top10")]["separates"], "no")
check("robotics consistency carries no verdict",
      {cp[("robotics", k)]["separates"] for k in ("top3", "top5", "top10")}, {"n/a"})

# ---- scorable class counts
cls = {d: len({r["fault"] for r in R if r["domain"] == d and r["rule"] == "top3"})
       for d in ("robotics", "healthcare", "industrial")}
check("scorable classes per domain", cls,
      {"robotics": 3, "healthcare": 14, "industrial": 5})
check("  summing to", sum(cls.values()), 22)

print("\n%s" % ("all %d claims verified" % nchecks if ok
                else "SOME CLAIMS FAILED -- a sentence needs changing, not a number"))
sys.exit(0 if ok else 1)
