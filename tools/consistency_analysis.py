"""
How much do different methods agree on which variables explain an anomaly?

The supervisor's second question. Relevance asks whether a method points at the
right variables; consistency asks whether the methods point at the same ones.
Both matter for his claim, and they are independent: nine configurations could
agree perfectly on a wrong answer, or each be partly right about different
variables.

Agreement is the mean pairwise Jaccard index over the selected sets. For the
causal pipeline that is the 9 configurations in a domain, 3 algorithms crossed
with 3 regression variants. Chance agreement is reported beside it, because
Jaccard rises on its own as the selected set grows relative to the variable
count: two random sets of 5 drawn from 12 leads overlap far more than two drawn
from 244 robot sensors, and without the baseline the ECG would look like the
most consistent domain purely by arithmetic.
"""
import csv, io, os, glob, statistics, itertools

SHAP = "Baselines/outputs/shap_full.csv"


def load(path):
    by = {}
    for r in csv.DictReader(io.open(path, encoding="utf-8")):
        by.setdefault((r["Model Variant"], r["Fault Name"]), []).append(
            (int(r["Rank"]), r["Variable"]))
    return {k: [v for _, v in sorted(vs)] for k, vs in by.items()}


def load_shap(path):
    """Same as load, but keyed by dataset as well, since the file holds three."""
    by = {}
    for r in csv.DictReader(io.open(path, encoding="utf-8")):
        by.setdefault((r.get("Dataset", ""), r["Model Variant"], r["Fault Name"]),
                      []).append((int(r["Rank"]), r["Variable"]))
    return {k: [v for _, v in sorted(vs)] for k, vs in by.items()}


def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A | B) else 0.0


def main():
    sets = {}
    nvars = {}
    for path in sorted(glob.glob("*/*/outputs/attribution_full.csv")):
        parts = path.replace(chr(92), "/").split("/")
        if parts[0] == "Baselines":
            continue
        algo, folder = parts[0], parts[1].lower()
        dom = ("robotics" if "robotic" in folder else
               "healthcare" if "healthcare" in folder else "industrial")
        for (variant, fault), ranked in load(path).items():
            f = fault.replace(".csv", "")
            nvars.setdefault((dom, f, "causal"), []).append(len(ranked))
            for k in (3, 5, 10):
                sets.setdefault((dom, f, k, "causal"), {})[(algo, variant)] = ranked[:k]

    # The neural baselines get their own bucket rather than joining the causal
    # one: mixing them would measure whether the two families agree with each
    # other, when the question is how well each family agrees with itself.
    # SHAP now covers all three datasets, and each row carries the dataset it
    # was computed on.
    DOM = {"Robotics (Pepper)": "robotics",
           "Healthcare (PTB-XL)": "healthcare",
           "Industrial (TEP)": "industrial"}
    if os.path.exists(SHAP):
        for (dom_label, variant, fault), ranked in load_shap(SHAP).items():
            dom = DOM.get(dom_label)
            if dom is None:
                continue
            f = fault.replace(".csv", "")
            nvars.setdefault((dom, f, "neural"), []).append(len(ranked))
            for k in (3, 5, 10):
                sets.setdefault((dom, f, k, "neural"), {})[("SHAP", variant)] = ranked[:k]

    out = [["domain", "family", "fault", "top_k", "n_configs", "mean_jaccard",
            "chance_jaccard", "lift", "unanimous_vars"]]
    for (dom, fault, k, family), cfgs in sorted(sets.items()):
        keys = sorted(cfgs)
        if len(keys) < 2:
            continue
        pairs = [jaccard(cfgs[a], cfgs[b]) for a, b in itertools.combinations(keys, 2)]
        m = statistics.mean(pairs)
        # expected Jaccard for two random k-subsets of n variables
        n = statistics.mean(nvars[(dom, fault, family)])
        exp_int = k * k / n if n else 0
        chance = exp_int / (2 * k - exp_int) if (2 * k - exp_int) > 0 else 0
        inter = set(cfgs[keys[0]])
        for key in keys[1:]:
            inter &= set(cfgs[key])
        out.append([dom, family, fault, k, len(keys), round(m, 4), round(chance, 4),
                    round(m / chance, 2) if chance else "", len(inter)])

    with io.open("tools/consistency_by_class.csv", "w", encoding="utf-8",
                 newline="") as f:
        csv.writer(f).writerows(out)
    print("  wrote tools/consistency_by_class.csv  (%d rows)" % (len(out) - 1))
    specificity()


def specificity():
    """How much of a family's agreement is about the class in front of it.

    Agreement on its own is cheap: a method that returns nearly the same answer
    whatever it is asked scores highly on it. The informative quantity is the
    margin between agreement within a class, measured across configurations, and
    agreement between classes, measured within one configuration. A family whose
    two numbers are close is not explaining the class, it is reporting a fixed
    ranking of the variables.
    """
    out = [["domain", "family", "top_k", "within_fault_jaccard",
            "between_fault_jaccard", "excess"]]
    srcs = []
    for path in sorted(glob.glob("*/*/outputs/attribution_full.csv")):
        parts = path.replace(chr(92), "/").split("/")
        if parts[0] == "Baselines":
            continue
        fold = parts[1].lower()
        dom = ("robotics" if "robotic" in fold else
               "healthcare" if "healthcare" in fold else "industrial")
        srcs.append((dom, parts[0], load(path)))
    if os.path.exists(SHAP):
        lab = {"robotics": "Robotics (Pepper)", "healthcare": "Healthcare (PTB-XL)",
               "industrial": "Industrial (TEP)"}
        for dom, dlabel in lab.items():
            d = {(v, f): r for (ds, v, f), r in load_shap(SHAP).items() if ds == dlabel}
            if d:
                srcs.append((dom, "SHAP", d))

    for dom, fam, d in srcs:
        byf = {}
        for (variant, fault), ranked in d.items():
            byf.setdefault(fault.replace(".csv", ""), {})[variant] = ranked
        for k in (3, 5, 10):
            within = [statistics.mean(jaccard(g[a][:k], g[b][:k])
                                      for a, b in itertools.combinations(sorted(g), 2))
                      for g in byf.values() if len(g) >= 2]
            variants = sorted({v for g in byf.values() for v in g})
            between = []
            for v in variants:
                sets = [g[v][:k] for g in byf.values() if v in g]
                if len(sets) >= 2:
                    between.append(statistics.mean(jaccard(a, b)
                                   for a, b in itertools.combinations(sets, 2)))
            if not within or not between:
                continue
            w, b = statistics.mean(within), statistics.mean(between)
            out.append([dom, fam, k, round(w, 4), round(b, 4), round(w - b, 4)])

    with io.open("tools/specificity_by_method.csv", "w", encoding="utf-8",
                 newline="") as f:
        csv.writer(f).writerows(out)
    print("  wrote tools/specificity_by_method.csv  (%d rows)" % (len(out) - 1))


if __name__ == "__main__":
    main()
