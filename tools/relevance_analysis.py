"""
Share of the top-attributed variables that are actually relevant to each anomaly.

The supervisor asked for this per dataset, method and anomaly class, over several
ways of choosing "the top ones". A percentage on its own cannot be read: on the
robot, 71 of 244 variables carry LED or tablet in the name, so a method choosing
at random scores 29 per cent. Every figure here is therefore reported beside the
chance level for that class, and the lift over it.

Ground truth is external to the data in all three domains, which is the point.
Robotics: the subsystem each injected attack targets, as the supervisor defined
it -- wheels by name, LED and tablet, joints by axis and temperature.
Healthcare: the standard lead territory of each diagnosis, recovered by matching
every anomaly block back to its PTB-XL record. Six of the twenty classes are
non-specific ST/T or conduction findings that do not localise to leads at all;
they are excluded and counted, not scored against an invented answer.
Industrial: the faults whose documented cause names a specific actuator.
"""
import csv, io, os, glob, statistics

ROBOT = {
    "WheelsControl": lambda v: "wheel" in v.lower(),
    "LedsControl":   lambda v: v.lower().startswith(("led", "tablet")),
    "JointControl":  lambda v: any(t in v.lower()
                                   for t in ("roll", "pitch", "yaw", "temp")),
}

TERRITORY = {
    "IMI":   ["II", "III", "AVF"],
    "ASMI":  ["V1", "V2", "V3", "V4"],
    "LMI":   ["I", "AVL", "V5", "V6"],
    "LAFB":  ["I", "AVL"],
    "LPFB":  ["II", "III", "AVF"],
    "IRBBB": ["V1", "V2"],
    "RVH":   ["V1", "V2"],
    "RAO/RAE": ["II", "V1"],
    "LVH":   ["I", "AVL", "V5", "V6", "V1"],
    "AFLT":  ["II", "III", "AVF", "V1"],
    "AFIB":  ["II", "III", "AVF", "V1"],
}

TEP = {"Fault_4": "XMV_10", "Fault_5": "XMV_11", "Fault_7": "XMV_4",
       "Fault_11": "XMV_10", "Fault_14": "XMV_10"}


def healthcare_truth():
    out = {}
    p = "tools/healthcare_attack_classes.csv"
    if not os.path.exists(p):
        return out
    for r in csv.DictReader(io.open(p, encoding="utf-8")):
        leads = set()
        for code in r["scp_codes"].split("|"):
            leads |= set(TERRITORY.get(code, []))
        if leads:
            out[r["attack"]] = leads
    return out


# The supervisor asked whether the twenty healthcare classes could be grouped
# into significantly fewer. They can, and it matters: pooling all of them hides
# that the localisation works on infarctions and fails on conduction and rhythm
# statements. The grouping is the PTB-XL diagnostic superclass of the dominant
# finding, with the atrial-flutter records placed in a rhythm group because
# PTB-XL assigns rhythm statements no diagnostic superclass at all.
HC_GROUP = {
    "Attack_1": "MI", "Attack_12": "MI", "Attack_17": "MI", "Attack_20": "MI",
    "Attack_10": "CD", "Attack_13": "CD", "Attack_14": "CD", "Attack_18": "CD",
    "Attack_9": "HYP",
    "Attack_2": "RHYTHM", "Attack_3": "RHYTHM", "Attack_4": "RHYTHM",
    "Attack_6": "RHYTHM", "Attack_11": "RHYTHM",
}


def group_of(domain, fault):
    """The coarser class a fault is reported under, or the fault itself."""
    if domain == "healthcare":
        return HC_GROUP.get(fault.replace(".csv", ""), "")
    return ""


SHAP = "Baselines/outputs/shap_full.csv"

HC = healthcare_truth()


def relevant(domain, fault, var):
    if domain == "robotics":
        f = ROBOT.get(fault.replace(".csv", ""))
        return f(var) if f else None
    if domain == "healthcare":
        t = HC.get(fault)
        return (var.upper() in t) if t else None
    if domain == "industrial":
        t = TEP.get(fault)
        return (var == t) if t else None
    return None


DATASET_DOMAIN = {"Robotics (Pepper)": "robotics",
                  "Healthcare (PTB-XL)": "healthcare",
                  "Industrial (TEP)": "industrial"}


def shap_domains():
    """Which domains the SHAP export actually covers."""
    return {DATASET_DOMAIN[r["Dataset"]]
            for r in csv.DictReader(io.open(SHAP, encoding="utf-8"))
            if r.get("Dataset") in DATASET_DOMAIN}


def rows_for(path, domain=None):
    by = {}
    for r in csv.DictReader(io.open(path, encoding="utf-8")):
        if domain is not None and DATASET_DOMAIN.get(r.get("Dataset")) != domain:
            continue
        key = (r["Model Variant"], r["Fault Name"])
        by.setdefault(key, []).append((int(r["Rank"]), r["Variable"],
                                       float(r["Score"])))
    for k in by:
        by[k].sort()
    return by


def select(ranked, rule):
    if rule.startswith("top"):
        return [v for _, v, _ in ranked[:int(rule[3:])]]
    scores = [s for _, _, s in ranked]
    if len(scores) < 2:
        return [v for _, v, _ in ranked]
    cut = statistics.mean(scores) + statistics.pstdev(scores)
    picked = [v for _, v, s in ranked if s >= cut]
    return picked or [ranked[0][1]]


def main():
    out = [["domain", "algorithm", "variant", "fault", "group", "rule",
            "pool", "n_selected", "n_relevant", "relevance_pct",
            "chance_pct", "lift"]]
    skipped = 0
    sources = []
    for path in sorted(glob.glob("*/*/outputs/attribution_full.csv")):
        parts = path.replace(chr(92), "/").split("/")
        if parts[0] == "Baselines":
            continue
        folder = parts[1].lower()
        sources.append((path, parts[0],
                        "robotics" if "robotic" in folder else
                        "healthcare" if "healthcare" in folder else
                        "industrial"))
    # The neural baselines are scored against the same ground truth as the
    # causal methods, so the two families of attribution are comparable on one
    # number.  SHAP now covers all three datasets, and which domain each row
    # belongs to is read from its Dataset column instead of being assumed.
    if os.path.exists(SHAP):
        for dom in sorted(shap_domains()):
            sources.append((SHAP, "SHAP", dom))

    for path, algo, domain in sources:
        for (variant, fault), ranked in sorted(
                rows_for(path, domain if algo == "SHAP" else None).items()):
            allvars = [v for _, v, _ in ranked]
            flags = [relevant(domain, fault, v) for v in allvars]
            if any(f is None for f in flags):
                skipped += 1
                continue
            chance = 100.0 * sum(flags) / len(flags)
            for rule in ("top3", "top5", "top10", "std1"):
                sel = select(ranked, rule)
                hit = sum(1 for v in sel if relevant(domain, fault, v))
                pct = 100.0 * hit / len(sel) if sel else 0.0
                out.append([domain, algo, variant, fault.replace(".csv", ""),
                            group_of(domain, fault),
                            rule, len(allvars), len(sel), hit, round(pct, 2),
                            round(chance, 2),
                            round(pct / chance, 2) if chance else ""])
    with io.open("tools/relevance_by_class.csv", "w", encoding="utf-8",
                 newline="") as f:
        csv.writer(f).writerows(out)
    print("  wrote tools/relevance_by_class.csv  (%d rows)" % (len(out) - 1))
    print("  class-variant combinations with no ground truth, excluded: %d"
          % skipped)


if __name__ == "__main__":
    main()
