"""
Prove that the 27 configurations differ only along the three permitted axes.

The reviewer's constraint was explicit: "use this same algorithm, just changing the dataset,
the causal discovery algorithm or the regression type, and nothing else". That is a claim
about 11,000 lines of code, and prose cannot establish it. This checks it mechanically.

What is checked

  1. RLS update            byte-identical across all 27 after normalising whitespace
  2. Threshold rule        every script takes the 95th percentile of fault-free scores
  3. No label leakage      no script selects its threshold against evaluation labels
  4. Scoring symmetry      normal and attack data enter the scorer the same way
  5. Attribution depth     every script reports the same number of top variables
  6. Per-domain constants  differ only per dataset or per regression variant
  7. Score persistence     every configuration writes scores_<variant>.npz for the bootstrap
  8. Cache guards          GES and CAM-UV refuse a cached graph built at another subsample

Exit code is non-zero if any check fails, so this can gate a build.

Usage:  python tools/audit_pipeline.py
"""

import ast
import glob
import hashlib
import io
import os
import re
import sys
from collections import defaultdict

SCRIPTS = sorted(glob.glob("*/*/anomaly_detection*.py"))


def src(p):
    return io.open(p, encoding="utf-8").read()


def folder(p):
    return p.replace("\\", "/").split("/")[1]


def domain(p):
    f = folder(p).lower()
    for d in ("robotic", "healthcare", "industrial"):
        if d in f:
            return "robotics" if d == "robotic" else d
    return "?"


def variant(p):
    b = os.path.basename(p)
    return "polynomial" if "polynomial" in b else ("rbf" if "rbf" in b else "linear")


def algorithm(p):
    """Which causal discovery algorithm a script belongs to, from its top-level directory."""
    top = p.replace("\\", "/").split("/")[0].upper()
    return "PCMCI" if top.startswith("PCMCI") else ("GES" if top.startswith("GES") else "CAM-UV")


def function_source(text, name):
    """Return the normalised source of a top-level function, or None."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(text, node)
            if seg is None:
                return None
            # strip comments and blank lines so formatting differences do not register
            lines = [l.split("#")[0].rstrip() for l in seg.split("\n")]
            return "\n".join(l for l in lines if l.strip())
    return None


def check_rls():
    groups = defaultdict(list)
    missing = []
    for p in SCRIPTS:
        s = function_source(src(p), "_rls_update")
        if s is None:
            missing.append(p)
        else:
            groups[hashlib.sha256(s.encode()).hexdigest()[:12]].append(p)
    ok = not missing and len(groups) == 1
    detail = f"{len(groups)} distinct implementation(s) across {len(SCRIPTS)} scripts"
    if missing:
        detail += f"; MISSING in {len(missing)}: {missing[:3]}"
    if len(groups) > 1:
        for h, ps in groups.items():
            detail += f"\n      {h}: {len(ps)} scripts e.g. {ps[0]}"
    return ok, detail


def check_regex(label, pattern, want_present, exclude_comments=True):
    """want_present=True: every script must match. False: no script may match."""
    rx = re.compile(pattern)
    bad = []
    for p in SCRIPTS:
        lines = src(p).split("\n")
        if exclude_comments:
            lines = [l for l in lines if not l.lstrip().startswith("#")]
        hit = any(rx.search(l) for l in lines)
        if hit != want_present:
            bad.append(p)
    return not bad, f"{len(SCRIPTS)-len(bad)}/{len(SCRIPTS)} scripts" + (
        f"; offenders: {[os.path.join(folder(b), os.path.basename(b)) for b in bad[:4]]}" if bad else "")


def check_constants():
    """Constants may vary per dataset or per regression variant, never per algorithm."""
    NAMES = ["DETECTION_THRESHOLD_MULTIPLIER", "CAUSAL_STRENGTH_MULTIPLIER",
             "RLS_LAMBDA", "TRAINING_FRAC", "THRESHOLD_PERCENTILE"]
    vals = defaultdict(dict)
    for p in SCRIPTS:
        text = src(p)
        for n in NAMES:
            m = re.search(rf"^{n}\s*=\s*([0-9.]+)", text, re.M)
            if m:
                vals[n][(domain(p), variant(p), folder(p))] = m.group(1)
    problems = []
    for n, d in vals.items():
        by_axis = defaultdict(set)
        for (dom, var, fold), v in d.items():
            by_axis[(dom, var)].add(v)
        for key, s in by_axis.items():
            if len(s) > 1:
                problems.append(f"{n} differs across algorithms at {key}: {sorted(s)}")
    return not problems, ("; ".join(problems) if problems else
                          f"{len(vals)} constants, no cross-algorithm mismatch")


def check_cache_guards():
    """GES and CAM-UV must refuse a cached graph built at a different subsample."""
    need = [p for p in SCRIPTS if folder(p).startswith(("GES", "CAM_UV"))]
    bad = [p for p in need if "SUBSAMPLE_EXPECTED" not in src(p)]
    return not bad, f"{len(need)-len(bad)}/{len(need)} GES and CAM-UV scripts guard the cache"


def check_discovery_knobs():
    """Discovery settings may vary per dataset, never between variants of the same dataset.

    GES robotics passes maxP=1 and adds seeded jitter to break perfect collinearity, which
    healthcare and industrial do not need. That is a per-dataset accommodation and allowed.
    What would not be allowed is the same dataset using different settings across its three
    regression variants, since regression type is supposed to be the only thing changing.
    """
    import re as _re
    KNOBS = {
        "ges maxP": _re.compile(r"ges\([^)]*maxP\s*=\s*(\d+)"),
        "collinearity jitter": _re.compile(r"np\.random\.normal\(0,\s*([0-9.e-]+)"),
        "CAMUV alpha": _re.compile(r"^CAMUV_ALPHA\s*=\s*([0-9.]+)", _re.M),
        "CAMUV num_explanatory": _re.compile(r"^CAMUV_NUM_EXPLANATORY\s*=\s*(\d+)", _re.M),
        "top variance sensors": _re.compile(r"^TOP_VARIANCE_SENSORS\s*=\s*(\d+)", _re.M),
    }
    by_folder = defaultdict(lambda: defaultdict(set))
    for p_ in SCRIPTS:
        text = src(p_)
        for name, rx in KNOBS.items():
            m = rx.search(text)
            by_folder[folder(p_)][name].add(m.group(1) if m else "absent")

    problems = []
    for fold, knobs in sorted(by_folder.items()):
        for name, vals in knobs.items():
            if len(vals) > 1:
                problems.append(f"{fold}: {name} differs across its variants: {sorted(vals)}")
    summary = []
    for fold, knobs in sorted(by_folder.items()):
        setl = [f"{n}={list(v)[0]}" for n, v in sorted(knobs.items()) if list(v)[0] != "absent"]
        if setl:
            summary.append(f"{fold}: " + ", ".join(setl))
    if problems:
        detail = "; ".join(problems)
    else:
        detail = ("consistent within every folder\n         "
                  + ("\n         ").join(summary))
    return not problems, detail


def check_regression_knobs_across_algorithms():
    """Regression settings may vary by regression type; they may not vary by algorithm.

    The ridge penalty is not uniform across the 27. On robotics it takes three values --
    10.0 linear, 0.01 polynomial, 1.0 RBF -- because the polynomial and random-Fourier
    design matrices are conditioned very differently from the raw one, while healthcare and
    industrial hold it at 10.0 throughout.

    Varying a regression knob by regression type is varying along a declared axis, so it is
    permitted. What would destroy the primary comparison is varying it by *algorithm*: if
    PCMCI robotics polynomial used a different penalty from GES robotics polynomial, the
    algorithm ranking in Chapter 6 would partly be a ranking of regularisation strength.

    This check therefore pins the invariant the headline result depends on: within each
    (dataset, regression type) cell, all three algorithms must agree. The residual
    consequence -- that the *variant* comparison on robotics also varies the penalty -- is
    not something a checker can fix, and is declared as a limitation in the thesis.
    """
    import re as _re
    KNOBS = {
        "ridge alpha": _re.compile(r"^RIDGE_ALPHA\s*=\s*([0-9.]+)", _re.M),
        "RLS lambda": _re.compile(r"^RLS_LAMBDA\s*=\s*([0-9.]+)", _re.M),
        "polynomial degree": _re.compile(r"^POLY_DEGREE\s*=\s*(\d+)", _re.M),
        "RBF gamma": _re.compile(r"^RBF_GAMMA\s*=\s*([0-9.e-]+)", _re.M),
    }
    cells = defaultdict(lambda: defaultdict(dict))
    for p_ in SCRIPTS:
        text = src(p_)
        key = (domain(p_), variant(p_))
        for name, rx in KNOBS.items():
            m = rx.search(text)
            cells[key][name][algorithm(p_)] = m.group(1) if m else "absent"

    problems = []
    for key in sorted(cells):
        for name, byalgo in sorted(cells[key].items()):
            vals = set(byalgo.values())
            if len(vals) > 1:
                problems.append(f"{key[0]}/{key[1]}: {name} differs by algorithm: {byalgo}")

    # Report the by-variant spread as information, since it is a declared limitation.
    spread = []
    for dom in sorted({k[0] for k in cells}):
        per = {}
        for key in cells:
            if key[0] == dom:
                per[key[1]] = list(cells[key]["ridge alpha"].values())[0]
        if len(set(per.values())) > 1:
            spread.append(f"{dom} ridge alpha varies by variant: "
                          + ", ".join(f"{v}={per[v]}" for v in sorted(per)))
    detail = ("; ".join(problems) if problems else
              "all regression knobs agree across algorithms within every cell"
              + ("\n         declared: " + "; ".join(spread) if spread else ""))
    return not problems, detail


def check_subsample_policy():
    """Report the temporal resolution each algorithm actually scores at.

    This is the most consequential non-uniformity in the study and it cannot be checked
    away, so it is checked *out loud*. PCMCI derives its subsample rate adaptively from the
    frequency content of the signal, following the tigramite workflow; GES and CAM-UV
    use a fixed rate of 10. The rates that result differ sharply:

        domain        PCMCI   GES   CAM-UV
        robotics         74    10          10
        healthcare        7    10          10
        industrial        1    10          10

    So within a domain the three algorithms do not score the same series: on industrial
    PCMCI sees every sample while the others see every tenth. The consequence is that RQ1
    compares each algorithm *as it is intended to be applied*, preprocessing included,
    rather than isolating the discovery step alone.

    Forcing a common rate is not a free fix in either direction. PCMCI robotics forced to
    10 falls from 0.7825 to 0.6813; GES healthcare forced to 1 falls from 0.8114 to 0.6184;
    and a 160-variable GES run at rate 1 did not converge in 66 minutes. The ablations
    bound the effect, and the thesis declares it as the primary threat to internal validity.

    What this check enforces is the part that *is* controllable: within one folder, all
    three regression variants must score at the same rate, and the cached graph must record
    it, so a variant comparison is never a resolution comparison.
    """
    import numpy as _np
    rates = {}
    for p_ in glob.glob("*/*/**/*.npz", recursive=True) + glob.glob("*/*/*.npz"):
        if "scores_" in p_:
            continue
        # The eight-lead ablation keeps its cache beside the main one. Both match the
        # glob, and the ablation sorts last, so without this the healthcare row reported
        # 8/8/8 -- the ablation's width -- where Appendix A and Chapter 5 both say the
        # main configuration runs over 12. The ablation is a separate experiment, not a
        # variable-matching fact about the grid this check describes.
        if "leads8" in p_.replace("\\", "/"):
            continue
        try:
            z = _np.load(p_, allow_pickle=True)
        except Exception:
            continue
        if "subsample" not in z:
            continue
        fold = p_.replace("\\", "/").split("/")[1]
        rates.setdefault(fold, set()).add(int(z["subsample"]))

    problems = [f"{f}: cached graphs disagree on subsample {sorted(v)}"
                for f, v in sorted(rates.items()) if len(v) > 1]
    missing = [folder(p) for p in SCRIPTS if folder(p) not in rates]
    if missing:
        problems.append("no cached rate recorded for " + ", ".join(sorted(set(missing))))

    if problems:
        return False, "; ".join(problems)
    table = ", ".join(f"{f}={list(v)[0]}" for f, v in sorted(rates.items()))
    return True, ("one rate per folder, recorded in the cache\n         declared: "
                  + table + "\n         PCMCI selects adaptively; GES and CAM-UV are fixed "
                  "at 10 -- see the limitation in the thesis")


def check_variable_matching():
    """Report whether the three algorithms see the same variables within each domain.

    They do on two domains and do not on the third:

        domain        PCMCI          GES        CAM-UV
        healthcare       12           12               12     matched
        industrial       52           52               52     matched
        robotics        244           35               35     NOT matched

    The 35-sensor cap is declared only in the GES and CAM-UV robotics scripts, and it exists
    because neither algorithm is tractable at the full width -- a 160-variable GES run did
    not converge in 66 minutes of CPU. PCMCI has no such limit and runs on all 244.

    So the cap is a necessary accommodation rather than an arbitrary choice, but it still
    means the robotics comparison is between algorithms solving differently sized problems.
    Combined with the scoring resolution -- 74 against 10 on the same domain -- the robotics
    row of RQ1 cannot attribute its margin to the discovery algorithm.

    The consequence for the thesis is a narrowing of the claim rather than a retraction:
    RQ1's evidence rests on healthcare and industrial, which are variable-matched, and
    robotics is reported as confounded. The headline conclusion survives, since the winners
    on the two matched domains are still different algorithms (GES on healthcare, PCMCI on
    industrial) and the ranking is still unstable.

    This check does not fail on the mismatch, which is a design fact rather than a defect.
    It fails if a domain that *is* matched stops being matched.
    """
    import numpy as _np
    seen = defaultdict(dict)
    for p_ in glob.glob("*/*/**/*.npz", recursive=True) + glob.glob("*/*/*.npz"):
        if "scores_" in p_:
            continue
        # Skip the eight-lead ablation cache. It matches the same glob as the main
        # cache and sorts after it, so the last-one-wins assignment below was
        # reporting healthcare as 8/8/8 -- the ablation width -- into Appendix C,
        # against the 12 that Appendix A and Chapter 5 both state. The ablation is a
        # separate experiment and is not evidence about the grid this check covers.
        if "leads8" in p_.replace("\\", "/"):
            continue
        try:
            z = _np.load(p_, allow_pickle=True)
        except Exception:
            continue
        if "nonconst" not in z:
            continue
        parts = p_.replace("\\", "/").split("/")
        dom = ("robotics" if "robotic" in parts[1]
               else "healthcare" if "healthcare" in parts[1] else "industrial")
        algo = ("PCMCI" if parts[0].startswith("PCMCI")
                else "GES" if parts[0].startswith("GES") else "CAM-UV")
        seen[dom][algo] = len(z["nonconst"])

    KNOWN_UNMATCHED = {"robotics"}
    problems, report = [], []
    for dom in sorted(seen):
        counts = seen[dom]
        matched = len(set(counts.values())) == 1
        report.append(f"{dom}=" + "/".join(str(counts.get(a, "?"))
                                           for a in ("PCMCI", "GES", "CAM-UV")))
        if not matched and dom not in KNOWN_UNMATCHED:
            problems.append(f"{dom} lost variable matching: {counts}")
        if matched and dom in KNOWN_UNMATCHED:
            problems.append(f"{dom} is now matched; remove it from KNOWN_UNMATCHED")

    detail = ("; ".join(problems) if problems else
              "PCMCI/GES/CAM-UV widths -- " + ", ".join(report)
              + "\n         declared: robotics is not variable-matched (GES and CAM-UV are"
                " capped at 35 for tractability); RQ1 rests on the other two domains")
    return not problems, detail


def check_graph_sparsification():
    """Report how each PCMCI folder turns a p-value matrix into a graph.

    Significance alone does not decide an edge. The pipeline also drops any link whose
    absolute effect size falls below a multiple of the mean absolute effect size:

        normal_matrix = val_matrix * (p_matrix < ALPHA) \
                        * (abs(val_matrix) > CAUSAL_STRENGTH_MULTIPLIER * mean(abs(val_matrix)))

    The multiplier is 1.0 on healthcare and robotics and 0.0 on industrial, where the second
    factor is therefore always true and the filter does nothing. That is a per-domain setting,
    which is a permitted axis, but it changes how dense a graph the detector is handed and so
    it is declared rather than left to be discovered.

    Measured, in tools/pcmci_graph_density.csv, the filter turns out to be nearly inert
    everywhere: healthcare 66 -> 63 pairs of 66, robotics 18958 unchanged, industrial 511
    unchanged. The density that reaches the detector is set by significance, not by strength.

    Worth carrying into the discussion: the healthcare graph links 63 of 66 possible pairs,
    so nearly every variable is a parent of nearly every other and the regressions have
    almost no structure to exploit. That is a candidate explanation for PCMCI trailing GES
    on this domain, and it is consistent with the independent-lead ablation improving the
    polynomial variant from 0.5075 to 0.6355 once the four algebraically dependent leads go.

    What this check enforces is the controllable part: within one folder all three variants
    must sparsify identically, so a variant comparison is never a density comparison.
    """
    per_folder = defaultdict(set)
    for p_ in SCRIPTS:
        if algorithm(p_) != "PCMCI":
            continue
        m = re.search(r"^CAUSAL_STRENGTH_MULTIPLIER\s*=\s*([0-9.]+)", src(p_), re.M)
        per_folder[folder(p_)].add(m.group(1) if m else "absent")

    problems = [f"{f}: variants disagree {sorted(v)}"
                for f, v in sorted(per_folder.items()) if len(v) > 1]
    if problems:
        return False, "; ".join(problems)
    if not per_folder:
        return False, "no PCMCI scripts found"
    table = ", ".join(f"{f}={list(v)[0]}" for f, v in sorted(per_folder.items()))
    return True, ("one sparsification rule per folder\n         declared: " + table
                  + "\n         0.0 disables the strength filter; measured effect is small"
                    " -- see tools/pcmci_graph_density.csv")


def check_feature_scaling():
    """Every variant of a dataset must agree on whether the design matrix is scaled.

    Ridge penalises coefficients, so fitting it on a raw design matrix spreads the penalty
    across parents according to their units. Four linear scripts -- GES and CAM-UV, robotics
    and industrial -- previously omitted the StandardScaler their own polynomial and RBF
    siblings applied, which made regression type not the only thing differing inside those
    folders.
    """
    by_folder = defaultdict(dict)
    for p_ in SCRIPTS:
        by_folder[folder(p_)][variant(p_)] = "StandardScaler()" in src(p_)
    problems = [f"{fold}: {sorted(k for k, v in d.items() if v)} scale, "
                f"{sorted(k for k, v in d.items() if not v)} do not"
                for fold, d in sorted(by_folder.items()) if len(set(d.values())) > 1]
    n = sum(1 for d in by_folder.values() for v in d.values() if v)
    return not problems, ("; ".join(problems) if problems else
                          f"{n}/{len(SCRIPTS)} scripts scale, consistent within every folder")


def check_metrics_arithmetic():
    """Recompute precision, recall, F1 and accuracy from each row's confusion matrix.

    The reported figures should follow from TP/FP/TN/FN by definition. If they do not, either
    the confusion matrix or the summary statistic was produced by different code than the
    other, which is the kind of drift that survives every visual check.
    """
    import csv as _csv
    bad, n = [], 0
    for p_ in sorted(glob.glob("*/*/outputs/metrics_summary.csv")
                     + glob.glob("Baselines/outputs/metrics_summary.csv")):
        for r in _csv.DictReader(io.open(p_, encoding="utf-8")):
            try:
                tp, fp, tn, fn = (int(r[k]) for k in ("TP", "FP", "TN", "FN"))
            except (KeyError, ValueError):
                continue
            n += 1
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
            for name, got, want in (("Precision", r.get("Precision"), prec),
                                    ("Recall", r.get("Recall"), rec),
                                    ("F1", r.get("F1"), f1),
                                    ("Accuracy", r.get("Accuracy"), acc)):
                if got is None:
                    continue
                if abs(float(got) - want) > 5e-4:
                    bad.append(f"{os.path.basename(os.path.dirname(os.path.dirname(p_)))}"
                               f"/{r.get('Model Variant')}: {name} {got} vs {want:.4f}")
    return not bad, (f"{n} rows, all consistent with their confusion matrices" if not bad
                     else "; ".join(bad[:4]))


def check_intercept_in_drift():
    """Every configuration must track the same thing as "coefficient drift".

    The baseline vector fine_coeffs and the online regressor x must agree on whether an
    intercept term is present. PCMCI robotics RBF stored model.coef_ alone and fed X[t] alone
    -- internally consistent, but it was the only one of 27 whose drift vector excluded the
    intercept, so the quantity being monitored was not the same quantity as everywhere else.
    """
    bad = []
    for p_ in SCRIPTS:
        s = src(p_)
        has_b0 = "model.intercept_" in s and "fine_coeffs[var] = np.concatenate" in s
        # a leading 1s column, however it is spelled: a per-timestep concatenate([[1.0], x])
        # or a whole-matrix column_stack([np.ones(...), X])
        prepends = bool(re.search(r"concatenate\(\s*\[\s*\[\s*1\.0\s*\]", s)
                        or re.search(r"column_stack\(\s*\[\s*np\.ones\(", s))
        if has_b0 != prepends:
            bad.append(f"{folder(p_)}/{os.path.basename(p_)} "
                       f"(coeffs have b0: {has_b0}, design matrix has a 1s column: {prepends})")
    return not bad, (f"{len(SCRIPTS)-len(bad)}/{len(SCRIPTS)} scripts agree" +
                     (f"; mismatched: {bad}" if bad else ", all include the intercept"))


def check_attribution_depth():
    """Every reporting helper must rank the same number of variables.

    This check exists because its predecessor did not do what its name said. It matched
    `var_indices[:3]` in the detection scripts and reported "attribution depth uniform
    (top-3)". That truncation is the three-parent regression cap in `fit_normal_coeffs`;
    it has nothing to do with attribution, which happens in the reporting helpers and
    ranks up to 15. The thesis then repeated the check's label as fact.

    A check that passes on a quantity it is not measuring is worse than no check, so the
    depth is now read from the file that actually sets it -- and both depths are read,
    because there are two. The helper ranks every monitored target, plots the top 15, and
    exports only the top 3 to the CSV the thesis tables are built from. Checking one and
    reporting it as "the" attribution depth is how the first error happened.
    """
    plotted, exported = {}, {}
    for p_ in sorted(glob.glob("*/*/reporting_helper.py")):
        text = src(p_)
        key = os.path.dirname(p_).replace("\\", "/")
        m = re.search(r"top_n\s*=\s*min\(\s*(\d+)", text)
        plotted[key] = int(m.group(1)) if m else None
        exported[key] = len(re.findall(r"'Top Variable (\d+)'", text))
    if not plotted:
        return False, "no reporting helpers found"
    missing = sorted(k for k, v in plotted.items() if v is None)
    if missing:
        return False, f"no plotted depth found in: {missing[:4]}"
    pv, ev = set(plotted.values()), set(exported.values())
    if len(pv) > 1:
        return False, f"plotted depths disagree: {sorted(pv)}"
    if len(ev) > 1 or 0 in ev:
        return False, f"exported depths disagree: {sorted(ev)}"
    return True, (f"{len(plotted)}/{len(plotted)} helpers plot top-{pv.pop()} "
                  f"and export top-{ev.pop()}")


def check_attribution_formula():
    """The attribution ranking must be constant within a domain.

    Robotics ranks by the raw L2 norm of the per-target residual, which is the XAI course
    lab's own formula and is required there: the reviewer asked that PCMCI-linear on Pepper
    reproduce the lab exactly. Healthcare and industrial divide that norm by the variable's
    own fault-free baseline, so leads and sensors with naturally large amplitude cannot
    dominate the ranking regardless of what was actually perturbed.

    That is a per-dataset difference and permitted. What would break the comparison is a
    domain whose three algorithms, or one algorithm's three variants, disagreed -- then a
    cross-model attribution table would not be like for like.
    """
    by_domain = defaultdict(dict)
    for p_ in SCRIPTS:
        s = src(p_)
        scaled = "normal_residual_scale=normal_residual_scale" in s
        by_domain[domain(p_)][f"{folder(p_)}/{variant(p_)}"] = scaled
    problems = []
    for dom, d in sorted(by_domain.items()):
        if len(set(d.values())) > 1:
            yes = sorted(k for k, v in d.items() if v)
            no = sorted(k for k, v in d.items() if not v)
            problems.append(f"{dom}: {yes} normalise, {no} do not")
    summary = "; ".join(
        f"{dom} = {'baseline-normalised RMS' if list(d.values())[0] else 'raw L2 norm (lab formula)'}"
        for dom, d in sorted(by_domain.items()))
    return not problems, ("; ".join(problems) if problems else summary)


def main():
    checks = [
        ("RLS update identical everywhere", check_rls),
        ("95th-percentile fault-free threshold",
         lambda: check_regex("", r"percentile\s*\(.*(95|THRESHOLD_PERCENTILE)", True)),
        ("no threshold fitted on evaluation labels",
         lambda: check_regex("", r"precision_recall_curve|argmax\s*\(\s*f1", False)),
        ("three-parent regression cap applied uniformly",
         lambda: check_regex("", r"^\s*var_indices\s*=\s*var_indices\[:3\]", True)),
        ("attribution depth uniform", check_attribution_depth),
        ("per-variant score files written",
         lambda: check_regex("", r"scores_|save_scores", True)),
        ("per-domain constants consistent", check_constants),
        ("cached graphs guarded by subsample", check_cache_guards),
        ("discovery knobs vary by dataset, not by variant", check_discovery_knobs),
        ("feature scaling consistent within each folder", check_feature_scaling),
        ("intercept present in both coefficients and design matrix", check_intercept_in_drift),
        ("attribution formula constant within each domain", check_attribution_formula),
        ("regression knobs agree across algorithms", check_regression_knobs_across_algorithms),
        ("one scoring resolution per folder", check_subsample_policy),
        ("variable matching within each domain", check_variable_matching),
        ("graph sparsification rule per folder", check_graph_sparsification),
        ("reported metrics follow from the confusion matrix", check_metrics_arithmetic),
    ]
    print(f"auditing {len(SCRIPTS)} detection scripts\n")
    failed = 0
    for label, fn in checks:
        ok, detail = fn()
        failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        print(f"         {detail}")
    print()
    if failed:
        print(f"{failed} check(s) FAILED")
    else:
        # Deliberately narrower than it used to read. The uniformity claim holds for the
        # detector -- one RLS, one threshold rule, one attribution depth, one scoring path --
        # and for the regression knobs across algorithms. It does not hold for the inputs:
        # PCMCI selects its own scoring resolution, and on robotics it also sees 244
        # variables where GES and CAM-UV see 35. Those are reported by their own checks and
        # declared in the thesis, so the summary must not paper over them.
        print("all checks passed")
        print("  detector, threshold, attribution and regression knobs are uniform "
              "across all 27")
        print("  inputs are NOT uniform: scoring resolution varies by algorithm in every "
              "domain,")
        print("  and robotics is additionally not variable-matched (244 vs 35) -- both "
              "declared")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
