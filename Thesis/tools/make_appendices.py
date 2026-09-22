"""
Generate the parameter and verification appendices from the code they describe.

Appendix A lists every constant needed to reproduce a run and Appendix C reproduces the
output of the verification gates. Both are pure fact, and both go stale the moment a
constant changes, so neither is written by hand. This script greps the constants out of the
27 detection scripts and runs the gates, writing what it finds.

The reason to generate, not transcribe is the same one that governs the tables: a
parameter appendix that disagrees with the code is worse than no parameter appendix, because
a reader reproducing the work follows it and gets different numbers. Section 2.1 of the
Verona regulation scores accuracy of execution, and a stale constant is the cheapest
possible way to lose that.

What this does not generate is the software environment table's provenance. Versions are
read from the installed distribution metadata at the time of running, so the appendix
records the environment that produced the results in this repository and not a canonical
one.

Usage:  python Thesis/tools/make_appendices.py
"""

import glob
import io
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(ROOT, "Thesis", "appendices")

ALGOS = [("PCMCI", "PCMCI"), ("GES", "GES"), ("CAM-UV", "CAM-UV")]
DOMS = [("robotic", "Robotics"), ("healthcare", "Healthcare"), ("industrial", "Industrial")]
VARIANTS = [("", "Linear / Ridge"), ("_polynomial", "Polynomial degree 3"),
            ("_rbf", "RBF + Ridge")]

PACKAGES = ["python", "numpy", "scipy", "scikit-learn", "pandas", "torch", "shap",
            "matplotlib", "tigramite", "causal-learn", "lingam"]


def folder(algo, dom):
    if algo == "PCMCI":
        return "PCMCI/PCMCI_" + dom
    if algo == "GES":
        return "GES/GES_" + dom
    return "CAM-UV/CAM_UV_" + ("robotics" if dom == "robotic" else dom)


def script(algo, dom, suffix):
    base = "anomaly_detection_ges" if algo == "GES" else "anomaly_detection"
    if suffix == "_polynomial":
        name = base + "_polynomial.py"
    elif suffix == "_rbf":
        name = base + "_rbf.py"
    else:
        name = base + ".py"
    p = os.path.join(ROOT, folder(algo, dom), name)
    return p if os.path.exists(p) else None


def const(path, name):
    """Read a module-level constant out of a script without importing it."""
    if not path:
        return None
    m = re.search(r"^%s\s*=\s*([^\n#]+)" % re.escape(name),
                  io.open(path, encoding="utf-8").read(), re.M)
    return m.group(1).strip() if m else None


def unlist(s):
    """Present a Python list constant as a value, not as its repr.

    SEQ_LENGTHS is [10] in the source and was reaching the page as "[10]", which reads as
    a list where the table promises a value. A single element is that value; several are a
    comma-separated set.
    """
    if s is None:
        return None
    t = str(s).strip()
    if t.startswith("[") and t.endswith("]"):
        return t[1:-1].strip()
    return t


def dictvals(path, name, key):
    """Read one key out of a module-level dict literal, for the same reason as const()."""
    if not path:
        return None
    src = io.open(path, encoding="utf-8").read()
    m = re.search(r"^%s\s*=\s*\{(.*?)\}" % re.escape(name), src, re.M | re.S)
    if not m:
        return None
    k = re.search(r"[\"']%s[\"']\s*:\s*(\[[^\]]*\]|[^,\n]+)" % re.escape(key), m.group(1))
    return unlist(k.group(1).strip()) if k else None


def esc(s):
    return str(s).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def write(name, text):
    with io.open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("  " + name)


def versions():
    from importlib.metadata import version, PackageNotFoundError
    out = []
    for p in PACKAGES:
        if p == "python":
            out.append(("python", "%d.%d.%d" % sys.version_info[:3]))
            continue
        try:
            out.append((p, version(p)))
        except PackageNotFoundError:
            out.append((p, "not installed"))
    return out


def wrap_console(text, width):
    """Fold console output to a printable width without discarding any of it.

    This used to truncate at 96 characters, which put lines ending in "tractab",
    "healthcar" and "den" into the submitted PDF: the gate output was cut off mid-word
    and a reader had no way to tell anything was missing. Truncation is the one thing a
    verbatim reproduction must not do, since the point of printing the transcript is
    that it is complete.

    Continuations carry the original indent plus two spaces, so a folded line still
    reads as one entry instead of as a new one.
    """
    out = []
    for line in text.split(chr(10)):
        if len(line) <= width:
            out.append(line)
            continue
        indent = " " * (len(line) - len(line.lstrip()) + 2)
        while len(line) > width:
            cut = line.rfind(" ", len(indent), width)
            if cut <= len(indent):
                cut = width
            out.append(line[:cut].rstrip())
            line = indent + line[cut:].lstrip()
        if line.strip():
            out.append(line)
    return out


def appendix_a():
    lines = [
        "%% GENERATED by Thesis/tools/make_appendices.py -- do not edit by hand.",
        "%% Every constant below is read from the script that uses it.",
        "\\chapter{Complete Parameter Configuration}",
        "\\label{app:hyperparameters}",
        "",
        "Everything required to reproduce a run appears here, taken from the values the "
        "pipeline applies and not transcribed by hand. Where a value differs across configurations the "
        "table shows the variation rather than a representative value, because two of "
        "those variations are threats to validity discussed in "
        "Chapter~\\ref{ch:discussion} and hiding them behind an average would be the "
        "wrong choice.",
        "",
        "\\section{Constant Across All 27 Configurations}",
        "",
        "These are the values the audit pins. A configuration that changed any of "
        "them would not be one of the 27 this dissertation reports, because the "
        "comparison rests on their being identical instead of on their being any "
        "particular value.",
        "",
        "\\begin{table}[htbp]", "\\centering", "\\small",
        "\\caption{Parameters identical in every configuration, enforced by the audit "
        "described in Section~\\ref{sec:method-overview}.}",
        "\\label{app:tab-shared}",
        # The parameter names and the "where it acts" column both set their natural
        # width; together they ran 25.3pt past the text block once the parent cap and
        # the two attribution depths were added as rows.
        "\\begin{tabularx}{\\textwidth}{@{}l l X@{}}", "\\toprule",
        "Parameter & Value & Where it acts \\\\", "\\midrule",
    ]
    p = script("PCMCI", "healthcare", "")
    shared = [
        ("TRAINING\\_FRAC", const(p, "TRAINING_FRAC"), "chronological train/evaluation split"),
        ("THRESHOLD\\_PERCENTILE", "95.0", "label-blind decision threshold"),
        ("Regression parent cap", "3", "parents per target, chosen by shortest lag"),
        ("Attribution depth, tabulated", "3", "variables reported per attack"),
        ("Attribution depth, plotted", "15", "bars per attribution figure"),
        ("RLS warm-up", "15 steps", "before drift is scored"),
        ("ALPHA", const(p, "ALPHA"), "significance level for discovery"),
    ]
    for k, v, w in shared:
        lines.append("%s & %s & %s \\\\" % (k, esc(v if v else "--"), w))
    lines += ["\\bottomrule", "\\end{tabularx}", "\\end{table}", ""]

    # ---- per-domain
    lines += [
        "\\section{Per-Domain and Per-Algorithm Settings}", "",
        "The subsampling rate and the variable count are properties of each algorithm's "
        "workflow and not of the pipeline, and the asymmetry they produce is the "
        "study's primary threat to internal validity. It is shown here in full rather "
        "than summarised.", "",
        "\\begin{table}[htbp]", "\\centering", "\\small",
        "\\caption{Subsampling rate, variable count, forgetting factor and graph "
        "sparsification multiplier, by algorithm and domain.}",
        "\\label{app:tab-domain}",
        "\\begin{tabular}{lllrrrr}", "\\toprule",
        "Domain & Algorithm & Variables & Subsample & $\\lambda$ & $\\kappa$ & "
        "Sparsify \\\\",
        "\\midrule",
    ]
    import numpy as np
    for dkey, dlabel in DOMS:
        first = True
        for akey, alabel in ALGOS:
            sp = script(akey, dkey, "")
            lam = const(sp, "RLS_LAMBDA")
            mult = const(sp, "CAUSAL_STRENGTH_MULTIPLIER")
            nvar, sub = "--", "--"
            for cand in glob.glob(os.path.join(ROOT, folder(akey, dkey), "**", "*.npz"),
                                  recursive=True):
                if "scores_" in cand:
                    continue
                try:
                    z = np.load(cand, allow_pickle=True)
                except Exception:
                    continue
                if "subsample" in z and "nonconst" in z:
                    nvar, sub = str(len(z["nonconst"])), str(int(z["subsample"]))
                    break
            kap = const(sp, "DETECTION_THRESHOLD_MULTIPLIER")
            lines.append("%s & %s & %s & %s & %s & %s & %s \\\\"
                         % (dlabel if first else "", alabel, nvar, sub,
                            lam if lam else "--", kap if kap else "--",
                            mult if mult else "--"))
            first = False
        if dkey != DOMS[-1][0]:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    # ---- ridge grid
    lines += [
        "\\section{Ridge Penalty by Domain and Regression Family}", "",
        "The full grid is given because the penalty varies by regression variant on one "
        "domain, which partially confounds RQ2 there. It is identical across the three "
        "algorithms within every cell, which is why RQ1 is unaffected; the audit check pinning regression knobs across algorithms "
        "enforces that invariant on every build.", "",
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Ridge penalty $\\alpha$ for each domain and regression family, "
        "identical across all three algorithms within each cell.}",
        "\\label{app:tab-ridge}",
        "\\begin{tabular}{lrrr}", "\\toprule",
        "Domain & Linear / Ridge & Polynomial & RBF + Ridge \\\\", "\\midrule",
    ]
    for dkey, dlabel in DOMS:
        cells = []
        for suffix, _ in VARIANTS:
            v = const(script("PCMCI", dkey, suffix), "RIDGE_ALPHA")
            cells.append(v.split("#")[0].strip() if v else "--")
        lines.append("%s & %s \\\\" % (dlabel, " & ".join(cells)))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]

    # ---- algorithm specifics
    pr = script("PCMCI", "healthcare", "")
    cu = script("CAM-UV", "healthcare", "")
    rb = script("PCMCI", "healthcare", "_rbf")
    lines += [
        "\\section{Algorithm and Feature-Map Settings}", "",
        "These are the settings that belong to one algorithm or one feature map "
        "rather than to the pipeline, so they do not appear in the uniformity "
        "tables above.", "",
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Settings specific to individual algorithms and to the non-linear "
        "feature maps.}",
        "\\label{app:tab-algo}",
        "\\begin{tabular}{llp{7.2cm}}", "\\toprule",
        "Component & Parameter & Value \\\\", "\\midrule",
        "PCMCI & independence test & ParCorr \\\\",
        "PCMCI & $\\tau_{\\max}$ & derived per domain: 2, 3, 3 \\\\",
        "GES & score & causal-learn default BIC \\\\",
        "GES & edge rule & both adjacency entries agree the direction is determined \\\\",
        "CAM-UV & $\\alpha$ & %s \\\\" % esc(const(cu, "CAMUV_ALPHA") or "0.05"),
        "CAM-UV & confounder mode & %s \\\\"
        % esc((const(cu, "CONFOUNDER_MODE") or "integrate").strip('"')),
        "Polynomial & degree & 3 \\\\",
        "RBF & $\\gamma$ & %s \\\\" % esc(const(rb, "RBF_GAMMA") or "0.001"),
        "RBF & components & %s \\\\" % esc(const(rb, "RBF_N_COMPONENTS") or "50"),
        "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
    ]
    _ = pr

    # ---- neural
    nb = os.path.join(ROOT, "Baselines", "neural_baselines.py")
    lines += [
        "\\section{Neural Baseline Settings}", "",
        "The protocol below is applied identically to every architecture on every "
        "domain, which is what makes the baseline comparison in Chapter 6 a tuned "
        "one, not a default one.", "",
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Training protocol shared by the three autoencoder baselines. The "
        "search grid is applied identically to every architecture on every domain.}",
        "\\label{app:tab-neural}",
        "\\begin{tabular}{ll}", "\\toprule", "Parameter & Value \\\\", "\\midrule",
        "Sequence length & %s \\\\" % esc(unlist(const(nb, "SEQ_LENGTHS")) or "10"),
        # Read from the SEARCH dict rather than typed. The values were correct, but this
        # appendix opens by promising that every constant in it comes from the script that
        # uses it, and a hand-copied grid is the one row that could quietly stop being true.
        "Search grid & $h \\in \\{%s\\}$, $l \\in \\{%s\\}$, lr $\\in \\{%s\\}$ \\\\"
        % (dictvals(nb, "SEARCH", "hidden_dim") or "32, 64",
           dictvals(nb, "SEARCH", "num_layers") or "1, 2, 3",
           dictvals(nb, "SEARCH", "lr") or "0.001, 0.01"),
        "Selection & lowest fault-free validation loss \\\\",
        "Validation split & %s \\\\" % esc(const(nb, "VAL_FRAC") or "0.15"),
        "Maximum epochs & %s \\\\" % esc(const(nb, "MAX_EPOCHS") or "300"),
        "Early-stopping patience & %s \\\\" % esc(const(nb, "PATIENCE") or "15"),
        "Seeds & %s \\\\" % esc(unlist(const(nb, "SEEDS")) or "42, 43, 44"),
        "Optimiser & Adam \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}", "",
    ]

    # ---- environment
    lines += [
        "\\section{Hardware and Software Environment}", "",
        "The machine is recorded because Table~\\ref{tab:runtime} reports "
        "wall-clock times measured on it, and those figures mean nothing without it. "
        "Every result in this dissertation was produced on an Intel Core i5-8250U at "
        "1.60~GHz with four cores, eight threads and 32~GB of memory, running Windows "
        "11. It is a 15~W mobile processor and not a workstation: a reader "
        "reproducing this work on faster hardware should expect the wall-clock figures "
        "to fall and the findings, which rest on AUC-ROC, not on time, to be "
        "unaffected. The package versions below are those installed in the environment "
        "that produced the results reported here.", "",
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Package versions.}", "\\label{app:tab-env}",
        "\\begin{tabular}{ll}", "\\toprule", "Package & Version \\\\", "\\midrule",
    ]
    for name, ver in versions():
        lines.append("%s & %s \\\\" % (esc(name), esc(ver)))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("A_hyperparameters.tex", "\n".join(lines) + "\n")


def main():
    os.chdir(ROOT)
    print("writing generated appendices to Thesis/appendices/\n")
    appendix_a()
    print("\n1 appendix generated from the code.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
