"""
Place every figure and table into the chapter that needs it, and leave the prose to be written.

The method this supports: put the visual anchors on the page first, then write a paragraph
above each one saying what the reader is looking at and a paragraph below saying why it
matters. That turns a blank chapter into a set of small, concrete writing tasks, each with
the evidence already sitting next to it.

So this emits float environments -- \\includegraphics, \\input of the generated tables,
labels, and captions -- into Thesis/skeleton/, one file per chapter. What it deliberately
does not emit is the argument. Every float is followed by

    % WRITE ABOVE: ...
    % WRITE BELOW: ...

markers naming the specific question that paragraph has to answer, drawn from the
measurements rather than invented. The markers are comments, so the document compiles with
them still in and they disappear from the PDF; a chapter is finished when none are left.

Captions are factual, not interpretive: they say which algorithm, domain and variant a
figure shows, because that is a label rather than a claim. Anything that reads as an
argument belongs in the prose the author writes. Rewrite the captions anyway -- a caption in
the author's own words alongside paragraphs in the author's own words reads as one voice,
and this cannot supply that.

Nothing here is destructive. It writes into Thesis/skeleton/ and never touches
Thesis/chapters/, so drafted prose cannot be overwritten. Copy across what you want.

Usage:  python Thesis/tools/make_skeleton.py
"""

import glob
import io
import os
import sys

ALGOS = [("pcmci", "PCMCI"), ("ges", "GES"), ("camuv", "CAM-UV")]
DOMAINS = [("robotic", "Robotics (Pepper)"), ("robotics", "Robotics (Pepper)"),
           ("healthcare", "Healthcare (PTB-XL ECG)"), ("industrial", "Industrial (TEP)")]
VARIANTS = [("linear_ridge", "linear/ridge"),
            ("polynomial_degree_3", "polynomial degree 3"),
            ("rbf_ridge", "RBF + ridge")]
FAULTS = [("wheelscontrol", "WheelsControl"), ("ledscontrol", "LedsControl"),
          ("jointcontrol", "JointControl")]


def fig(path, caption, label, above, below, width="0.9"):
    return """\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=%s\\textwidth]{%s}
  \\caption{%s}
  \\label{fig:%s}
\\end{figure}

%% WRITE ABOVE: %s
%% WRITE BELOW: %s

""" % (width, path, caption, label, above, below)


def table(name, above, below):
    return """\\input{tables/%s}

%% WRITE ABOVE: %s
%% WRITE BELOW: %s

""" % (name, above, below)


def have(stem):
    return os.path.exists(os.path.join("Thesis", "figures", stem + ".png"))


def chapter5():
    out = ["% Chapter 5 floats. Prose is yours.\n\n\\section{Validation of the implementation}\n\n"]
    out.append(table("tbl_validation",
                     "what each validation checks and why a closed-form answer exists for it",
                     "the RLS-to-OLS agreement at lambda=1 is 7.478e-10 -- say what that "
                     "rules out, namely a filter that only appears to work"))
    out.append("\n\\section{Statistical power of the industrial discovery regime}\n\n")
    out.append(table("tbl_tep_power",
                     "the 70/30 split leaves 35 samples for 52 variables at subsample 10",
                     "density rises 0.0238 to 0.1112 as samples rise 35 to 350 -- the "
                     "sparsity is absent power, not selectivity. Name which it is."))
    return "".join(out)


def chapter6():
    out = ["% Chapter 6 floats. Prose is yours.\n\n\\section{Headline comparison}\n\n"]
    out.append(table("tbl_main_results",
                     "how to read the table: 27 configurations, three axes",
                     "which algorithm wins in each domain, and whether the margin survives "
                     "Table~\\ref{tab:bootstrap}"))
    for key, label in (("robotic", "Robotics"), ("healthcare", "Healthcare"),
                       ("industrial", "Industrial")):
        out.append(table("tbl_detail_%s" % key,
                         "the %s domain in full: F1, AUC-ROC, AUC-PR per configuration"
                         % label.lower(),
                         "where F1 and AUC-ROC disagree, and why a fixed 95th-percentile "
                         "threshold makes that possible"))
    out.append("\n\\section{Uncertainty}\n\n")
    out.append(table("tbl_bootstrap",
                     "why a moving-block bootstrap and not the analytical interval",
                     "the analytical interval is narrower on every configuration -- by 54\\% "
                     "and 63\\% on the two worst. Say what reporting it would have claimed."))
    out.append("\n\\section{Baselines}\n\n")
    out.append(table("tbl_baselines",
                     "the three tuned neural architectures and the trivial control, same "
                     "threshold rule throughout",
                     "the trivial z-score scores 0.9286 on industrial against the best "
                     "causal 0.9151. This is the paragraph that changes the narrative -- "
                     "write it as a finding, not an apology."))
    out.append("\n\\section{Attribution}\n\n")
    out.append(table("tbl_attributions",
                     "what the causal attribution ranks and how the ranking is computed",
                     "9 of 9 on the LED requirement, 8 of 9 on the joint requirement"))
    out.append(table("tbl_shap",
                     "the same three faults under post-hoc SHAP",
                     "SHAP recovers a tablet sensor for the LED fault in two of three "
                     "architectures. Contrast with the row above and say what that shows "
                     "about post-hoc versus intrinsic attribution."))
    for arch, nice in (("lstm_autoencoder", "LSTM"), ("tcn_autoencoder", "TCN"),
                       ("transformer_autoencoder", "Transformer")):
        for fkey, fname in FAULTS:
            stem = "baseline_shap_%s_%s" % (arch, fkey)
            if have(stem):
                out.append(fig("%s.png" % stem,
                               "SHAP attribution for the %s autoencoder on the %s fault."
                               % (nice, fname),
                               stem.replace("_", "-"),
                               "which variable this architecture blames for %s" % fname,
                               "whether that is the subsystem actually attacked"))
    out.append("\n\\section{Confounders}\n\n")
    out.append(table("tbl_confounder",
                     "the confounder ablation and what it is testing",
                     "what survives and what does not"))
    return "".join(out)


def chapter_attribution_figures():
    """The 27 per-configuration attribution plots. Most belong in an appendix."""
    out = ["% Appendix: per-configuration attribution figures.\n"
           "% 27 of these is too many for a results chapter. Promote the two or three you\n"
           "% actually discuss into Chapter 6 and leave the rest here as evidence.\n\n"]
    for akey, aname in ALGOS:
        for dkey, dname in DOMAINS:
            for vkey, vname in VARIANTS:
                stem = "%s_%s_%s" % (akey, dkey, vkey)
                if not have(stem):
                    continue
                out.append(fig("%s.png" % stem,
                               "Top attributed variables, %s, %s, %s."
                               % (aname, dname, vname),
                               stem.replace("_", "-"),
                               "(appendix -- caption may be enough)",
                               "(appendix -- caption may be enough)"))
    return "".join(out)


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    os.chdir(root)
    dest = os.path.join("Thesis", "skeleton")
    if not os.path.isdir(dest):
        os.makedirs(dest)

    files = {
        "ch5_floats.tex": chapter5(),
        "ch6_floats.tex": chapter6(),
        "appendix_attributions.tex": chapter_attribution_figures(),
    }
    for name, body in files.items():
        with io.open(os.path.join(dest, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(body)

    total_w = sum(b.count("% WRITE") for b in files.values())
    n_fig = sum(b.count("includegraphics") for b in files.values())
    n_tab = sum(b.count("\\input{tables/") for b in files.values())
    print("wrote %d files to %s\n" % (len(files), dest))
    for name, body in sorted(files.items()):
        print("  %-28s %2d figures  %2d tables  %2d writing prompts"
              % (name, body.count("includegraphics"), body.count("\\input{tables/"),
                 body.count("% WRITE")))
    print("\n  %d figures and %d tables placed, %d paragraphs to write." % (n_fig, n_tab, total_w))

    placed = set()
    for b in files.values():
        for line in b.split("\n"):
            if "includegraphics" in line:
                placed.add(line.split("{")[-1].rstrip("}"))
    all_figs = {os.path.basename(p) for p in glob.glob("Thesis/figures/*.png")}
    missing = sorted(all_figs - placed)
    if missing:
        print("\n  not placed anywhere (%d) -- decide whether each earns a page:" % len(missing))
        for m in missing:
            print("    %s" % m)

    print("\n  Nothing in Thesis/chapters/ was touched. Copy across what you want.")
    print("  A chapter is finished when it has no %% WRITE markers left.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
