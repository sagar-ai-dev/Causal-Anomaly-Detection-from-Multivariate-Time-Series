"""
Settle the title page.

Three changes were requested. Two apply; the third had nothing to act on, and this script
says so rather than reporting success for a no-op.

  title       The three candidates in the comment block at the foot of titlepage.tex are
              replaced by the chosen one. The alternatives stay in the file as a comment,
              because the reasoning for the choice is worth keeping.

  candidate   AUTHOR NAME becomes the real name. The matricola stays a placeholder until
              the number is supplied -- see the warning this prints, and the build check
              that now refuses to let a placeholder reach a submitted document.

  draft mode  No draft option exists. main.tex reads

                  \\documentclass[12pt,a4paper,twoside,openright]{report}

              and the string "draft" appears nowhere under Thesis/. graphicx is loaded
              normally with \\graphicspath{{figures/}}, so figures are not being suppressed
              by a class option. If figures are rendering as empty frames, the cause is
              elsewhere -- most often the figures directory not being uploaded alongside
              the sources.

Each edit is anchored to exact text and asserted, so a stale anchor fails loudly instead
of leaving the page unchanged while the script claims otherwise. Idempotent.

Usage:  python Thesis/tools/finalise_titlepage.py [--apply]
"""

import io
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

TITLEPAGE = "Thesis/frontmatter/titlepage.tex"
MAIN = "Thesis/main.tex"

MATRICOLA_PLACEHOLDER = "[INSERT YOUR VR NUMBER HERE]"

OLD_TITLE = """{\\LARGE\\bfseries
WHEN DOES CAUSAL STRUCTURE\\\\[0.35em]
HELP DETECT ANOMALIES?\\par}
\\vspace{0.9cm}
{\\large A Controlled Comparison of Three Causal Discovery\\\\[0.25em]
Algorithms Across Robotics, Healthcare\\\\[0.25em]
and Industrial Process Data\\par}"""

NEW_TITLE = """{\\LARGE\\bfseries
Causal anomaly detection from\\\\[0.35em]
multivariate time series\\par}"""

OLD_TODO = """%  TODO before submitting:
%    - confirm your name spelling and matricola number
%    - confirm the academic year
%    - settle the title (three candidates are listed at the bottom of
%      this file; the active one is the first)"""

NEW_TODO = """%  TODO before submitting:
%    - the matricola is still a placeholder. check_latex.py fails the
%      build while it remains, so this cannot be submitted by accident
%    - confirm the academic year
%  The title is settled; the alternatives are kept at the foot of this
%  file so the choice stays legible."""

EDITS = [
    (TITLEPAGE, OLD_TITLE, NEW_TITLE, "title"),
    # The `{}` is load-bearing. LaTeX skips whitespace, including a newline, when it
    # looks for the optional argument of `\\`, so `\\` followed by a line opening with
    # `[` is read as a line break whose vertical space is the bracketed text. With a
    # placeholder in the brackets that is not a dimension and the build fails; with a
    # bare number it would silently insert vertical space instead of printing it. The
    # empty group ends the `\\` command before the bracket is reached.
    (TITLEPAGE, "  AUTHOR NAME\\\\\n  (VR\\,MATRICOLA)",
     "  Sagar Lal\\\\\n  {}" + MATRICOLA_PLACEHOLDER, "candidate name and matricola"),
    (TITLEPAGE, OLD_TODO, NEW_TODO, "the to-do block"),
]


def main():
    apply = "--apply" in sys.argv
    os.chdir(ROOT)

    applied, already, missing = 0, 0, []

    for path, old, new, what in EDITS:
        text = io.open(path, encoding="utf-8", newline="").read()
        if new in text:
            already += 1
            print("  already done   %s" % what)
            continue
        if old not in text:
            missing.append((path, what))
            print("  ANCHOR MISSING %s" % what)
            continue
        text = text.replace(old, new, 1)
        if apply:
            io.open(path, "w", encoding="utf-8", newline="").write(text)
        applied += 1
        print("  updated        %s" % what)

    # Third request: there is no draft option to remove.
    main_src = io.open(MAIN, encoding="utf-8").read()
    cls = [l for l in main_src.split("\n") if l.startswith("\\documentclass")]
    print("\n  draft mode:")
    print("    %s" % (cls[0] if cls else "no \\documentclass found"))
    if "draft" in main_src:
        print("    a 'draft' string is present -- inspect it by hand")
    else:
        print("    no draft option present, so nothing to remove; figures are not")
        print("    being suppressed by a class option")

    print("\n  applied %d, already done %d, anchors missing %d"
          % (applied, already, len(missing)))

    if missing:
        for path, what in missing:
            print("    %s :: %s" % (path, what))
        return 1

    if apply:
        print("\n  WARNING: the matricola is the placeholder %s." % MATRICOLA_PLACEHOLDER)
        print("  A placeholder on a title page is the kind of thing that ships, so")
        print("  check_latex.py now fails while it is there.")
    else:
        print("\n  dry run; pass --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
