"""
Fix the three build warnings the Overleaf log reported by name.

  headheight        fancyhdr wants at least 13.6pt and the document gives it 12pt, so
                    every page with a running head reports the warning. The fix goes to
                    geometry rather than to a bare \\setlength, because geometry has
                    already computed the page layout by the time fancyhdr loads; setting
                    the length afterwards changes the header box without changing the
                    layout geometry reserved for it.

  float too large   Appendix B scales its figures by width alone. That is safe only while
                    every figure is wider than it is tall, and one is not:
                    baseline_shap_feature_importance.png has an aspect ratio of 2.52, so
                    at 0.7\\textwidth it wants 26.5cm of a 23.7cm text block and overflows
                    by the 114.93pt the log reports. Constraining height as well, with
                    keepaspectratio, makes the whole appendix safe rather than patching
                    the one figure that happens to fail today.

  empty journal     runge2023tigramite is a software package entered as @article, so
                    BibTeX finds no journal field and warns. It is not an article; @misc
                    with howpublished is the entry type for software.

Idempotent: each edit checks for its own result first.

Usage:  python Thesis/tools/fix_compile_warnings.py [--apply]
"""

import io
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
B = chr(92)


def load(p):
    return io.open(os.path.join(ROOT, p), encoding="utf-8", newline="").read()


def store(p, s):
    io.open(os.path.join(ROOT, p), "w", encoding="utf-8", newline="").write(s)


def fix_headheight(apply):
    p = "Thesis/main.tex"
    s = load(p)
    if "headheight" in s:
        print("  headheight   already set")
        return 0
    old = ("[a4paper,inner=3.5cm,outer=2.5cm,top=3cm,bottom=3cm]{geometry}")
    new = ("[a4paper,inner=3.5cm,outer=2.5cm,top=3cm,bottom=3cm,headheight=14pt]{geometry}")
    if old not in s:
        print("  headheight   ANCHOR MISSING in main.tex")
        return 1
    if apply:
        store(p, s.replace(old, new, 1))
    print("  headheight   14pt passed to geometry")
    return 0


def fix_appendix_floats(apply):
    p = "Thesis/appendices/B_full_results.tex"
    s = load(p)
    if "keepaspectratio" in s:
        print("  B floats     already constrained")
        return 0
    # Every figure in this appendix is a full-width plate; capping the height at
    # 0.78\textheight leaves room for the caption on the same page.
    pat = re.compile(re.escape(B + "includegraphics[width=") + r"([0-9.]+)"
                     + re.escape(B + "textwidth]"))
    n = len(pat.findall(s))
    if not n:
        print("  B floats     ANCHOR MISSING: no width-only includegraphics found")
        return 1
    out = pat.sub(lambda m: (B + "includegraphics[width=" + m.group(1) + B + "textwidth,"
                             + "height=0.78" + B + "textheight,keepaspectratio]"), s)
    if apply:
        store(p, out)
    print("  B floats     height capped on %d figures" % n)
    return 0


def fix_tigramite(apply):
    p = "Thesis/refs.bib"
    s = load(p)
    if "@misc{runge2023tigramite" in s:
        print("  tigramite    already @misc")
        return 0
    m = re.search(r"@article\{runge2023tigramite,(.*?)\n\}", s, re.S)
    if not m:
        print("  tigramite    ANCHOR MISSING in refs.bib")
        return 1
    entry = ("@misc{runge2023tigramite,\n"
             "  title        = {Tigramite: Causal inference and causal discovery for time\n"
             "                  series datasets},\n"
             "  author       = {Runge, Jakob and others},\n"
             "  year         = {2023},\n"
             "  howpublished = {Software package},\n"
             "  url          = {https://github.com/jakobrunge/tigramite}\n"
             "}")
    if apply:
        store(p, s[:m.start()] + entry + s[m.end():])
    print("  tigramite    @article -> @misc with howpublished")
    return 0


def main():
    apply = "--apply" in sys.argv
    os.chdir(ROOT)
    bad = 0
    bad += fix_headheight(apply)
    bad += fix_appendix_floats(apply)
    bad += fix_tigramite(apply)
    if bad:
        print("\n  %d anchor(s) not found; nothing written for those" % bad)
        return 1
    print("\n  dry run; pass --apply to write" if not apply else "\n  written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
