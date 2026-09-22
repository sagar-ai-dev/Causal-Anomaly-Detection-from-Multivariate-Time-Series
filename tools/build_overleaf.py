"""
Bundle the LaTeX sources for Overleaf.

Overleaf wants the document and nothing else. The repository carries the pipelines,
the cached graphs, the verification tools and the notes that were written while the
thesis was being built, and none of that compiles or belongs in an upload. This
packages exactly what main.tex reaches: the root file, the bibliography, and the
frontmatter, chapters, appendices, tables and figures it pulls in.

Left out on purpose:
    Thesis/tools/        make_tables.py and the checkers. The tables they generate
                         are included as .tex, so Overleaf never needs to run them.
    Thesis/*.md          working notes, scaffolds and the defence preparation.
    everything above Thesis/, which is the research code.

The gates run first and the bundle is refused if any fails, because an archive that
compiles but quotes a number no CSV supports is the failure mode worth preventing.
Figures no \\includegraphics reaches are reported rather than silently shipped: 51
PNGs accumulated over the project and not all of them are still in the document.

Writes Thesis_overleaf.zip beside the repository. Upload it to Overleaf with
"New Project -> Upload Project", and set main.tex as the compile target.

Usage:  python tools/build_overleaf.py [--skip-checks]
"""
import io
import os
import re
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THESIS = os.path.join(ROOT, "Thesis")
OUT = os.path.join(ROOT, "Thesis_overleaf.zip")

# Everything main.tex can reach. tools/ is absent by intent, not by oversight.
DIRS = ["frontmatter", "chapters", "appendices", "tables", "figures"]
ROOT_FILES = ["main.tex", "refs.bib"]

GATES = [
    ("Thesis/tools/check_latex.py", "every cross-reference, label and float resolves"),
    ("tools/verify_docs.py", "every decimal figure traces to a generated CSV"),
    ("tools/verify_integers.py", "every counted claim traces to a generated CSV"),
    ("tools/audit_pipeline.py", "the 27 configurations remain uniform downstream"),
    ("tools/verify_claims.py", "the attribution claims are true, not just traceable"),
]

GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def run_gates():
    ok = True
    for script, what in GATES:
        r = subprocess.run([sys.executable, script], cwd=ROOT,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        mark = "PASS" if r.returncode == 0 else "FAIL"
        print("  [%s] %-28s %s" % (mark, os.path.basename(script), what))
        if r.returncode != 0:
            ok = False
            print(r.stdout.decode("utf-8", "replace")[-1500:])
    return ok


def referenced_figures():
    """Every image main.tex actually asks for, by basename without extension."""
    wanted = set()
    for sub in DIRS + [""]:
        d = os.path.join(THESIS, sub)
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if not name.endswith(".tex"):
                continue
            body = io.open(os.path.join(d, name), encoding="utf-8").read()
            for m in GRAPHIC.finditer(body):
                wanted.add(os.path.splitext(os.path.basename(m.group(1)))[0])
    return wanted


def main():
    skip = "--skip-checks" in sys.argv
    if skip:
        print("checks skipped by request\n")
    else:
        print("running the gates before bundling:")
        if not run_gates():
            print("\nrefusing to bundle: a gate failed.")
            return 1
        print()

    wanted = referenced_figures()
    used, unused = [], []
    for name in sorted(os.listdir(os.path.join(THESIS, "figures"))):
        stem, ext = os.path.splitext(name)
        if ext.lower() == ".tex":
            used.append(name)          # \input figures, reached by name not by image
        elif stem in wanted:
            used.append(name)
        else:
            unused.append(name)

    if os.path.exists(OUT):
        os.remove(OUT)

    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ROOT_FILES:
            z.write(os.path.join(THESIS, f), f)
            n += 1
        for sub in DIRS:
            d = os.path.join(THESIS, sub)
            for name in sorted(os.listdir(d)):
                if sub == "figures" and name not in used:
                    continue
                if not name.lower().endswith((".tex", ".png", ".pdf", ".jpg", ".bib")):
                    continue
                z.write(os.path.join(d, name), sub + "/" + name)
                n += 1

    size = os.path.getsize(OUT) / 1024.0 / 1024.0
    print("wrote %s" % OUT)
    print("  %d files, %.1f MB" % (n, size))
    for sub in DIRS:
        d = os.path.join(THESIS, sub)
        c = len([x for x in os.listdir(d)
                 if (sub != "figures" or x in used)
                 and x.lower().endswith((".tex", ".png", ".pdf", ".jpg", ".bib"))])
        print("    %-14s %3d" % (sub + "/", c))
    if unused:
        print("\n  %d figures in Thesis/figures are not reached by any "
              "\\includegraphics and were left out:" % len(unused))
        for name in unused:
            print("      %s" % name)
    print("\nUpload to Overleaf with New Project -> Upload Project, then set "
          "main.tex as the compile target.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
