r"""
Package the dissertation for Overleaf as Thesis_overleaf.zip in the repository root.

Overleaf compiles whatever it is given, so a file left out of the archive does not
fail here -- it fails there, as a missing \input or a blank figure box, after the
upload. The archive is therefore checked against the source before it is written:
every \input, \include and \includegraphics target reachable from main.tex must
resolve to a file going into the zip, or nothing is written at all.

Tooling is excluded on purpose. tools/, skeleton/ and the working notes are how the
document is produced, not part of it, and Overleaf has no use for them.

Run:  python Thesis/tools/make_overleaf_zip.py
"""

import io
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
THESIS = os.path.dirname(HERE)
ROOT = os.path.dirname(THESIS)
OUT = os.path.join(ROOT, "Thesis_overleaf.zip")

DIRS = ("frontmatter", "chapters", "appendices", "tables", "figures")
ROOT_FILES = ("main.tex", "refs.bib")

INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def strip_comments(text):
    out = []
    for line in text.split("\n"):
        cut, esc = None, False
        for i, ch in enumerate(line):
            if ch == "\\":
                esc = not esc
            elif ch == "%" and not esc:
                cut = i
                break
            else:
                esc = False
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def collect():
    """Every file the archive will carry, as paths relative to Thesis/."""
    members = [f for f in ROOT_FILES if os.path.exists(os.path.join(THESIS, f))]
    for d in DIRS:
        full = os.path.join(THESIS, d)
        if not os.path.isdir(full):
            continue
        for name in sorted(os.listdir(full)):
            if name.lower().endswith((".tex", ".png", ".pdf", ".jpg", ".jpeg", ".bib")):
                members.append(d + "/" + name)
    return sorted(members)


def unresolved(members):
    """
    Targets referenced from the archived .tex files that the archive lacks.

    main.tex sets a graphics path, so a figure is written as
    {fig_density_vs_auc.png} rather than {figures/fig_density_vs_auc.png} and a
    literal path comparison reports all 42 of them missing. A target therefore
    counts as resolved when its basename is present anywhere in the archive,
    with or without its extension, which is what the graphics path does for
    figures and what a relative input does for the rest.
    """
    have, stems = set(), set()
    for m in members:
        have.add(m)
        if "." in m:
            have.add(m.rsplit(".", 1)[0])
        base = m.split("/")[-1]
        stems.add(base)
        if "." in base:
            stems.add(base.rsplit(".", 1)[0])
    missing = []
    for m in members:
        if not m.endswith(".tex"):
            continue
        text = strip_comments(io.open(os.path.join(THESIS, m),
                                      encoding="utf-8", errors="replace").read())
        for pat in (INPUT, GRAPHIC):
            for target in pat.findall(text):
                t = target.replace(chr(92), "/").strip()
                if t in have or t + ".tex" in have or t + ".png" in have:
                    continue
                base = t.split("/")[-1]
                if base in stems:
                    continue
                missing.append((m, target))
    return missing


def main():
    members = collect()
    if "main.tex" not in members:
        print("main.tex not found under %s" % THESIS)
        return 1

    gaps = unresolved(members)
    if gaps:
        print("targets referenced but not in the archive:")
        for src, target in gaps:
            print("  %-34s %s" % (src, target))
        print("\nnothing written -- Overleaf would fail on these after the upload.")
        return 1

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for m in members:
            z.write(os.path.join(THESIS, m), m)

    tex = sum(1 for m in members if m.endswith(".tex"))
    img = sum(1 for m in members if m.endswith(".png"))
    print("wrote %s" % OUT)
    print("  %d files: %d tex, %d png, refs.bib" % (len(members), tex, img))
    print("  %.1f MB" % (os.path.getsize(OUT) / 1048576.0))
    print("  every " + chr(92) + "input and " + chr(92)
          + "includegraphics target resolves inside the archive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
