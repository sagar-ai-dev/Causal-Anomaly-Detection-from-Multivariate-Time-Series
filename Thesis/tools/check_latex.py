"""
Structural validation of the dissertation source, without needing a LaTeX install.

Catches the three failures that waste the most time in Overleaf: unbalanced braces,
mismatched environments, and \\input targets that do not exist. None of these are subtle
once the compiler reports them, but the compiler reports them by line number in a merged
document, and finding the actual file takes longer than this check does.

Run before every upload:  python Thesis/tools/check_latex.py
"""

import glob
import io
import os
import re
import subprocess
import sys
from collections import Counter

BEGIN = re.compile(r"\\begin\{([A-Za-z]+\*?)\}")
END = re.compile(r"\\end\{([A-Za-z]+\*?)\}")
INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")


def strip_comments(text):
    """Remove LaTeX comments, respecting escaped percent signs."""
    out = []
    for line in text.split("\n"):
        cut, esc = None, False
        for j, c in enumerate(line):
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
            elif c == "%":
                cut = j
                break
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def brace_depth(text):
    """Final brace depth, and whether it ever went negative."""
    depth, floor, esc = 0, 0, False
    for c in text:
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            floor = min(floor, depth)
    return depth, floor




def empty_sections(files):
    """Sections whose whole body is a float, with no sentence anywhere in them.

    Appendix D.5 shipped in one compile as a heading followed immediately by a table and
    then the next appendix: a section with a title, a number in the table of contents, and
    nothing said in it. In the source the heading and the table sit two lines apart and
    look deliberate; on the page the omission is obvious.

    A section is bounded by the next heading of its own level or higher, so one built out
    of subsections is not empty. List environments count as body for the same reason: a
    limitations section that is a bulleted list says something. Only floats and the markup
    inside them are discounted.
    """
    bs = chr(92)
    rank = {'chapter': 0, 'section': 1, 'subsection': 2}
    head = re.compile(r'(?m)^' + re.escape(bs) + r'(chapter|section|subsection)[*]?{([^}]*)}')
    lists = ('itemize', 'enumerate', 'description')
    skip = tuple(bs + k for k in ('label', 'begin', 'end', 'centering', 'small', 'caption',
                                  'toprule', 'midrule', 'bottomrule', 'input',
                                  'includegraphics', 'resizebox', 'clearpage')) + ('%',)
    bad = []
    for f in files:
        lines = io.open(f, encoding='utf-8').read().split(chr(10))
        marks = [(i, head.match(l)) for i, l in enumerate(lines) if head.match(l)]
        for n, (i, m) in enumerate(marks):
            if m.group(1) == 'chapter':
                continue
            end = len(lines)
            for j, m2 in marks[n + 1:]:
                if rank[m2.group(1)] <= rank[m.group(1)]:
                    end = j
                    break
            depth, prose = 0, False
            for l in lines[i + 1:end]:
                t = l.strip()
                if t.startswith(bs + 'begin{'):
                    if not any(t.startswith(bs + 'begin{' + e) for e in lists):
                        depth += 1
                elif t.startswith(bs + 'end{'):
                    if not any(t.startswith(bs + 'end{' + e) for e in lists):
                        depth = max(0, depth - 1)
                elif not depth and t and not t.startswith(skip) and not head.match(l):
                    prose = True
                    break
            if not prose:
                bad.append((f.replace(chr(92), '/'), m.group(2)))
    return bad

def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.chdir(root)
    # skeleton/ holds scaffolds that main.tex never includes. Sweeping them in makes the
    # label check report a reference as resolving when the only definition sits in a file
    # that is not part of the document -- a false pass that reaches the PDF as "??".
    files = sorted(f for f in glob.glob("*.tex") + glob.glob("*/*.tex")
                   if "skeleton" not in f.replace("\\", "/").split("/"))
    if not files:
        print("no .tex files found")
        return 1

    problems = 0
    print("%-46s %7s %6s  %s" % ("file", "braces", "envs", "status"))
    print("-" * 78)
    for f in files:
        text = strip_comments(io.open(f, encoding="utf-8").read())
        depth, floor = brace_depth(text)
        begins, ends = BEGIN.findall(text), END.findall(text)
        envs_ok = Counter(begins) == Counter(ends)
        ok = depth == 0 and floor == 0 and envs_ok
        problems += not ok
        print("%-46s %7d %6d  %s" % (f, depth, len(begins), "ok" if ok else "PROBLEM"))
        if not envs_ok:
            unclosed = Counter(begins) - Counter(ends)
            unopened = Counter(ends) - Counter(begins)
            if unclosed:
                print("      unclosed environments: %s" % dict(unclosed))
            if unopened:
                print("      unopened environments: %s" % dict(unopened))
        if floor < 0:
            print("      a closing brace appears before its opening brace")
        elif depth != 0:
            print("      %d brace(s) left open" % depth)

    # Every \input and \includegraphics target must resolve THE WAY LATEX RESOLVES IT:
    # relative to the directory of the root document, not the file doing the including.
    # A chapter in chapters/ that writes \input{../tables/x} therefore points outside the
    # project entirely, compiles locally in an editor that guesses, and fails in Overleaf.
    missing = []
    for f in files:
        text = strip_comments(io.open(f, encoding="utf-8").read())
        for m in INPUT.findall(text):
            if not (os.path.exists(m) or os.path.exists(m + ".tex")):
                missing.append((f, m, "input"))
        for m in GRAPHIC.findall(text):
            # \graphicspath{{figures/}} means a bare name is looked up there too
            cands = [m, os.path.join("figures", m)]
            if not any(os.path.exists(c) or
                       any(os.path.exists(c + e) for e in (".png", ".pdf", ".jpg"))
                       for c in cands):
                missing.append((f, m, "graphic"))

    print("\n%d files checked, %d with structural problems" % (len(files), problems))
    if missing:
        print("\nunresolved targets:")
        for f, m, kind in missing:
            print("  %-40s %s -> %s" % (f, kind, m))
    else:
        print("all input and graphics targets resolve")

    # Cross-references. LaTeX does not fail on an undefined \ref -- it typesets "??" and
    # carries on, so a broken reference reaches the printed thesis silently. Forward
    # references to sections not yet written are expected during drafting and are reported
    # rather than treated as failures; what matters is that the list is visible and empties
    # before submission.
    label_re = re.compile(r"\\label\{([^}]+)\}")
    ref_re = re.compile(r"\\(?:ref|autoref|cref|Cref|eqref)\{([^}]+)\}")
    defined, used = set(), {}
    for f in files:
        text = strip_comments(io.open(f, encoding="utf-8").read())
        defined |= set(label_re.findall(text))
        for m in ref_re.findall(text):
            used.setdefault(m, set()).add(os.path.basename(f))
    dangling = {k: v for k, v in used.items() if k not in defined}
    print("\n%d labels defined, %d referenced" % (len(defined), len(used)))
    if dangling:
        print("unresolved cross-references (typeset as ?? in the PDF):")
        for k, v in sorted(dangling.items()):
            print("  %-32s referenced from %s" % (k, ", ".join(sorted(v))))
    else:
        print("every cross-reference resolves")

    # Citations, which fail the same silent way. An undefined key typesets as [?] and
    # bibtex reports it in a log nobody reads. A comma-separated \cite{a,b,c} counts as
    # three keys, so the list is split before checking.
    bib = next((c for c in ("Thesis/refs.bib", "refs.bib") if os.path.exists(c)), None)
    keys = set()
    if bib:
        keys = set(re.findall(r"@\w+\{([^,\s]+)\s*,", io.open(bib, encoding="utf-8").read()))
    cite_re = re.compile(r"\\(?:cite|citep|citet|autocite|parencite)\*?"
                         r"(?:\[[^\]]*\])*\{([^}]+)\}")
    cited = {}
    for f in files:
        text = strip_comments(io.open(f, encoding="utf-8").read())
        for group in cite_re.findall(text):
            for k in (x.strip() for x in group.split(",")):
                if k:
                    cited.setdefault(k, set()).add(os.path.basename(f))
    if not keys:
        print("\nno refs.bib found; citations not checked")
    else:
        bad = {k: v for k, v in cited.items() if k not in keys}
        print("\n%d bibliography keys, %d cited" % (len(keys), len(cited)))
        if bad:
            print("undefined citations (typeset as [?] in the PDF):")
            for k, v in sorted(bad.items()):
                print("  %-32s cited from %s" % (k, ", ".join(sorted(v))))
        else:
            print("every citation resolves")

    # Resolving a key is not the same as the command existing. main.tex loads
    # neither natbib nor biblatex, so IT_CMDS below are undefined control
    # sequences that fail the compile -- and the key inside them resolves
    # perfectly well, so the check above passes them.
    NATBIB_ONLY = ("citet", "citep", "citeauthor", "citeyear", "citealt",
                   "citealp", "textcite", "parencite", "autocite")
    preamble = ""
    if os.path.exists("main.tex"):
        whole = strip_comments(io.open("main.tex", encoding="utf-8").read())
        cut = whole.find(chr(92) + "begin{document}")
        preamble = whole[:cut if cut > 0 else len(whole)]
    has_pkg = ("natbib" in preamble) or ("biblatex" in preamble)
    undefined = []
    if not has_pkg:
        scan = [f for f in files if "chapters" in f.replace(chr(92), "/")
                or "appendices" in f.replace(chr(92), "/")
                or "frontmatter" in f.replace(chr(92), "/")]
        for f in scan + ["main.tex"]:
            text = strip_comments(io.open(f, encoding="utf-8").read())
            for cmd in NATBIB_ONLY:
                for m in re.finditer(re.escape(chr(92) + cmd) + r"(?![A-Za-z])", text):
                    undefined.append((os.path.basename(f), cmd))
    if undefined:
        print(chr(10) + "citation commands used without natbib or biblatex:")
        for f, cmd in sorted(set(undefined)):
            print("  %-30s %s%s" % (f, chr(92), cmd))
        print("  these are undefined control sequences; the key resolves but the "
              "compile does not")
    else:
        print("no citation command the preamble does not define")

    # Float hygiene. Two failures LaTeX will not report usefully: a generated table that
    # no chapter includes never appears in the PDF at all, and one included twice defines
    # its label twice, so every reference to it resolves to whichever copy came last.
    # Both had happened here and neither surfaced until they were looked for.
    gen = sorted(glob.glob("tables/tbl_*.tex"))
    doc = [f for f in files
           if "chapters" in f.replace("\\", "/") or "appendices" in f.replace("\\", "/")]
    inc = Counter()
    for f in doc:
        text = strip_comments(io.open(f, encoding="utf-8").read())
        for m in INPUT.findall(text):
            inc[os.path.basename(m)] += 1
    bad = [(os.path.basename(g)[:-4], inc.get(os.path.basename(g)[:-4], 0)) for g in gen]
    bad = [(n, c) for n, c in bad if c != 1]
    print("\n%d generated tables" % len(gen))
    if bad:
        print("tables not included exactly once:")
        for n, c in bad:
            print("  %-30s included %d time(s)" % (n, c))
    else:
        print("every generated table is included exactly once")

    # A float nothing refers to is placed wherever LaTeX likes and read by nobody.
    tab_orphan = sorted(k for k in defined if k.startswith("tab:") and k not in used)
    if tab_orphan:
        print("\ntable labels never referenced from the text:")
        for k in tab_orphan:
            print("  %s" % k)
    else:
        print("every table label is referenced at least once")

    # The same for figures, with one legitimate exception. Appendix plates are cited as
    # a range -- "Figures B.1 to B.27" -- so only the two endpoints carry a \ref, and
    # demanding one per figure would flag 30 correctly written references. A figure is
    # therefore reported only when it is neither referenced nor enclosed by referenced
    # figures in its own file. Figure 6.6 reached the PDF with nothing pointing at it,
    # which is how it came to sit three pages from the argument it belonged to.
    fig_orphan = []
    for f in doc:
        text = strip_comments(io.open(f, encoding="utf-8").read())
        seq = re.findall(r"\\label\{(fig:[^}]*)\}", text)
        for i, lab in enumerate(seq):
            if lab in used:
                continue
            covered = (any(x in used for x in seq[:i])
                       and any(x in used for x in seq[i + 1:]))
            if not covered:
                fig_orphan.append((os.path.basename(f), lab))
    if fig_orphan:
        print("\nfigures neither referenced nor inside a referenced range:")
        for f, k in fig_orphan:
            print("  %-26s %s" % (f, k))
    else:
        print("every figure is referenced or falls inside a referenced range")

    # Placeholder text that must never reach a submitted document. main.tex is scanned
    # too: its hypersetup block carried pdfauthor={AUTHOR NAME} into the PDF's
    # properties long after the title page had been filled in, because this gate only
    # looked at the files that produce visible pages. The title page is the
    # first thing an examiner sees and the last thing anyone rereads, so an unfilled slot
    # there is both the most damaging and the easiest to miss.
    PLACEHOLDERS = ("INSERT YOUR", "AUTHOR NAME", "VR\\,MATRICOLA", "TODO:", "XXX",
                    "FIXME", "LOREM IPSUM")
    holes = []
    for path in (["main.tex"] if os.path.exists("main.tex") else []) + sorted(
                       glob.glob(os.path.join("frontmatter", "*.tex"))
                       + glob.glob(os.path.join("chapters", "*.tex"))
                       + glob.glob(os.path.join("appendices", "*.tex"))):
        body = strip_comments(io.open(path, encoding="utf-8").read())
        for token in PLACEHOLDERS:
            if token in body:
                holes.append((path, token))

    if holes:
        print("\nplaceholder text still present outside comments:")
        for path, token in holes:
            print("  %-42s %s" % (path.replace("\\", "/"), token))
        print("  These reach the PDF. Fill them before submitting.")
        print("")
    # The attributed quantity is the coefficient drift ||w(t) - w_0||, not a
    # regression residual. The code calls its arrays err_*/residuals -- the true
    # residual e_t that _rls_update returns is discarded at the call site -- and
    # that naming leaked into Sections 4.7 and 6.5, which described the ranking as
    # a residual. That contradicts the premise of Section 4.6, where drift and
    # residual magnitude are the two things the detector exists to tell apart.
    ATTR_SLIPS = ("attack-window residual", "residual ranking",
                  "residual-based attribution", "root mean square residual")
    slips = []
    for path in sorted(glob.glob(os.path.join("chapters", "*.tex"))
                       + glob.glob(os.path.join("appendices", "*.tex"))):
        body = strip_comments(io.open(path, encoding="utf-8").read())
        for token in ATTR_SLIPS:
            if token in body:
                slips.append((path, token))
    if slips:
        print("the attribution is described as ranking residuals, not drift:")
        for path, token in slips:
            print("  %-42s %s" % (path.replace(chr(92), "/"), token))
        print("  It ranks ||w(t) - w_0||. Section 4.6 rests on the distinction.")
    else:
        print("attribution is described as ranking coefficient drift throughout")

    if not holes:
        print("\nno placeholder text in main.tex, frontmatter, chapters or appendices")


    # A TikZ style named after a reserved key silently redefines that key, and the
    # error surfaces far from its cause. Naming one "out" -- the outgoing angle in
    # to[out=30,in=150] -- made pgfkeys report "/tikz/out requires a value" and took
    # the rest of the picture down with it, then the enclosing resizebox divided by a
    # zero-size box and the chapter that included it lost its figure environment. The
    # cheapest possible guard against a whole class of failures that only a compiler
    # would otherwise reveal.
    RESERVED_TIKZ = {"at", "in", "out", "to", "pos", "scale", "shift", "rotate",
                     "anchor", "above", "below", "left", "right", "draw", "fill",
                     "text", "font", "name", "label", "node", "every", "color",
                     "opacity", "sloped", "midway", "looseness", "dashed", "dotted",
                     "solid", "thick", "thin", "xshift", "yshift", "clip", "bend",
                     "distance", "width", "height", "transform"}
    clashes = []
    for path in sorted(glob.glob(os.path.join("figures", "*.tex"))):
        src = io.open(path, encoding="utf-8").read()
        for m in re.finditer(r"(?m)^\s*([A-Za-z@ ]+?)/\.style\s*=", src):
            nm = m.group(1).strip()
            if nm.lower() in RESERVED_TIKZ:
                clashes.append((os.path.basename(path), nm))

    if clashes:
        print("\nTikZ style names that collide with reserved keys:")
        for fn, nm in clashes:
            print("  %-28s %s" % (fn, nm))
        print("  Rename them. A reserved key used as a style redefines the key.")
    else:
        print("no TikZ style shadows a reserved key")

    # Running heads. fancyhdr sets the left mark flush left and the right mark flush
    # right in one box with no separator, so two long marks do not push each other
    # apart -- they overlap, and the page prints one on top of the other. This is what
    # happened when the two-sided LE/RO slots were carried over to a one-sided layout
    # unchanged. The check measures the real chapter and section titles rather than
    # assuming, and only applies when both slots are actually in use.
    heads = []
    if os.path.exists("main.tex"):
        mt = re.sub(r"(?m)^%.*$", "", io.open("main.tex", encoding="utf-8").read())
        left = "fancyhead[L]" in mt or "fancyhead[LE]" in mt
        right = "fancyhead[R]" in mt or "fancyhead[RO]" in mt
        if left and right and "oneside" in mt:
            chaps, secs = [], []
            for f in sorted(glob.glob(os.path.join("chapters", "*.tex"))):
                src = io.open(f, encoding="utf-8").read()
                chaps += re.findall(r"\\chapter{([^}]*)}", src)
                secs += re.findall(r"\\section{([^}]*)}", src)
            if chaps and secs:
                # "Chapter N. " and "N.M. " prefixes, 4.6pt per character at small
                widest = (max(len(c) for c in chaps) + 11
                          + max(len(s) for s in secs) + 6) * 4.6
                if widest > 426:
                    heads.append((widest, max(chaps, key=len), max(secs, key=len)))

    if heads:
        w, c, s = heads[0]
        print("\nrunning heads will overlap: both marks are set on every page")
        print("  widest pair needs about %.0fpt of a 426pt line" % w)
        print("  chapter: %s" % c)
        print("  section: %s" % s)
        print("  Keep one mark, or shorten the titles.")
    elif os.path.exists("main.tex"):
        print("running heads cannot collide")

    # chapter* sets no page mark. Where the surrounding pagestyle is fancy the
    # header then keeps whichever mark was left standing, which is how page two
    # of the List of Abbreviations came to be titled "List of Tables" -- it
    # follows \listoftables, and inherited its mark. Only files included while
    # fancy is in force can show the fault, so the pagestyle in effect at each
    # \input is tracked rather than assumed; the abstract and acknowledgements
    # sit under \pagestyle{empty} and carry no header to get wrong.
    unmarked = []
    if os.path.exists("main.tex"):
        body = strip_comments(io.open("main.tex", encoding="utf-8").read())
        style = "plain"
        for line in body.split(chr(10)):
            m = re.search(r"\\pagestyle\{([^}]*)\}", line)
            if m:
                style = m.group(1).strip()
            m = re.search(r"\\(?:input|include)\{([^}]+)\}", line)
            if m and style == "fancy":
                target = m.group(1).strip()
                for cand in (target, target + ".tex"):
                    if os.path.exists(cand):
                        text = strip_comments(io.open(cand, encoding="utf-8").read())
                        for star in re.finditer(r"\\chapter\*\{([^}]*)\}", text):
                            tail = text[star.end():star.end() + 400]
                            if "markboth" not in tail and "markright" not in tail:
                                unmarked.append((cand, star.group(1)))
                        break
    if unmarked:
        print(chr(10) + "starred chapters under fancy that set no running head:")
        for f, title in unmarked:
            print("  %-34s %s" % (os.path.basename(f), title))
        print("  each inherits the previous mark; add "
              + chr(92) + "markboth after the " + chr(92) + "chapter*")
    else:
        print("every starred chapter under fancy sets its own running head")

    body = [f for f in files
            if f.replace(chr(92), "/").split("/")[0] in ("chapters", "appendices")]
    hollow = empty_sections(body)
    if hollow:
        print("" + chr(10) + "SECTIONS WITH NO PROSE (a heading, a float, and nothing said):")
        for f, t in hollow:
            print("  %s: %s" % (f, t))
    else:
        print("every section says something besides its own title")

    return 1 if (problems or missing or bad or holes or clashes or heads or hollow) else 0


if __name__ == "__main__":
    sys.exit(main())
