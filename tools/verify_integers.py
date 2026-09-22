"""
Check that every counted claim in the dissertation exists in a generated CSV.

verify_docs.py covers the decimal figures. It says so in its own output, and until now that
disclaimer was the whole story: variable counts, sample counts, edge counts and ranks were
covered by nothing. Those are the claims a reader checks first, because they are the ones
that can be counted by hand -- 27 configurations, 244 variables, 54 of 108 comparisons, the
78th-ranked sensor. A wrong one is not a rounding difference, it is a wrong statement about
the experiment.

The method is the same as its decimal sibling: build a pool from every integer-valued cell
in every pipeline CSV, pull every integer out of the documents, and require each to be in
the pool or in ALLOW with a reason.

What is deliberately NOT scanned, because it is markup rather than argument:

    cite, ref, label and input keys   runge2019pcmci is a citation key, not the year 2019
    optional arguments               [htbp] and [0.35em] are placement and spacing
    texttt spans                     LaserF01X is a sensor name, not the number 01
    verbatim blocks                  Appendix C is captured gate output. It is a transcript
                                     of what ran, checked by rerunning the gate, not prose
    single digits                    "three axes" and "9 of 9" read better spelled or in
                                     context, and one-digit tokens are almost all prose

Digit grouping is undone before scanning, so the 7{,}143 samples the chapters typeset are
checked as 7143. Written the other way the thousands separator hid the number from the
scan entirely, which would have left the largest counts in the document unchecked.

Usage:  python tools/verify_integers.py
"""

import csv
import glob
import io
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Integers quoted in the argument that no CSV holds, each with the reason it is right.
# A number belongs here only when it can be checked another way and has been.
ALLOW = {
    "564": "candidate healthcare edges before sparsification: 12*12*3 lagged plus 12*11 "
           "contemporaneous = 432 + 132. A structural count of the search space, not a "
           "measurement, so no CSV stores it",
    "281": "ratio of KneePicthTemp's fault-free baseline to LedEarR3's, 7.5380 / 0.0268, "
           "derived in the sentence that quotes it",
    "976": "datasets in Schmidl et al., read from that paper rather than produced here",
    "512164": "the author's matricola on the title page",
    "2025": "the academic year on the title page",
    "2026": "the academic year on the title page",
    "8250": "the model number in 'Intel Core i5-8250U', the processor Appendix A "
            "records as having produced these results. A name, not a measurement",
}

# Markup whose digits are identifiers, placement or spacing rather than claims.
STRIP = [
    re.compile(r"\\begin\{(verbatim|lstlisting)\}.*?\\end\{\1\}", re.S),
    re.compile(r"\\(?:cite|ref|label|input|includegraphics|texttt|eqref|autoref)\*?\s*\{[^}]*\}"),
    re.compile(r"\[[^\]]*\]"),
    re.compile(r"%.*"),
]

# 7{,}143 and 7\,143 are one number typeset with a thousands separator.
GROUP = re.compile(r"(?<=\d)(?:\{,\}|\\,)(?=\d)")

INT = re.compile(r"(?<![.,\d])\d{2,6}(?![.,\d])")


def pool():
    """Every integer a pipeline wrote, including the rounded reading of a decimal."""
    vals, seen = set(), []
    sources = (glob.glob("*/*/outputs/metrics_summary.csv")
               + glob.glob("*/*/outputs_leads8/metrics_summary.csv")
               + glob.glob("Baselines/outputs/*.csv")
               + glob.glob("CAM-UV/*.csv")
               + glob.glob("CAM-UV/*/*.csv")
               + glob.glob("tools/*.csv"))
    for p in sources:
        if not os.path.exists(p):
            continue
        seen.append(p)
        try:
            for row in csv.DictReader(io.open(p, encoding="utf-8")):
                for v in row.values():
                    cell = (v or "").strip()
                    try:
                        f = float(cell)
                    except ValueError:
                        continue
                    if f == int(f):
                        vals.add(str(int(f)))
                    vals.add(str(int(round(f))))
                    # a share stored as 0.982 answers a claim of 98 per cent
                    vals.add(str(int(round(f * 100))))
        except Exception:
            continue
    return vals, seen


def main():
    os.chdir(ROOT)
    vals, seen = pool()
    docs = (sorted(glob.glob("Thesis/chapters/*.tex"))
            + sorted(glob.glob("Thesis/frontmatter/*.tex"))
            + sorted(glob.glob("Thesis/appendices/*.tex")))

    total, misses = 0, []
    for d in docs:
        raw = io.open(d, encoding="utf-8").read()
        for pat in STRIP:
            raw = pat.sub(" ", raw)
        raw = GROUP.sub("", raw)
        for n in INT.findall(raw):
            total += 1
            if n not in vals and n not in ALLOW:
                misses.append((d, n))

    print("value pool: %d distinct integers from %d CSVs" % (len(vals), len(seen)))
    print("documents:  %d" % len(docs))
    print("checked:    %d counted claims" % total)
    print("scope:      integers of two to six digits, thousands separators undone,")
    print("            outside citation keys, optional arguments, typewriter spans and")
    print("            captured gate transcripts. Single digits are not scanned.")

    if misses:
        distinct = sorted(set(n for _d, n in misses), key=lambda x: (len(x), x))
        print("\nUNACCOUNTED (%d occurrences, %d distinct):" % (len(misses), len(distinct)))
        for d, n in misses:
            print("  %s: %s" % (d, n))
        print("\nAdd to ALLOW with the reason it is right, or correct the document.")
        return 1

    print("\nall %d counted claims trace to a generated CSV or a documented exception"
          % total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
