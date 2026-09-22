"""
Check that every metric quoted in the documentation exists in a generated CSV.

A thesis README drifts from its results the moment a pipeline is rerun and a number is not
carried across by hand. This walks README.md and every FINAL_RESULTS.md, pulls out each
four-decimal figure, and confirms it appears in one of the CSVs the pipelines actually wrote.

Value pool:
    */*/outputs/metrics_summary.csv     the 27 causal configurations
    Baselines/outputs/*.csv             neural architectures and the trivial control
    tools/bootstrap_intervals.csv       block-bootstrap confidence intervals

Figures that are computed in prose rather than tabulated -- z-scores, correlations, retained
variance, condition numbers, lead-identity errors -- are listed in ALLOW with a reason, so an
unexplained number is a failure rather than something to squint at.

Exit code is non-zero if anything is unaccounted for, so this can gate a commit.

Usage:  python tools/verify_docs.py
"""
import csv
import glob
import io
import os
import re
import sys

NUM = re.compile(r"\b\d\.\d{4}\b")
# Two-decimal figures were outside the net entirely: 12.42 candidate parents, 291.67
# samples per variable, 95.24 per cent prevalence. They are quoted in the argument and
# were checked only by hand. A fraction in a CSV and a percentage in the prose are the
# same claim, so the pool below carries both readings of every value.
NUM2 = re.compile(r"(?<![.\d])\d{1,3}\.\d{2}(?![\d.])")
# One-decimal figures were the last uncovered precision, and the runtime table quotes 27
# of them. Four digits before the point, because seconds run past 100 where the metrics
# never do.
NUM1 = re.compile(r"(?<![.\d])\d{1,4}\.\d(?![\d.])")
# Optional arguments carry layout numbers -- figure widths, [0.35em] skips, [htbp] --
# which are typography rather than claims and must not be scanned.
OPTARG = re.compile(r"\[[^\]]*\]")
# A width is typography, not a measurement: {0.45\\textwidth} must not be scanned.
WIDTH = re.compile(r"\d*\.?\d+\s*\\(?:text|line|column)width")
# A cross-reference is navigation, not a measurement. The markdown notes write section
# numbers as -5.3 and Section 5.3, which at one decimal are indistinguishable from a
# figure unless the reference itself is removed first.
SECREF = re.compile(r"(?:§|[Ss]ections?\s+|[Cc]hapters?\s+)\d+(?:\.\d+)*")
# A code listing is a reproduction of source, not a claim about a measurement. It is
# checked by being extracted from the running code at generation time, which is a
# stronger guarantee than matching its literals against a CSV would be.
CODE = re.compile(r"\\begin\{(lstlisting|verbatim)\}.*?\\end\{\1\}", re.S)
# Megabytes are a property of a file on disk, not a result read from a CSV.
UNIT = re.compile(r"\d+(?:\.\d+)?\s*(?:MB|GB|KB|kB|MiB|GiB)\b")

ALLOW = {
    # analytical Hanley-McNeil / bootstrap comparisons quoted in prose
    "0.5000": "chance level, referred to as 0.5",
    # Resolution of the class-level bootstrap: an interval excludes zero only when all
    # n classes agree in sign, so the smallest two-sided p the design can reach is
    # 2^(1-n). This is a property of the design and not a measurement, so it is derived
    # here rather than read from a CSV: 2^-4 = 0.0625 at the plant's five documented
    # actuators. The n=3 and n=14 values, 0.25 and 0.00012, are quoted alongside it.
    "0.0625": "2^(1-5), the smallest two-sided p reachable with five anomaly classes",
    # lab-reference reproduction, from _lab_reference*.py not a metrics CSV
    "0.9585": "lab script as supplied, F1 (see section 4)",
    "0.9204": "lab script recall / fixed-script recall (section 4)",
    "0.8927": "lab script with Ridge, F1 (section 4) -- also a CAM-UV AUC",
    "0.8812": "lab script with symmetric scoring, F1 (section 4)",
    "1.0000": "lab script precision before the Ridge fix (section 4)",
    "0.8667": "lab script precision after the Ridge fix (section 4)",
    "0.7876": "lab script recall under symmetric scoring (section 4)",
    # margins quoted in Thesis/RQ3_TUNED_BASELINES.md. Each is the difference of two
    # figures that are themselves in the pool, so the check cannot match them directly;
    # the arithmetic is stated here so it can be re-derived rather than trusted.
    "0.1299": "0.8114 GES healthcare 12-lead minus 0.6815 best untuned neural",
    "0.1037": "0.8114 GES healthcare 12-lead minus 0.7077 best tuned neural",
    "0.0104": "0.8856 best tuned neural robotics minus 0.8752 best causal",
    "0.0846": "0.9997 best tuned neural industrial minus 0.9151 best causal",
    "0.0341": "0.7418 GES healthcare 8-lead minus 0.7077 best tuned neural",
    "0.1280": "0.6355 PCMCI healthcare polynomial 8-lead minus 0.5075 at 12 leads",
    # ablations reported in prose, not tabulated as configurations
    "0.6184": "GES healthcare forced to subsample=1 (section 7.4)",
    "0.4992": "GES healthcare polynomial forced to subsample=1 (section 7.4)",
    "0.6813": "PCMCI robotics forced to subsample=10 (section 7.4)",
    "0.4856": "RidgeCV probe on PCMCI healthcare (section 2.2)",
    "0.6327": "GES healthcare on 8 independent leads, F1 (limitation 11)",
    "0.7667": "GES healthcare on 8 independent leads, AUC (limitation 11)",
    "0.6288": "expanding-window OLS swap, AUC (section 4)",
    "0.5644": "expanding-window OLS swap, F1 (section 4)",
    # neural search history
    "0.7072": "untuned LSTM default, F1 (section 5)",
    "0.7600": "untuned LSTM default, AUC (section 5)",
    "0.8000": "tuned LSTM in the earlier LSTM-only search",
    "0.8807": "tuned LSTM in the earlier LSTM-only search",
    "0.8566": "untuned LSTM default, AUC-PR",
    # AUC deltas quoted in section 7.1a -- differences between two tabulated figures
    "0.1757": "GES robotics linear AUC change from the feature-scaling fix",
    "0.0499": "CAM-UV industrial linear AUC change from the feature-scaling fix",
    "0.0244": "CAM-UV robotics linear AUC change from the feature-scaling fix",
    "0.0186": "GES industrial linear AUC change from the feature-scaling fix",
    # pre-fix figures quoted in the section 7.1b before/after table
    "0.8804": "PCMCI robotics RBF F1 before the intercept fix (also an LSTM AUC)",
    "0.9390": "PCMCI robotics RBF AUC-PR before the intercept fix",
    "0.8519": "PCMCI robotics RBF AUC-ROC before the intercept fix",
    # pre-fix figures quoted in the section 7.1a before/after table
    "0.7663": "GES robotics linear AUC before the feature-scaling fix",
    "0.6741": "CAM-UV robotics linear AUC before the feature-scaling fix",
    "0.8593": "GES industrial linear AUC before the feature-scaling fix",
    # per-folder decision thresholds (an operating point, not a score)
    "0.0128": "GES healthcare linear threshold",
    "0.0222": "GES healthcare polynomial threshold",
    "0.0283": "GES healthcare RBF threshold",
    "0.0005": "GES robotics RLS gain referenced in prose",
    "0.0002": "CAM-UV healthcare AUC delta, a difference not a score",
    # figures each folder README explicitly labels as superseded, kept as a correction record
    "0.9980": "superseded GES healthcare AUC, flagged in that README",
    "0.9576": "superseded GES industrial AUC range, flagged in that README",
    "0.9965": "superseded GES industrial AUC range, flagged in that README",
    "0.9218": "superseded GES robotics AUC, flagged in that README",
    "0.8331": "superseded CAM-UV robotics F1 from the 59-variable run, flagged in that README",
    "0.7048": "superseded CAM-UV robotics AUC from the 59-variable run, flagged in that README",
    # correlations, variance, misc prose
    "0.2770": "correlation between variable magnitude and drift (section 4)",
    "1.43": "ratio 10/7 of the two healthcare subsampling rates, derived in the prose",
    "7.75": "z under the analytical interval the study rejects, quoted to show it is too narrow",
    "26.67": "same, the healthcare analytical z range the bootstrap overturns",
    "94.55": "PCA variance retained at K=35, a preprocessing figure no metrics CSV carries",
    "99.50": "superseded GES industrial F1, the README says so in the same sentence",
    "95.85": "superseded GES robotics F1, the README says so in the same sentence",
    # One-decimal figures that are derived in place or read from outside the CSV pool.
    "93.3": "14 of 15 precordial pairs, the derivation given in the same table row",
    "95.6": "variance retained at K = 40, from the PCA sweep quoted beside K = 35 and "
            "K = 30 rather than from a metrics CSV",
    "70.5": "share of robotics pairs CAM-UV flagged as confounded, quoted in the "
            "industrial README as the contrast that makes CAM-UV competitive there "
            "and unusable on robotics. It belongs to the robotics run, and no CSV in "
            "this repository stores it",
    "23.6": "the RBF penalty on PCMCI industrial, 151.5 minus 127.9, subtracted in the "
            "sentence that quotes it",
}


def pool():
    vals, seen, numeric = set(), [], set()
    sources = (glob.glob("*/*/outputs/metrics_summary.csv")
               + glob.glob("Baselines/outputs/*.csv")
               + glob.glob("CAM-UV/*.csv")
               + glob.glob("CAM-UV/*/*.csv")
               + glob.glob("*/*/outputs_leads8/metrics_summary.csv")
               + glob.glob("tools/*.csv"))
    for p in sources:
        if not os.path.exists(p):
            continue
        seen.append(p)
        for row in csv.DictReader(io.open(p, encoding="utf-8")):
            for v in row.values():
                cell = str(v).strip() if v else ""
                if not cell:
                    continue
                if NUM.fullmatch(cell):
                    vals.add(cell)
                try:
                    numeric.add(float(cell))
                except ValueError:
                    pass
    vals2, vals1 = set(), set()
    for f in numeric:
        vals2.add('%.2f' % f)
        vals2.add('%.2f' % (f * 100.0))
        vals2.add('%.2f' % (f / 100.0))
        vals1.add('%.1f' % f)
        vals1.add('%.1f' % (f * 100.0))
        vals1.add('%.1f' % (f / 100.0))
    return vals, seen, vals2, vals1


def main():
    vals, seen, vals2, vals1 = pool()
    # Thesis/*.md is included deliberately. Those files carry the corrected mathematics,
    # the algorithm-selection argument and the tuned-baseline comparison, and every figure
    # in them was transcribed by hand from a CSV -- which is exactly the transcription this
    # script exists to police. Exempting the documents that argue the thesis would defeat
    # the point of having the check at all.
    docs = (["README.md", "Baselines/README.md"]
            + sorted(glob.glob("*/*/FINAL_RESULTS.md"))
            + sorted(glob.glob("*/*/README.md"))
            + sorted(glob.glob("Thesis/*.md"))
            + sorted(glob.glob("Thesis/chapters/*.tex"))
            + sorted(glob.glob("Thesis/frontmatter/*.tex"))
            + sorted(glob.glob("Thesis/appendices/*.tex")))
    total = bad = 0
    misses = []
    for d in docs:
        if not os.path.exists(d):
            continue
        raw = io.open(d, encoding="utf-8").read()
        raw = CODE.sub(" ", raw)
        for n in NUM.findall(raw):
            total += 1
            if n not in vals and n not in ALLOW:
                bad += 1
                misses.append((d, n))
        stripped = UNIT.sub(" ", SECREF.sub(" ", WIDTH.sub(" ", OPTARG.sub(" ", CODE.sub(" ", raw)))))
        for n in NUM2.findall(stripped):
            total += 1
            if n not in vals2 and n not in vals and n not in ALLOW:
                bad += 1
                misses.append((d, n))
        for n in NUM1.findall(stripped):
            total += 1
            if n not in vals1 and n not in ALLOW:
                bad += 1
                misses.append((d, n))
    print(f"value pool: {len(vals)} distinct figures from {len(seen)} CSVs")
    print(f"documents:  {len([d for d in docs if os.path.exists(d)])}")
    print(f"checked:    {total} quoted figures")
    print("scope:      four-, two- and one-decimal figures, the last two matched against")
    print("            both the fraction and the percentage reading of each CSV value.")
    print("            Integer claims -- variable counts, sample counts, edge counts,")
    print("            ranks -- are covered by tools/verify_integers.py instead.")
    if misses:
        print(f"\nUNACCOUNTED ({len(misses)}):")
        for d, n in misses:
            print(f"  {d}: {n}")
        print("\nAdd to ALLOW with a reason, or correct the document.")
    else:
        print(f"\nall {total} quoted figures trace to a generated CSV or a documented"
              " exception")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())
