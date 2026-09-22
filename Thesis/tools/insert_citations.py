"""
Attach sources to claims that already had none.

Every reference used here was already in refs.bib and already relevant; what was missing
was the `\\cite` marker joining it to the sentence that needs it. Two groups:

  Chapter 1 had no citations at all. It asserts what the prevailing detection paradigm
  does, describes how a deep autoencoder behaves, and states Einthoven's identity. Each is
  a claim about the world rather than about this work, so each needs a source.

  Eight further references were sitting uncited while the thesis used what they describe.
  Two are datasets -- PTB-XL and the Tennessee Eastman process -- where citation is a
  condition of use rather than a courtesy. The rest are the methods the pipeline runs:
  recursive least squares, the ridge penalty, the moving-block bootstrap, the analytical
  Hanley-McNeil interval, the Hilbert-Schmidt criterion, and the precision-recall result
  that explains why AUC-PR is not comparable across domains.

Each edit is anchored to an exact sentence and asserted. A missed anchor raises rather
than silently doing nothing, because a script that reports success while changing nothing
is worse than one that fails: the citation would still be absent and nobody would look
again.

Idempotent: an anchor that already carries its citation is skipped.

Usage:  python Thesis/tools/insert_citations.py [--apply]
"""

import io
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# (file, anchor text, text to insert immediately after it, why)
EDITS = [
    # ---------------------------------------------------------------- Chapter 1
    ("Thesis/chapters/01_introduction.tex",
     "searching for excursions beyond expected bounds",
     "~\\cite{chandola2009survey,blazquezgarcia2021review}",
     "the claim that magnitude-based detection is the prevailing paradigm"),

    ("Thesis/chapters/01_introduction.tex",
     "returning a reconstruction error aggregated over a latent vector",
     "~\\cite{malhotra2016lstmad,su2019omnianomaly,audibert2020usad}",
     "the description of how reconstruction autoencoders report anomalies"),

    ("Thesis/chapters/01_introduction.tex",
     "The limb leads satisfy Einthoven's identity",
     "~\\cite{malmivuo1995bioelectromagnetism}",
     "Einthoven's identity, asserted here and in Chapters 5 and 7"),

    # ---------------------------------------------------------------- Chapter 3
    ("Thesis/chapters/03_background.tex",
     "To do so continuously it uses a recursive least squares filter",
     "~\\cite{haykin2002adaptive}",
     "the RLS recursion the detector is built on"),

    ("Thesis/chapters/03_background.tex",
     "Residual independence is assessed with the Hilbert-Schmidt independence criterion",
     "~\\cite{gretton2005hsic}",
     "the independence test CAM-UV uses"),

    ("Thesis/chapters/03_background.tex",
     "All three variants therefore apply an $L_2$ ridge penalty",
     "~\\cite{hoerl1970ridge}",
     "the ridge penalty applied in every regression variant"),

    ("Thesis/chapters/03_background.tex",
     "quantified with a moving-block bootstrap of 2000 draws",
     "~\\cite{kunsch1989blockbootstrap}",
     "the block bootstrap the confidence intervals come from"),

    ("Thesis/chapters/03_background.tex",
     "and AUC-PR depends directly on the positive rate",
     "~\\cite{davis2006prroc}",
     "the precision-recall result behind the incomparability argument"),

    # ---------------------------------------------------------------- Chapter 5
    ("Thesis/chapters/05_experimental_setup.tex",
     "The primary evaluation on the PTB-XL ECG dataset",
     "~\\cite{wagner2020ptbxl,goldberger2000physionet}",
     "PTB-XL and the PhysioNet platform paper its data use agreement requires"),

    ("Thesis/chapters/05_experimental_setup.tex",
     "governed by Einthoven's and Goldberger's equations",
     "~\\cite{malmivuo1995bioelectromagnetism,goldberger1942augmented}",
     "both named lead systems: Einthoven's identity and Goldberger's augmented leads"),

    ("Thesis/chapters/05_experimental_setup.tex",
     "The Tennessee Eastman Process (TEP) dataset",
     "~\\cite{downs1993tep,rieth2017tepdata}",
     "the TEP process definition and the simulation data actually used"),

    # ---------------------------------------------------------------- Chapter 6
    ("Thesis/chapters/06_results.tex",
     "rather than the analytical Hanley--McNeil interval",
     "~\\cite{hanley1982roc}",
     "the analytical interval the bootstrap is compared against"),
]


def main():
    apply = "--apply" in sys.argv
    os.chdir(ROOT)

    by_file = {}
    for path, anchor, cite, why in EDITS:
        by_file.setdefault(path, []).append((anchor, cite, why))

    inserted, skipped, missing = 0, 0, []

    for path, edits in by_file.items():
        text = io.open(path, encoding="utf-8", newline="").read()
        original = text
        print("\n%s" % path)
        for anchor, cite, why in edits:
            if anchor not in text:
                missing.append((path, anchor))
                print("   MISSING ANCHOR  %s" % anchor[:64])
                continue
            if anchor + cite in text:
                skipped += 1
                print("   already cited   %s" % why)
                continue
            # Guard against an anchor that occurs more than once: cite the first.
            if text.count(anchor) > 1:
                print("   note: anchor occurs %d times, citing the first"
                      % text.count(anchor))
            text = text.replace(anchor, anchor + cite, 1)
            inserted += 1
            print("   + %-46s %s" % (cite[6:-1][:44], why))
        if apply and text != original:
            io.open(path, "w", encoding="utf-8", newline="").write(text)

    print("\n  inserted %d, already present %d, anchors not found %d"
          % (inserted, skipped, len(missing)))
    if missing:
        print("\n  These anchors were not found. The wording they target has changed,")
        print("  so the citation was NOT inserted and the claim is still unsourced:")
        for path, anchor in missing:
            print("    %s :: %s" % (path, anchor[:70]))
        return 1
    if not apply:
        print("\n  dry run; pass --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
