"""
Build the submission archive.

Packages the repository as `Causal_anomaly_detection_from_multivariate_time_series.zip` in the parent directory,
excluding the virtualenv, git metadata, bytecode caches and any previous archive. Everything
else ships, including the cached causal graphs and the generated outputs, so the reviewer can
read every result without running anything.

Before packaging it re-runs every check that gates correctness and refuses to build if any
fails:

    pyflakes                        over every script in the archive
    tools/verify_docs.py            every decimal figure quoted anywhere traces to a CSV
    tools/verify_integers.py        every counted claim quoted anywhere traces to a CSV
    tools/audit_pipeline.py         the 27 configurations differ only by the three permitted axes
    tools/validate_algorithms.py    RLS and the GES edge rule match closed-form answers
    tools/check_lab_requirements.py the XAI lab attribution requirement still holds

Shipping an archive whose documentation quotes numbers no CSV contains, or whose pipeline has
drifted out of uniformity, is the specific failure this is meant to prevent.

One archive is produced:

    Causal_anomaly_detection_from_multivariate_time_series.zip

It carries all code, the cached causal graphs, every generated output and figure, the full
dissertation source, and the verification tools -- everything needed to read and check the
results without rerunning anything.

It excludes the raw sensor inputs, for two reasons. They are triplicated, since each of the
PCMCI, GES and CAM-UV folders keeps its own copy of the same Pepper, ECG and TEP files so
that each runs standalone, and they dominate at roughly 370 MB uncompressed. And they are
not needed to verify a single number in the dissertation, because the graphs are cached and
the outputs are all present. PTB-XL and the Tennessee Eastman data are publicly available;
the Pepper recordings come from the XAI course laboratory.

Usage:  python tools/build_submission.py [--skip-checks]
"""

import io
import os
import subprocess
import sys
import zipfile

# The archive is named for the dissertation it accompanies, so that a file sitting in an
# examiner's downloads folder identifies itself. The title is not retyped here: it is read
# off the title page, and the build stops if that page no longer says what this expects,
# rather than shipping an archive named after a title the document does not carry.
TITLE = "Causal anomaly detection from multivariate time series"
NAME = TITLE.replace(" ", "_") + ".zip"
NAME_CODE = TITLE.replace(" ", "_") + "_code_only.zip"
EXCLUDE_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
                # Agent and editor state. Not part of the work, and not something an
                # examiner should be handed.
                ".claude", ".vscode", ".idea"}
EXCLUDE_EXT = {".pyc", ".pyo",
               # Every archive, not merely the one being written. A Thesis.zip cut for an
               # Overleaf upload was being packaged inside the submission, so the archive
               # carried a second, older copy of the dissertation. A reviewer opening the
               # nested one would have read superseded text and had no way to know it.
               ".zip"}

# Raw inputs, dropped from the code-only archive. Cached graphs (*.npz) and everything
# under outputs/ are kept, so all results stay readable.
DATA_DIRS = ("pepper_csv", "Healthcare dataset", "TEP")
DATA_EXT = (".csv", ".pkl")


def is_raw_data(rel):
    parts = rel.replace("\\", "/").split("/")
    if "outputs" in parts:
        return False
    return any(d in parts for d in DATA_DIRS) and rel.endswith(DATA_EXT)


def gather(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1] in EXCLUDE_EXT or fn == NAME:
                continue
            full = os.path.join(dirpath, fn)
            yield full, os.path.relpath(full, root)


def check(cmd, label):
    print(f"  {label} ... ", end="", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    ok = r.returncode == 0 and not r.stdout.strip().startswith("UNACCOUNTED")
    if label.startswith("pyflakes"):
        ok = not r.stdout.strip()
    print("ok" if ok else "FAILED")
    if not ok:
        print((r.stdout + r.stderr).strip()[:2000])
    return ok


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.chdir(root)

    # The archive is named for the title page, so confirm the title page still carries it.
    # A renamed dissertation shipping under its old name is a small error that survives all
    # the way to the examiner's desk, and it costs nothing to rule out here.
    page = os.path.join("Thesis", "frontmatter", "titlepage.tex")
    if os.path.exists(page):
        flat = " ".join(io.open(page, encoding="utf-8").read().split())
        for fragment in ("Causal anomaly detection from", "multivariate time series"):
            if fragment not in flat:
                print("title page no longer reads %r." % fragment)
                print("Update TITLE in this script to match it, then rebuild.")
                return 1

    if "--skip-checks" not in sys.argv:
        print("pre-build checks")
        scripts = [f for f, _ in gather(root) if f.endswith(".py")]
        ok = check([sys.executable, "-m", "pyflakes"] + scripts, "pyflakes over every script")
        ok &= check([sys.executable, "tools/verify_docs.py"], "every quoted figure traces to a CSV")
        ok &= check([sys.executable, "tools/verify_integers.py"], "every counted claim traces to a CSV")
        ok &= check([sys.executable, "tools/audit_pipeline.py"], "27 configurations differ only by the three axes")
        ok &= check([sys.executable, "tools/validate_algorithms.py"], "algorithms match closed-form answers")
        ok &= check([sys.executable, "tools/check_lab_requirements.py"], "XAI lab attribution requirement holds")
        ok &= check([sys.executable, "tools/verify_orientation.py"], "wrappers distinguish (obs, vars) from its transpose")
        if not ok:
            print("\nrefusing to build. Fix the above, or pass --skip-checks deliberately.")
            return 1

    files = sorted(gather(root), key=lambda x: x[1])
    parent = os.path.dirname(root)

    # One archive, not two.
    #
    # There used to be a full copy and a code-only copy, which meant the reviewer had to be
    # told which one to open and the two could drift apart. The raw sensor data is the only
    # thing that separated them, and it is the one part nobody needs in order to read the
    # results: the cached causal graphs, every generated output, every figure and the whole
    # dissertation are all here, so a reader can verify the numbers without rerunning
    # anything. The raw inputs are also triplicated -- each of the three algorithm folders
    # keeps its own copy of the same Pepper, ECG and TEP files, since each runs standalone --
    # so including them would triple the archive to no benefit.
    #
    # Where to get the raw data, if a full rerun is wanted, is documented in the README.
    subset = [f for f in files if not is_raw_data(f[1])]
    dropped = [f for f in files if is_raw_data(f[1])]

    out = os.path.join(parent, NAME)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for full, rel in subset:
            z.write(full, os.path.join("Anomaly_detection", rel))
    mb = os.path.getsize(out) / 1024 / 1024

    # Remove the superseded second archive so only one can ever be picked up.
    stale = os.path.join(parent, NAME_CODE)
    if os.path.exists(stale):
        os.remove(stale)
        print(f"\nremoved superseded {NAME_CODE}")

    print(f"\n{NAME}")
    print(f"  {len(subset)} files, {mb:.1f} MB"
          + ("  (fits a 25 MB mail attachment)" if mb < 25 else "  (share via Drive)"))
    print("  contains: all code, cached causal graphs, every generated output and figure,")
    print("            the full dissertation source, and the verification tools")
    print(f"  excludes: {len(dropped)} raw input files, triplicated across the three algorithm")
    print("            folders; see README for where to obtain them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
