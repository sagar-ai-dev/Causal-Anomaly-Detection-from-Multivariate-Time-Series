"""
Stage only the files that changed, laid out the way Overleaf's folders are.

build_overleaf.py makes a complete archive for a fresh project. This is the other
case: an Overleaf project that already exists and must keep its URL, its history and
its share links. Uploading 96 files to replace 29 works but makes the project history
unreadable, so this copies out just what differs from a given commit.

The layout under Overleaf_update/ mirrors the project, so each subfolder's contents
can be dragged into the folder of the same name in Overleaf, which replaces the files
it already has and adds the ones it does not.

Usage:  python tools/stage_overleaf_update.py [SINCE_COMMIT]
        (default: 2049c6126, the last point an upload list was given)
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "Overleaf_update")
SINCE = sys.argv[1] if len(sys.argv) > 1 else "2049c6126"


def main():
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", SINCE, "HEAD", "--", "Thesis/"],
        cwd=ROOT, stdout=subprocess.PIPE, check=True).stdout.decode().split("\n")

    # Thesis/tools/ is the table generator and the checkers. Overleaf never runs them;
    # it reads the .tex they produce, which is in tables/.
    files = [f for f in changed
             if f.strip() and not f.startswith("Thesis/tools/")
             and not f.startswith("Thesis/skeleton/")
             and os.path.exists(os.path.join(ROOT, f))]

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)

    by_dir = {}
    for f in files:
        rel = f[len("Thesis/"):]
        dst = os.path.join(OUT, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(ROOT, f), dst)
        by_dir.setdefault(os.path.dirname(rel) or "(project root)", []).append(
            os.path.basename(rel))

    print("staged %d changed files in %s\n" % (len(files), OUT))
    for d in sorted(by_dir):
        names = sorted(by_dir[d])
        print("  %-14s %2d file%s" % (d + "/", len(names), "" if len(names) == 1 else "s"))
        for n in names:
            print("       %s" % n)
        print()

    # A file Overleaf does not have yet has to be added rather than replaced, and it is
    # the one that breaks the build if it is missed.
    added = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=A", SINCE, "HEAD", "--", "Thesis/"],
        cwd=ROOT, stdout=subprocess.PIPE, check=True).stdout.decode().split("\n")
    added = [a for a in added if a.strip() and not a.startswith("Thesis/tools/")]
    if added:
        print("  NEW -- these do not exist in Overleaf yet and must be added:")
        for a in added:
            print("       %s" % a[len("Thesis/"):])
        print()

    deleted = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=D", SINCE, "HEAD", "--", "Thesis/"],
        cwd=ROOT, stdout=subprocess.PIPE, check=True).stdout.decode().split("\n")
    deleted = [d for d in deleted if d.strip()]
    if deleted:
        print("  DELETED locally -- remove from Overleaf if the project still has them:")
        for d in deleted:
            print("       %s" % d[len("Thesis/"):])
    return 0


if __name__ == "__main__":
    sys.exit(main())
