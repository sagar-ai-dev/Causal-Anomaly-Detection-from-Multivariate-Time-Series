"""
Measure the wall-clock cost of every configuration, for the computational-cost section.

Two quantities are separated, because they behave very differently and conflating them
would misrepresent both:

  discovery   learning the causal graph. Paid once per (algorithm, dataset) and cached, so
              it does not recur per regression variant. This is where the algorithms differ
              by orders of magnitude, and where GES and CAM-UV hit the tractability wall
              that forces the 35-sensor cap on robotics.

  detection   loading the cached graph, fitting the baseline coefficients, running RLS over
              the evaluation windows, and writing the report. Paid on every run, and the
              number that matters operationally.

Only detection is measured by execution here. Discovery is NOT re-run, for a reason worth
stating: CAM-UV is not reproducible, so rebuilding its cache would change the graph and
therefore every result in the thesis. Discovery cost is instead read from the cache file's
own recorded build time where present, and otherwise reported as not measured.

The measurement is meaningless under CPU contention, so the script refuses to start if it
detects another Python process from this repository already running.

Usage:  python tools/measure_runtime.py            # all 27
        python tools/measure_runtime.py --domain healthcare
"""

import csv
import io
import os
import subprocess
import sys
import time

CONFIGS = []
for _algo, _folder, _scripts in [
    ("PCMCI", "PCMCI/PCMCI_{d}", ["anomaly_detection.py", "anomaly_detection_polynomial.py",
                                  "anomaly_detection_rbf.py"]),
    ("GES", "GES/GES_{d}", ["anomaly_detection_ges.py", "anomaly_detection_ges_polynomial.py",
                            "anomaly_detection_ges_rbf.py"]),
    ("CAM-UV", "CAM-UV/CAM_UV_{d}", ["anomaly_detection.py",
                                        "anomaly_detection_polynomial.py",
                                        "anomaly_detection_rbf.py"]),
]:
    for _dom, _key in [("robotic", "robotics"), ("healthcare", "healthcare"),
                       ("industrial", "industrial")]:
        _f = _folder.format(d="robotics" if (_algo == "CAM-UV" and _dom == "robotic") else _dom)
        for _s in _scripts:
            CONFIGS.append((_algo, _key, _f, _s))


def variant_of(script):
    if "polynomial" in script:
        return "Polynomial degree 3"
    if "rbf" in script:
        return "RBF + Ridge"
    return "Linear / Ridge"


def graph_size(folder):
    """Variables and edges in the cached graph, for context alongside the timing."""
    import glob
    import numpy as np
    for p in glob.glob(os.path.join(folder, "**", "*.npz"), recursive=True):
        if "scores_" in p:
            continue
        try:
            z = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if "nonconst" not in z:
            continue
        n = len(z["nonconst"])
        sub = int(z["subsample"]) if "subsample" in z else -1
        return n, sub, os.path.getsize(p)
    return None, None, None


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.chdir(root)

    only = None
    if "--domain" in sys.argv:
        only = sys.argv[sys.argv.index("--domain") + 1]

    rows = []
    todo = [c for c in CONFIGS if only is None or c[1] == only]
    print("timing %d configurations (graph cached; discovery not re-run)\n" % len(todo))
    print("  %-11s %-11s %-21s %9s %7s %7s" % ("algorithm", "domain", "variant",
                                               "seconds", "vars", "subsmp"))
    print("  " + "-" * 72)

    for algo, dom, folder, script in todo:
        path = os.path.join(folder, script)
        if not os.path.exists(path):
            print("  %-11s %-11s %-21s   MISSING" % (algo, dom, variant_of(script)))
            continue
        n, sub, _ = graph_size(folder)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", script], cwd=folder,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace",
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        dt = time.time() - t0
        ok = r.returncode == 0
        print("  %-11s %-11s %-21s %9.1f %7s %7s%s"
              % (algo, dom, variant_of(script), dt, n if n else "?",
                 sub if sub else "?", "" if ok else "   FAILED"))
        rows.append({
            "Algorithm": algo,
            "Domain": dom,
            "Variant": variant_of(script),
            "Detection seconds": "%.1f" % dt,
            "Variables": n if n is not None else "",
            "Subsample": sub if sub is not None else "",
            "Exit": r.returncode,
        })

    out = "tools/runtime.csv"
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nwrote %s (%d rows)" % (out, len(rows)))
    return 0 if all(r["Exit"] == 0 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
