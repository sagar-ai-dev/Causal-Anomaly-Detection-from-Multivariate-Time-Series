"""
Run the nine healthcare configurations restricted to the eight independent ECG leads.

This is the rerun the reviewer asked for as the last experiment: with III, aVR, aVL and aVF
removed, the causal graph can no longer recover Einthoven's and Goldberger's identities, so
whatever detection performance survives is attributable to physiology rather than algebra.

The linear variant of each algorithm runs first because it builds the causal graph; the
polynomial and RBF variants reuse the cache it writes. Caches and outputs carry a _leads8
suffix, so nothing here can overwrite the twelve-lead results.

Usage:  python tools/run_ecg_ablation.py
"""

import io
import os
import subprocess
import sys
import time

ORDER = [
    ("PCMCI", "PCMCI/PCMCI_healthcare", ["anomaly_detection.py",
                                         "anomaly_detection_polynomial.py",
                                         "anomaly_detection_rbf.py"]),
    ("GES", "GES/GES_healthcare", ["anomaly_detection_ges.py",
                                   "anomaly_detection_ges_polynomial.py",
                                   "anomaly_detection_ges_rbf.py"]),
    ("CAM-UV", "CAM-UV/CAM_UV_healthcare", ["anomaly_detection.py",
                                               "anomaly_detection_polynomial.py",
                                               "anomaly_detection_rbf.py"]),
]


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.chdir(root)

    env = dict(os.environ)
    env["ECG_INDEPENDENT_LEADS"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    log = io.open("tools/ecg_ablation.log", "w", encoding="utf-8", newline="\n")
    failures = []
    t_all = time.time()

    for algo, folder, scripts in ORDER:
        for s in scripts:
            path = os.path.join(folder, s)
            if not os.path.exists(path):
                print("  MISSING %s" % path)
                continue
            label = "%s / %s" % (algo, s.replace("anomaly_detection", "").strip("_.py") or "linear")
            print("  running %-42s " % label, end="", flush=True)
            t0 = time.time()
            r = subprocess.run([sys.executable, "-u", s], cwd=folder, env=env,
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace")
            dt = time.time() - t0
            ok = r.returncode == 0
            print("%s  %6.1fs" % ("ok" if ok else "FAILED", dt))
            log.write("=" * 78 + "\n%s  (%.1fs, exit %d)\n" % (label, dt, r.returncode)
                      + "=" * 78 + "\n" + (r.stdout or "") + (r.stderr or "") + "\n")
            log.flush()
            if not ok:
                failures.append(label)

    log.close()
    print()
    print("total %.1f min, log in tools/ecg_ablation.log" % ((time.time() - t_all) / 60))
    if failures:
        print("FAILED: " + ", ".join(failures))
        return 1
    print("outputs written to */*_healthcare/outputs_leads8/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
