"""
Numerical validation of the algorithm implementations, against closed-form answers.

Reading code and judging it correct is not evidence. Each check below constructs a problem
whose answer is known independently and confirms the implementation reaches it.

  1. RLS with no forgetting equals ordinary least squares.
     With lam = 1.0 and P0 = I/alpha, recursive least squares is algebraically identical to
     ridge regression with penalty alpha, and approaches OLS as alpha -> 0. If the recursion
     is wrong -- a missing lambda, a transposed outer product, a gain divided by the wrong
     denominator -- the estimate will not match, so this pins the update rule exactly.

  2. RLS with forgetting tracks a coefficient that changes.
     The entire detector rests on the claim that the filter follows a drifting relationship.
     A step change is planted in a known coefficient and the estimate must follow it, while
     a no-forgetting filter must lag.

  3. Coefficient drift separates a broken relationship from an intact one.
     This is the detection premise itself: the score must rise when the *relationship*
     changes, and must not rise merely because a variable's values move.

  4. GES keeps only unambiguously directed CPDAG edges.
     causal-learn returns a CPDAG in which adj[i,j] == -1 and adj[j,i] == 1 encodes a
     directed edge i -> j, while an undirected edge is symmetric. Retaining an undirected
     edge as if it were directed would invent causal direction the algorithm did not find.

Usage:  python tools/validate_algorithms.py
"""

import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath("PCMCI/PCMCI_robotic"))

# Figures printed below are also written to a CSV so tools/verify_docs.py can
# trace every number the README quotes from this file back to a generated source.
MEASURED = {}


def load_rls():
    """Import _rls_update from the robotics linear script without running its pipeline."""
    p = "PCMCI/PCMCI_robotic/anomaly_detection.py"
    src = open(p, encoding="utf-8").read()
    ns = {"np": np}
    start = src.index("def _rls_update")
    end = src.index("\ndef ", start + 1)
    exec(compile(src[start:end], p, "exec"), ns)
    return ns["_rls_update"]


def check_rls_equals_ols(rls):
    rng = np.random.default_rng(0)
    n, d = 4000, 4
    X = rng.normal(size=(n, d))
    w_true = np.array([1.5, -2.0, 0.5, 3.0])
    y = X @ w_true + 0.01 * rng.normal(size=n)

    alpha = 1e-6                      # P0 = I/alpha, so the ridge penalty is negligible
    w = np.zeros(d)
    P = np.eye(d) / alpha
    for t in range(n):
        w, P, _ = rls(X[t], y[t], w, P, lam=1.0)

    w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    err = float(np.max(np.abs(w - w_ols)))
    ok = err < 1e-6
    MEASURED["rls_vs_ols_max_abs_diff"] = err
    return ok, (f"max |w_RLS - w_OLS| = {err:.3e} over {n} samples, {d} parameters "
                f"(lam=1.0, P0=I/{alpha:g})")


def check_rls_tracks_drift(rls):
    rng = np.random.default_rng(1)
    n, d = 4000, 3
    X = rng.normal(size=(n, d))
    w_a = np.array([1.0, 0.0, -1.0])
    w_b = np.array([1.0, 2.5, -1.0])          # coefficient 1 steps from 0.0 to 2.5 midway
    y = np.array([X[t] @ (w_a if t < n // 2 else w_b) for t in range(n)])
    y += 0.01 * rng.normal(size=n)

    out = {}
    for lam in (0.995, 1.0):
        w, P = np.zeros(d), np.eye(d) / 1e-6
        for t in range(n):
            w, P, _ = rls(X[t], y[t], w, P, lam=lam)
        out[lam] = float(w[1])

    forget, no_forget = out[0.995], out[1.0]
    MEASURED["rls_forgetting_estimate"] = forget
    MEASURED["rls_no_forgetting_estimate"] = no_forget
    ok = abs(forget - 2.5) < 0.05 and abs(no_forget - 2.5) > abs(forget - 2.5)
    return ok, (f"true post-change value 2.5 -- lam=0.995 reaches {forget:.4f}, "
                f"lam=1.0 lags at {no_forget:.4f}")


def check_drift_separates_cause(rls):
    """A broken relationship must raise drift; a rescaled but intact one must not."""
    rng = np.random.default_rng(2)
    n, d = 3000, 3

    def run(mode):
        X = rng.normal(size=(n, d))
        w = np.array([1.0, -0.5, 2.0])
        y = X @ w
        if mode == "broken":                  # relationship changes
            y[n // 2:] = X[n // 2:] @ np.array([-2.0, 3.0, 0.5])
        elif mode == "rescaled":              # values move, relationship intact
            X[n // 2:] *= 5.0
            y[n // 2:] = X[n // 2:] @ w
        y = y + 0.01 * rng.normal(size=n)
        wv, P = w.astype(float).copy(), np.eye(d) / 10.0     # start at the true coefficients
        base = wv.copy()
        drift = []
        for t in range(n):
            wv, P, _ = rls(X[t], y[t], wv, P, lam=0.995)
            drift.append(np.linalg.norm(wv - base))
        return float(np.mean(drift[n // 2:]))

    broken, rescaled, intact = run("broken"), run("rescaled"), run("intact")
    MEASURED["drift_broken_relationship"] = broken
    MEASURED["drift_rescaled_intact"] = rescaled
    MEASURED["drift_unchanged"] = intact
    ok = broken > 10 * intact and broken > 3 * rescaled
    return ok, (f"mean drift after the change -- broken {broken:.4f}, "
                f"rescaled-but-intact {rescaled:.4f}, unchanged {intact:.4f}")


def check_ges_edge_extraction():
    """The CPDAG rule the GES scripts use must keep directed edges and drop undirected ones."""
    import glob
    import re
    rx = re.compile(r"(\w+)\[i,\s*j\]\s*==\s*-1\s*and\s*\1\[j,\s*i\]\s*==\s*1")
    scripts = sorted(glob.glob("GES/*/anomaly_detection_ges*.py"))
    missing = [s for s in scripts if not rx.search(io.open(s, encoding="utf-8").read())]
    present = scripts and not missing

    # the rule itself, applied to a CPDAG with one directed and one undirected edge
    adj = np.zeros((3, 3))
    adj[0, 1], adj[1, 0] = -1, 1          # 0 -> 1 directed
    adj[1, 2], adj[2, 1] = -1, -1         # 1 -- 2 undirected
    kept = [(i, j) for i in range(3) for j in range(3)
            if i != j and adj[i, j] == -1 and adj[j, i] == 1]
    ok = present and kept == [(0, 1)]
    return ok, (f"rule present in {len(scripts)-len(missing)}/{len(scripts)} GES scripts"
                + (f" (missing: {missing})" if missing else "")
                + f"; applied to a test CPDAG it keeps {kept} and drops the undirected pair")


def main():
    rls = load_rls()
    checks = [
        ("RLS with lam=1 equals ordinary least squares", lambda: check_rls_equals_ols(rls)),
        ("RLS with forgetting tracks a drifting coefficient", lambda: check_rls_tracks_drift(rls)),
        ("drift separates a broken relationship from a rescaled one",
         lambda: check_drift_separates_cause(rls)),
        ("GES keeps only unambiguously directed CPDAG edges", check_ges_edge_extraction),
    ]
    failed = 0
    for label, fn in checks:
        ok, detail = fn()
        failed += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        print(f"         {detail}")
    print()
    print(f"{len(checks)-failed}/{len(checks)} algorithm checks passed")

    import csv
    out = "tools/algorithm_validation.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value"])
        for k, v in MEASURED.items():
            # A quantity whose whole point is that it is negligible must not be rounded to
            # 0.0000: the RLS-vs-OLS agreement is 7.5e-10, and four decimal places would
            # discard the very evidence the check exists to provide.
            w.writerow([k, f"{v:.3e}" if 0 < abs(v) < 1e-4 else f"{v:.4f}"])
    print(f"wrote {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
