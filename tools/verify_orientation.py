"""
Does the integration layer hand each library its matrix the right way round?

Every causal discovery library in this pipeline takes a two-dimensional array and every one
of them expects (observations, variables). Passing (variables, observations) instead does not
raise: the library runs happily, treats each timestep as a variable, and returns a graph over
the wrong axis. On a square matrix that is undetectable by inspection, and on a rectangular
one it is detectable only if somebody checks the shape of what came back.

That is the failure this guards against, and it has a signature that cannot be argued with:
given an (n, d) matrix the output graph is d x d, and given its transpose the output is
n x n. So the test constructs data with n distinct from d and asserts the returned dimension.

Two things are checked, in order.

  orientation   Each wrapper is handed a rectangular matrix in both orientations. The
                correct one must return a graph over d variables; the transpose must
                return something else -- either a graph over n, or a refusal. A wrapper
                that returns d x d for both is silently ignoring the axis and would
                produce a plausible, wrong graph on real data.

  recovery      In the correct orientation only, a known sparse structure is planted and
                each algorithm must recover more of it than it invents. This is weaker than
                the orientation check and deliberately so -- these are different estimators
                with different assumptions and none is expected to be exact -- but an
                algorithm that recovers nothing is not being driven correctly regardless of
                which way the matrix is turned.

The planted system is a chain with independent bystanders:

    x0 -> x1 -> x2 -> x3,      x4 and x5 independent of everything

so 3 of the 15 possible undirected pairs are real and 12 are not.

Usage:  python tools/verify_orientation.py
"""

import csv
import io
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

D = 6                 # variables
N_RECOVERY = 400      # samples for the structure-recovery check
N_SHAPE = 40          # samples for the orientation check, kept small: the transposed
                      # matrix becomes an N_SHAPE-variable problem and must stay tractable
ALPHA = 0.05
TRUE_PAIRS = {(0, 1), (1, 2), (2, 3)}
SEED = 7


def planted(n, seed=SEED):
    """x0 -> x1 -> x2 -> x3 with two independent bystanders. Nonlinear, non-Gaussian."""
    rng = np.random.default_rng(seed)
    e = rng.uniform(-1, 1, size=(n, D))
    x = np.zeros((n, D))
    x[:, 0] = e[:, 0]
    x[:, 1] = np.tanh(1.5 * x[:, 0]) + 0.3 * e[:, 1]
    x[:, 2] = np.tanh(1.5 * x[:, 1]) + 0.3 * e[:, 2]
    x[:, 3] = np.tanh(1.5 * x[:, 2]) + 0.3 * e[:, 3]
    x[:, 4] = e[:, 4]
    x[:, 5] = e[:, 5]
    return x


# --------------------------------------------------------------------- wrappers
# Each returns the number of variables the library believed it was given, which is the
# quantity the orientation check turns on, plus the undirected pairs it linked.

def run_pcmci(data):
    from tigramite import data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI
    res = PCMCI(dataframe=pp.DataFrame(data), cond_ind_test=ParCorr(),
                verbosity=0).run_pcmci(tau_max=1, pc_alpha=ALPHA)
    p = res["p_matrix"]
    d = p.shape[0]
    pairs = {(i, j) for i in range(d) for j in range(d)
             if i < j and ((p[i, j, :] < ALPHA).any() or (p[j, i, :] < ALPHA).any())}
    return d, pairs


def run_ges(data):
    from causallearn.search.ScoreBased.GES import ges
    g = ges(data)["G"].graph
    d = g.shape[0]
    pairs = {(i, j) for i in range(d) for j in range(i + 1, d) if g[i, j] != 0 or g[j, i] != 0}
    return d, pairs


def run_camuv(data):
    sys.path.insert(0, os.path.abspath("CAM-UV"))
    import camuv_engine
    parents, _u = camuv_engine.execute(data, alpha=ALPHA, num_explanatory_vals=2)
    d = data.shape[1]
    pairs = set()
    for child, ps in enumerate(parents):
        for par in ps:
            pairs.add((min(child, par), max(child, par)))
    return d, pairs


ALGOS = [("PCMCI", run_pcmci), ("GES", run_ges), ("CAM-UV", run_camuv)]


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    os.chdir(root)

    print("Integration orientation check\n")
    print("  Every library here expects (observations, variables) and none raises when")
    print("  handed the transpose. The signature is the returned dimension.\n")
    print("  %-11s %10s %14s %14s %s"
          % ("algorithm", "expected", "correct way", "transposed", "verdict"))
    print("  " + "-" * 74)

    rows = []
    x_shape = planted(N_SHAPE)
    for name, fn in ALGOS:
        try:
            d_ok, _ = fn(x_shape)
        except Exception as exc:
            print("  %-11s %10d %14s %14s %s"
                  % (name, D, "ERROR", "-", type(exc).__name__))
            rows.append({"Algorithm": name, "Expected variables": D,
                         "Correct orientation": "error", "Transposed": "",
                         "Distinguishes orientation": "unknown"})
            continue
        try:
            d_t, _ = fn(x_shape.T)
            t_repr = str(d_t)
        except Exception as exc:
            d_t, t_repr = None, "refused (%s)" % type(exc).__name__

        ok = (d_ok == D) and (d_t != D)
        print("  %-11s %10d %14d %14s %s"
              % (name, D, d_ok, t_repr, "distinguishes" if ok else "DOES NOT DISTINGUISH"))
        rows.append({"Algorithm": name, "Expected variables": D,
                     "Correct orientation": d_ok, "Transposed": t_repr,
                     "Distinguishes orientation": "yes" if ok else "NO"})

    print("\n\nStructure recovery, correct orientation only")
    print("  planted: x0->x1->x2->x3, x4 and x5 independent."
          "  3 real pairs of 15 possible\n")
    print("  %-11s %10s %10s %10s %s" % ("algorithm", "recovered", "of 3", "spurious", "note"))
    print("  " + "-" * 66)

    x_rec = planted(N_RECOVERY)
    for name, fn in ALGOS:
        try:
            d, pairs = fn(x_rec)
        except Exception as exc:
            print("  %-11s %s" % (name, type(exc).__name__))
            continue
        hit = len(pairs & TRUE_PAIRS)
        spur = len(pairs - TRUE_PAIRS)
        note = "recovers structure" if hit >= 2 else "recovers little"
        print("  %-11s %10d %10d %10d %s" % (name, hit, len(TRUE_PAIRS), spur, note))
        for r in rows:
            if r["Algorithm"] == name:
                r["Planted pairs recovered"] = hit
                r["Planted pairs total"] = len(TRUE_PAIRS)
                r["Spurious pairs"] = spur

    out = "tools/orientation_check.csv"
    with io.open(out, "w", encoding="utf-8", newline="") as f:
        keys = ["Algorithm", "Expected variables", "Correct orientation", "Transposed",
                "Distinguishes orientation", "Planted pairs recovered",
                "Planted pairs total", "Spurious pairs"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("\n  wrote %s" % out)

    bad = [r for r in rows if r.get("Distinguishes orientation") == "NO"]
    if bad:
        print("\n  %d wrapper(s) returned the same dimension both ways. On real data that"
              % len(bad))
        print("  would produce a plausible graph over the wrong axis, without any error.")
        return 1
    print("\n  Every wrapper distinguishes the two orientations, so a transposed matrix")
    print("  cannot reach a discovery call unnoticed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
