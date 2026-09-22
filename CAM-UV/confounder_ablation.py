"""
Does TS-CAM-UV recover a confounder it is given?

The `U` set -- pairs sharing an unobserved common cause -- is the entire reason for choosing
CAM-UV over PCMCI and GES, so whether the method can recover a confounder it is actually
given is worth establishing separately from what it returns on real data.

(An earlier draft of this docstring said `U` comes back empty on all three thesis datasets.
That was wrong and confounder_structure.py corrects it: `U` is not empty anywhere, and on
robotics it is the entire graph, 480 of 595 possible pairs. The real finding is that `U` is
overactive where samples are plentiful and inert where they are not, which is a statement
about statistical power rather than about the method failing.)

An earlier version of this file concluded that CAM-UV never recovers a planted confounder.
That conclusion was wrong, and the error was in the test rather than the method. Both systems
are kept here so the difference is visible.

    A. proxy / linear-gaussian   (the original test, retained as a negative control)

        u latent;  x1 = 0.8*x0 + 0.4*u + e,  x2 = 0.7*x1 + e,  x3 = 0.6*u + 0.1*e

       Two independent defects. Every relation is linear and every noise term Gaussian,
       which is the classically non-identifiable regime for this family of methods. And
       x3 = 0.6*u + 0.1*e makes x3 a near-perfect observation of u -- u explains about 97%
       of its variance -- so u is effectively observed and a direct edge x3 -> x1 is the
       defensible answer, not a latent confounder. The test could not have passed.

    B. latent / nonlinear        (a system CAM-UV is actually specified for)

        u latent;  x1 = f(x0) + e,  x2 = g(x1) + c*u + e,  x3 = f(x0) + c*u + e
        with f, g nonlinear and all noise uniform

       u confounds {x2, x3}, and both have strong observed parents, so neither is a proxy
       for u. This is the regime the paper addresses: nonlinear additive relations with
       non-Gaussian noise.

Every configuration is run through two independent implementations of the same algorithm --
`lingam.camuv` (wrapped by camuv_engine) and `causallearn`'s CAMUV -- so a disagreement would
point at the wrapper rather than the method.

Usage:  python CAM-UV/confounder_ablation.py      (from the repository root)
"""

import itertools
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camuv_engine  # noqa: E402

from causallearn.search.FCMBased.lingam.CAMUV import execute as causallearn_execute  # noqa: E402

SAMPLE_SIZES = [200, 500, 1000, 1500]
ALPHAS = [0.01, 0.05, 0.10]
NUM_EXPLANATORY = [2, 3]
CONFOUND_STRENGTH = 0.5


def system_proxy(n, seed=0, noise=0.1):
    """A. The original test: linear, Gaussian, and x3 is a near-perfect proxy for u."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)
    x0 = rng.normal(size=n)
    x1 = 0.8 * x0 + 0.4 * u + noise * rng.normal(size=n)
    x2 = 0.7 * x1 + noise * rng.normal(size=n)
    x3 = 0.6 * u + noise * rng.normal(size=n)
    return np.column_stack([x0, x1, x2, x3])


def system_latent(n, seed=0, c=CONFOUND_STRENGTH):
    """B. Nonlinear, non-Gaussian, u confounds {x2, x3}, neither of which proxies u."""
    rng = np.random.default_rng(seed)
    q = lambda: rng.uniform(-1, 1, n)
    f = lambda v: np.sin(2 * v) + 0.5 * v ** 2
    g = lambda v: 0.3 * v ** 3 + 0.5 * v
    u = q()
    x0 = q()
    x1 = f(x0) + 0.3 * q()
    x2 = g(x1) + c * u + 0.3 * q()
    x3 = f(x0) + c * u + 0.3 * q()
    return np.column_stack([x0, x1, x2, x3])


SYSTEMS = {
    "A proxy / linear-gaussian": (system_proxy, {0: set(), 1: {0}, 2: {1}, 3: set()}, {1, 3}),
    "B latent / nonlinear":      (system_latent, {0: set(), 1: {0}, 2: {1}, 3: {0}}, {2, 3}),
}

IMPLS = {
    "lingam": lambda X, a, k: camuv_engine.execute(X, alpha=a, num_explanatory_vals=k),
    "causallearn": lambda X, a, k: causallearn_execute(X, a, k),
}


def main():
    for label, (make, truth_p, truth_u) in SYSTEMS.items():
        print(f"\n{'='*94}\n {label}   -- correct confounded pair {sorted(truth_u)}\n{'='*94}")
        print(f"{'impl':12} {'n':>5} {'alpha':>6} {'k':>2}  {'U recovered':26} {'correct?':>9}  parents")
        print("-" * 94)
        tally = {k: [0, 0] for k in IMPLS}
        for n, a, k in itertools.product(SAMPLE_SIZES, ALPHAS, NUM_EXPLANATORY):
            X = make(n)
            for impl, fn in IMPLS.items():
                P, U = fn(X, a, k)
                sets = [set(s) for s in U]
                hit = truth_u in sets
                nc = sum(1 for i in truth_p if set(P[i]) == truth_p[i])
                tally[impl][0] += hit
                tally[impl][1] += 1
                print(f"{impl:12} {n:5d} {a:6.2f} {k:2d}  {str([sorted(s) for s in sets]):26} "
                      f"{'YES' if hit else 'no':>9}  {nc}/{len(truth_p)}")
        print()
        for impl, (hits, tot) in tally.items():
            print(f"  {impl:12} recovered the planted confounder in {hits}/{tot} configurations")


if __name__ == "__main__":
    main()
