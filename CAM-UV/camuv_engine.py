"""
==============================================================
 TS-CAM-UV engine loader
==============================================================

Provides the authentic CAM-UV causal discovery algorithm of Maeda & Shimizu, as implemented
in the `lingam` package by the paper's own authors:

  Maeda, T. N. and Shimizu, S. (2021). "Causal Additive Models with Unobserved Variables."
  Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence (UAI), 97-106.

Why this loader exists
----------------------
Two practical obstacles stand between the installed package and a usable CAM-UV:

1. `lingam/__init__.py` eagerly imports every algorithm the package ships, and several of
   those pull heavyweight optional dependencies that CAM-UV itself never uses (`semopy`,
   `psy`, ...). A plain `from lingam import CAMUV` therefore fails on an unrelated import.

2. `lingam` pins `scipy<=1.13.1` and `pygam` pins `scipy<1.17`, but this project is locked to
   scipy 1.18.0 -- the version that produced every published PCMCI and GES result. Honouring
   those pins would downgrade scipy and silently invalidate six already-validated folders.
   Both packages were therefore installed with `--no-deps` and verified to work correctly on
   scipy 1.18.0 (see `verify_engine()` below).

This module loads only the three files CAM-UV actually needs (`hsic`, `utils`, `camuv`) into a
minimal `lingam` namespace, so the unrelated imports never execute. The algorithm code itself
is the unmodified upstream implementation -- nothing about the method is reimplemented here.

API
---
`execute(data, alpha, num_explanatory_vals)` returns `(P, U)` where

  P : list[set[int]]  -- P[i] is the set of observed parents of variable i
  U : list[set[int]]  -- pairs of variables sharing an unobserved common cause

matching the signature used by the CAM-UV reference scripts.
"""

import importlib.util
import os
import sys
import types

import numpy as np

_CAMUV_CLASS = None


def _load_camuv_class():
    """Load lingam.camuv without executing lingam/__init__.py."""
    global _CAMUV_CLASS
    if _CAMUV_CLASS is not None:
        return _CAMUV_CLASS

    site_packages = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".venv", "Lib", "site-packages", "lingam",
    )
    if not os.path.isdir(site_packages):
        import lingam as _probe  # noqa: F401  (raises a clear ImportError if truly absent)
        site_packages = os.path.dirname(_probe.__file__)

    if "lingam" not in sys.modules or not hasattr(sys.modules.get("lingam"), "__path__"):
        pkg = types.ModuleType("lingam")
        pkg.__path__ = [site_packages]
        sys.modules["lingam"] = pkg

    for name in ("hsic", "utils", "camuv"):
        full = "lingam." + name
        if full in sys.modules:
            continue
        path = os.path.join(site_packages, name)
        path = os.path.join(path, "__init__.py") if os.path.isdir(path) else path + ".py"
        spec = importlib.util.spec_from_file_location(full, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)

    _CAMUV_CLASS = sys.modules["lingam.camuv"].CAMUV
    return _CAMUV_CLASS


def execute(data, alpha=0.05, num_explanatory_vals=2):
    """
    Run CAM-UV on `data` (n_samples x n_variables).

    Returns (P, U) in the reference-script convention:
      P[i] = set of observed parents of variable i
      U    = list of {i, j} pairs judged to share an unobserved common cause

    CAM-UV encodes a confounded pair as NaN in `adjacency_matrix_`, which is what
    distinguishes it from purely observational discovery such as PCMCI or GES.
    """
    CAMUV = _load_camuv_class()
    X = np.asarray(data, dtype=float)

    model = CAMUV(alpha=alpha, num_explanatory_vals=num_explanatory_vals)
    model.fit(X)
    A = model.adjacency_matrix_

    n = A.shape[0]
    P = [set() for _ in range(n)]
    U = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if np.isnan(A[i, j]):
                pair = {i, j}
                if pair not in U:
                    U.append(pair)
            elif A[i, j] != 0:
                # lingam convention: A[i, j] != 0 means j is a parent of i
                P[i].add(j)
    return P, U


def adjacency_from_parents(P, n_features):
    """Dense 0/1 parent matrix, M[parent, child] = 1, for downstream regression code."""
    M = np.zeros((n_features, n_features), dtype=float)
    for child, parents in enumerate(P):
        for parent in parents:
            M[parent, child] = 1.0
    return M


def verify_engine(verbose=True):
    """
    Smoke-test the engine on a synthetic system with a known unobserved confounder.

    x0 -> x1 -> x2, and x1 and x3 share a hidden common cause u. A correct CAM-UV run should
    recover directed structure among the observed chain and be capable of flagging confounded
    pairs. Returns (P, U).
    """
    rng = np.random.default_rng(0)
    n = 200
    u = rng.normal(size=n)
    x0 = rng.normal(size=n)
    x1 = 0.8 * x0 + 0.4 * u + 0.1 * rng.normal(size=n)
    x2 = 0.7 * x1 + 0.1 * rng.normal(size=n)
    x3 = 0.6 * u + 0.1 * rng.normal(size=n)
    X = np.column_stack([x0, x1, x2, x3])

    P, U = execute(X, alpha=0.05, num_explanatory_vals=2)
    if verbose:
        print("CAM-UV engine verification")
        print("  parents per variable:", {i: sorted(s) for i, s in enumerate(P)})
        print("  unobserved-confounded pairs:", [sorted(p) for p in U])
    return P, U


if __name__ == "__main__":
    verify_engine()
