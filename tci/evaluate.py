"""Solver-in-the-loop comparison of troubled-cell indicators.

The benchmark is the thesis test (sections 4.8 / 5.9): advect a box profile
    u0 = 1 on [0.4, 0.6], 0 elsewhere on [0, 1]
with speed a = 1 to T = 0.5 under periodic BCs, using each indicator to
decide where the minmod limiter acts. Reported per indicator:

- l1_error, l2_error : cell-mean error against the exact advected box
- total_variation    : TV of the final cell means (exact box: 2.0)
- flagged_pct        : mean percentage of cells flagged per time step
- runtime_s          : wall-clock time of the full solve
"""

import time

import numpy as np

from tci.solvers.dg1d import DG1D


def box_initial(x):
    return np.where((x >= 0.4) & (x <= 0.6), 1.0, 0.0)


def exact_advected(x, a, T, domain=(0.0, 1.0), u0=box_initial):
    lo, hi = domain
    return u0(lo + np.mod(x - a * T - lo, hi - lo))


def total_variation(v):
    return float(np.sum(np.abs(np.diff(v))))


def run_indicator(indicator, K=100, N=1, a=1.0, T=0.5, domain=(0.0, 1.0), cfl=0.375):
    """Run the box benchmark with one indicator; returns (metrics, artifacts)."""
    solver = DG1D(*domain, K=K, N=N)
    u0 = solver.project(box_initial)

    t0 = time.perf_counter()
    u, history = solver.advect(
        u0, a, T, indicator=indicator, cfl=cfl, record_flags=True
    )
    runtime = time.perf_counter() - t0

    v = solver.cell_means(u)
    centers = 0.5 * (solver.VX[:-1] + solver.VX[1:])
    v_exact = exact_advected(centers, a, T, domain)

    metrics = {
        "l1_error": float(np.mean(np.abs(v - v_exact))),
        "l2_error": float(np.sqrt(np.mean((v - v_exact) ** 2))),
        "total_variation": total_variation(v),
        "flagged_pct": 100.0 * float(np.mean([f.mean() for _, f in history])),
        "runtime_s": runtime,
    }
    artifacts = {"solver": solver, "u": u, "history": history, "v_exact": v_exact}
    return metrics, artifacts


def compare_indicators(indicators, **kwargs):
    """indicators: dict name -> Indicator. Returns dict name -> metrics."""
    results = {}
    for name, ind in indicators.items():
        metrics, _ = run_indicator(ind, **kwargs)
        results[name] = metrics
    return results


def format_table(results):
    cols = ["l1_error", "l2_error", "total_variation", "flagged_pct", "runtime_s"]
    header = f"{'indicator':<12}" + "".join(f"{c:>18}" for c in cols)
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        lines.append(
            f"{name:<12}" + "".join(f"{m[c]:>18.5f}" for c in cols)
        )
    return "\n".join(lines)
