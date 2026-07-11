"""Solver-in-the-loop comparison of troubled-cell indicators.

Benchmarks (each uses the indicator to decide where the minmod limiter acts):

- box       : thesis test (secs. 4.8/5.9) — box profile advected to T = 0.5
- burgers   : u0 = 0.5 + sin(2 pi x), shock forms; fine-grid reference
- sod       : Sod shock tube vs the exact Riemann solution (density)
- shu_osher : shock/entropy-wave interaction vs a fine-grid reference

Reported per indicator:

- l1_error, l2_error : cell-mean error against the reference
- total_variation    : TV of the final cell means
- flagged_pct        : mean percentage of cells flagged per time step
- runtime_s          : wall-clock time of the full solve
"""

import time

import numpy as np

from tci.solvers.burgers import BurgersDG1D
from tci.solvers.dg1d import DG1D
from tci.solvers.euler import EulerDG1D, shu_osher_initial, sod_initial
from tci.solvers.riemann import sod_exact


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


# ---------------------------------------------------------------------------
# Problem registry: box (advection), burgers, sod, shu_osher
# ---------------------------------------------------------------------------

_REFERENCE_CACHE = {}


def _metrics_from(field_means, ref_means, history, runtime):
    return {
        "l1_error": float(np.mean(np.abs(field_means - ref_means))),
        "l2_error": float(np.sqrt(np.mean((field_means - ref_means) ** 2))),
        "total_variation": total_variation(field_means),
        "flagged_pct": 100.0 * float(np.mean([f.mean() for _, f in history]))
        if history
        else 0.0,
        "runtime_s": runtime,
    }


def _centers(solver):
    return 0.5 * (solver.VX[:-1] + solver.VX[1:])


def _burgers_reference(T, K_ref=2000):
    """Fine minmod-limited solve, linearly interpolated to coarse centers."""
    from tci.indicators.classical import MinmodIndicator

    key = ("burgers", T, K_ref)
    if key not in _REFERENCE_CACHE:
        s = BurgersDG1D(0.0, 1.0, K=K_ref, N=1)
        u = s.solve(
            s.project(lambda x: 0.5 + np.sin(2 * np.pi * x)),
            T,
            indicator=MinmodIndicator(),
        )
        _REFERENCE_CACHE[key] = (_centers(s), s.cell_means(u))
    return _REFERENCE_CACHE[key]


def _shu_osher_reference(T, K_ref=1600):
    from tci.indicators.classical import MinmodIndicator

    key = ("shu_osher", T, K_ref)
    if key not in _REFERENCE_CACHE:
        s = EulerDG1D(-5.0, 5.0, K=K_ref, N=2)
        U = s.solve(shu_osher_initial, T, indicator=MinmodIndicator())
        _REFERENCE_CACHE[key] = (_centers(s), s.cell_means(U[:, :, 0]))
    return _REFERENCE_CACHE[key]


def run_benchmark(problem, indicator, K=None, N=None, T=None, cfl=None):
    """Run one indicator on one named problem; returns (metrics, artifacts)."""
    if problem == "box":
        return run_indicator(
            indicator,
            K=K or 100,
            N=1 if N is None else N,
            T=0.5 if T is None else T,
            cfl=cfl or 0.375,
        )

    if problem == "burgers":
        K, N, T = K or 100, 1 if N is None else N, 0.3 if T is None else T
        solver = BurgersDG1D(0.0, 1.0, K=K, N=N)
        u0 = solver.project(lambda x: 0.5 + np.sin(2 * np.pi * x))
        t0 = time.perf_counter()
        u, history = solver.solve(
            u0, T, indicator=indicator, cfl=cfl or 0.375, record_flags=True
        )
        runtime = time.perf_counter() - t0
        xr, vr = _burgers_reference(T)
        ref = np.interp(_centers(solver), xr, vr)
        v = solver.cell_means(u)
        metrics = _metrics_from(v, ref, history, runtime)
        return metrics, {"solver": solver, "u": u, "history": history, "v_exact": ref}

    if problem == "sod":
        K, N, T = K or 200, 1 if N is None else N, 0.2 if T is None else T
        solver = EulerDG1D(0.0, 1.0, K=K, N=N)
        t0 = time.perf_counter()
        U, history = solver.solve(
            sod_initial, T, indicator=indicator, cfl=cfl or 0.2, record_flags=True
        )
        runtime = time.perf_counter() - t0
        rho = solver.cell_means(U[:, :, 0])
        rho_ex, _, _ = sod_exact(_centers(solver), T)
        metrics = _metrics_from(rho, rho_ex, history, runtime)
        return metrics, {
            "solver": solver,
            "u": U,
            "history": history,
            "v_exact": rho_ex,
        }

    if problem == "shu_osher":
        K, N, T = K or 300, 2 if N is None else N, 1.8 if T is None else T
        solver = EulerDG1D(-5.0, 5.0, K=K, N=N)
        t0 = time.perf_counter()
        U, history = solver.solve(
            shu_osher_initial,
            T,
            indicator=indicator,
            cfl=cfl or 0.2,
            record_flags=True,
        )
        runtime = time.perf_counter() - t0
        rho = solver.cell_means(U[:, :, 0])
        xr, vr = _shu_osher_reference(T)
        ref = np.interp(_centers(solver), xr, vr)
        metrics = _metrics_from(rho, ref, history, runtime)
        return metrics, {"solver": solver, "u": U, "history": history, "v_exact": ref}

    raise ValueError(f"unknown problem {problem!r}")


def compare_on(problem, indicators, **kwargs):
    """indicators: dict name -> Indicator. Returns dict name -> metrics.

    An indicator that under-flags can let the solve blow up (the solver
    raises RuntimeError); that is a result, reported as {"diverged": True}.
    """
    results = {}
    for name, ind in indicators.items():
        try:
            results[name] = run_benchmark(problem, ind, **kwargs)[0]
        except RuntimeError:
            results[name] = {"diverged": True}
    return results


def format_table(results):
    cols = ["l1_error", "l2_error", "total_variation", "flagged_pct", "runtime_s"]
    header = f"{'indicator':<12}" + "".join(f"{c:>18}" for c in cols)
    lines = [header, "-" * len(header)]
    for name, m in results.items():
        if m.get("diverged"):
            lines.append(f"{name:<12}" + f"{'DIVERGED (solution blew up)':>18}")
        else:
            lines.append(f"{name:<12}" + "".join(f"{m[c]:>18.5f}" for c in cols))
    return "\n".join(lines)
