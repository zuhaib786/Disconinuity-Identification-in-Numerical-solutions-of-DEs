"""Conservative slope limiting for P1 fields on triangular meshes."""

import numpy as np


def neighbor_mean_bounds(solver, u):
    means = solver.cell_means(u)
    neighbors = solver.all_neighbors
    cells = np.arange(solver.K)[:, None]
    safe_neighbors = np.where(neighbors >= 0, neighbors, cells)
    neighbor_values = means[safe_neighbors]
    lower = np.minimum(means, np.min(neighbor_values, axis=1))
    upper = np.maximum(means, np.max(neighbor_values, axis=1))
    return means, lower, upper


def limit_p1(solver, u, flags):
    """Barth-Jespersen limiter; preserves every flagged cell's mean exactly."""
    u = np.asarray(u, dtype=float).copy()
    flags = np.asarray(flags, dtype=bool)
    if flags.shape != (solver.K,):
        raise ValueError(f"flags have shape {flags.shape}, expected {(solver.K,)}")
    means, lower, upper = neighbor_mean_bounds(solver, u)
    delta = u - means[None, :]
    ratios = np.full_like(delta, np.inf)
    np.divide(
        upper[None, :] - means[None, :],
        delta,
        out=ratios,
        where=delta > 0,
    )
    np.divide(
        lower[None, :] - means[None, :],
        delta,
        out=ratios,
        where=delta < 0,
    )
    theta = np.clip(np.min(ratios, axis=0), 0.0, 1.0)
    theta = np.where(flags, theta, 1.0)
    limited = means[None, :] + theta[None, :] * delta
    limited[:, ~flags] = u[:, ~flags]
    return limited


def limit_p1_system(solver, U, flags):
    """Fused component-wise Barth-Jespersen limiting for ``(3, K, C)``."""
    U = np.asarray(U, dtype=float)
    flags = np.asarray(flags, dtype=bool)
    if U.ndim != 3 or U.shape[:2] != (3, solver.K):
        raise ValueError(f"U has shape {U.shape}, expected (3, {solver.K}, C)")
    if flags.shape != (solver.K,):
        raise ValueError(f"flags have shape {flags.shape}, expected {(solver.K,)}")
    means, lower, upper = neighbor_mean_bounds(solver, U)
    delta = U - means[None, :, :]
    ratios = np.full_like(delta, np.inf)
    np.divide(
        upper[None, :, :] - means[None, :, :],
        delta,
        out=ratios,
        where=delta > 0,
    )
    np.divide(
        lower[None, :, :] - means[None, :, :],
        delta,
        out=ratios,
        where=delta < 0,
    )
    theta = np.clip(np.min(ratios, axis=0), 0.0, 1.0)
    theta = np.where(flags[:, None], theta, 1.0)
    limited = means[None, :, :] + theta[None, :, :] * delta
    limited[:, ~flags, :] = U[:, ~flags, :]
    return limited
