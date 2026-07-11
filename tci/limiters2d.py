"""Conservative slope limiting for P1 fields on triangular meshes."""

import numpy as np


def neighbor_mean_bounds(solver, u):
    means = solver.cell_means(u)
    lower = means.copy()
    upper = means.copy()
    for cell in range(solver.K):
        neighbors = solver.all_neighbors[cell]
        neighbors = neighbors[neighbors >= 0]
        if len(neighbors):
            values = means[np.concatenate([[cell], neighbors])]
            lower[cell], upper[cell] = np.min(values), np.max(values)
    return means, lower, upper


def limit_p1(solver, u, flags):
    """Barth-Jespersen limiter; preserves every flagged cell's mean exactly."""
    u = np.asarray(u, dtype=float).copy()
    flags = np.asarray(flags, dtype=bool)
    if flags.shape != (solver.K,):
        raise ValueError(f"flags have shape {flags.shape}, expected {(solver.K,)}")
    means, lower, upper = neighbor_mean_bounds(solver, u)
    for cell in np.flatnonzero(flags):
        delta = u[:, cell] - means[cell]
        theta = 1.0
        positive = delta > 0
        negative = delta < 0
        if np.any(positive):
            theta = min(theta, float(np.min((upper[cell] - means[cell]) / delta[positive])))
        if np.any(negative):
            theta = min(theta, float(np.min((lower[cell] - means[cell]) / delta[negative])))
        u[:, cell] = means[cell] + max(0.0, theta) * delta
    return u
