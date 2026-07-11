"""Framework-independent graph construction for one-dimensional meshes."""

from __future__ import annotations

import numpy as np


def line_graph_edges(
    n_cells: int, *, bidirectional: bool = True, periodic: bool = False
) -> np.ndarray:
    """Return a PyG-compatible ``(2, n_edges)`` integer edge array."""

    if n_cells < 1:
        raise ValueError("n_cells must be positive")
    if n_cells == 1:
        return np.empty((2, 0), dtype=np.int64)

    source = np.arange(n_cells - 1, dtype=np.int64)
    target = source + 1
    if periodic:
        source = np.concatenate((source, np.array([n_cells - 1])))
        target = np.concatenate((target, np.array([0])))
    if bidirectional:
        source, target = (
            np.concatenate((source, target)),
            np.concatenate((target, source)),
        )
    return np.vstack((source, target))


def cell_features(
    dofs: np.ndarray,
    vertices: np.ndarray,
    *,
    normalize_solution: bool = True,
    include_geometry: bool = True,
) -> np.ndarray:
    """Convert ``(n_dofs, n_cells)`` DG values into node-feature rows."""

    solution = np.asarray(dofs, dtype=np.float64)
    mesh = np.asarray(vertices, dtype=np.float64).reshape(-1)
    if solution.ndim != 2:
        raise ValueError("dofs must have shape (n_dofs, n_cells)")
    if mesh.size != solution.shape[1] + 1:
        raise ValueError("vertices must contain one more entry than there are cells")
    widths = np.diff(mesh)
    if np.any(widths <= 0):
        raise ValueError("vertices must be strictly increasing")

    if normalize_solution:
        lower, upper = float(solution.min()), float(solution.max())
        scale = upper - lower
        solution = np.zeros_like(solution) if scale == 0.0 else (solution - lower) / scale

    features = solution.T
    if include_geometry:
        relative_width = widths / np.mean(widths)
        centers = 0.5 * (mesh[:-1] + mesh[1:])
        domain_scale = mesh[-1] - mesh[0]
        relative_center = (centers - mesh[0]) / domain_scale
        features = np.column_stack((features, relative_width, relative_center))
    return features

