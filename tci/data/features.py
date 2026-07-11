"""Fixed-stencil per-cell features in the style of Ray & Hesthaven (JCP
2018): for each cell k the feature vector is

    (v_{k-1}, v_k, v_{k+1}, u_k(x_{k-1/2}), u_k(x_{k+1/2}))

i.e. the three neighbouring cell means and the two interface values of the
local polynomial, min-max normalized per sample. This is the MLP baseline's
input; unlike the GNN it cannot see beyond the fixed 3-cell stencil.
"""

import numpy as np

from tci.solvers.nodal import reference_element


def cell_means_from_nodal(u):
    """Cell means of a nodal field (Np, K) without a solver instance."""
    _, _, inv_v, _, _ = reference_element(u.shape[0] - 1)
    return (inv_v @ u)[0] / np.sqrt(2.0)


def stencil_features(u, bc="periodic"):
    """(K, 5) feature matrix for a nodal field u of shape (Np, K)."""
    v = cell_means_from_nodal(u)
    if bc == "periodic":
        vm1, vp1 = np.roll(v, 1), np.roll(v, -1)
    else:
        vm1 = np.concatenate(([v[0]], v[:-1]))
        vp1 = np.concatenate((v[1:], [v[-1]]))
    X = np.stack([vm1, v, vp1, u[0, :], u[-1, :]], axis=1)

    lo, hi = float(X.min()), float(X.max())
    scale = hi - lo if hi - lo > 1e-12 else 1.0
    return ((X - lo) / scale).astype(np.float32)


def stencil_features2d(u, mesh, neighbors=None):
    """Fixed-width, permutation-invariant P1 triangle stencil features."""
    from tci.data.graphs import normalize_features, triangle_geometry_features

    nodal = normalize_features(u)
    means = np.mean(nodal, axis=1)
    if neighbors is None:
        neighbors = mesh.neighbors
    neighbor_means = np.repeat(means[:, None], 3, axis=1)
    valid = neighbors >= 0
    neighbor_means[valid] = means[neighbors[valid]]
    neighbor_means.sort(axis=1)

    face_midpoints = np.column_stack(
        [
            0.5 * (nodal[:, face] + nodal[:, (face + 1) % 3])
            for face in range(3)
        ]
    )
    face_midpoints.sort(axis=1)
    neighbor_face_means = neighbor_means.copy()
    jumps = np.sort(np.abs(face_midpoints - neighbor_face_means), axis=1)
    geometry = triangle_geometry_features(mesh)
    return np.column_stack(
        [
            means,
            np.min(nodal, axis=1),
            np.max(nodal, axis=1),
            neighbor_means,
            jumps,
            geometry,
        ]
    ).astype(np.float32)
