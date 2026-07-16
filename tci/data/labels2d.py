"""Resolution- and amplitude-aware reference labels for `data-v3` (plan 6.1).

Labels are computed from the reference field, never from the network input.
A cell is troubled when the face-mean jump of the projected reference is large
relative to the Phase 3 one-ring robust scale *and* does not decay under one
uniform refinement the way the ``O(h^2)`` jump of a smooth field does.
"""

import numpy as np

from tci.data.graphs import one_ring_robust_scale
from tci.mesh import TriangleMesh

# Frozen on the held-out calibration batch of `scripts/calibrate_v3_labels.py`;
# the evidence is `runs/paper/phase6-label-calibration.json`.  The plan's
# starting values (0.5, 3.0) are rejected there: alpha=0.5 recovers only 16.5%
# of the cells a reference discontinuity bisects.
ALPHA = 0.15
GAMMA = 2.0
QUADRATURE_ORDER = 8

# M_K^{-1} for the nodal P1 mass matrix ``M_K = |K| B / 12`` on a triangle.
_INVERSE_MASS_UNIT = 3.0 * np.array([[3.0, -1.0, -1.0], [-1.0, 3.0, -1.0], [-1.0, -1.0, 3.0]])


def duffy_quadrature(order=QUADRATURE_ORDER):
    """Barycentric points and area-fraction weights of a Duffy triangle rule."""
    gauss, weights = np.polynomial.legendre.leggauss(int(order))
    r_values = 0.5 * (gauss + 1.0)
    r_weights = 0.5 * weights
    points, area_weights = [], []
    for r, wr in zip(r_values, r_weights):
        for s, ws in zip(r_values, r_weights):
            points.append([1.0 - r, r * (1.0 - s), r * s])
            area_weights.append(2.0 * r * wr * ws)
    return np.asarray(points), np.asarray(area_weights)


def project_p1(mesh, field, order=QUADRATURE_ORDER):
    """L2 projection of ``field`` onto the discontinuous P1 space: ``(3, K)``.

    Unlike vertex interpolation, this projection is two-valued on a face that a
    discontinuity crosses, so its face-mean jumps detect the discontinuity.
    """
    vertices = mesh.points[mesh.cells]
    barycentric, area_weights = duffy_quadrature(order)
    load = np.zeros((mesh.K, 3))
    for point, area_weight in zip(barycentric, area_weights):
        xy = np.einsum("i,kid->kd", point, vertices)
        values = np.asarray(field(xy[:, 0], xy[:, 1]), dtype=float)
        if values.shape != (mesh.K,):
            raise ValueError(f"field returned {values.shape}, expected {(mesh.K,)}")
        load += (area_weight * mesh.areas * values)[:, None] * point[None, :]
    inverse_mass = _INVERSE_MASS_UNIT[None, :, :] / mesh.areas[:, None, None]
    return np.einsum("kij,kj->ik", inverse_mass, load)


def face_mean_jumps(mesh, coefficients):
    """Largest absolute face-mean jump over the interior faces of each cell."""
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.shape != (3, mesh.K):
        raise ValueError(
            f"coefficients have shape {coefficients.shape}, expected {(3, mesh.K)}"
        )
    cells = np.arange(mesh.K)
    jumps = np.zeros((mesh.K, 3))
    for face in range(3):
        neighbors = mesh.neighbors[:, face]
        neighbor_faces = mesh.neighbor_faces[:, face]
        connected = neighbors >= 0
        interior = cells[connected]
        neighbor = neighbors[connected]
        neighbor_face = neighbor_faces[connected]
        inside = 0.5 * (
            coefficients[face, interior] + coefficients[(face + 1) % 3, interior]
        )
        outside = 0.5 * (
            coefficients[neighbor_face, neighbor]
            + coefficients[(neighbor_face + 1) % 3, neighbor]
        )
        jumps[connected, face] = np.abs(outside - inside)
    return np.max(jumps, axis=1)


def uniform_refine(mesh):
    """Split every triangle into four by its edge midpoints.

    Returns the refined mesh and a ``(K, 4)`` array of child cell indices.
    """
    midpoint_index = {}
    midpoints = []
    for a, b in np.unique(np.sort(mesh.face_vertices.reshape(-1, 2), axis=1), axis=0):
        midpoint_index[(int(a), int(b))] = len(mesh.points) + len(midpoints)
        midpoints.append(0.5 * (mesh.points[a] + mesh.points[b]))
    points = np.concatenate([mesh.points, np.asarray(midpoints)])

    cells = []
    for a, b, c in mesh.cells:
        ab = midpoint_index[tuple(sorted((int(a), int(b))))]
        bc = midpoint_index[tuple(sorted((int(b), int(c))))]
        ca = midpoint_index[tuple(sorted((int(c), int(a))))]
        cells.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
    children = np.arange(4 * mesh.K, dtype=np.int64).reshape(mesh.K, 4)
    return TriangleMesh(points, cells), children


def label_cells(mesh, field, alpha=ALPHA, gamma=GAMMA, order=QUADRATURE_ORDER, refinement=None):
    """Label cells troubled by the predeclared `data-v3` reference rule.

    ``refinement`` optionally supplies a cached ``uniform_refine(mesh)`` result,
    which is worth reusing across the snapshots of one trajectory.
    """
    coarse = project_p1(mesh, field, order)
    jumps = face_mean_jumps(mesh, coarse)
    scales, _ = one_ring_robust_scale(np.mean(coarse, axis=0), mesh.neighbors)

    fine_mesh, children = uniform_refine(mesh) if refinement is None else refinement
    fine = project_p1(fine_mesh, field, order)
    child_jumps = np.max(face_mean_jumps(fine_mesh, fine)[children], axis=1)

    amplitude_positive = jumps >= alpha * scales
    refinement_persistent = jumps <= gamma * child_jumps
    labels = amplitude_positive & refinement_persistent
    diagnostics = {
        "jump": jumps,
        "scale": scales,
        "child_jump": child_jumps,
        "amplitude_positive": amplitude_positive,
        "refinement_persistent": refinement_persistent,
    }
    return labels, diagnostics
