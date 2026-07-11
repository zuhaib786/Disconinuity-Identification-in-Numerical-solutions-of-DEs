"""Classical troubled-cell indicators for P1 triangular DG fields."""

import numpy as np

from tci.indicators.base import Indicator
from tci.limiters2d import neighbor_mean_bounds


class MinmodIndicator2D(Indicator):
    """Flags cells whose P1 vertex values exceed neighboring mean bounds."""

    def __init__(self, m_tvb=0.0):
        self.m_tvb = float(m_tvb)

    def flag(self, solver, u):
        _, lower, upper = neighbor_mean_bounds(solver, u)
        tolerance = self.m_tvb * solver.mesh.areas
        return np.any(
            (u < lower[None, :] - tolerance[None, :])
            | (u > upper[None, :] + tolerance[None, :]),
            axis=0,
        )


class KXRCFIndicator2D(Indicator):
    """Inflow-face jump indicator scaled by cell size and solution magnitude."""

    def __init__(self, threshold=1.0):
        self.threshold = float(threshold)

    def flag(self, solver, u):
        means = solver.cell_means(u)
        norm = np.maximum(np.max(np.abs(u), axis=0), 1e-12)
        values = np.zeros((solver.K, 3))
        face_points = solver.mesh.face_points
        face_velocity = solver.velocity_values(
            face_points[:, :, :, 0], face_points[:, :, :, 1], 0.0
        )
        beta = np.mean(
            np.sum(
                face_velocity * solver.mesh.face_normals[:, :, None, :], axis=3
            ),
            axis=2,
        )
        for face in range(3):
            neighbor = solver.all_neighbors[:, face]
            valid = (neighbor >= 0) & (beta[:, face] < 0.0)
            jump = np.zeros(solver.K)
            jump[valid] = np.abs(means[valid] - means[neighbor[valid]])
            values[:, face] = (
                jump
                * solver.mesh.edge_lengths[:, face]
                / (np.sqrt(solver.mesh.areas) * norm)
            )
        return np.max(values, axis=1) > self.threshold
