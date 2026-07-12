"""Positivity-protected fixed-P1 DG solver for the 2D Euler equations."""

import time

import numpy as np

from tci.limiters2d import limit_p1
from tci.solvers.dg2d import AdvectionDG2D

GAMMA = 1.4


class EulerTimeoutError(TimeoutError):
    """Wall-clock timeout carrying a restartable Euler state."""

    def __init__(self, U, current_time, final_time, steps):
        super().__init__(
            f"2D Euler solve timed out at t={current_time:.6g}/{final_time:.6g} "
            f"after {steps} accepted steps"
        )
        self.U = U
        self.current_time = float(current_time)
        self.final_time = float(final_time)
        self.steps = int(steps)


def primitive_to_conserved_2d(rho, u, v, p, gamma=GAMMA):
    return np.stack(
        [rho, rho * u, rho * v, p / (gamma - 1) + 0.5 * rho * (u * u + v * v)],
        axis=-1,
    )


def conserved_to_primitive_2d(U, gamma=GAMMA):
    rho = U[..., 0]
    u = U[..., 1] / rho
    v = U[..., 2] / rho
    p = (gamma - 1) * (
        U[..., 3] - 0.5 * (U[..., 1] ** 2 + U[..., 2] ** 2) / rho
    )
    return rho, u, v, p


def euler_flux_2d(U, gamma=GAMMA):
    rho, u, v, p = conserved_to_primitive_2d(U, gamma)
    fx = np.stack(
        [U[..., 1], U[..., 1] * u + p, U[..., 2] * u, (U[..., 3] + p) * u],
        axis=-1,
    )
    fy = np.stack(
        [U[..., 2], U[..., 1] * v, U[..., 2] * v + p, (U[..., 3] + p) * v],
        axis=-1,
    )
    return fx, fy


def normal_flux_2d(U, normal, gamma=GAMMA):
    fx, fy = euler_flux_2d(U, gamma)
    return fx * normal[..., 0, None] + fy * normal[..., 1, None]


def pressure_2d(U, gamma=GAMMA):
    return conserved_to_primitive_2d(U, gamma)[3]


def positivity_scale(U, gamma=GAMMA, rho_floor=1e-12, p_floor=1e-12):
    """Scale nodal deviations toward admissible cell means, conservatively."""
    U = np.asarray(U, dtype=float).copy()
    means = np.mean(U, axis=0)
    mean_rho = means[:, 0]
    mean_pressure = pressure_2d(means, gamma)
    if np.any(mean_rho <= rho_floor) or np.any(mean_pressure <= p_floor):
        raise ValueError("Euler stage has a non-admissible cell mean")
    for cell in range(U.shape[1]):
        rho_min = float(np.min(U[:, cell, 0]))
        if rho_min < rho_floor:
            theta = min(
                1.0,
                (mean_rho[cell] - rho_floor) / (mean_rho[cell] - rho_min),
            )
            U[:, cell] = means[cell] + theta * (U[:, cell] - means[cell])
        if np.min(pressure_2d(U[:, cell], gamma)) < p_floor:
            direction = U[:, cell] - means[cell]
            theta = 1.0
            for node in range(3):
                if pressure_2d(U[node, cell], gamma) >= p_floor:
                    continue
                lo, hi = 0.0, 1.0
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    trial = means[cell] + mid * direction[node]
                    if pressure_2d(trial, gamma) >= p_floor:
                        lo = mid
                    else:
                        hi = mid
                theta = min(theta, lo)
            U[:, cell] = means[cell] + theta * direction
    return U


def limit_euler_p1(solver, U, flags):
    limited = np.empty_like(U)
    for component in range(4):
        limited[:, :, component] = limit_p1(solver, U[:, :, component], flags)
    return positivity_scale(
        limited, solver.gamma, solver.rho_floor, solver.p_floor
    )


class EulerDG2D:
    """P1 triangular DG with LLF flux, SSP-RK3, limiting and positivity."""

    def __init__(
        self,
        mesh,
        gamma=GAMMA,
        boundary_state=None,
        periodic=(False, False),
        rho_floor=1e-12,
        p_floor=1e-12,
    ):
        geometry = AdvectionDG2D(mesh, velocity=(0.0, 0.0), periodic=periodic)
        self.mesh = mesh
        self.K = mesh.K
        self.basis_gradients = geometry.basis_gradients
        self.mass_inv = geometry.mass_inv
        self.lift = geometry.lift
        self.all_neighbors = geometry.all_neighbors
        self.all_neighbor_faces = geometry.all_neighbor_faces
        self.gamma = float(gamma)
        self.boundary_state = boundary_state
        self.rho_floor = float(rho_floor)
        self.p_floor = float(p_floor)

    @property
    def nodes(self):
        return self.mesh.points[self.mesh.cells].transpose(1, 0, 2)

    def project(self, function):
        nodes = self.nodes
        U = np.asarray(function(nodes[:, :, 0], nodes[:, :, 1]), dtype=float)
        if U.shape != (3, self.K, 4):
            raise ValueError(f"projected Euler state has shape {U.shape}, expected {(3, self.K, 4)}")
        return U

    def cell_means(self, U):
        return np.mean(U, axis=0)

    def integral(self, U):
        return np.sum(self.mesh.areas[:, None] * self.cell_means(U), axis=0)

    def _face_traces(self, U, face, time_value):
        local_nodes = [face, (face + 1) % 3]
        U_minus = U[local_nodes].transpose(1, 0, 2)
        U_plus = U_minus.copy()
        neighbors = self.all_neighbors[:, face]
        neighbor_faces = self.all_neighbor_faces[:, face]
        connected = neighbors >= 0
        if np.any(connected):
            neighbor = neighbors[connected]
            neighbor_face = neighbor_faces[connected]
            U_plus[connected, 0] = U[(neighbor_face + 1) % 3, neighbor]
            U_plus[connected, 1] = U[neighbor_face, neighbor]
        boundary = ~connected
        if self.boundary_state is not None and np.any(boundary):
            xy = self.mesh.face_points[boundary, face]
            normals = self.mesh.face_normals[boundary, face]
            values = np.asarray(
                self.boundary_state(
                    xy[:, :, 0],
                    xy[:, :, 1],
                    time_value,
                    U_minus[boundary],
                    normals,
                ),
                dtype=float,
            )
            if values.shape != U_plus[boundary].shape:
                raise ValueError("boundary_state returned an incompatible Euler state")
            U_plus[boundary] = values
        return U_minus, U_plus

    def rhs(self, U, time_value=0.0):
        U = np.asarray(U, dtype=float)
        if U.shape != (3, self.K, 4):
            raise ValueError(f"U has shape {U.shape}, expected {(3, self.K, 4)}")
        fx, fy = euler_flux_2d(U, self.gamma)
        divergence = np.einsum("ikc,ki->kc", fx, self.basis_gradients[:, :, 0])
        divergence += np.einsum("ikc,ki->kc", fy, self.basis_gradients[:, :, 1])
        rhs = np.broadcast_to(-divergence, (3, self.K, 4)).copy()
        for face in range(3):
            U_minus, U_plus = self._face_traces(U, face, time_value)
            normal = self.mesh.face_normals[:, face, None, :]
            fn_minus = normal_flux_2d(U_minus, normal, self.gamma)
            fn_plus = normal_flux_2d(U_plus, normal, self.gamma)
            rho_m, um, vm, pm = conserved_to_primitive_2d(U_minus, self.gamma)
            rho_p, up, vp, pp = conserved_to_primitive_2d(U_plus, self.gamma)
            nx = normal[..., 0]
            ny = normal[..., 1]
            speed_m = np.abs(um * nx + vm * ny) + np.sqrt(self.gamma * pm / rho_m)
            speed_p = np.abs(up * nx + vp * ny) + np.sqrt(self.gamma * pp / rho_p)
            alpha = np.maximum(speed_m, speed_p)
            numerical = 0.5 * (fn_minus + fn_plus) - 0.5 * alpha[..., None] * (
                U_plus - U_minus
            )
            correction = fn_minus - numerical
            rhs += np.einsum("kiq,kqc->ikc", self.lift[:, face], correction)
        return rhs

    def stable_dt(self, U, cfl=0.1):
        if cfl <= 0:
            raise ValueError("cfl must be positive")
        rho, u, v, p = conserved_to_primitive_2d(U, self.gamma)
        if np.any(rho <= 0) or np.any(p <= 0):
            raise ValueError("stable_dt requires an admissible Euler state")
        speed = np.sqrt(u * u + v * v) + np.sqrt(self.gamma * p / rho)
        cell_speed = np.max(speed, axis=0)
        rates = 3.0 * cell_speed / (2.0 * self.mesh.areas) * np.sum(
            self.mesh.edge_lengths, axis=1
        )
        return cfl / float(np.max(rates))

    def solve(
        self,
        U0,
        final_time,
        indicator=None,
        cfl=0.1,
        max_seconds=None,
        max_retries=12,
        start_time=0.0,
        progress_callback=None,
        progress_interval=100,
    ):
        U = self.project(U0) if callable(U0) else np.asarray(U0, dtype=float).copy()
        U = positivity_scale(U, self.gamma, self.rho_floor, self.p_floor)
        deadline = None if max_seconds is None else time.monotonic() + max_seconds
        current_time = float(start_time)
        steps = 0
        while current_time < final_time - 1e-14:
            if deadline is not None and time.monotonic() >= deadline:
                raise EulerTimeoutError(U, current_time, final_time, steps)
            dt = min(self.stable_dt(U, cfl), final_time - current_time)
            candidate = None
            for retry in range(max_retries + 1):
                try:
                    U1 = U + dt * self.rhs(U, current_time)
                    U1 = self._postprocess(U1, indicator)
                    U2 = 0.75 * U + 0.25 * (
                        U1 + dt * self.rhs(U1, current_time + dt)
                    )
                    U2 = self._postprocess(U2, indicator)
                    candidate = U / 3.0 + 2.0 / 3.0 * (
                        U2 + dt * self.rhs(U2, current_time + 0.5 * dt)
                    )
                    candidate = self._postprocess(candidate, indicator)
                    break
                except (ValueError, FloatingPointError):
                    if retry == max_retries:
                        raise RuntimeError("2D Euler stage remained non-admissible")
                    dt *= 0.5
            assert candidate is not None
            U = candidate
            current_time += dt
            steps += 1
            if progress_callback is not None and steps % progress_interval == 0:
                progress_callback(current_time, U, steps)
        if progress_callback is not None:
            progress_callback(current_time, U, steps)
        return U

    def _postprocess(self, U, indicator):
        if indicator is not None:
            flags = indicator.flag(self, U[:, :, 0])
            return limit_euler_p1(self, U, flags)
        return positivity_scale(U, self.gamma, self.rho_floor, self.p_floor)


def reflective_boundary_2d(x, y, time_value, U_minus, normal):
    U_plus = U_minus.copy()
    momentum = U_minus[..., 1:3]
    normal_momentum = np.sum(momentum * normal[:, None, :], axis=2)
    U_plus[..., 1:3] = momentum - 2.0 * normal_momentum[..., None] * normal[:, None, :]
    return U_plus
