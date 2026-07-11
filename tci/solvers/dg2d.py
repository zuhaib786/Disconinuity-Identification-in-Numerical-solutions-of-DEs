"""Fixed-P1 nodal DG solver for scalar advection on triangular meshes."""

import time

import numpy as np

from tci.mesh import TriangleMesh


class AdvectionDG2D:
    """Solve ``u_t + velocity . grad(u) = 0`` on affine triangles.

    The three nodal values in each cell live at its vertices, but remain
    discontinuous across shared faces. Boundary data are imposed only where
    the velocity enters the domain; outgoing faces use the interior trace.
    """

    N = 1
    Np = 3
    reference_mass = np.array(
        [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]
    ) / 6.0
    reference_mass_inv = np.array(
        [[3.0, -1.0, -1.0], [-1.0, 3.0, -1.0], [-1.0, -1.0, 3.0]]
    ) * 1.5
    face_mass_unit = np.array([[2.0, 1.0], [1.0, 2.0]]) / 6.0

    def __init__(
        self,
        mesh: TriangleMesh,
        velocity=(1.0, 0.0),
        boundary_func=None,
        periodic=(False, False),
    ):
        self.mesh = mesh
        self.K = mesh.K
        self.velocity = velocity
        if not callable(velocity):
            self.velocity = np.asarray(velocity, dtype=float)
            if self.velocity.shape != (2,) or not np.all(np.isfinite(self.velocity)):
                raise ValueError("velocity must be a finite vector of shape (2,)")
        self.boundary_func = boundary_func
        self.periodic = tuple(bool(value) for value in periodic)
        self.periodic_neighbors, self.periodic_neighbor_faces = mesh.periodic_face_map(
            self.periodic
        )
        self.all_neighbors = mesh.neighbors.copy()
        self.all_neighbor_faces = mesh.neighbor_faces.copy()
        periodic_mask = self.periodic_neighbors >= 0
        self.all_neighbors[periodic_mask] = self.periodic_neighbors[periodic_mask]
        self.all_neighbor_faces[periodic_mask] = self.periodic_neighbor_faces[periodic_mask]

        vertices = mesh.points[mesh.cells]
        twice_area = 2.0 * mesh.areas
        self.basis_gradients = np.empty((self.K, 3, 2))
        self.basis_gradients[:, 0] = np.column_stack(
            [vertices[:, 1, 1] - vertices[:, 2, 1], vertices[:, 2, 0] - vertices[:, 1, 0]]
        ) / twice_area[:, None]
        self.basis_gradients[:, 1] = np.column_stack(
            [vertices[:, 2, 1] - vertices[:, 0, 1], vertices[:, 0, 0] - vertices[:, 2, 0]]
        ) / twice_area[:, None]
        self.basis_gradients[:, 2] = np.column_stack(
            [vertices[:, 0, 1] - vertices[:, 1, 1], vertices[:, 1, 0] - vertices[:, 0, 0]]
        ) / twice_area[:, None]

        self.mass_inv = (
            2.0
            * self.reference_mass_inv[None, :, :]
            / mesh.areas[:, None, None]
        )
        self.lift = np.empty((self.K, 3, 3, 2))
        for face in range(3):
            restriction_t = np.zeros((3, 2))
            restriction_t[face, 0] = 1.0
            restriction_t[(face + 1) % 3, 1] = 1.0
            face_load = restriction_t @ self.face_mass_unit
            self.lift[:, face] = np.einsum(
                "kij,jq,k->kiq",
                self.mass_inv,
                face_load,
                mesh.edge_lengths[:, face],
            )

    @property
    def nodes(self):
        """Physical DG nodes with shape ``(3, K, 2)``."""
        return self.mesh.points[self.mesh.cells].transpose(1, 0, 2)

    def project(self, function):
        nodes = self.nodes
        values = np.asarray(function(nodes[:, :, 0], nodes[:, :, 1]), dtype=float)
        if values.shape != (3, self.K):
            raise ValueError(f"projected function returned {values.shape}, expected {(3, self.K)}")
        return values

    def cell_means(self, u):
        return np.mean(u, axis=0)

    def integral(self, u):
        return float(np.sum(self.mesh.areas * self.cell_means(u)))

    def velocity_values(self, x, y, time=0.0):
        shape = np.broadcast_shapes(np.shape(x), np.shape(y))
        if callable(self.velocity):
            values = np.asarray(self.velocity(x, y, time), dtype=float)
        else:
            values = np.broadcast_to(self.velocity, shape + (2,))
        if values.shape != shape + (2,) or not np.all(np.isfinite(values)):
            raise ValueError(f"velocity field returned {values.shape}, expected {shape + (2,)}")
        return values

    def l2_error(self, u, exact):
        target = self.project(exact) if callable(exact) else np.asarray(exact, dtype=float)
        error = (u - target).T
        mass = self.mesh.areas[:, None, None] * np.array(
            [[[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]]
        ) / 12.0
        return float(np.sqrt(np.sum(np.einsum("ki,kij,kj->k", error, mass, error))))

    def rhs(self, u, time=0.0):
        u = np.asarray(u, dtype=float)
        if u.shape != (3, self.K):
            raise ValueError(f"u has shape {u.shape}, expected {(3, self.K)}")

        nodes = self.nodes
        nodal_velocity = self.velocity_values(nodes[:, :, 0], nodes[:, :, 1], time)
        flux = u[:, :, None] * nodal_velocity
        divergence = np.einsum("ikd,kid->k", flux, self.basis_gradients)
        rhs = np.broadcast_to(-divergence, (3, self.K)).copy()

        cells = np.arange(self.K)
        for face in range(3):
            local_nodes = [face, (face + 1) % 3]
            u_minus = u[local_nodes].T
            beta = np.sum(
                nodal_velocity[local_nodes].transpose(1, 0, 2)
                * self.mesh.face_normals[:, face, None, :],
                axis=2,
            )
            u_plus = u_minus.copy()
            neighbors = self.all_neighbors[:, face]
            neighbor_faces = self.all_neighbor_faces[:, face]
            connected = neighbors >= 0
            if np.any(connected):
                neighbor = neighbors[connected]
                neighbor_face = neighbor_faces[connected]
                u_plus[connected, 0] = u[(neighbor_face + 1) % 3, neighbor]
                u_plus[connected, 1] = u[neighbor_face, neighbor]
            boundary = ~connected
            if self.boundary_func is not None and np.any(boundary):
                xy = self.mesh.face_points[boundary, face]
                boundary_values = np.asarray(
                    self.boundary_func(xy[:, :, 0], xy[:, :, 1], time), dtype=float
                )
                if boundary_values.shape == ():
                    boundary_values = np.full((np.count_nonzero(boundary), 2), float(boundary_values))
                if boundary_values.shape != (np.count_nonzero(boundary), 2):
                    raise ValueError("boundary_func must return one value per face node")
                u_plus[boundary] = boundary_values
            correction = np.where(beta < 0.0, beta * (u_minus - u_plus), 0.0)
            rhs += np.einsum("kiq,kq->ik", self.lift[:, face], correction)
        return rhs

    def stable_dt(self, cfl=0.2, time=0.0):
        if cfl <= 0:
            raise ValueError("cfl must be positive")
        face_points = self.mesh.face_points
        face_velocity = self.velocity_values(
            face_points[:, :, :, 0], face_points[:, :, :, 1], time
        )
        normal_speed = np.max(
            np.abs(np.sum(face_velocity * self.mesh.face_normals[:, :, None, :], axis=3)),
            axis=2,
        )
        rates = 3.0 / (2.0 * self.mesh.areas) * np.sum(
            normal_speed * self.mesh.edge_lengths, axis=1
        )
        max_rate = float(np.max(rates))
        return np.inf if max_rate == 0.0 else cfl / max_rate

    def solve(
        self,
        u0,
        final_time,
        cfl=0.2,
        indicator=None,
        limiter=None,
        record_flags=False,
        max_seconds=None,
    ):
        if final_time < 0:
            raise ValueError("final_time must be nonnegative")
        u = self.project(u0) if callable(u0) else np.asarray(u0, dtype=float).copy()
        if u.shape != (3, self.K):
            raise ValueError(f"u0 has shape {u.shape}, expected {(3, self.K)}")
        if final_time == 0:
            return (u, []) if record_flags else u
        if limiter is None:
            from tci.limiters2d import limit_p1

            limiter = limit_p1

        history = []
        time = 0.0
        deadline = None if max_seconds is None else time_module() + float(max_seconds)
        while time < final_time - 1e-14:
            if deadline is not None and time_module() >= deadline:
                raise TimeoutError(
                    f"2D solve exceeded its {max_seconds:.1f}s wall-clock limit at t={time:.4g}/{final_time}"
                )
            nominal_dt = self.stable_dt(cfl, time)
            if not np.isfinite(nominal_dt):
                return (u, history) if record_flags else u
            dt = min(nominal_dt, final_time - time)
            u1 = u + dt * self.rhs(u, time)
            if indicator is not None:
                flags = indicator.flag(self, u1)
                u1 = limiter(self, u1, flags)
            u2 = 0.75 * u + 0.25 * (u1 + dt * self.rhs(u1, time + dt))
            if indicator is not None:
                flags = indicator.flag(self, u2)
                u2 = limiter(self, u2, flags)
            u = u / 3.0 + 2.0 / 3.0 * (
                u2 + dt * self.rhs(u2, time + 0.5 * dt)
            )
            flags = None
            if indicator is not None:
                flags = indicator.flag(self, u)
                u = limiter(self, u, flags)
            time += dt
            if record_flags and flags is not None:
                history.append((time, flags))
        return (u, history) if record_flags else u


def time_module():
    return time.perf_counter()
