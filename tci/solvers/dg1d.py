"""1D nodal discontinuous Galerkin solvers for hyperbolic conservation laws

    u_t + f(u)_x = 0

with periodic, inflow or transmissive boundary conditions, low-storage RK4
time stepping, and pluggable troubled-cell indicator + limiter.

``DG1DBase`` holds the mesh, reference operators and the generic solve loop;
``DG1D`` is the linear advection solver (f(u) = a u, upwind flux) used
throughout the thesis. See tci.solvers.burgers / tci.solvers.euler for the
nonlinear problems.
"""

import numpy as np

from tci.solvers.nodal import reference_element

# Carpenter-Kennedy low-storage five-stage RK4.
RK4A = np.array(
    [
        0.0,
        -567301805773.0 / 1357537059087.0,
        -2404267990393.0 / 2016746695238.0,
        -3550918686646.0 / 2091501179385.0,
        -1275806237668.0 / 842570457699.0,
    ]
)
RK4B = np.array(
    [
        1432997174477.0 / 9575080441755.0,
        5161836677717.0 / 13612068292357.0,
        1720146321549.0 / 2090206949498.0,
        3134564353537.0 / 4481467310338.0,
        2277821191437.0 / 14882151754819.0,
    ]
)
RK4C = np.array(
    [
        0.0,
        1432997174477.0 / 9575080441755.0,
        2526269341429.0 / 6820363962896.0,
        2006345519317.0 / 3224310063776.0,
        2802321613138.0 / 2924317926251.0,
    ]
)

BCS = ("periodic", "inflow", "transmissive")


class DG1DBase:
    """Mesh, reference-element operators and the generic RK solve loop.

    Subclasses implement ``rhs(u, time)`` and ``max_speed(u)`` (used for the
    CFL time step), and optionally ``local_velocity(u)`` (used by KXRCF to
    pick the inflow face).
    """

    def __init__(self, xmin, xmax, K, N, bc="periodic", bc_func=None):
        if bc not in BCS:
            raise ValueError(f"unknown bc {bc!r}")
        if bc == "inflow" and bc_func is None:
            raise ValueError("bc='inflow' requires bc_func")
        self.xmin, self.xmax = float(xmin), float(xmax)
        self.K, self.N = int(K), int(N)
        self.Np = self.N + 1
        self.bc = bc
        self.bc_func = bc_func

        self.r, self.V, self.invV, self.Dr, self.LIFT = reference_element(self.N)

        # Physical nodes: x[:, k] are the Np nodes of element k.
        self.VX = np.linspace(self.xmin, self.xmax, self.K + 1)
        va, vb = self.VX[:-1], self.VX[1:]
        self.h = vb - va
        self.x = va[None, :] + 0.5 * (1 + self.r[:, None]) * self.h[None, :]

        # Geometric factors (affine map).
        self.J = self.Dr @ self.x
        self.rx = 1.0 / self.J
        self.Fscale = 1.0 / self.J[[0, -1], :]
        # Outward normals at the two faces of every element.
        self.nx = np.array([[-1.0], [1.0]]) @ np.ones((1, self.K))

    # ------------------------------------------------------------------
    def cell_means(self, u):
        """Cell averages of a nodal field u of shape (Np, K)."""
        uh = self.invV @ u
        # P_0 = 1/sqrt(2) on [-1, 1], so the mean is c_0 / sqrt(2).
        return uh[0] / np.sqrt(2.0)

    def project(self, f):
        """Interpolate a callable onto the DG nodes."""
        return np.asarray(f(self.x), dtype=float)

    def cells_containing(self, points):
        """Boolean mask (K,) of elements containing any of the given points."""
        mask = np.zeros(self.K, dtype=bool)
        for p in np.atleast_1d(points):
            if self.xmin <= p <= self.xmax:
                k = min(int(np.searchsorted(self.VX, p, side="right")) - 1, self.K - 1)
                mask[max(k, 0)] = True
        return mask

    def traces(self, u):
        """Interior (M) and exterior (P) values at the two faces, shape (2, K).

        Periodic BCs wrap; otherwise the boundary exterior value equals the
        interior one (transmissive; inflow solvers override the boundary
        flux afterwards).
        """
        uM = np.vstack([u[0, :], u[-1, :]])
        if self.bc == "periodic":
            uP = np.vstack([np.roll(u[-1, :], 1), np.roll(u[0, :], -1)])
        else:
            uP = np.vstack(
                [
                    np.concatenate(([u[0, 0]], u[-1, :-1])),
                    np.concatenate((u[0, 1:], [u[-1, -1]])),
                ]
            )
        return uM, uP

    # ------------------------------------------------------------------
    def rhs(self, u, time):
        raise NotImplementedError

    def max_speed(self, u):
        raise NotImplementedError

    def local_velocity(self, u):
        """Characteristic velocity per cell (K,), used to pick inflow faces."""
        raise NotImplementedError

    def solve(
        self,
        u0,
        final_time,
        indicator=None,
        limiter=None,
        cfl=0.375,
        record_flags=False,
    ):
        """Evolve u0 (nodal array (Np, K) or callable) to ``final_time``.

        When ``indicator`` is given, after every RK stage the flagged cells
        are limited with ``limiter`` (default: minmod MUSCL limiter). The
        time step is recomputed each step from ``max_speed`` (CFL).

        Returns u, or (u, history) when record_flags is True; history is a
        list of (time, flags) per time step.
        """
        from tci.limiters import limit_cells

        u = self.project(u0) if callable(u0) else np.array(u0, dtype=float)
        if limiter is None:
            limiter = limit_cells

        xmin_node = np.min(np.abs(self.x[0, :] - self.x[1, :]))

        history = []
        resu = np.zeros_like(u)
        time = 0.0
        # A blowing-up solution drives max_speed up and dt towards zero;
        # cap the step count so that surfaces as an error, not a hang.
        max_steps = int(50 * np.ceil(final_time * self.max_speed(u) / (cfl * xmin_node)) + 1e4)
        steps = 0
        while time < final_time - 1e-12:
            steps += 1
            if steps > max_steps or not np.all(np.isfinite(u)):
                raise RuntimeError(
                    f"solve diverged: step {steps}, t={time:.3g}/{final_time}, "
                    f"max_speed={self.max_speed(u):.3g} (under-limited blow-up?)"
                )
            dt = min(cfl / self.max_speed(u) * xmin_node, final_time - time)
            if dt < 1e-9 * final_time:
                raise RuntimeError(
                    f"solve diverged: dt collapsed at t={time:.3g}/{final_time} "
                    f"(max_speed={self.max_speed(u):.3g}; under-limited blow-up?)"
                )
            flags = None
            for stage in range(5):
                rhsu = self.rhs(u, time + RK4C[stage] * dt)
                resu = RK4A[stage] * resu + dt * rhsu
                u = u + RK4B[stage] * resu
                if indicator is not None:
                    flags = indicator.flag(self, u)
                    u = limiter(self, u, flags)
            time += dt
            if record_flags and flags is not None:
                history.append((time, flags))

        if record_flags:
            return u, history
        return u


class DG1D(DG1DBase):
    """Linear advection u_t + a u_x = 0 with the upwind flux.

    The speed ``a`` is passed to :meth:`advect` (kept from the original
    API); ``solve`` uses the last speed set.
    """

    def __init__(self, xmin, xmax, K, N, bc="periodic", bc_func=None):
        super().__init__(xmin, xmax, K, N, bc=bc, bc_func=bc_func)
        self.a = 1.0

    def advec_rhs(self, u, time, a, alpha=0.0):
        """RHS of the semi-discrete advection equation.

        alpha = 0 gives the upwind flux, alpha = 1 the central flux.
        """
        uM, uP = self.traces(u)
        du = (uM - uP) * (a * self.nx - (1 - alpha) * np.abs(a * self.nx)) / 2.0

        if self.bc == "inflow":
            assert self.bc_func is not None
            if a >= 0:
                uin = self.bc_func(self.xmin - a * time)
                du[0, 0] = (
                    (u[0, 0] - uin)
                    * (a * self.nx[0, 0] - (1 - alpha) * np.abs(a * self.nx[0, 0]))
                    / 2.0
                )
                du[1, -1] = 0.0
            else:
                uin = self.bc_func(self.xmax - a * time)
                du[1, -1] = (
                    (u[-1, -1] - uin)
                    * (a * self.nx[1, -1] - (1 - alpha) * np.abs(a * self.nx[1, -1]))
                    / 2.0
                )
                du[0, 0] = 0.0

        return -a * self.rx * (self.Dr @ u) + self.LIFT @ (self.Fscale * du)

    def rhs(self, u, time):
        return self.advec_rhs(u, time, self.a)

    def max_speed(self, u):
        return np.abs(self.a)

    def local_velocity(self, u):
        return np.full(self.K, self.a)

    def advect(self, u0, a, final_time, **kwargs):
        """Advect u0 with speed ``a`` to ``final_time`` (see ``solve``)."""
        self.a = float(a)
        return self.solve(u0, final_time, **kwargs)
