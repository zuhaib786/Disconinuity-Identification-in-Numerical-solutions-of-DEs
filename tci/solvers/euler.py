"""1D compressible Euler equations in conservation form

    U_t + F(U)_x = 0,   U = (rho, rho u, E),
    F = (rho u, rho u^2 + p, (E + p) u),   p = (gamma - 1)(E - rho u^2 / 2)

solved with nodal DG, a local Lax-Friedrichs flux, SSP-RK3 time stepping and
per-stage limiting of the conserved variables in cells flagged by a
troubled-cell indicator applied to the density field.

State arrays have shape (Np, K, 3). Standard test cases (Sod, Lax,
Shu-Osher) are provided as initial-condition helpers.
"""

import numpy as np

from tci.solvers.dg1d import DG1DBase

GAMMA = 1.4


def primitive_to_conserved(rho, u, p, gamma=GAMMA):
    return np.stack([rho, rho * u, p / (gamma - 1) + 0.5 * rho * u * u], axis=-1)


def conserved_to_primitive(U, gamma=GAMMA):
    rho = U[..., 0]
    u = U[..., 1] / rho
    p = (gamma - 1) * (U[..., 2] - 0.5 * rho * u * u)
    return rho, u, p


def euler_flux(U, gamma=GAMMA):
    rho, u, p = conserved_to_primitive(U, gamma)
    return np.stack(
        [U[..., 1], U[..., 1] * u + p, (U[..., 2] + p) * u], axis=-1
    )


class EulerDG1D(DG1DBase):
    """Nodal DG for the 1D Euler system (transmissive BCs by default)."""

    def __init__(self, xmin, xmax, K, N, bc="transmissive", gamma=GAMMA):
        if bc not in ("transmissive", "periodic"):
            raise ValueError("EulerDG1D supports transmissive or periodic bc")
        super().__init__(xmin, xmax, K, N, bc=bc)
        self.gamma = gamma

    # -- system traces: apply the scalar trace helper per component -------
    def _traces3(self, U):
        UM = np.empty((2, self.K, 3))
        UP = np.empty((2, self.K, 3))
        for c in range(3):
            UM[:, :, c], UP[:, :, c] = self.traces(U[:, :, c])
        return UM, UP

    def sound_speed(self, U):
        rho, _, p = conserved_to_primitive(U, self.gamma)
        return np.sqrt(self.gamma * np.maximum(p, 1e-13) / np.maximum(rho, 1e-13))

    def max_speed(self, u):
        U = u
        _, u, _ = conserved_to_primitive(U, self.gamma)
        return float(np.max(np.abs(u) + self.sound_speed(U)))

    def local_velocity(self, u):
        """Mean flow velocity per cell. Indicators receive only the density
        field, so the full state is taken from the current solve (stashed by
        ``_limit``) when a 2-D array is passed."""
        U = u if u.ndim == 3 else self._U
        return self.cell_means(U[:, :, 1]) / self.cell_means(U[:, :, 0])

    def rhs(self, u, time):
        U = u
        FU = euler_flux(U, self.gamma)
        UM, UP = self._traces3(U)
        FM, FP = euler_flux(UM, self.gamma), euler_flux(UP, self.gamma)

        _, uM, pM = conserved_to_primitive(UM, self.gamma)
        _, uP, pP = conserved_to_primitive(UP, self.gamma)
        cM = np.sqrt(self.gamma * np.maximum(pM, 1e-13) / np.maximum(UM[..., 0], 1e-13))
        cP = np.sqrt(self.gamma * np.maximum(pP, 1e-13) / np.maximum(UP[..., 0], 1e-13))
        lam = np.maximum(np.abs(uM) + cM, np.abs(uP) + cP)[..., None]

        dF = (FM - FP) * self.nx[..., None] / 2.0 - lam / 2.0 * (UM - UP)

        out = np.empty_like(U)
        for c in range(3):
            out[:, :, c] = -self.rx * (self.Dr @ FU[:, :, c]) + self.LIFT @ (
                self.Fscale * dF[:, :, c]
            )
        return out

    # -- SSP-RK3 with per-stage limiting ----------------------------------
    def _limit(self, U, indicator, limiter):
        if indicator is None:
            return U, None
        self._U = U  # full state for local_velocity during detection
        flags = indicator.flag(self, U[:, :, 0])  # detect on density
        for c in range(3):
            U[:, :, c] = limiter(self, U[:, :, c], flags)
        return U, flags

    def solve(
        self,
        u0,
        final_time,
        indicator=None,
        limiter=None,
        cfl=0.2,
        record_flags=False,
    ):
        from tci.limiters import limit_cells

        U = self.project(u0) if callable(u0) else np.array(u0, dtype=float)
        if limiter is None:
            limiter = limit_cells

        xmin_node = np.min(np.abs(self.x[0, :] - self.x[1, :]))
        history = []
        time = 0.0
        # Cap the step count so blow-up surfaces as an error, not a hang.
        max_steps = int(50 * np.ceil(final_time * self.max_speed(U) / (cfl * xmin_node)) + 1e4)
        steps = 0
        while time < final_time - 1e-12:
            steps += 1
            if steps > max_steps or not np.all(np.isfinite(U)):
                raise RuntimeError(
                    f"solve diverged: step {steps}, t={time:.3g}/{final_time}, "
                    f"max_speed={self.max_speed(U):.3g} (under-limited blow-up?)"
                )
            dt = min(cfl / self.max_speed(U) * xmin_node, final_time - time)
            if dt < 1e-9 * final_time:
                raise RuntimeError(
                    f"solve diverged: dt collapsed at t={time:.3g}/{final_time} "
                    f"(max_speed={self.max_speed(U):.3g}; under-limited blow-up?)"
                )

            U1 = U + dt * self.rhs(U, time)
            U1, _ = self._limit(U1, indicator, limiter)
            U2 = 0.75 * U + 0.25 * (U1 + dt * self.rhs(U1, time + dt))
            U2, _ = self._limit(U2, indicator, limiter)
            U = U / 3.0 + 2.0 / 3.0 * (U2 + dt * self.rhs(U2, time + dt / 2))
            U, flags = self._limit(U, indicator, limiter)

            time += dt
            if record_flags and flags is not None:
                history.append((time, flags))

        if record_flags:
            return U, history
        return U


# -- standard shock-tube initial conditions -------------------------------
def sod_initial(x):
    """Sod shock tube on [0, 1], diaphragm at 0.5; run to T = 0.2."""
    rho = np.where(x < 0.5, 1.0, 0.125)
    u = np.zeros_like(x)
    p = np.where(x < 0.5, 1.0, 0.1)
    return primitive_to_conserved(rho, u, p)


def lax_initial(x):
    """Lax problem on [0, 1], diaphragm at 0.5; run to T = 0.13."""
    rho = np.where(x < 0.5, 0.445, 0.5)
    u = np.where(x < 0.5, 0.698, 0.0)
    p = np.where(x < 0.5, 3.528, 0.571)
    return primitive_to_conserved(rho, u, p)


def shu_osher_initial(x):
    """Shu-Osher shock/entropy-wave interaction on [-5, 5]; run to T = 1.8."""
    rho = np.where(x < -4.0, 3.857143, 1.0 + 0.2 * np.sin(5.0 * x))
    u = np.where(x < -4.0, 2.629369, 0.0)
    p = np.where(x < -4.0, 10.33333, 1.0)
    return primitive_to_conserved(rho, u, p)
