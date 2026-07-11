"""1D inviscid Burgers equation  u_t + (u^2/2)_x = 0  with a local
Lax-Friedrichs flux (Hesthaven & Warburton, chapter 5)."""

import numpy as np

from tci.solvers.dg1d import DG1DBase


class BurgersDG1D(DG1DBase):
    def rhs(self, u, time):
        fu = 0.5 * u * u
        uM, uP = self.traces(u)
        fM, fP = 0.5 * uM * uM, 0.5 * uP * uP
        c = np.maximum(np.abs(uM), np.abs(uP))  # local LF speed per face
        # Consistent with the upwind advection flux when f(u) = a u.
        df = (fM - fP) * self.nx / 2.0 - c / 2.0 * (uM - uP)
        return -self.rx * (self.Dr @ fu) + self.LIFT @ (self.Fscale * df)

    def max_speed(self, u):
        return max(float(np.max(np.abs(u))), 1e-12)

    def local_velocity(self, u):
        return self.cell_means(u)
