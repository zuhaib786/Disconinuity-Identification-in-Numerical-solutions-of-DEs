"""Classical troubled-cell indicators: minmod/TVB and KXRCF."""

import numpy as np

from tci.indicators.base import Indicator
from tci.limiters import _neighbor_means
from tci.solvers.nodal import minmod, minmod_tvb


class MinmodIndicator(Indicator):
    """Flags cells where the minmod-reconstructed interface values differ
    from the DG interface values (the detection step of SlopeLimitN).

    m_tvb > 0 gives the TVB-modified variant (Cockburn & Shu): cells whose
    interface deviation is below M h^2 are never flagged.
    """

    def __init__(self, m_tvb=0.0, eps=1e-8):
        self.m_tvb = float(m_tvb)
        self.eps = float(eps)

    def flag(self, solver, u):
        v = solver.cell_means(u)
        vm1, vp1 = _neighbor_means(solver, v)
        ue1, ue2 = u[0, :], u[-1, :]

        args1 = np.array([v - ue1, v - vm1, vp1 - v])
        args2 = np.array([ue2 - v, v - vm1, vp1 - v])
        if self.m_tvb > 0:
            ve1 = v - minmod_tvb(args1, self.m_tvb, solver.h)
            ve2 = v + minmod_tvb(args2, self.m_tvb, solver.h)
        else:
            ve1 = v - minmod(args1)
            ve2 = v + minmod(args2)

        return (np.abs(ve1 - ue1) > self.eps) | (np.abs(ve2 - ue2) > self.eps)


class KXRCFIndicator(Indicator):
    """Krivodonova et al. (KXRCF) shock detector.

    I_k = |jump of u across the inflow face| / (h^((N+1)/2) * ||u||_inf,cell);
    the cell is flagged when I_k exceeds ``threshold`` (usually 1).
    """

    def __init__(self, threshold=1.0, a=1.0):
        self.threshold = float(threshold)
        # Sign of the advection speed decides which face is inflow.
        self.a = float(a)

    def flag(self, solver, u):
        if solver.bc == "periodic":
            left_nb = np.roll(u[-1, :], 1)
            right_nb = np.roll(u[0, :], -1)
        else:
            left_nb = np.concatenate(([u[0, 0]], u[-1, :-1]))
            right_nb = np.concatenate((u[0, 1:], [u[-1, -1]]))

        if self.a >= 0:
            jump = np.abs(u[0, :] - left_nb)
        else:
            jump = np.abs(u[-1, :] - right_nb)

        norm = np.max(np.abs(u), axis=0)
        scale = solver.h ** ((solver.N + 1) / 2.0) * norm
        with np.errstate(divide="ignore", invalid="ignore"):
            ind = np.where(scale > 1e-14, jump / scale, 0.0)
        return ind > self.threshold
