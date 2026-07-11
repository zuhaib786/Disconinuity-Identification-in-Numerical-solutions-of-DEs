"""Slope limiting applied to cells flagged by a troubled-cell indicator.

Port of SlopeLimitN / SlopeLimitLin from the Hesthaven-Warburton codes: in
each flagged cell the solution is replaced by its linear part with a
minmod-limited slope against the neighbouring cell means.
"""

import numpy as np

from tci.solvers.nodal import minmod


def _neighbor_means(solver, v):
    """Cell means of the left/right neighbours (periodic wrap or clamped)."""
    if solver.bc == "periodic":
        vm1 = np.roll(v, 1)
        vp1 = np.roll(v, -1)
    else:
        vm1 = np.concatenate(([v[0]], v[:-1]))
        vp1 = np.concatenate((v[1:], [v[-1]]))
    return vm1, vp1


def limit_cells(solver, u, flags):
    """Return u with the flagged cells replaced by minmod-limited linears."""
    ids = np.where(flags)[0]
    if ids.size == 0:
        return u

    Np = solver.Np
    v = solver.cell_means(u)
    vm1, vp1 = _neighbor_means(solver, v)

    # Linear part (modes 0 and 1) of the flagged cells.
    uh = solver.invV @ u[:, ids]
    uh[2:] = 0.0
    ul = solver.V @ uh

    x1 = solver.x[:, ids]
    h = x1[-1, :] - x1[0, :]
    x0 = np.ones((Np, 1)) @ (x1[0, :] + h / 2).reshape(1, -1)
    hn = np.ones((Np, 1)) @ h.reshape(1, -1)
    ux = (2.0 / hn) * (solver.Dr @ ul)

    slope = minmod(
        np.array([ux[1, :], (vp1[ids] - v[ids]) / h, (v[ids] - vm1[ids]) / h])
    )
    ulimit = u.copy()
    ulimit[:, ids] = np.ones((Np, 1)) @ v[ids].reshape(1, -1) + (x1 - x0) * (
        np.ones((Np, 1)) @ slope.reshape(1, -1)
    )
    return ulimit
