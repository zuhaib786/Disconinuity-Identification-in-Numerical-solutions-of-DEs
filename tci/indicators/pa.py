"""Polynomial annihilation discontinuity detection (Archibald, Gelb & Yoon).

Vectorized reimplementation of legacy/1D/polynomialAnnhilation.py: for each
evaluation point the method builds the m-th order annihilation operator on
the m+1 nearest data points; |L_m f| is O(h) away from discontinuities and
O([f]) (the jump) at them.
"""

import numpy as np

from tci.indicators.base import Indicator


def _factorial(m):
    out = 1
    for i in range(2, m + 1):
        out *= i
    return out


def _stencil_windows(n_data, centers, m):
    """Start index of the (m+1)-wide window of data points nearest to each
    center index, replicating the alternating left/right growth of the
    original implementation."""
    starts = np.empty(len(centers), dtype=int)
    for out_i, idx in enumerate(centers):
        lo, hi = idx, idx + 1
        grow_right = True
        while hi - lo < m + 1:
            if not grow_right and lo - 1 >= 0:
                lo -= 1
            elif grow_right and hi < n_data:
                hi += 1
            grow_right = not grow_right
        starts[out_i] = lo
    return starts


def polynomial_annihilation(x_data, f_data, m=5, x_eval=None, eval_index=None):
    """Evaluate L_m f at points between data samples.

    x_data, f_data : 1-D arrays of sample locations and values
    m : annihilation order (stencil of m+1 nearest points)
    x_eval, eval_index : evaluation points and, for each, the index i such
        that the point lies in [x_data[i-1], x_data[i]). Default: midpoints
        between consecutive data points (eval_index = 1..n-1), matching the
        original code's per-point labelling.

    Returns |L_m f| at the evaluation points.
    """
    x_data = np.asarray(x_data, dtype=float).ravel()
    f_data = np.asarray(f_data, dtype=float).ravel()
    n = x_data.size
    if x_eval is None:
        eval_index = np.arange(1, n)
        x_eval = 0.5 * (x_data[:-1] + x_data[1:])
    else:
        x_eval = np.asarray(x_eval, dtype=float).ravel()
        eval_index = np.asarray(eval_index, dtype=int).ravel()

    starts = _stencil_windows(n, eval_index, m)
    offsets = np.arange(m + 1)
    stencil = starts[:, None] + offsets[None, :]          # (E, m+1)
    xs = x_data[stencil]                                   # (E, m+1)
    fs = f_data[stencil]                                   # (E, m+1)

    # Solve  sum_j c_j x_j^i = m! * delta_{i,m}  for each evaluation point.
    powers = xs[:, None, :] ** np.arange(m + 1)[None, :, None]  # (E, m+1, m+1)
    rhs = np.zeros((len(x_eval), m + 1))
    rhs[:, m] = _factorial(m)
    coeffs = np.linalg.solve(powers, rhs[..., None])[..., 0]   # (E, m+1)

    # Normalization: sum of coefficients over stencil points >= x_eval.
    right_of = xs >= x_eval[:, None]
    q = np.sum(np.where(right_of, coeffs, 0.0), axis=1)
    q = np.where(np.abs(q) > 1e-14, q, 1.0)

    return np.abs(np.sum(coeffs * fs, axis=1) / q)


class PAIndicator(Indicator):
    """Troubled-cell indicator from polynomial annihilation on cell means.

    L_m is evaluated at every interior element interface using the m+1
    nearest cell centers; both neighbouring cells are flagged when
    |L_m f| > threshold.
    """

    def __init__(self, m=5, threshold=1.0):
        self.m = int(m)
        self.threshold = float(threshold)

    def flag(self, solver, u):
        centers = 0.5 * (solver.VX[:-1] + solver.VX[1:])
        means = solver.cell_means(u)
        interfaces = solver.VX[1:-1]
        vals = polynomial_annihilation(
            centers,
            means,
            m=self.m,
            x_eval=interfaces,
            eval_index=np.arange(1, solver.K),
        )
        flags = np.zeros(solver.K, dtype=bool)
        hit = vals > self.threshold
        flags[:-1] |= hit
        flags[1:] |= hit
        return flags
