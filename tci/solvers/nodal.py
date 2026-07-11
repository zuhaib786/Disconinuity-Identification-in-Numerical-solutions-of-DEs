"""Nodal DG building blocks (Hesthaven & Warburton, "Nodal Discontinuous
Galerkin Methods", 2008). Ported from the MATLAB codes in DG-1D/BookVersion
and the Python translation in legacy/DG-1D/Appendix.py, with the global
state removed.
"""

from functools import lru_cache

import numpy as np
from scipy.special import gamma


def jacobi_p(x, alpha, beta, n):
    """Normalized Jacobi polynomial P_n^{(alpha,beta)} evaluated at x.

    Returns a 1-D array of the same length as x.
    """
    x = np.asarray(x, dtype=float).ravel()
    pl = np.zeros((n + 1, x.size))

    gamma0 = (
        2 ** (alpha + beta + 1)
        / (alpha + beta + 1)
        * gamma(alpha + 1)
        * gamma(beta + 1)
        / gamma(alpha + beta + 1)
    )
    pl[0] = 1.0 / np.sqrt(gamma0)
    if n == 0:
        return pl[0]

    gamma1 = (alpha + 1) * (beta + 1) / (alpha + beta + 3) * gamma0
    pl[1] = ((alpha + beta + 2) * x / 2 + (alpha - beta) / 2) / np.sqrt(gamma1)
    if n == 1:
        return pl[1]

    aold = 2 / (2 + alpha + beta) * np.sqrt(
        (alpha + 1) * (beta + 1) / (alpha + beta + 3)
    )
    for i in range(n - 1):
        h1 = 2 * (i + 1) + alpha + beta
        anew = (
            2
            / (h1 + 2)
            * np.sqrt(
                (i + 2)
                * (i + 2 + alpha + beta)
                * (i + 2 + alpha)
                * (i + 2 + beta)
                / (h1 + 1)
                / (h1 + 3)
            )
        )
        bnew = -(alpha**2 - beta**2) / h1 / (h1 + 2)
        pl[i + 2] = (-aold * pl[i] + (x - bnew) * pl[i + 1]) / anew
        aold = anew
    return pl[n]


def grad_jacobi_p(x, alpha, beta, n):
    """Derivative of the normalized Jacobi polynomial at x (1-D array)."""
    x = np.asarray(x, dtype=float).ravel()
    if n == 0:
        return np.zeros_like(x)
    return np.sqrt(n * (n + alpha + beta + 1)) * jacobi_p(
        x, alpha + 1, beta + 1, n - 1
    )


def jacobi_gq(alpha, beta, n):
    """Gauss quadrature points and weights for Jacobi (alpha, beta), order n."""
    if n == 0:
        return (
            np.array([(alpha - beta) / (alpha + beta + 2)]),
            np.array([2.0]),
        )

    k = np.arange(1, n + 1, dtype=float)
    h1 = 2 * np.arange(n + 1, dtype=float) + alpha + beta
    diag_main = -0.5 * (alpha**2 - beta**2) / (h1 + 2) / h1
    if alpha + beta < 1e-4:
        diag_main[0] = 0.0
    diag_off = (
        2
        / (h1[:n] + 2)
        * np.sqrt(
            k
            * (k + alpha + beta)
            * (k + alpha)
            * (k + beta)
            / (h1[:n] + 1)
            / (h1[:n] + 3)
        )
    )
    J = np.diag(diag_main) + np.diag(diag_off, 1)
    J = J + np.triu(J, 1).T

    d, v = np.linalg.eigh(J)
    w = (
        v[0] ** 2
        * 2 ** (alpha + beta + 1)
        / (alpha + beta + 1)
        * gamma(alpha + 1)
        * gamma(beta + 1)
        / gamma(alpha + beta + 1)
    )
    return d, w


@lru_cache(maxsize=None)
def jacobi_gl(n):
    """Gauss-Lobatto points for Legendre (alpha = beta = 0), order n."""
    if n == 1:
        return np.array([-1.0, 1.0])
    xint, _ = jacobi_gq(1, 1, n - 2)
    return np.concatenate(([-1.0], np.sort(xint), [1.0]))


def vandermonde_1d(n, r):
    """Vandermonde matrix V_{ij} = phi_j(r_i) for the modal Legendre basis."""
    r = np.asarray(r, dtype=float).ravel()
    v = np.zeros((r.size, n + 1))
    for j in range(n + 1):
        v[:, j] = jacobi_p(r, 0, 0, j)
    return v


def grad_vandermonde_1d(n, r):
    r = np.asarray(r, dtype=float).ravel()
    dv = np.zeros((r.size, n + 1))
    for j in range(n + 1):
        dv[:, j] = grad_jacobi_p(r, 0, 0, j)
    return dv


def dmatrix_1d(n, r, v):
    """Differentiation matrix Dr on the reference element."""
    return grad_vandermonde_1d(n, r) @ np.linalg.inv(v)


@lru_cache(maxsize=None)
def reference_element(n):
    """Cached reference-element operators for polynomial order n.

    Returns (r, V, invV, Dr, LIFT).
    """
    r = jacobi_gl(n)
    v = vandermonde_1d(n, r)
    inv_v = np.linalg.inv(v)
    dr = dmatrix_1d(n, r, v)

    np_ = n + 1
    emat = np.zeros((np_, 2))
    emat[0, 0] = 1.0
    emat[-1, 1] = 1.0
    lift = v @ (v.T @ emat)
    return r, v, inv_v, dr, lift


def minmod(v):
    """Row-wise minmod of a (m, n) array, returns shape (n,).

    m(v_1..v_m) = s * min|v_i| if all signs agree (s = common sign), else 0.
    """
    v = np.atleast_2d(np.asarray(v, dtype=float))
    m = v.shape[0]
    s = np.sum(np.sign(v), axis=0) / m
    out = np.zeros(v.shape[1])
    ids = np.abs(s) == 1
    out[ids] = s[ids] * np.min(np.abs(v[:, ids]), axis=0)
    return out


def minmod_tvb(v, m_tvb, h):
    """TVB-modified minmod (Shu): pass through v_1 when |v_1| <= M h^2."""
    v = np.atleast_2d(np.asarray(v, dtype=float))
    out = minmod(v)
    passthrough = np.abs(v[0]) <= m_tvb * h**2
    out[passthrough] = v[0, passthrough]
    return out
