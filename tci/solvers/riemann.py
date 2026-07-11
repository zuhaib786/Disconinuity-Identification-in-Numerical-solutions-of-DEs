"""Exact Riemann solver for the 1D Euler equations (Toro, "Riemann Solvers
and Numerical Methods for Fluid Dynamics", chapter 4). Used as the reference
solution for shock-tube benchmarks (Sod, Lax)."""

import numpy as np

GAMMA = 1.4


def _pressure_function(p, rho_k, p_k, c_k, gamma):
    """f_K(p) and its derivative for the star-region pressure iteration."""
    if p > p_k:  # shock
        a = 2.0 / ((gamma + 1) * rho_k)
        b = (gamma - 1) / (gamma + 1) * p_k
        sq = np.sqrt(a / (p + b))
        f = (p - p_k) * sq
        df = sq * (1.0 - 0.5 * (p - p_k) / (p + b))
    else:  # rarefaction
        f = (
            2.0
            * c_k
            / (gamma - 1)
            * ((p / p_k) ** ((gamma - 1) / (2 * gamma)) - 1.0)
        )
        df = 1.0 / (rho_k * c_k) * (p / p_k) ** (-(gamma + 1) / (2 * gamma))
    return f, df


def star_state(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma=GAMMA, tol=1e-10):
    """Pressure and velocity in the star region (Newton iteration)."""
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)
    du = u_r - u_l

    p = max(0.5 * (p_l + p_r) - 0.125 * du * (rho_l + rho_r) * (c_l + c_r), tol)
    for _ in range(60):
        f_l, df_l = _pressure_function(p, rho_l, p_l, c_l, gamma)
        f_r, df_r = _pressure_function(p, rho_r, p_r, c_r, gamma)
        dp = (f_l + f_r + du) / (df_l + df_r)
        p = max(p - dp, tol)
        if abs(dp) < tol * p:
            break
    f_l, _ = _pressure_function(p, rho_l, p_l, c_l, gamma)
    f_r, _ = _pressure_function(p, rho_r, p_r, c_r, gamma)
    u = 0.5 * (u_l + u_r) + 0.5 * (f_r - f_l)
    return p, u


def discontinuity_speeds(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma=GAMMA):
    """Speeds of the jump discontinuities in an Euler Riemann solution.

    The contact is always included. Shock speeds are included when the
    corresponding nonlinear wave is a shock; rarefaction heads and tails are
    omitted because the exact solution is continuous across them.
    """
    p_s, u_s = star_state(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma)
    gm1, gp1 = gamma - 1.0, gamma + 1.0
    speeds = []
    if p_s > p_l:
        c_l = np.sqrt(gamma * p_l / rho_l)
        speeds.append(
            u_l
            - c_l
            * np.sqrt(gp1 / (2 * gamma) * p_s / p_l + gm1 / (2 * gamma))
        )
    speeds.append(u_s)
    if p_s > p_r:
        c_r = np.sqrt(gamma * p_r / rho_r)
        speeds.append(
            u_r
            + c_r
            * np.sqrt(gp1 / (2 * gamma) * p_s / p_r + gm1 / (2 * gamma))
        )
    return np.asarray(speeds)


def sample(xi, rho_l, u_l, p_l, rho_r, u_r, p_r, gamma=GAMMA):
    """Solution (rho, u, p) of the Riemann problem at similarity points
    xi = x/t. xi may be an array."""
    xi = np.asarray(xi, dtype=float)
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)
    p_s, u_s = star_state(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma)

    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    p = np.empty_like(xi)

    gm1, gp1 = gamma - 1.0, gamma + 1.0

    left = xi <= u_s
    # ---- left of the contact -------------------------------------------
    if p_s > p_l:  # left shock
        rho_sl = rho_l * (p_s / p_l + gm1 / gp1) / (gm1 / gp1 * p_s / p_l + 1.0)
        s_l = u_l - c_l * np.sqrt(gp1 / (2 * gamma) * p_s / p_l + gm1 / (2 * gamma))
        pre = left & (xi < s_l)
        post = left & ~(xi < s_l)
        rho[pre], u[pre], p[pre] = rho_l, u_l, p_l
        rho[post], u[post], p[post] = rho_sl, u_s, p_s
    else:  # left rarefaction
        rho_sl = rho_l * (p_s / p_l) ** (1.0 / gamma)
        c_sl = c_l * (p_s / p_l) ** (gm1 / (2 * gamma))
        head, tail = u_l - c_l, u_s - c_sl
        pre = left & (xi < head)
        fan = left & (xi >= head) & (xi <= tail)
        post = left & (xi > tail)
        rho[pre], u[pre], p[pre] = rho_l, u_l, p_l
        u[fan] = 2.0 / gp1 * (c_l + gm1 / 2.0 * u_l + xi[fan])
        c_fan = c_l - gm1 / 2.0 * (u[fan] - u_l)
        rho[fan] = rho_l * (c_fan / c_l) ** (2.0 / gm1)
        p[fan] = p_l * (c_fan / c_l) ** (2.0 * gamma / gm1)
        rho[post], u[post], p[post] = rho_sl, u_s, p_s

    right = ~left
    # ---- right of the contact ------------------------------------------
    if p_s > p_r:  # right shock
        rho_sr = rho_r * (p_s / p_r + gm1 / gp1) / (gm1 / gp1 * p_s / p_r + 1.0)
        s_r = u_r + c_r * np.sqrt(gp1 / (2 * gamma) * p_s / p_r + gm1 / (2 * gamma))
        post = right & (xi < s_r)
        pre = right & ~(xi < s_r)
        rho[post], u[post], p[post] = rho_sr, u_s, p_s
        rho[pre], u[pre], p[pre] = rho_r, u_r, p_r
    else:  # right rarefaction
        rho_sr = rho_r * (p_s / p_r) ** (1.0 / gamma)
        c_sr = c_r * (p_s / p_r) ** (gm1 / (2 * gamma))
        head, tail = u_r + c_r, u_s + c_sr
        post = right & (xi < tail)
        fan = right & (xi >= tail) & (xi <= head)
        pre = right & (xi > head)
        rho[post], u[post], p[post] = rho_sr, u_s, p_s
        u[fan] = 2.0 / gp1 * (-c_r + gm1 / 2.0 * u_r + xi[fan])
        c_fan = c_r + gm1 / 2.0 * (u[fan] - u_r)
        rho[fan] = rho_r * (c_fan / c_r) ** (2.0 / gm1)
        p[fan] = p_r * (c_fan / c_r) ** (2.0 * gamma / gm1)
        rho[pre], u[pre], p[pre] = rho_r, u_r, p_r

    return rho, u, p


def sod_exact(x, t, x0=0.5, gamma=GAMMA):
    """Exact (rho, u, p) of the Sod problem at time t."""
    if t <= 0:
        raise ValueError("t must be positive")
    return sample((np.asarray(x) - x0) / t, 1.0, 0.0, 1.0, 0.125, 0.0, 0.1, gamma)
