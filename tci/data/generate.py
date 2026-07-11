"""Randomized training data for troubled-cell indicators.

Port of the thesis data-generation procedures (sections 4.5.1 and 5.7.2 and
legacy/1D/OneDimData.py): piecewise random Fourier series with up to
``max_disc`` jump discontinuities, sampled either exactly on the DG nodes
("exact" mode) or taken from a limited DG solve of the advection equation
("numerical" mode). Random Euler Riemann solves provide nonlinear training
states. Mesh length is randomized per sample for the GNN.
"""

from dataclasses import dataclass, field

import numpy as np

from tci.solvers.dg1d import DG1D


@dataclass
class Sample:
    """One training example: nodal values u (Np, K) and per-cell labels (K,)."""

    u: np.ndarray
    labels: np.ndarray
    x: np.ndarray | None = None
    disc_locs: np.ndarray = field(default_factory=lambda: np.array([]))


def random_piecewise_fourier(
    rng, domain=(0.0, 1.0), max_disc=5, n_fourier=15, coeff_sigma=1.0
):
    """Random piecewise-smooth function with jump discontinuities.

    Each piece is a0 + sum_n (a_n cos(n x) + b_n sin(n x)) with iid normal
    coefficients; a fresh draw of coefficients across every discontinuity
    creates O(1) jumps. Returns (f, disc_locs) with f vectorized.
    """
    lo, hi = domain
    n_disc = int(rng.integers(0, max_disc + 1))
    margin = 0.02 * (hi - lo)
    locs = np.sort(rng.uniform(lo + margin, hi - margin, size=n_disc))

    n_pieces = n_disc + 1
    a0 = rng.normal(0, coeff_sigma, size=n_pieces)
    a = rng.normal(0, coeff_sigma, size=(n_pieces, n_fourier))
    b = rng.normal(0, coeff_sigma, size=(n_pieces, n_fourier))
    modes = np.arange(1, n_fourier + 1)

    def f(x):
        x = np.asarray(x, dtype=float)
        piece = np.searchsorted(locs, x, side="right")
        phase = modes[:, None] * x.ravel()[None, :]  # (n_fourier, len(x))
        pc = piece.ravel()
        vals = (
            a0[pc]
            + np.sum(a[pc].T * np.cos(phase), axis=0)
            + np.sum(b[pc].T * np.sin(phase), axis=0)
        )
        return vals.reshape(x.shape)

    return f, locs


def _crop(u, labels, rng, min_cells=10, keep_clean_prob=0.3):
    """Random contiguous window of cells (thesis step 7). Windows with no
    troubled cell are kept only with probability ``keep_clean_prob``."""
    K = u.shape[1]
    if K <= min_cells:
        return u, labels
    for _ in range(8):
        w = int(rng.integers(min_cells, K + 1))
        s = int(rng.integers(0, K - w + 1))
        cl, cu = labels[s : s + w], u[:, s : s + w]
        if cl.any() or rng.random() < keep_clean_prob:
            return cu, cl
    return u, labels


def generate_exact_samples(
    n_samples,
    N=1,
    k_range=(50, 150),
    domain=(0.0, 1.0),
    max_disc=5,
    n_fourier=15,
    seed=0,
    crop=False,
):
    """Samples from exact function values on the DG nodes (no solver noise)."""
    rng = np.random.default_rng(seed)
    kmin, kmax = k_range
    out = []
    for _ in range(n_samples):
        K = int(rng.integers(kmin, kmax + 1))
        solver = DG1D(*domain, K=K, N=N)
        f, locs = random_piecewise_fourier(
            rng, domain=domain, max_disc=max_disc, n_fourier=n_fourier
        )
        u = solver.project(f)
        labels = solver.cells_containing(locs)
        if crop:
            u, labels = _crop(u, labels, rng)
        out.append(Sample(u=u, labels=labels, x=None, disc_locs=locs))
    return out


def generate_numerical_samples(
    n_samples,
    N=1,
    k_range=(50, 150),
    domain=(0.0, 1.0),
    max_disc=5,
    n_fourier=15,
    seed=0,
    crop=False,
    a_range=(0.5, 1.0),
    t_range=(0.1, 0.3),
    cfl=0.375,
):
    """Samples taken from minmod-limited DG solves of the advection equation
    (thesis section 5.7.2): oscillations near jumps are present in u, and
    labels come from the exactly advected discontinuity locations."""
    from tci.indicators.classical import MinmodIndicator

    rng = np.random.default_rng(seed)
    lo, hi = domain
    width = hi - lo
    kmin, kmax = k_range
    indicator = MinmodIndicator()
    out = []
    for _ in range(n_samples):
        K = int(rng.integers(kmin, kmax + 1))
        solver = DG1D(lo, hi, K=K, N=N)
        f, locs = random_piecewise_fourier(
            rng, domain=domain, max_disc=max_disc, n_fourier=n_fourier
        )
        a = float(rng.uniform(*a_range))
        T = float(rng.uniform(*t_range))
        u = solver.advect(f, a, T, indicator=indicator, cfl=cfl)
        assert isinstance(u, np.ndarray)
        adv_locs = lo + np.mod(locs + a * T - lo, width)
        labels = solver.cells_containing(adv_locs)
        if crop:
            u, labels = _crop(u, labels, rng)
        out.append(Sample(u=u, labels=labels, x=None, disc_locs=adv_locs))
    return out


def generate_euler_riemann_samples(
    n_samples,
    N=1,
    k_range=(50, 150),
    domain=(0.0, 1.0),
    seed=0,
    crop=False,
    rho_range=(0.2, 2.0),
    velocity_range=(-0.5, 0.5),
    pressure_range=(0.1, 2.0),
    diaphragm_range=(0.35, 0.65),
    t_range=(0.03, 0.1),
    cfl=0.2,
    max_attempts=20,
    **_ignored,
):
    """Density fields from minmod-limited random Euler Riemann problems.

    Labels follow exact shock and contact trajectories. Rarefaction edges are
    not labeled because the exact solution is continuous there.
    """
    from tci.indicators.classical import MinmodIndicator
    from tci.solvers.euler import EulerDG1D, primitive_to_conserved
    from tci.solvers.riemann import discontinuity_speeds

    rng = np.random.default_rng(seed)
    lo, hi = domain
    width = hi - lo
    kmin, kmax = k_range
    out = []
    for sample_id in range(n_samples):
        for _ in range(max_attempts):
            K = int(rng.integers(kmin, kmax + 1))
            x0 = lo + float(rng.uniform(*diaphragm_range)) * width
            T = float(rng.uniform(*t_range))
            rho_l, rho_r = rng.uniform(*rho_range, size=2)
            vel_l, vel_r = rng.uniform(*velocity_range, size=2)
            p_l, p_r = rng.uniform(*pressure_range, size=2)
            speeds = discontinuity_speeds(
                rho_l, vel_l, p_l, rho_r, vel_r, p_r
            )
            locs = x0 + T * speeds
            if np.any((locs <= lo) | (locs >= hi)):
                continue

            solver = EulerDG1D(lo, hi, K=K, N=N)

            def initial(x):
                left = x < x0
                rho = np.where(left, rho_l, rho_r)
                vel = np.where(left, vel_l, vel_r)
                pressure = np.where(left, p_l, p_r)
                return primitive_to_conserved(rho, vel, pressure)

            try:
                U = solver.solve(initial, T, indicator=MinmodIndicator(), cfl=cfl)
            except RuntimeError as exc:
                if not str(exc).startswith("solve diverged:"):
                    raise
                continue
            assert isinstance(U, np.ndarray)
            u = U[:, :, 0]
            if not np.all(np.isfinite(u)) or np.min(u) <= 0:
                continue
            labels = solver.cells_containing(locs)
            if crop:
                u, labels = _crop(u, labels, rng)
            out.append(Sample(u=u, labels=labels, x=None, disc_locs=locs))
            break
        else:
            raise RuntimeError(
                f"could not generate admissible Euler sample {sample_id} "
                f"after {max_attempts} attempts"
            )
    return out
