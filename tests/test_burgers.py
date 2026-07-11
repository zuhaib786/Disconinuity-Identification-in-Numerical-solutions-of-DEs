import numpy as np

from tci.indicators.classical import MinmodIndicator
from tci.solvers.burgers import BurgersDG1D


def test_shock_speed_rankine_hugoniot():
    """Riemann data (1, 0): the shock moves at speed (f(1)-f(0))/(1-0) = 1/2."""
    s = BurgersDG1D(0.0, 1.0, K=200, N=1, bc="transmissive")
    u0 = s.project(lambda x: np.where(x < 0.25, 1.0, 0.0))
    u = s.solve(u0, 0.5, indicator=MinmodIndicator())
    v = s.cell_means(u)
    centers = 0.5 * (s.VX[:-1] + s.VX[1:])
    shock_pos = centers[np.argmin(np.abs(v - 0.5))]
    assert abs(shock_pos - 0.5) < 0.02
    exact = np.where(centers < 0.5, 1.0, 0.0)
    assert np.mean(np.abs(v - exact)) < 0.01


def test_conservation_periodic():
    """Total mass is conserved through shock formation (periodic BCs)."""
    s = BurgersDG1D(0.0, 1.0, K=100, N=2)
    u0 = s.project(lambda x: 0.5 + np.sin(2 * np.pi * x))
    u = s.solve(u0, 0.3, indicator=MinmodIndicator())
    assert np.all(np.isfinite(u))
    assert abs(np.mean(s.cell_means(u)) - 0.5) < 1e-10


def test_smooth_prebreaking_accuracy():
    """Before the shock forms (t < 1/(2 pi)) the solution stays smooth and
    refined meshes give smaller error against a fine reference."""
    T = 0.1
    ref = BurgersDG1D(0.0, 1.0, K=800, N=2)
    uref = ref.solve(ref.project(lambda x: 0.5 + np.sin(2 * np.pi * x)), T)
    ref_centers = 0.5 * (ref.VX[:-1] + ref.VX[1:])
    ref_means = ref.cell_means(uref)

    errs = []
    for K in (25, 50):
        s = BurgersDG1D(0.0, 1.0, K=K, N=2)
        u = s.solve(s.project(lambda x: 0.5 + np.sin(2 * np.pi * x)), T)
        centers = 0.5 * (s.VX[:-1] + s.VX[1:])
        errs.append(
            np.mean(np.abs(s.cell_means(u) - np.interp(centers, ref_centers, ref_means)))
        )
    assert errs[1] < errs[0] / 4
