import numpy as np

from tci.indicators.classical import MinmodIndicator
from tci.solvers.euler import (
    EulerDG1D,
    conserved_to_primitive,
    primitive_to_conserved,
    shu_osher_initial,
    sod_initial,
)
from tci.solvers.riemann import discontinuity_speeds, sod_exact, star_state


def test_primitive_conserved_roundtrip():
    rho = np.array([1.0, 0.125])
    u = np.array([0.3, -0.2])
    p = np.array([1.0, 0.1])
    U = primitive_to_conserved(rho, u, p)
    r2, u2, p2 = conserved_to_primitive(U)
    assert np.allclose([r2, u2, p2], [rho, u, p])


def test_riemann_star_state_sod():
    """Toro's reference values for the Sod problem."""
    p_s, u_s = star_state(1.0, 0.0, 1.0, 0.125, 0.0, 0.1)
    assert abs(p_s - 0.30313) < 1e-4
    assert abs(u_s - 0.92745) < 1e-4


def test_sod_discontinuity_speeds_are_contact_and_shock():
    speeds = discontinuity_speeds(1.0, 0.0, 1.0, 0.125, 0.0, 0.1)
    assert speeds.shape == (2,)  # left wave is a continuous rarefaction
    assert abs(speeds[0] - 0.92745) < 1e-4
    assert abs(speeds[1] - 1.75216) < 1e-4


def test_sod_shock_tube():
    s = EulerDG1D(0.0, 1.0, K=200, N=1)
    U = s.solve(sod_initial, 0.2, indicator=MinmodIndicator())
    assert isinstance(U, np.ndarray)
    rho = s.cell_means(U[:, :, 0])
    centers = 0.5 * (s.VX[:-1] + s.VX[1:])
    rho_ex, _, _ = sod_exact(centers, 0.2)
    assert np.all(np.isfinite(U))
    assert rho.min() > 0
    assert np.mean(np.abs(rho - rho_ex)) < 0.01


def test_shu_osher_stability():
    s = EulerDG1D(-5.0, 5.0, K=200, N=2)
    U = s.solve(shu_osher_initial, 1.8, indicator=MinmodIndicator())
    assert isinstance(U, np.ndarray)
    rho = s.cell_means(U[:, :, 0])
    assert np.all(np.isfinite(U))
    assert rho.min() > 0.5 and rho.max() < 5.0


def test_kxrcf_local_velocity_on_euler():
    """KXRCF with a=None asks the Euler solver for per-cell velocities."""
    from tci.indicators.classical import KXRCFIndicator

    s = EulerDG1D(0.0, 1.0, K=100, N=1)
    U = s.solve(sod_initial, 0.1, indicator=KXRCFIndicator(threshold=1.0))
    assert np.all(np.isfinite(U))
