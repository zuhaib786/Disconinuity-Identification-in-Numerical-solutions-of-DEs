import numpy as np

from tci.evaluate import box_initial
from tci.indicators.classical import KXRCFIndicator, MinmodIndicator
from tci.indicators.pa import PAIndicator, polynomial_annihilation
from tci.solvers.dg1d import DG1D

# Box edges start at cells 40/60 and advect to ~45/65 by T = 0.05 (a = 1).
K, T = 100, 0.05
JUMPS_AT_T = np.array([45, 65])


def _near_jumps(flags, tol):
    """All flagged cells lie within tol cells of a (periodic) jump location."""
    ids = np.where(flags)[0]
    if ids.size == 0:
        return False
    d = np.abs(ids[:, None] - JUMPS_AT_T[None, :])
    d = np.minimum(d, K - d)
    return bool(np.all(d.min(axis=1) <= tol))


def test_minmod_flags_near_fronts_in_loop():
    solver = DG1D(0.0, 1.0, K=K, N=1)
    _, hist = solver.advect(
        solver.project(box_initial),
        1.0,
        T,
        indicator=MinmodIndicator(),
        record_flags=True,
    )
    flags = hist[-1][1]
    assert _near_jumps(flags, tol=10)
    assert flags.sum() <= 12


def test_kxrcf_flags_near_fronts_in_loop():
    """KXRCF fires on the (smeared) fronts; its band is wider than minmod's."""
    solver = DG1D(0.0, 1.0, K=K, N=2)
    _, hist = solver.advect(
        solver.project(box_initial),
        1.0,
        T,
        indicator=KXRCFIndicator(threshold=1.0, a=1.0),
        record_flags=True,
    )
    assert _near_jumps(hist[-1][1], tol=20)


def test_pa_flags_box_edges_on_projection():
    solver = DG1D(0.0, 1.0, K=K, N=1)
    u0 = solver.project(box_initial)
    flags = PAIndicator(m=5, threshold=0.3).flag(solver, u0)
    ids = np.where(flags)[0]
    assert ids.size > 0 and flags.sum() <= 8
    # Initial jumps sit at cells 40/60; the m+1-wide stencil spreads flags
    # up to ~m/2 cells from the jump.
    d = np.abs(ids[:, None] - np.array([40, 60])[None, :]).min(axis=1)
    assert np.all(d <= 5)


def test_indicators_quiet_on_smooth_data():
    solver = DG1D(0.0, 1.0, K=K, N=2)
    u = solver.advect(solver.project(lambda x: np.sin(2 * np.pi * x)), 1.0, T)
    assert not KXRCFIndicator(a=1.0).flag(solver, u).any()
    assert PAIndicator(m=5, threshold=0.3).flag(solver, u).sum() <= 4
    # Plain minmod famously clips smooth extrema: a handful of flags is expected.
    assert MinmodIndicator().flag(solver, u).sum() <= 8


def test_polynomial_annihilation_jump_detection():
    x = np.linspace(-1, 1, 201)
    f = np.where(x < 0.3, np.sin(x), np.sin(x) + 5.0)
    vals = polynomial_annihilation(x, f, m=5)
    mids = 0.5 * (x[:-1] + x[1:])
    peak = mids[np.argmax(vals)]
    assert abs(peak - 0.3) < 0.02
    far = np.abs(mids - 0.3) > 0.1
    assert np.max(vals[far]) < 1.0


def test_tvb_flags_no_more_than_plain_minmod():
    solver = DG1D(0.0, 1.0, K=K, N=2)
    u = solver.advect(solver.project(box_initial), 1.0, T)  # unlimited, oscillatory
    plain = MinmodIndicator().flag(solver, u).sum()
    tvb = MinmodIndicator(m_tvb=50.0).flag(solver, u).sum()
    assert plain > 0
    assert tvb <= plain
