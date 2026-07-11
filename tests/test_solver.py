import numpy as np
import pytest

from tci.solvers.dg1d import DG1D


def test_smooth_advection_periodic_accuracy():
    """A sine wave advected one full period returns to itself."""
    solver = DG1D(0.0, 1.0, K=20, N=4)
    u0 = solver.project(lambda x: np.sin(2 * np.pi * x))
    u = solver.advect(u0, a=1.0, final_time=1.0)
    assert np.max(np.abs(u - u0)) < 1e-4


def test_convergence_with_resolution():
    """Error decreases fast with mesh refinement for smooth data."""
    errs = []
    for K in (10, 20):
        solver = DG1D(0.0, 1.0, K=K, N=3)
        u0 = solver.project(lambda x: np.sin(2 * np.pi * x))
        u = solver.advect(u0, a=1.0, final_time=0.5)
        exact = solver.project(lambda x: np.sin(2 * np.pi * (x - 0.5)))
        errs.append(np.sqrt(np.mean((u - exact) ** 2)))
    assert errs[1] < errs[0] / 8  # at least ~3rd order


def test_inflow_bc():
    solver = DG1D(0.0, 2.0, K=40, N=4, bc="inflow", bc_func=np.sin)
    u0 = solver.project(np.sin)
    u = solver.advect(u0, a=1.0, final_time=0.5)
    exact = solver.project(lambda x: np.sin(x - 0.5))
    assert np.max(np.abs(u - exact)) < 1e-3


def test_cell_means():
    solver = DG1D(0.0, 1.0, K=10, N=3)
    u = solver.project(lambda x: 3.0 * np.ones_like(x))
    assert np.allclose(solver.cell_means(u), 3.0)


def test_cells_containing():
    solver = DG1D(0.0, 1.0, K=10, N=1)
    mask = solver.cells_containing([0.05, 0.55])
    assert mask[0] and mask[5] and mask.sum() == 2


def test_bad_bc_raises():
    with pytest.raises(ValueError):
        DG1D(0, 1, K=10, N=1, bc="nonsense")
    with pytest.raises(ValueError):
        DG1D(0, 1, K=10, N=1, bc="inflow")
