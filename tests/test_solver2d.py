import numpy as np
import pytest

from tci.mesh import random_delaunay_mesh, rectangular_mesh
from tci.solvers.dg2d import AdvectionDG2D
from tci.indicators.classical2d import KXRCFIndicator2D, MinmodIndicator2D
from tci.limiters2d import limit_p1, neighbor_mean_bounds


def test_p1_reference_mass_matrix():
    mass = AdvectionDG2D.reference_mass
    assert np.allclose(mass, mass.T)
    assert np.all(np.linalg.eigvalsh(mass) > 0)
    assert np.isclose(np.ones(3) @ mass @ np.ones(3), 2.0)
    assert np.allclose(AdvectionDG2D.reference_mass_inv @ mass, np.eye(3))


def test_exact_linear_derivative_and_free_stream():
    solver = AdvectionDG2D(rectangular_mesh(nx=3, ny=2), velocity=(1.0, 0.3))
    linear = solver.project(lambda x, y: 2.0 + x - 3.0 * y)
    # -(1, .3) dot grad(2 + x - 3y) = -0.1.
    assert np.allclose(solver.rhs(linear), -0.1)

    constant = np.full_like(linear, 4.0)
    assert np.allclose(solver.rhs(constant), 0.0)
    final = solver.solve(constant, 0.1)
    assert np.array_equal(final, constant)
    assert np.isclose(solver.integral(constant), 4.0)


def test_free_stream_on_unstructured_mesh():
    mesh = random_delaunay_mesh(n_interior=20, boundary_divisions=(3, 3), seed=4)
    solver = AdvectionDG2D(mesh, velocity=(-0.4, 0.7))
    constant = np.full((3, mesh.K), 2.5)
    assert np.max(np.abs(solver.rhs(constant))) < 1e-12
    assert np.allclose(solver.solve(constant, 0.05), constant)


def test_zero_velocity_leaves_state_unchanged():
    solver = AdvectionDG2D(rectangular_mesh(nx=2, ny=2), velocity=(0.0, 0.0))
    u0 = solver.project(lambda x, y: np.sin(x + y))
    assert np.isinf(solver.stable_dt())
    assert np.array_equal(solver.solve(u0, 1.0), u0)


def test_periodic_rhs_and_time_integration_are_conservative():
    mesh = rectangular_mesh(nx=5, ny=4)
    solver = AdvectionDG2D(mesh, velocity=(0.7, -0.2), periodic=(True, True))
    rng = np.random.default_rng(3)
    u0 = rng.normal(size=(3, mesh.K))
    assert abs(solver.integral(solver.rhs(u0))) < 1e-12
    initial_mass = solver.integral(u0)
    final = solver.solve(u0, 0.1)
    assert abs(solver.integral(final) - initial_mass) < 1e-12


def test_rotational_velocity_preserves_constant_state():
    velocity = lambda x, y, time: np.stack(
        [-2 * np.pi * (y - 0.5), 2 * np.pi * (x - 0.5)], axis=-1
    )
    solver = AdvectionDG2D(
        rectangular_mesh(nx=4, ny=4),
        velocity=velocity,
        boundary_func=lambda x, y, time: np.ones_like(x) * 3.0,
    )
    constant = np.full((3, solver.K), 3.0)
    assert np.max(np.abs(solver.rhs(constant))) < 1e-12
    assert np.allclose(solver.solve(constant, 0.05), constant)


def test_p1_limiter_preserves_means_and_enforces_neighbor_bounds():
    solver = AdvectionDG2D(rectangular_mesh(nx=2, ny=2))
    u = np.tile(np.linspace(0.0, 1.0, solver.K), (3, 1))
    u[:, 2] += np.array([-3.0, 0.0, 3.0])
    before = solver.cell_means(u)
    limited = limit_p1(solver, u, np.arange(solver.K) == 2)
    assert np.allclose(solver.cell_means(limited), before)
    _, lower, upper = neighbor_mean_bounds(solver, u)
    assert np.min(limited[:, 2]) >= lower[2] - 1e-14
    assert np.max(limited[:, 2]) <= upper[2] + 1e-14


def test_2d_indicator_runs_inside_solver_and_records_flags():
    solver = AdvectionDG2D(
        rectangular_mesh(nx=4, ny=4), velocity=(1.0, 0.0), periodic=(True, True)
    )
    u, history = solver.solve(
        lambda x, y: np.where(x < 0.5, 1.0, 0.0),
        0.02,
        indicator=MinmodIndicator2D(),
        record_flags=True,
    )
    assert np.all(np.isfinite(u))
    assert history and all(flags.shape == (solver.K,) for _, flags in history)
    assert KXRCFIndicator2D().flag(solver, u).shape == (solver.K,)


def test_solver_wall_clock_limit_is_enforced():
    solver = AdvectionDG2D(rectangular_mesh(nx=2, ny=2), velocity=(1.0, 0.0))
    with pytest.raises(TimeoutError, match="wall-clock limit"):
        solver.solve(lambda x, y: x + y, 1.0, max_seconds=0.0)


def test_smooth_inflow_convergence_is_second_order_asymptotically():
    velocity = np.array([0.8, 0.3])
    final_time = 0.05

    def exact(x, y, time):
        return np.sin(2 * np.pi * (x - velocity[0] * time)) * np.cos(
            2 * np.pi * (y - velocity[1] * time)
        )

    errors = []
    for n in (8, 16, 32):
        solver = AdvectionDG2D(
            rectangular_mesh(nx=n, ny=n),
            velocity=velocity,
            boundary_func=lambda x, y, time: exact(x, y, time),
        )
        u = solver.solve(lambda x, y: exact(x, y, 0.0), final_time, cfl=0.1)
        errors.append(solver.l2_error(u, lambda x, y: exact(x, y, final_time)))

    rates = np.log2(np.asarray(errors[:-1]) / errors[1:])
    assert errors[2] < errors[1] < errors[0]
    assert rates[-1] > 1.8
