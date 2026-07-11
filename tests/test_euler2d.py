import numpy as np

from tci.indicators.classical2d import MinmodIndicator2D
from tci.mesh import perturbed_delaunay_mesh, rectangular_mesh
from tci.solvers.euler2d import (
    EulerDG2D,
    conserved_to_primitive_2d,
    euler_flux_2d,
    normal_flux_2d,
    positivity_scale,
    primitive_to_conserved_2d,
)


def constant_state(x, y):
    return primitive_to_conserved_2d(
        np.ones_like(x),
        np.full_like(x, 0.3),
        np.full_like(x, -0.2),
        np.ones_like(x),
    )


def test_primitive_conserved_roundtrip_2d():
    rho = np.array([1.0, 0.4])
    u = np.array([0.2, -0.1])
    v = np.array([-0.3, 0.4])
    p = np.array([1.0, 0.2])
    U = primitive_to_conserved_2d(rho, u, v, p)
    assert np.allclose(conserved_to_primitive_2d(U), [rho, u, v, p])


def test_normal_flux_matches_cartesian_fluxes():
    U = primitive_to_conserved_2d(
        np.array([1.0]), np.array([0.3]), np.array([-0.2]), np.array([1.0])
    )
    fx, fy = euler_flux_2d(U)
    assert np.allclose(normal_flux_2d(U, np.array([[1.0, 0.0]])), fx)
    assert np.allclose(normal_flux_2d(U, np.array([[0.0, 1.0]])), fy)


def test_euler_free_stream_on_structured_and_unstructured_meshes():
    for mesh in (
        rectangular_mesh(nx=3, ny=3),
        perturbed_delaunay_mesh(nx=3, ny=3, seed=2),
    ):
        solver = EulerDG2D(mesh, periodic=(True, True))
        U = solver.project(constant_state)
        assert np.max(np.abs(solver.rhs(U))) < 1e-11


def test_periodic_euler_rhs_is_conservative():
    solver = EulerDG2D(rectangular_mesh(nx=3, ny=3), periodic=(True, True))
    nodes = solver.nodes
    U = primitive_to_conserved_2d(
        1.0 + 0.02 * np.sin(2 * np.pi * nodes[:, :, 0]),
        0.2 + 0.01 * np.cos(2 * np.pi * nodes[:, :, 1]),
        np.zeros((3, solver.K)),
        np.ones((3, solver.K)),
    )
    assert np.max(np.abs(solver.integral(solver.rhs(U)))) < 1e-11


def test_positivity_scaling_repairs_nodes_and_preserves_mean():
    mean = primitive_to_conserved_2d(1.0, 0.0, 0.0, 1.0)
    U = np.tile(mean, (3, 1, 1))
    U[:, 0] += np.array(
        [[-1.2, 0.0, 0.0, -2.0], [0.6, 0.0, 0.0, 1.0], [0.6, 0.0, 0.0, 1.0]]
    )
    before = np.mean(U, axis=0)
    fixed = positivity_scale(U)
    rho, _, _, p = conserved_to_primitive_2d(fixed)
    assert np.min(rho) > 0 and np.min(p) > 0
    assert np.allclose(np.mean(fixed, axis=0), before)


def test_short_four_quadrant_solve_stays_admissible():
    mesh = rectangular_mesh(nx=4, ny=4)
    solver = EulerDG2D(mesh)

    def initial(x, y):
        rho = np.where((x >= 0.5) & (y >= 0.5), 1.5, 0.5323)
        rho = np.where((x < 0.5) & (y < 0.5), 0.138, rho)
        u = np.where(x < 0.5, 1.206, 0.0)
        v = np.where(y < 0.5, 1.206, 0.0)
        p = np.where((x >= 0.5) & (y >= 0.5), 1.5, 0.3)
        p = np.where((x < 0.5) & (y < 0.5), 0.029, p)
        return primitive_to_conserved_2d(rho, u, v, p)

    U = solver.solve(
        initial,
        0.002,
        indicator=MinmodIndicator2D(),
        cfl=0.05,
        max_seconds=10.0,
    )
    rho, _, _, p = conserved_to_primitive_2d(U)
    assert np.all(np.isfinite(U))
    assert np.min(rho) > 0 and np.min(p) > 0
