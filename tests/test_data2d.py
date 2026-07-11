import numpy as np

from tci.data.generate2d import (
    cells_cut_by_circle,
    cells_cut_by_line,
    generate_exact_2d_samples,
    generate_mixed_2d_samples,
    generate_numerical_2d_samples,
)
from tci.mesh import TriangleMesh, rectangular_mesh


def test_line_cut_labels_match_triangle_vertex_ranges():
    mesh = rectangular_mesh(nx=4, ny=3)
    point = np.array([0.43, 0.2])
    normal = np.array([1.0, 0.0])
    labels = cells_cut_by_line(mesh, point, normal)
    x = mesh.points[mesh.cells, 0]
    expected = (np.min(x, axis=1) <= point[0]) & (np.max(x, axis=1) >= point[0])
    assert np.array_equal(labels, expected)
    assert labels.any() and not labels.all()


def test_circle_contained_inside_triangle_is_detected():
    mesh = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
    labels = cells_cut_by_circle(mesh, center=(0.2, 0.2), radius=0.05)
    assert np.array_equal(labels, [True])


def test_circle_outside_triangle_is_not_detected():
    mesh = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], [[0, 1, 2]])
    labels = cells_cut_by_circle(mesh, center=(2.0, 2.0), radius=0.1)
    assert np.array_equal(labels, [False])


def test_exact_2d_samples_are_variable_and_reproducible():
    first = generate_exact_2d_samples(
        4, n_interior_range=(10, 20), boundary_divisions=(3, 3), seed=9
    )
    second = generate_exact_2d_samples(
        4, n_interior_range=(10, 20), boundary_divisions=(3, 3), seed=9
    )
    assert len({sample.mesh.K for sample in first}) > 1
    for a, b in zip(first, second):
        assert a.u.shape == (3, a.mesh.K)
        assert a.labels.shape == (a.mesh.K,)
        assert a.labels.dtype == bool and a.labels.any()
        assert np.all(np.isfinite(a.u))
        assert np.array_equal(a.mesh.points, b.mesh.points)
        assert np.array_equal(a.mesh.cells, b.mesh.cells)
        assert np.array_equal(a.u, b.u)
        assert np.array_equal(a.labels, b.labels)


def test_structured_2d_generation_mode():
    sample = generate_exact_2d_samples(
        1, mesh_type="structured", structured_range=(3, 3), curves=("line",), seed=2
    )[0]
    assert sample.mesh.K == 18
    assert sample.curve == "line"


def test_numerical_2d_generation_is_bounded_and_reproducible():
    first = generate_numerical_2d_samples(
        2,
        mesh_range=(3, 4),
        time_range=(0.005, 0.01),
        max_steps=100,
        max_seconds_per_sample=2.0,
        max_generation_seconds=10.0,
        seed=5,
    )
    second = generate_numerical_2d_samples(
        2,
        mesh_range=(3, 4),
        time_range=(0.005, 0.01),
        max_steps=100,
        max_seconds_per_sample=2.0,
        max_generation_seconds=10.0,
        seed=5,
    )
    for a, b in zip(first, second):
        assert a.source == "unlimited" and a.time > 0
        assert a.u.shape == (3, a.mesh.K) and np.all(np.isfinite(a.u))
        assert a.labels.any()
        assert np.array_equal(a.u, b.u)
        assert np.array_equal(a.labels, b.labels)


def test_mixed_2d_contains_exact_and_numerical_sources():
    samples = generate_mixed_2d_samples(
        4,
        exact_fraction=0.5,
        mesh_range=(3, 3),
        time_range=(0.005, 0.005),
        max_steps=100,
        max_generation_seconds=10.0,
        seed=2,
    )
    assert {sample.source for sample in samples} == {"exact", "unlimited"}
