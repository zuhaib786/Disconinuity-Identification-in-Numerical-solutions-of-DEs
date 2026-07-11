import numpy as np
import pytest

from tci.data.graphs import mesh_edge_index
from tci.mesh import (
    TriangleMesh,
    delaunay_mesh,
    perturbed_delaunay_mesh,
    forward_step_mesh,
    random_delaunay_mesh,
    rectangular_mesh,
)


def test_single_triangle_orientation_geometry_and_normals():
    points = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    mesh = TriangleMesh(points, [[0, 1, 2]])  # clockwise input
    assert np.array_equal(mesh.cells, [[0, 2, 1]])
    assert np.allclose(mesh.areas, 0.5)
    assert np.allclose(mesh.edge_lengths[0], [1.0, np.sqrt(2.0), 1.0])
    assert np.allclose(mesh.inradii, 1.0 / (2.0 + np.sqrt(2.0)))
    assert np.allclose(mesh.circumradii, np.sqrt(2.0) / 2.0)
    assert mesh.interior_faces.shape == (0, 4)
    assert mesh.boundary_faces.shape == (3, 2)
    assert np.all(mesh.neighbors == -1)
    assert mesh.graph_edge_index().shape == (2, 0)

    midpoint = np.mean(mesh.face_points[0], axis=1)
    outward = np.sum(mesh.face_normals[0] * (midpoint - mesh.centroids[0]), axis=1)
    assert np.all(outward > 0)


@pytest.mark.parametrize("diagonal", ["left", "right", "alternating"])
def test_rectangular_mesh_topology_and_area(diagonal):
    nx, ny = 4, 3
    mesh = rectangular_mesh((0.0, 2.0), (-1.0, 1.0), nx, ny, diagonal)
    assert mesh.K == 2 * nx * ny
    assert len(mesh.boundary_faces) == 2 * nx + 2 * ny
    assert 3 * mesh.K == 2 * len(mesh.interior_faces) + len(mesh.boundary_faces)
    assert np.isclose(np.sum(mesh.areas), 4.0)
    boundary_lengths = mesh.edge_lengths[
        mesh.boundary_faces[:, 0], mesh.boundary_faces[:, 1]
    ]
    assert np.isclose(np.sum(boundary_lengths), 8.0)
    assert np.all(mesh.areas > 0)


def test_neighbor_reciprocity_and_graph_edges():
    mesh = rectangular_mesh(nx=3, ny=2)
    for cell, face, neighbor, neighbor_face in mesh.interior_faces:
        assert mesh.neighbors[cell, face] == neighbor
        assert mesh.neighbor_faces[cell, face] == neighbor_face
        assert mesh.neighbors[neighbor, neighbor_face] == cell
        assert np.array_equal(
            mesh.face_vertices[cell, face],
            mesh.face_vertices[neighbor, neighbor_face][::-1],
        )
    edges = mesh_edge_index(mesh)
    assert edges.shape == (2, 2 * len(mesh.graph_edges))
    directed = {(int(edge[0]), int(edge[1])) for edge in edges.T}
    for a, b in mesh.graph_edges:
        assert (a, b) in directed and (b, a) in directed


def test_periodic_face_pairing_on_structured_and_delaunay_meshes():
    for mesh in (
        rectangular_mesh(nx=4, ny=3),
        random_delaunay_mesh(n_interior=15, boundary_divisions=(4, 3), seed=2),
    ):
        neighbors, faces = mesh.periodic_face_map((True, True))
        paired = np.argwhere(neighbors >= 0)
        assert len(paired) == len(mesh.boundary_faces)
        for cell, face in paired:
            other = neighbors[cell, face]
            other_face = faces[cell, face]
            assert neighbors[other, other_face] == cell
            assert faces[other, other_face] == face


def test_partial_periodicity_leaves_other_boundaries_unpaired():
    mesh = rectangular_mesh(nx=3, ny=2)
    neighbors, _ = mesh.periodic_face_map((True, False))
    assert np.count_nonzero(neighbors >= 0) == 2 * 2


def test_geometry_features_are_scale_invariant_except_normalized_area():
    base = rectangular_mesh((0.0, 2.0), (0.0, 1.0), 3, 2)
    scaled = TriangleMesh(base.points * 1e9, base.cells)
    assert np.allclose(base.geometry_features(), scaled.geometry_features())
    raw = base.geometry_features(dimensionless=False)
    assert raw.shape == (base.K, 7)
    assert np.allclose(raw[:, 0], base.areas)


def test_equilateral_and_thin_triangle_quality():
    equilateral = TriangleMesh(
        [[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3.0) / 2.0]], [[0, 1, 2]]
    )
    thin = TriangleMesh([[0.0, 0.0], [1.0, 0.0], [0.5, 1e-6]], [[0, 1, 2]])
    assert equilateral.skewness[0] < 1e-14
    assert thin.skewness[0] > 0.999


def test_delaunay_mesh_and_seeded_random_mesh():
    points = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]]
    )
    mesh = delaunay_mesh(points)
    assert mesh.K == 4
    assert len(mesh.boundary_faces) == 4
    assert np.isclose(np.sum(mesh.areas), 1.0)

    first = random_delaunay_mesh(n_interior=20, boundary_divisions=(4, 3), seed=8)
    second = random_delaunay_mesh(n_interior=20, boundary_divisions=(4, 3), seed=8)
    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.cells, second.cells)
    assert np.isclose(np.sum(first.areas), 1.0)


def test_perturbed_delaunay_mesh_is_reproducible_and_quality_controlled():
    first = perturbed_delaunay_mesh(nx=12, ny=12, jitter=0.2, seed=4)
    second = perturbed_delaunay_mesh(nx=12, ny=12, jitter=0.2, seed=4)
    assert np.array_equal(first.points, second.points)
    assert np.array_equal(first.cells, second.cells)
    assert first.K == 288
    assert np.max(first.skewness) < 0.8


def test_forward_step_mesh_has_correct_nonconvex_area():
    mesh = forward_step_mesh(nx=30, ny=10)
    assert np.isclose(np.sum(mesh.areas), 2.52)
    inside_step = (mesh.centroids[:, 0] > 0.6) & (mesh.centroids[:, 1] < 0.2)
    assert not np.any(inside_step)


@pytest.mark.parametrize(
    "points,cells,match",
    [
        ([[0, 0], [1, 0], [0, 1]], [[0, 1, 3]], "out-of-range"),
        ([[0, 0], [1, 0], [0, 1]], [[0, 0, 2]], "distinct"),
        ([[0, 0], [1, 0], [0, 1]], [[0, 1, 2], [2, 1, 0]], "duplicate"),
        ([[0, 0], [1, 0], [2, 0]], [[0, 1, 2]], "degenerate"),
        ([[0, 0], [1, 0], [1, 0]], [[0, 1, 2]], "duplicate point"),
    ],
)
def test_invalid_meshes_are_rejected(points, cells, match):
    with pytest.raises(ValueError, match=match):
        TriangleMesh(points, cells)


def test_non_manifold_edge_is_rejected():
    points = [[0, 0], [1, 0], [0, 1], [0, -1], [0.5, 1]]
    cells = [[0, 1, 2], [1, 0, 3], [0, 1, 4]]
    with pytest.raises(ValueError, match="non-manifold"):
        TriangleMesh(points, cells)
