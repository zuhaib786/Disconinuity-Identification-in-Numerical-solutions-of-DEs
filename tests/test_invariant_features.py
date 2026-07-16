import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from tci.data.generate2d import generate_exact_2d_samples  # noqa: E402
from tci.data.graphs import (  # noqa: E402
    FEATURE_SCHEMAS,
    TriangleFeatureBuilder,
    expand_labels_by_hops,
    sample2d_to_data,
)
from tci.indicators.learned import GNN2DIndicator  # noqa: E402
from tci.mesh import TriangleMesh, rectangular_mesh  # noqa: E402
from tci.models import GNNDetector  # noqa: E402
from tci.solvers.dg2d import AdvectionDG2D  # noqa: E402
from tci.train import train  # noqa: E402


INVARIANT_SCHEMAS = (
    "invariant-node-v2",
    "invariant-edge-v2",
    "invariant-local-v2",
    "invariant-extrema-v3",
)


@pytest.mark.parametrize("schema", FEATURE_SCHEMAS)
def test_schema_dimensions_and_finite_features(schema):
    sample = generate_exact_2d_samples(
        1, n_interior_range=(8, 8), boundary_divisions=(2, 2), seed=4
    )[0]
    data = sample2d_to_data(sample, feature_schema=schema)
    spec = FEATURE_SCHEMAS[schema]
    assert data.x.shape == (sample.mesh.K, spec["node_dim"])
    assert torch.isfinite(data.x).all()
    if spec["edge_dim"] is None:
        assert getattr(data, "edge_attr", None) is None
    else:
        assert data.edge_attr.shape == (data.edge_index.shape[1], spec["edge_dim"])
        assert torch.isfinite(data.edge_attr).all()


@pytest.mark.parametrize("schema", INVARIANT_SCHEMAS)
def test_local_vertex_permutation_preserves_features_and_logits(schema):
    sample = generate_exact_2d_samples(
        1, n_interior_range=(12, 12), boundary_divisions=(3, 3), seed=2
    )[0]
    shifts = np.random.default_rng(19).integers(0, 3, size=sample.mesh.K)
    orders = np.stack([np.roll(np.arange(3), -shift) for shift in shifts])
    permuted_cells = np.take_along_axis(sample.mesh.cells, orders, axis=1)
    permuted_u = np.empty_like(sample.u)
    for cell, order in enumerate(orders):
        permuted_u[:, cell] = sample.u[order, cell]
    permuted_mesh = TriangleMesh(sample.mesh.points, permuted_cells)
    original = TriangleFeatureBuilder(sample.mesh, schema)
    permuted = TriangleFeatureBuilder(permuted_mesh, schema)
    original_x, original_edge_attr = original.build(sample.u)
    permuted_x, permuted_edge_attr = permuted.build(permuted_u)
    assert np.array_equal(original.edge_index, permuted.edge_index)
    assert np.array_equal(original_x, permuted_x)
    if original_edge_attr is not None:
        assert np.array_equal(original_edge_attr, permuted_edge_attr)

    model = GNNDetector(
        in_dim=FEATURE_SCHEMAS[schema]["node_dim"],
        hidden=4,
        heads=1,
        dropout=0.0,
        edge_dim=FEATURE_SCHEMAS[schema]["edge_dim"],
    ).eval()
    edge_index = torch.from_numpy(original.edge_index)
    with torch.no_grad():
        original_logits = model(
            torch.from_numpy(original_x),
            edge_index,
            None if original_edge_attr is None else torch.from_numpy(original_edge_attr),
        )
        permuted_logits = model(
            torch.from_numpy(permuted_x),
            edge_index,
            None if permuted_edge_attr is None else torch.from_numpy(permuted_edge_attr),
        )
    assert torch.equal(original_logits, permuted_logits)


def test_local_scaling_is_unchanged_by_remote_extremum():
    mesh = rectangular_mesh(nx=4, ny=4)
    rng = np.random.default_rng(5)
    u = rng.normal(size=(3, mesh.K))
    changed = u.copy()
    changed[:, -1] += 1e6
    assert mesh.K - 1 not in mesh.neighbors[0]

    local = TriangleFeatureBuilder(mesh, "invariant-local-v2")
    local_before, _ = local.build(u)
    local_after, _ = local.build(changed)
    assert np.array_equal(local_before[0], local_after[0])

    global_builder = TriangleFeatureBuilder(mesh, "invariant-node-v2")
    global_before, _ = global_builder.build(u)
    global_after, _ = global_builder.build(changed)
    assert not np.array_equal(global_before[0], global_after[0])


def test_extrema_features_are_bounded_neighbor_envelope_violations():
    mesh = rectangular_mesh(nx=2, ny=2)
    u = np.zeros((3, mesh.K))
    u[:, :] = np.linspace(0.2, 0.8, mesh.K)
    cell = next(index for index in range(mesh.K) if np.any(mesh.neighbors[index] >= 0))
    u[:, cell] = [0.0, 0.5, 1.0]
    features, edge_attr = TriangleFeatureBuilder(mesh, "invariant-extrema-v3").build(u)
    assert edge_attr is None
    assert features.shape == (mesh.K, 10)
    assert np.all(features[:, -2:] >= 0.0)
    assert np.all(features[:, -2:] < 1.0)
    assert np.any(features[cell, -2:] > 0.0)


def test_one_hop_label_expansion_is_exact_face_neighbor_union():
    mesh = rectangular_mesh(nx=3, ny=3)
    labels = np.zeros(mesh.K, dtype=bool)
    labels[mesh.K // 2] = True
    expanded = expand_labels_by_hops(labels, mesh, hops=1)
    expected = labels.copy()
    neighbors = mesh.neighbors[labels].ravel()
    expected[neighbors[neighbors >= 0]] = True
    assert np.array_equal(expanded, expected)
    assert np.array_equal(expand_labels_by_hops(labels, mesh, hops=0), labels)


def test_halo_override_changes_targets_but_not_features():
    sample = generate_exact_2d_samples(
        1, n_interior_range=(8, 8), boundary_divisions=(2, 2), seed=17
    )[0]
    halo = expand_labels_by_hops(sample.labels, sample.mesh, hops=1)
    original = sample2d_to_data(sample, "invariant-node-v2")
    expanded = sample2d_to_data(sample, "invariant-node-v2", labels=halo)
    assert torch.equal(original.x, expanded.x)
    assert torch.equal(original.edge_index, expanded.edge_index)
    assert torch.equal(original.y, torch.from_numpy(sample.labels.astype(np.float32)))
    assert torch.equal(expanded.y, torch.from_numpy(halo.astype(np.float32)))
    assert expanded.y.sum() >= original.y.sum()


def test_edge_aware_model_and_indicator_require_and_use_edge_attributes():
    sample = generate_exact_2d_samples(
        1, n_interior_range=(8, 8), boundary_divisions=(2, 2), seed=7
    )[0]
    data = sample2d_to_data(sample, "invariant-edge-v2")
    model = GNNDetector(in_dim=8, hidden=4, heads=1, edge_dim=6)
    with pytest.raises(ValueError, match="requires edge_attr"):
        model(data.x, data.edge_index)
    assert model(data.x, data.edge_index, data.edge_attr).shape == (sample.mesh.K,)

    model.checkpoint_metadata = {
        "feature_schema": "invariant-edge-v2",
        "edge_dim": 6,
    }
    solver = AdvectionDG2D(sample.mesh)
    flags = GNN2DIndicator(model=model).flag(solver, sample.u)
    assert flags.shape == (sample.mesh.K,)


def test_tiny_edge_schema_training_records_schema(tmp_path):
    config = {
        "data": {
            "mode": "exact2d",
            "seed": 0,
            "split_seed": 0,
            "n_samples": 12,
            "n_interior_range": [6, 8],
            "boundary_divisions": [2, 2],
            "val_fraction": 0.25,
        },
        "model": {
            "feature_schema": "invariant-edge-v2",
            "edge_dim": 6,
            "hidden": 4,
            "heads": 1,
            "dropout": 0.0,
        },
        "train": {"seed": 0, "epochs": 1, "batch_size": 3},
    }
    model, _ = train(config, tmp_path)
    history = json.loads((tmp_path / "metrics.json").read_text())
    assert model.hparams["in_dim"] == 8
    assert model.hparams["edge_dim"] == 6
    assert model.checkpoint_metadata["feature_schema"] == "invariant-edge-v2"
    assert history[-1]["best_epoch"] == 0


def test_tiny_halo_training_records_policy_and_original_validation(tmp_path):
    config = {
        "data": {
            "mode": "exact2d",
            "seed": 0,
            "split_seed": 0,
            "n_samples": 12,
            "n_interior_range": [6, 8],
            "boundary_divisions": [2, 2],
            "val_fraction": 0.25,
        },
        "model": {
            "feature_schema": "invariant-node-v2",
            "edge_dim": None,
            "hidden": 4,
            "heads": 1,
            "dropout": 0.0,
        },
        "train": {"seed": 0, "epochs": 1, "batch_size": 3, "label_halo": 1},
    }
    model, _ = train(config, tmp_path)
    policy = model.checkpoint_metadata["label_policy"]
    assert policy["training_hops"] == 1
    assert policy["validation_hops"] == 0
    assert policy["training_target_positive_cells"] > policy[
        "training_original_positive_cells"
    ]
