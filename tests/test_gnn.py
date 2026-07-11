import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from tci.data.generate import generate_exact_samples  # noqa: E402
from tci.data.graphs import (  # noqa: E402
    mesh_solution_features,
    path_edge_index,
    sample2d_to_data,
    sample_to_data,
)
from tci.data.generate2d import generate_exact_2d_samples  # noqa: E402
from tci.indicators.learned import GNN2DIndicator, GNNIndicator  # noqa: E402
from tci.indicators.learned import MLP2DIndicator  # noqa: E402
from tci.data.features import stencil_features2d  # noqa: E402
from tci.models import MLPDetector  # noqa: E402
from tci.models import GNNDetector  # noqa: E402
from tci.solvers.dg1d import DG1D  # noqa: E402
from tci.solvers.dg2d import AdvectionDG2D  # noqa: E402


def test_path_edge_index():
    e = path_edge_index(4)
    assert e.shape == (2, 6)


def test_forward_shapes():
    model = GNNDetector(in_dim=2)
    samples = generate_exact_samples(1, N=1, k_range=(40, 40), seed=0)
    d = sample_to_data(samples[0])
    logits = model(d.x, d.edge_index)
    assert logits.shape == (40,)


def test_2d_graph_features_and_forward_shape():
    sample = generate_exact_2d_samples(
        1, n_interior_range=(8, 8), boundary_divisions=(2, 2), seed=3
    )[0]
    features = mesh_solution_features(sample)
    assert features.shape == (sample.mesh.K, 10)
    assert np.all(features[:, 4:7] == np.sort(features[:, 4:7], axis=1))
    data = sample2d_to_data(sample)
    model = GNNDetector(in_dim=10, hidden=4, heads=1)
    assert model(data.x, data.edge_index).shape == (sample.mesh.K,)
    solver = AdvectionDG2D(sample.mesh)
    flags = GNN2DIndicator(model=model).flag(solver, sample.u)
    assert flags.shape == (sample.mesh.K,) and flags.dtype == bool
    mlp_features = stencil_features2d(sample.u, sample.mesh)
    assert mlp_features.shape == (sample.mesh.K, 16)
    mlp_flags = MLP2DIndicator(model=MLPDetector(in_dim=16, hidden=(4,))).flag(
        solver, sample.u
    )
    assert mlp_flags.shape == (sample.mesh.K,)


@pytest.mark.parametrize("conv", ["gat", "gcn", "sage"])
def test_all_conv_variants(conv):
    model = GNNDetector(in_dim=2, hidden=8, heads=2, conv=conv)
    samples = generate_exact_samples(1, N=1, k_range=(30, 30), seed=0)
    d = sample_to_data(samples[0])
    assert model(d.x, d.edge_index).shape == (30,)


def test_training_reduces_loss(tmp_path):
    from tci.train import train

    config = {
        "data": {"n_samples": 60, "k_range": [30, 50]},
        "train": {"epochs": 15},
    }
    model, final = train(config, tmp_path)
    import json

    history = json.loads((tmp_path / "metrics.json").read_text())
    assert history[-1]["train_loss"] < history[0]["train_loss"]
    assert (tmp_path / "model.pt").exists()

    # Round-trip and use as a solver indicator.
    loaded = GNNDetector.load(tmp_path / "model.pt")
    ind = GNNIndicator(model=loaded, threshold=0.1)
    solver = DG1D(0.0, 1.0, K=40, N=1)
    u = solver.project(lambda x: np.where(x < 0.5, 0.0, 1.0))
    flags = ind.flag(solver, u)
    assert flags.shape == (40,) and flags.dtype == bool


def test_tiny_2d_training_run(tmp_path):
    from tci.train import train

    config = {
        "data": {
            "mode": "exact2d",
            "n_samples": 12,
            "n_interior_range": [6, 10],
            "boundary_divisions": [2, 2],
            "val_fraction": 0.25,
        },
        "model": {"hidden": 4, "heads": 1, "layers": 2, "dropout": 0.0},
        "train": {"epochs": 1, "batch_size": 3},
    }
    model, metrics = train(config, tmp_path)
    assert model.hparams["in_dim"] == 10
    assert np.isfinite(metrics["train_loss"])
