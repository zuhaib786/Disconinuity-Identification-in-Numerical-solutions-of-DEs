import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from tci.data.generate import generate_exact_samples  # noqa: E402
from tci.data.graphs import path_edge_index, sample_to_data  # noqa: E402
from tci.indicators.learned import GNNIndicator  # noqa: E402
from tci.models import GNNDetector  # noqa: E402
from tci.solvers.dg1d import DG1D  # noqa: E402


def test_path_edge_index():
    e = path_edge_index(4)
    assert e.shape == (2, 6)


def test_forward_shapes():
    model = GNNDetector(in_dim=2)
    samples = generate_exact_samples(1, N=1, k_range=(40, 40), seed=0)
    d = sample_to_data(samples[0])
    logits = model(d.x, d.edge_index)
    assert logits.shape == (40,)


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
