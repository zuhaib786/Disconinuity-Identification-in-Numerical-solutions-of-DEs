import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from tci.data.features import stencil_features  # noqa: E402
from tci.data.generate import generate_exact_samples  # noqa: E402
from tci.indicators.learned import MLPIndicator  # noqa: E402
from tci.models import MLPDetector  # noqa: E402
from tci.solvers.dg1d import DG1D  # noqa: E402


def test_stencil_features_shape_and_range():
    samples = generate_exact_samples(1, N=1, k_range=(40, 40), seed=0)
    X = stencil_features(samples[0].u)
    assert X.shape == (40, 5)
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_mlp_forward_and_roundtrip(tmp_path):
    model = MLPDetector(in_dim=5, hidden=(16, 16))
    x = torch.rand(30, 5)
    assert model(x).shape == (30,)
    model.save(tmp_path / "m.pt")
    loaded = MLPDetector.load(tmp_path / "m.pt")
    assert torch.allclose(model(x), loaded(x))


def test_mlp_training_reduces_loss(tmp_path):
    import json

    from tci.train import train

    config = {
        "data": {"n_samples": 80, "k_range": [30, 50]},
        "model": {"type": "mlp", "hidden": [16, 16]},
        "train": {"epochs": 20, "threshold": 0.5},
    }
    model, final = train(config, tmp_path)
    history = json.loads((tmp_path / "metrics.json").read_text())
    assert history[-1]["train_loss"] < history[0]["train_loss"]

    ind = MLPIndicator(model=model, threshold=0.5)
    solver = DG1D(0.0, 1.0, K=40, N=1)
    u = solver.project(lambda x: np.where(x < 0.5, 0.0, 1.0))
    flags = ind.flag(solver, u)
    assert flags.shape == (40,) and flags.dtype == bool
