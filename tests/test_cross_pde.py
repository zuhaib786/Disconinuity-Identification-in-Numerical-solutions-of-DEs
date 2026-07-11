"""Cross-PDE generalization and robustness of the solve loop.

The advection-trained GNN applied inside a Burgers solve is the cheap smoke
test of the 'universal TCI' claim (the full Euler comparison lives in
scripts/run_benchmarks.py with the properly trained model — an
under-trained model can legitimately let a Sod solve blow up).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from tci.indicators.base import Indicator  # noqa: E402
from tci.indicators.learned import GNNIndicator  # noqa: E402
from tci.solvers.burgers import BurgersDG1D  # noqa: E402
from tci.solvers.euler import EulerDG1D, sod_initial  # noqa: E402
from tci.train import train  # noqa: E402


class NeverFlag(Indicator):
    def flag(self, solver, u):
        return np.zeros(solver.K, dtype=bool)


@pytest.fixture(scope="module")
def tiny_gnn(tmp_path_factory):
    out = tmp_path_factory.mktemp("gnn")
    config = {
        "data": {"n_samples": 150, "k_range": [40, 80]},
        "train": {"epochs": 40},
    }
    model, _ = train(config, out)
    return model


def test_gnn_runs_inside_burgers(tiny_gnn):
    ind = GNNIndicator(model=tiny_gnn, threshold=0.1)
    s = BurgersDG1D(0.0, 1.0, K=80, N=1)
    u = s.solve(
        s.project(lambda x: 0.5 + np.sin(2 * np.pi * x)), 0.3, indicator=ind
    )
    assert np.all(np.isfinite(u))
    assert abs(np.mean(s.cell_means(u)) - 0.5) < 1e-10


def test_blowup_raises_instead_of_hanging():
    """An indicator that never flags lets the unlimited Sod solve blow up;
    the solver must fail fast with RuntimeError, not loop as dt -> 0."""
    s = EulerDG1D(0.0, 1.0, K=100, N=2)
    with pytest.raises(RuntimeError, match="diverged"):
        s.solve(sod_initial, 0.2, indicator=NeverFlag())
