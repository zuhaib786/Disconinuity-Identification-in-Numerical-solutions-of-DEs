"""GNN troubled-cell indicator (requires tci[ml])."""

from tci.indicators.base import Indicator


class GNNIndicator(Indicator):
    """Wraps a trained GNNDetector for solver-in-the-loop use.

    Nodal values per cell are min-max normalized (per call) and classified
    over the path graph of cells; probability > threshold flags the cell.
    Thesis default threshold: 0.1.
    """

    def __init__(self, model_path=None, model=None, threshold=0.1):
        import torch  # noqa: F401  (fail early with a clear error)

        from tci.models import GNNDetector

        if model is None:
            if model_path is None:
                raise ValueError("provide model or model_path")
            model = GNNDetector.load(model_path)
        self.model = model.eval()
        self.threshold = float(threshold)

    def flag(self, solver, u):
        import torch

        from tci.data.graphs import normalize_features, path_edge_index

        feats = torch.from_numpy(normalize_features(u))
        edges = torch.from_numpy(path_edge_index(u.shape[1])).long()
        with torch.no_grad():
            prob = torch.sigmoid(self.model(feats, edges)).numpy()
        return prob > self.threshold


class MLPIndicator(Indicator):
    """Fixed-stencil MLP baseline (Ray & Hesthaven, JCP 2018 style)."""

    def __init__(self, model_path=None, model=None, threshold=0.5):
        from tci.models import MLPDetector

        if model is None:
            if model_path is None:
                raise ValueError("provide model or model_path")
            model = MLPDetector.load(model_path)
        self.model = model.eval()
        self.threshold = float(threshold)

    def flag(self, solver, u):
        import torch

        from tci.data.features import stencil_features

        feats = torch.from_numpy(stencil_features(u, bc=solver.bc))
        with torch.no_grad():
            prob = torch.sigmoid(self.model(feats)).numpy()
        return prob > self.threshold
