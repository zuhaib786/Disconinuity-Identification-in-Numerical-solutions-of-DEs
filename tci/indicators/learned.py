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

        expected = int(self.model.hparams["in_dim"])
        if u.shape[0] != expected:
            raise ValueError(
                f"GNN checkpoint expects {expected} nodal features (N={expected - 1}), "
                f"but solver supplied {u.shape[0]} (N={u.shape[0] - 1})"
            )
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


class GNN2DIndicator(Indicator):
    """GNN indicator using P1 values and triangular-cell geometry features."""

    def __init__(self, model_path=None, model=None, threshold=0.1):
        from tci.models import GNNDetector

        if model is None:
            if model_path is None:
                raise ValueError("provide model or model_path")
            model = GNNDetector.load(model_path)
        self.model = model.eval()
        self.threshold = float(threshold)
        self.feature_schema = self.model.checkpoint_metadata.get(
            "feature_schema", "ordered-global-v1"
        )
        self._mesh_cache = {}

    def flag(self, solver, u):
        import torch

        from tci.data.graphs import TriangleFeatureBuilder

        key = id(solver.mesh)
        builder = self._mesh_cache.get(key)
        if builder is None:
            builder = TriangleFeatureBuilder(solver.mesh, self.feature_schema)
            self._mesh_cache = {key: builder}
        features, edge_attr = builder.build(u)
        edges = torch.from_numpy(builder.edge_index).long()
        expected = int(self.model.hparams["in_dim"])
        if features.shape[1] != expected:
            raise ValueError(
                f"GNN checkpoint expects {expected} features, got {features.shape[1]}"
            )
        with torch.no_grad():
            probability = torch.sigmoid(
                self.model(
                    torch.from_numpy(features),
                    edges,
                    None if edge_attr is None else torch.from_numpy(edge_attr),
                )
            ).numpy()
        return probability > self.threshold


class MLP2DIndicator(Indicator):
    """Fixed-width, permutation-invariant triangle-stencil MLP baseline."""

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

        from tci.data.features import stencil_features2d

        features = torch.from_numpy(
            stencil_features2d(u, solver.mesh, solver.all_neighbors)
        )
        with torch.no_grad():
            probability = torch.sigmoid(self.model(features)).numpy()
        return probability > self.threshold
