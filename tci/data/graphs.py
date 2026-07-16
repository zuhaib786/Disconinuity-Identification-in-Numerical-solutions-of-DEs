"""Cell-graph construction for the GNN indicator (torch/PyG required)."""

import numpy as np


FEATURE_SCHEMAS = {
    "ordered-global-v1": {"node_dim": 10, "edge_dim": None},
    "invariant-node-v2": {"node_dim": 8, "edge_dim": None},
    "invariant-edge-v2": {"node_dim": 8, "edge_dim": 6},
    "invariant-local-v2": {"node_dim": 8, "edge_dim": 6},
    "invariant-extrema-v3": {"node_dim": 10, "edge_dim": None},
}


def one_ring_robust_scale(means, neighbors):
    """Phase 3 robust scale ``s_K`` and one-ring median of cell means.

    The one-ring of ``K`` contains ``K`` and its face neighbors; ``-1`` entries
    of ``neighbors`` mark boundary faces and are excluded.
    """
    means = np.asarray(means, dtype=float)
    valid = np.asarray(neighbors) >= 0
    neighbor_means = means[np.where(valid, neighbors, 0)]
    neighbor_means = np.where(valid, neighbor_means, np.nan)
    ring = np.column_stack([means, neighbor_means])
    center = np.nanmedian(ring, axis=1)
    mad = np.nanmedian(np.abs(ring - center[:, None]), axis=1)
    max_jump = np.nanmax(np.abs(ring - means[:, None]), axis=1)
    amplitude_floor = 1e-8 * np.maximum(1.0, np.nanmedian(np.abs(ring), axis=1))
    return np.maximum.reduce([mad, max_jump, amplitude_floor]), center


def normalize_features(u):
    """Per-sample min-max normalization of nodal values (thesis sec. 5.7.1).

    u: (Np, K) -> features (K, Np) in [0, 1].
    """
    lo, hi = float(np.min(u)), float(np.max(u))
    scale = hi - lo if hi - lo > 1e-12 else 1.0
    return ((u - lo) / scale).T.astype(np.float32)


def path_edge_index(K):
    """Bidirectional path-graph connectivity of K cells: (2, 2(K-1)) int array."""
    src = np.arange(K - 1)
    return np.vstack(
        [np.concatenate([src, src + 1]), np.concatenate([src + 1, src])]
    )


def mesh_edge_index(mesh):
    """Bidirectional face-adjacency graph for a triangular cell mesh."""
    return mesh.graph_edge_index()


def triangle_geometry_features(mesh):
    """Dimensionless geometry with edge lengths sorted per triangle."""
    geometry = mesh.geometry_features().copy()
    geometry[:, 1:4] = np.sort(geometry[:, 1:4], axis=1)
    return geometry.astype(np.float32)


def triangle_solution_features(u, mesh, geometry=None):
    """P1 solution and dimensionless, permutation-stable triangle geometry."""
    solution = normalize_features(u)
    if geometry is None:
        geometry = triangle_geometry_features(mesh)
    return np.concatenate([solution, geometry], axis=1).astype(np.float32)


class TriangleFeatureBuilder:
    """Schema-driven invariant node and directed-interface feature builder."""

    def __init__(self, mesh, feature_schema="ordered-global-v1"):
        if feature_schema not in FEATURE_SCHEMAS:
            raise ValueError(f"unknown feature schema {feature_schema!r}")
        self.mesh = mesh
        self.feature_schema = feature_schema
        self.spec = FEATURE_SCHEMAS[feature_schema]
        self.edge_index = mesh.graph_edge_index()
        self.geometry = triangle_geometry_features(mesh)
        self._prepare_node_geometry()
        self._prepare_gradient_operator()
        self._prepare_directed_faces()

    def _prepare_node_geometry(self):
        mesh = self.mesh
        neighbors = mesh.neighbors
        safe_neighbors = np.where(neighbors >= 0, neighbors, 0)
        neighbor_areas = mesh.areas[safe_neighbors]
        neighbor_areas = np.where(neighbors >= 0, neighbor_areas, np.nan)
        ring_areas = np.column_stack([mesh.areas, neighbor_areas])
        median_areas = np.nanmedian(ring_areas, axis=1)
        interior_fraction = np.mean(neighbors >= 0, axis=1)
        self.node_geometry = np.column_stack(
            [
                np.log(mesh.areas / median_areas),
                mesh.radius_ratio,
                1.0 - interior_fraction,
                interior_fraction,
            ]
        )
        self.safe_neighbors = safe_neighbors
        self.valid_neighbors = neighbors >= 0

    def _prepare_gradient_operator(self):
        vertices = self.mesh.points[self.mesh.cells]
        twice_area = 2.0 * self.mesh.areas
        gradients = np.empty((self.mesh.K, 3, 2))
        gradients[:, 0] = np.column_stack(
            [vertices[:, 1, 1] - vertices[:, 2, 1], vertices[:, 2, 0] - vertices[:, 1, 0]]
        ) / twice_area[:, None]
        gradients[:, 1] = np.column_stack(
            [vertices[:, 2, 1] - vertices[:, 0, 1], vertices[:, 0, 0] - vertices[:, 2, 0]]
        ) / twice_area[:, None]
        gradients[:, 2] = np.column_stack(
            [vertices[:, 0, 1] - vertices[:, 1, 1], vertices[:, 1, 0] - vertices[:, 0, 0]]
        ) / twice_area[:, None]
        self.basis_gradients = gradients

    def _prepare_directed_faces(self):
        edge_count = self.edge_index.shape[1]
        source_faces = np.empty(edge_count, dtype=np.int64)
        target_faces = np.empty(edge_count, dtype=np.int64)
        for edge in range(edge_count):
            source = self.edge_index[0, edge]
            target = self.edge_index[1, edge]
            source_match = np.flatnonzero(self.mesh.neighbors[source] == target)
            target_match = np.flatnonzero(self.mesh.neighbors[target] == source)
            if len(source_match) != 1 or len(target_match) != 1:
                raise ValueError("graph edge does not map to exactly one shared face")
            source_faces[edge] = source_match[0]
            target_faces[edge] = target_match[0]
        self.source_faces = source_faces
        self.target_faces = target_faces
        source = self.edge_index[0]
        target = self.edge_index[1]
        root_area = np.sqrt(self.mesh.areas[source])
        self.edge_geometry = np.column_stack(
            [
                self.mesh.edge_lengths[source, source_faces] / root_area,
                np.linalg.norm(
                    self.mesh.centroids[target] - self.mesh.centroids[source], axis=1
                )
                / root_area,
            ]
        )

    def _scales(self, u, means):
        if self.feature_schema != "invariant-local-v2":
            scale = max(float(np.max(u) - np.min(u)), 1e-8 * max(1.0, float(np.median(np.abs(means)))))
            return np.full(self.mesh.K, scale), np.full(self.mesh.K, float(np.median(means)))
        return one_ring_robust_scale(means, self.mesh.neighbors)

    def build(self, u):
        u = np.asarray(u, dtype=float)
        if u.shape != (3, self.mesh.K):
            raise ValueError(f"P1 field has shape {u.shape}, expected {(3, self.mesh.K)}")
        if self.feature_schema == "ordered-global-v1":
            return triangle_solution_features(u, self.mesh, self.geometry), None

        means = np.mean(u, axis=0)
        scales, centers = self._scales(u, means)
        gradients = np.einsum("ki,kid->kd", u.T, self.basis_gradients)
        h = np.sqrt(self.mesh.areas)
        node_features = np.column_stack(
            [
                (means - centers) / scales,
                np.std(u, axis=0) / scales,
                np.ptp(u, axis=0) / scales,
                h * np.linalg.norm(gradients, axis=1) / scales,
                self.node_geometry,
            ]
        ).astype(np.float32)
        if self.feature_schema == "invariant-extrema-v3":
            neighbor_means = means[self.safe_neighbors]
            neighbor_means = np.where(self.valid_neighbors, neighbor_means, np.nan)
            ring_means = np.column_stack([means, neighbor_means])
            lower_envelope = np.nanmin(ring_means, axis=1)
            upper_envelope = np.nanmax(ring_means, axis=1)
            lower_excess = np.maximum(lower_envelope - np.min(u, axis=0), 0.0) / scales
            upper_excess = np.maximum(np.max(u, axis=0) - upper_envelope, 0.0) / scales
            # Bound both nonnegative ratios in [0, 1) so a single extreme cell
            # cannot dominate the otherwise globally normalized node vector.
            extrema = np.column_stack(
                [
                    lower_excess / (1.0 + lower_excess),
                    upper_excess / (1.0 + upper_excess),
                ]
            ).astype(np.float32)
            node_features = np.column_stack([node_features, extrema]).astype(np.float32)
        if self.spec["edge_dim"] is None:
            return node_features, None

        source = self.edge_index[0]
        target = self.edge_index[1]
        source_faces = self.source_faces
        target_faces = self.target_faces
        source_nodes = np.stack([source_faces, (source_faces + 1) % 3], axis=1)
        # Neighbor face orientation is opposite, so reverse its local endpoints.
        target_nodes = np.stack([(target_faces + 1) % 3, target_faces], axis=1)
        edge_ids = np.arange(len(source))[:, None]
        source_trace = u[source_nodes, source[edge_ids]]
        target_trace = u[target_nodes, target[edge_ids]]
        trace_jump = target_trace - source_trace
        if self.feature_schema == "invariant-local-v2":
            edge_scale = np.maximum(scales[source], scales[target])
        else:
            edge_scale = scales[source]
        gradient_jump = gradients[target] - gradients[source]
        normal_jump = np.sum(
            gradient_jump * self.mesh.face_normals[source, source_faces], axis=1
        )
        edge_features = np.column_stack(
            [
                (means[target] - means[source]) / edge_scale,
                np.mean(trace_jump, axis=1) / edge_scale,
                np.max(np.abs(trace_jump), axis=1) / edge_scale,
                h[source] * normal_jump / edge_scale,
                self.edge_geometry,
            ]
        ).astype(np.float32)
        return node_features, edge_features


def mesh_solution_features(sample, feature_schema="ordered-global-v1"):
    features, _ = TriangleFeatureBuilder(sample.mesh, feature_schema).build(sample.u)
    return features


def expand_labels_by_hops(labels, mesh, hops=1):
    """Expand boolean cell labels over face adjacency by an exact hop count."""

    labels = np.asarray(labels, dtype=bool)
    if labels.shape != (mesh.K,):
        raise ValueError(f"labels have shape {labels.shape}, expected {(mesh.K,)}")
    if not isinstance(hops, (int, np.integer)) or hops < 0:
        raise ValueError("hops must be a nonnegative integer")
    expanded = labels.copy()
    frontier = labels.copy()
    for _ in range(int(hops)):
        neighbors = mesh.neighbors[frontier].ravel()
        neighbors = neighbors[neighbors >= 0]
        next_frontier = np.zeros(mesh.K, dtype=bool)
        next_frontier[neighbors] = True
        next_frontier &= ~expanded
        expanded |= next_frontier
        frontier = next_frontier
    return expanded


def sample2d_to_data(sample, feature_schema="ordered-global-v1", labels=None):
    """Convert a triangular-mesh sample into a PyG cell graph."""
    import torch
    from torch_geometric.data import Data

    builder = TriangleFeatureBuilder(sample.mesh, feature_schema)
    features, edge_attr = builder.build(sample.u)
    target = sample.labels if labels is None else np.asarray(labels)
    if target.shape != (sample.mesh.K,):
        raise ValueError(
            f"labels have shape {target.shape}, expected {(sample.mesh.K,)}"
        )
    kwargs = {
        "x": torch.from_numpy(features),
        "edge_index": torch.from_numpy(builder.edge_index).long(),
        "y": torch.from_numpy(target.astype(np.float32)),
    }
    if edge_attr is not None:
        kwargs["edge_attr"] = torch.from_numpy(edge_attr)
    return Data(**kwargs)


def sample_to_data(sample):
    """Convert a tci.data.Sample into a torch_geometric Data object."""
    import torch
    from torch_geometric.data import Data

    return Data(
        x=torch.from_numpy(normalize_features(sample.u)),
        edge_index=torch.from_numpy(path_edge_index(sample.u.shape[1])).long(),
        y=torch.from_numpy(sample.labels.astype(np.float32)),
    )
