"""Cell-graph construction for the GNN indicator (torch/PyG required)."""

import numpy as np


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


def mesh_solution_features(sample):
    return triangle_solution_features(sample.u, sample.mesh)


def sample2d_to_data(sample):
    """Convert a triangular-mesh sample into a PyG cell graph."""
    import torch
    from torch_geometric.data import Data

    return Data(
        x=torch.from_numpy(mesh_solution_features(sample)),
        edge_index=torch.from_numpy(mesh_edge_index(sample.mesh)).long(),
        y=torch.from_numpy(sample.labels.astype(np.float32)),
    )


def sample_to_data(sample):
    """Convert a tci.data.Sample into a torch_geometric Data object."""
    import torch
    from torch_geometric.data import Data

    return Data(
        x=torch.from_numpy(normalize_features(sample.u)),
        edge_index=torch.from_numpy(path_edge_index(sample.u.shape[1])).long(),
        y=torch.from_numpy(sample.labels.astype(np.float32)),
    )
