"""Learned troubled-cell detectors (requires tci[ml]).

GNNDetector default matches thesis Table 5.1: GATConv(in, 32, heads=8,
dropout=0.2) -> ELU -> GATConv(256, 1) -> sigmoid (applied in the loss / at
predict time); ``conv`` swaps the message-passing layer for ablations.

MLPDetector is the fixed-stencil baseline in the style of Ray & Hesthaven
(JCP 2018), operating on the 5 features of tci.data.features.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GNNDetector(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden=32,
        heads=8,
        dropout=0.2,
        conv="gat",
        layers=2,
        edge_dim=None,
    ):
        super().__init__()
        if edge_dim is not None and conv != "gat":
            raise ValueError("edge attributes are currently supported only by GAT")
        self.hparams = dict(
            in_dim=in_dim,
            hidden=hidden,
            heads=heads,
            dropout=dropout,
            conv=conv,
            layers=layers,
            edge_dim=edge_dim,
        )
        self.checkpoint_metadata = {}
        self.conv_type = conv
        self.convs = nn.ModuleList()
        dim = in_dim
        for _ in range(layers - 1):
            if conv == "gat":
                kwargs = {"edge_dim": edge_dim} if edge_dim is not None else {}
                self.convs.append(
                    GATConv(dim, hidden, heads=heads, dropout=dropout, **kwargs)
                )
                dim = hidden * heads
            elif conv == "gcn":
                self.convs.append(GCNConv(dim, hidden))
                dim = hidden
            elif conv == "sage":
                self.convs.append(SAGEConv(dim, hidden))
                dim = hidden
            else:
                raise ValueError(f"unknown conv {conv!r}")
        if conv == "gat":
            kwargs = {"edge_dim": edge_dim} if edge_dim is not None else {}
            self.convs.append(GATConv(dim, 1, heads=1, **kwargs))
        elif conv == "gcn":
            self.convs.append(GCNConv(dim, 1))
        else:
            self.convs.append(SAGEConv(dim, 1))

    def forward(self, x, edge_index, edge_attr=None):
        """Returns per-node logits of shape (num_nodes,)."""
        expected_edge_dim = self.hparams["edge_dim"]
        if expected_edge_dim is not None:
            if edge_attr is None:
                raise ValueError("edge-aware GNN checkpoint requires edge_attr")
            if edge_attr.ndim != 2 or edge_attr.shape[1] != expected_edge_dim:
                raise ValueError(
                    f"edge_attr has shape {tuple(edge_attr.shape)}, expected (*, {expected_edge_dim})"
                )
        for conv in self.convs[:-1]:
            x = F.elu(
                conv(x, edge_index, edge_attr=edge_attr)
                if self.conv_type == "gat"
                else conv(x, edge_index)
            )
        output = (
            self.convs[-1](x, edge_index, edge_attr=edge_attr)
            if self.conv_type == "gat"
            else self.convs[-1](x, edge_index)
        )
        return output.squeeze(-1)

    def save(self, path, metadata=None):
        checkpoint_metadata = dict(self.checkpoint_metadata)
        if metadata is not None:
            checkpoint_metadata.update(metadata)
        torch.save(
            {
                "hparams": self.hparams,
                "state_dict": self.state_dict(),
                "metadata": checkpoint_metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        model.checkpoint_metadata = dict(ckpt.get("metadata", {}))
        model.eval()
        return model


class MLPDetector(nn.Module):
    """Per-cell MLP on the fixed 5-feature stencil (Ray-Hesthaven style)."""

    def __init__(self, in_dim=5, hidden=(32, 32, 32), dropout=0.0):
        super().__init__()
        hidden = tuple(hidden)
        self.hparams = dict(in_dim=in_dim, hidden=list(hidden), dropout=dropout)
        self.checkpoint_metadata = {}
        layers = []
        dim = in_dim
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.LeakyReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """x: (num_cells, 5) -> per-cell logits (num_cells,)."""
        return self.net(x).squeeze(-1)

    def save(self, path, metadata=None):
        checkpoint_metadata = dict(self.checkpoint_metadata)
        if metadata is not None:
            checkpoint_metadata.update(metadata)
        torch.save(
            {
                "hparams": self.hparams,
                "state_dict": self.state_dict(),
                "metadata": checkpoint_metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        model.checkpoint_metadata = dict(ckpt.get("metadata", {}))
        model.eval()
        return model
