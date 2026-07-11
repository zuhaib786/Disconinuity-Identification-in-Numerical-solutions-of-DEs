"""GNN architectures for troubled-cell detection (requires tci[ml]).

The default matches thesis Table 5.1: GATConv(in, 32, heads=8, dropout=0.2)
-> ELU -> GATConv(256, 1) -> sigmoid (applied in the loss / at predict time).
``conv`` swaps the message-passing layer for ablations.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GNNDetector(nn.Module):
    def __init__(self, in_dim, hidden=32, heads=8, dropout=0.2, conv="gat", layers=2):
        super().__init__()
        self.hparams = dict(
            in_dim=in_dim,
            hidden=hidden,
            heads=heads,
            dropout=dropout,
            conv=conv,
            layers=layers,
        )
        self.conv_type = conv
        self.convs = nn.ModuleList()
        dim = in_dim
        for _ in range(layers - 1):
            if conv == "gat":
                self.convs.append(GATConv(dim, hidden, heads=heads, dropout=dropout))
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
            self.convs.append(GATConv(dim, 1, heads=1))
        elif conv == "gcn":
            self.convs.append(GCNConv(dim, 1))
        else:
            self.convs.append(SAGEConv(dim, 1))

    def forward(self, x, edge_index):
        """Returns per-node logits of shape (num_nodes,)."""
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
        return self.convs[-1](x, edge_index).squeeze(-1)

    def save(self, path):
        torch.save({"hparams": self.hparams, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        model = cls(**ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
