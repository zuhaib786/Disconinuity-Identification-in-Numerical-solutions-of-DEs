#!/usr/bin/env python
"""Plot the completed structured/unstructured rotation comparison tables."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path("runs/gnn2d-exact")
    data = {}
    for mesh in ("structured", "delaunay"):
        data[mesh] = json.loads((root / f"rotation-{mesh}-n12.json").read_text())
    data["structured"].update(
        json.loads((root / "rotation-baselines-structured-n12.json").read_text())
    )
    data["delaunay"].update(
        json.loads((root / "rotation-kxrcf-delaunay-n12.json").read_text())
    )
    data["delaunay"].update(
        json.loads((root / "rotation-mlp-delaunay-n12.json").read_text())
    )

    names = ["none", "minmod", "kxrcf", "mlp", "gnn"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    metrics = [
        ("l1_error", "L1 error"),
        ("undershoot", "Undershoot"),
        ("flagged_pct", "Cells flagged (%)"),
        ("runtime_s", "Runtime (s)"),
    ]
    x = np.arange(len(names))
    for axis, (metric, title) in zip(axes.ravel(), metrics):
        for offset, mesh in ((-0.18, "structured"), (0.18, "delaunay")):
            values = [data[mesh][name][metric] for name in names]
            axis.bar(x + offset, values, width=0.36, label=mesh)
        axis.set_xticks(x, names, rotation=25)
        axis.set_title(title)
    axes[0, 0].legend()
    Path("images").mkdir(exist_ok=True)
    fig.savefig("images/rotation2d-metrics.png", dpi=200)


if __name__ == "__main__":
    main()
