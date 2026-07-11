#!/usr/bin/env python
"""Generate paper-style final-state plots for bounded 2D rotation runs."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from tci.evaluate2d import run_slotted_rotation
from tci.indicators.learned import GNN2DIndicator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/gnn2d-exact/model.pt")
    parser.add_argument("--mesh", choices=["structured", "delaunay"], default="delaunay")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    indicator = GNN2DIndicator(model_path=args.model, threshold=args.threshold)
    _, artifacts = run_slotted_rotation(
        indicator,
        n=args.n,
        mesh_type=args.mesh,
        max_seconds=args.max_seconds,
    )
    solver, u = artifacts["solver"], artifacts["u"]
    means = solver.cell_means(u)
    triangulation = mtri.Triangulation(
        solver.mesh.points[:, 0], solver.mesh.points[:, 1], solver.mesh.cells
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    axes[0].tripcolor(triangulation, artifacts["v_exact"], shading="flat", vmin=0, vmax=1)
    axes[0].set_title("Initial / exact after one turn")
    image = axes[1].tripcolor(triangulation, means, shading="flat", vmin=0, vmax=1)
    axes[1].set_title(f"GNN, threshold={args.threshold:g}")
    for axis in axes:
        axis.set_aspect("equal")
        axis.set_xlabel("x")
        axis.set_ylabel("y")
    fig.colorbar(image, ax=axes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
