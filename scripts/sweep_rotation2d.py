#!/usr/bin/env python
"""Time-bounded GNN threshold sweep for 2D slotted-disk rotation."""

import argparse
import json
from pathlib import Path

from tci.evaluate2d import estimate_rotation, run_slotted_rotation
from tci.indicators.learned import GNN2DIndicator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="runs/gnn2d-exact/model.pt")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--meshes", nargs="+", choices=["structured", "delaunay"], default=["structured", "delaunay"])
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for mesh in args.meshes:
        for threshold in args.thresholds:
            indicator = GNN2DIndicator(model_path=args.model, threshold=threshold)
            estimate = estimate_rotation(indicator, args.n, mesh)
            if estimate["estimated_runtime_s"] > args.max_seconds:
                raise TimeoutError(f"estimate exceeds bound: {estimate}")
            metrics, _ = run_slotted_rotation(
                indicator,
                n=args.n,
                mesh_type=mesh,
                max_seconds=args.max_seconds,
            )
            rows.append({"mesh": mesh, "threshold": threshold, **metrics})
            args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
