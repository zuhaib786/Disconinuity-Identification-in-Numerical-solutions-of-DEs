#!/usr/bin/env python
"""Incremental, bounded multi-checkpoint and multi-resolution rotation grid."""

import argparse
import json
from pathlib import Path

from tci.evaluate2d import estimate_rotation, run_slotted_rotation
from tci.indicators.learned import GNN2DIndicator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[8, 12, 16])
    parser.add_argument("--meshes", nargs="+", choices=["structured", "delaunay"], default=["structured", "delaunay"])
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for model_path in args.models:
        for mesh in args.meshes:
            for n in args.resolutions:
                indicator = GNN2DIndicator(model_path=model_path, threshold=args.threshold)
                estimate = estimate_rotation(indicator, n, mesh)
                row = {"model": model_path, "mesh": mesh, "n": n, "estimate": estimate}
                if estimate["estimated_runtime_s"] > args.max_seconds:
                    row["status"] = "skipped_estimate"
                else:
                    try:
                        metrics, _ = run_slotted_rotation(
                            indicator,
                            n=n,
                            mesh_type=mesh,
                            max_seconds=args.max_seconds,
                        )
                        row.update(status="ok", metrics=metrics)
                    except TimeoutError as exc:
                        row.update(status="timeout", reason=str(exc))
                rows.append(row)
                args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
