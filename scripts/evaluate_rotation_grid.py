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
    if args.output.exists() and args.output.stat().st_size:
        try:
            rows = json.loads(args.output.read_text())
        except json.JSONDecodeError:
            print(f"Ignoring incomplete output file {args.output}", flush=True)
            rows = []
    completed = {
        (row["model"], row["mesh"], row["n"])
        for row in rows
        if "model" in row and "mesh" in row and "n" in row
    }
    total = len(args.models) * len(args.meshes) * len(args.resolutions)
    print(f"Loaded {len(completed)}/{total} completed rows", flush=True)
    for model_path in args.models:
        for mesh in args.meshes:
            for n in args.resolutions:
                key = (model_path, mesh, n)
                if key in completed:
                    print(f"Skipping completed row: {key}", flush=True)
                    continue
                indicator = GNN2DIndicator(model_path=model_path, threshold=args.threshold)
                estimate = estimate_rotation(indicator, n, mesh)
                row = {
                    "model": model_path,
                    "mesh": mesh,
                    "n": n,
                    "threshold": args.threshold,
                    "max_seconds": args.max_seconds,
                    "estimate": estimate,
                }
                print(
                    f"Starting {len(rows) + 1}/{total}: model={model_path}, "
                    f"mesh={mesh}, n={n}, estimate={estimate['estimated_runtime_s']:.1f}s",
                    flush=True,
                )
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
                completed.add(key)
                print(
                    f"Finished {len(rows)}/{total}: status={row['status']}",
                    flush=True,
                )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
