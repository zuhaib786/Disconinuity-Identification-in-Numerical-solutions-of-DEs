#!/usr/bin/env python
"""Compare unlimited, classical, and GNN limiting on slotted-disk rotation."""

import argparse
import json
import sys
from pathlib import Path

from tci.evaluate2d import estimate_rotation, run_slotted_rotation
from tci.indicators import OrIndicator
from tci.indicators.classical2d import KXRCFIndicator2D, MinmodIndicator2D
from tci.indicators.learned import GNN2DIndicator, MLP2DIndicator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--indicators",
        nargs="+",
        choices=["none", "minmod", "kxrcf", "gnn", "mlp", "gnn-kxrcf"],
        default=["none", "minmod", "gnn"],
    )
    parser.add_argument("--model", default="runs/gnn2d-exact/model.pt")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--mlp-model", default="runs/mlp2d-exact/model.pt")
    parser.add_argument("--mlp-threshold", type=float, default=0.5)
    parser.add_argument("--kxrcf-threshold", type=float, default=1.0)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--mesh", choices=["structured", "delaunay"], default="structured")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-seconds", type=float, default=300.0)
    args = parser.parse_args()

    results = {}
    for name in args.indicators:
        if name == "none":
            indicator = None
        elif name == "minmod":
            indicator = MinmodIndicator2D()
        elif name == "gnn":
            indicator = GNN2DIndicator(model_path=args.model, threshold=args.threshold)
        elif name == "mlp":
            indicator = MLP2DIndicator(
                model_path=args.mlp_model, threshold=args.mlp_threshold
            )
        elif name == "kxrcf":
            indicator = KXRCFIndicator2D(threshold=args.kxrcf_threshold)
        else:
            indicator = OrIndicator(
                GNN2DIndicator(model_path=args.model, threshold=args.threshold),
                KXRCFIndicator2D(threshold=args.kxrcf_threshold),
            )
        estimate = estimate_rotation(indicator, args.n, args.mesh, args.seed)
        print(f"{name}: estimate {estimate}", file=sys.stderr, flush=True)
        if estimate["estimated_runtime_s"] > args.max_seconds:
            raise TimeoutError(
                f"{name} estimate {estimate['estimated_runtime_s']:.1f}s exceeds "
                f"the {args.max_seconds:.1f}s limit; reduce --n or raise --max-seconds"
            )
        metrics, _ = run_slotted_rotation(
            indicator,
            n=args.n,
            mesh_type=args.mesh,
            seed=args.seed,
            max_seconds=args.max_seconds,
        )
        results[name] = metrics
    text = json.dumps(results, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
