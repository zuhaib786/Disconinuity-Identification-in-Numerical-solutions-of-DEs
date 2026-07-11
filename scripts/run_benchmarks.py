#!/usr/bin/env python
"""Compare troubled-cell indicators across benchmark problems.

The learned indicators (gnn, mlp) are trained on ADVECTION data only, so the
burgers/sod/shu_osher rows measure cross-PDE generalization — the "universal
TCI" claim.

Examples:
    python scripts/run_benchmarks.py --problems box burgers sod
    python scripts/run_benchmarks.py --problems all \
        --indicators minmod kxrcf pa gnn mlp \
        --gnn-model runs/gnn1d/model.pt --mlp-model runs/mlp1d/model.pt
"""

import argparse

from tci.evaluate import compare_on, format_table
from tci.indicators import get_indicator

ALL_PROBLEMS = ["box", "burgers", "sod", "shu_osher"]


def build(name, args):
    if name == "gnn":
        return get_indicator(
            "gnn", model_path=args.gnn_model, threshold=args.gnn_threshold
        )
    if name == "mlp":
        return get_indicator(
            "mlp", model_path=args.mlp_model, threshold=args.mlp_threshold
        )
    return get_indicator(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problems", nargs="+", default=["box"],
                   choices=ALL_PROBLEMS + ["all"])
    p.add_argument("--indicators", nargs="+",
                   default=["minmod", "kxrcf", "pa"],
                   choices=["minmod", "kxrcf", "pa", "gnn", "mlp"])
    p.add_argument("--gnn-model", default="runs/gnn1d/model.pt")
    p.add_argument("--mlp-model", default="runs/mlp1d/model.pt")
    p.add_argument("--gnn-threshold", type=float, default=0.1)
    p.add_argument("--mlp-threshold", type=float, default=0.5)
    args = p.parse_args()

    problems = ALL_PROBLEMS if "all" in args.problems else args.problems
    indicators = {name: build(name, args) for name in args.indicators}

    for problem in problems:
        print(f"\n=== {problem} ===")
        print(format_table(compare_on(problem, indicators)))


if __name__ == "__main__":
    main()
