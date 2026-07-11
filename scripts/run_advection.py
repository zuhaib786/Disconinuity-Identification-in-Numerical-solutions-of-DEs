#!/usr/bin/env python
"""Compare troubled-cell indicators on the box-advection benchmark.

Examples:
    python scripts/run_advection.py --indicators minmod kxrcf pa
    python scripts/run_advection.py --indicators gnn --model runs/gnn1d/model.pt --plot
"""

import argparse

from tci.evaluate import compare_indicators, format_table, run_indicator
from tci.indicators import get_indicator


def build(name, args):
    if name == "gnn":
        return get_indicator("gnn", model_path=args.model, threshold=args.threshold)
    if name == "kxrcf":
        return get_indicator("kxrcf", a=args.a)
    return get_indicator(name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--indicators",
        nargs="+",
        default=["minmod", "kxrcf", "pa"],
        choices=["minmod", "kxrcf", "pa", "gnn"],
    )
    p.add_argument("--model", default="runs/gnn1d/model.pt", help="GNN checkpoint")
    p.add_argument("--threshold", type=float, default=0.1, help="GNN flag threshold")
    p.add_argument("--K", type=int, default=100)
    p.add_argument("--N", type=int, default=1)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--T", type=float, default=0.5)
    p.add_argument("--plot", action="store_true", help="show solution + flag history")
    args = p.parse_args()

    indicators = {name: build(name, args) for name in args.indicators}
    results = compare_indicators(indicators, K=args.K, N=args.N, a=args.a, T=args.T)
    print(format_table(results))

    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2, len(indicators), figsize=(6 * len(indicators), 8), squeeze=False
        )
        for j, (name, ind) in enumerate(indicators.items()):
            _, art = run_indicator(ind, K=args.K, N=args.N, a=args.a, T=args.T)
            solver, u, history = art["solver"], art["u"], art["history"]
            centers = 0.5 * (solver.VX[:-1] + solver.VX[1:])

            ax = axes[0][j]
            ax.plot(solver.x.ravel("F"), u.ravel("F"), ".", ms=2, label="final")
            ax.plot(centers, art["v_exact"], "g-", lw=1, label="exact")
            ax.set_title(f"{name}: solution at T={args.T}")
            ax.legend()

            ax = axes[1][j]
            for i, (_, flags) in enumerate(history):
                cells = centers[flags]
                ax.plot([i] * len(cells), cells, "b.", ms=1)
            ax.set_title(f"{name}: flagged cells per step")
            ax.set_xlabel("time step")
            ax.set_ylabel("x")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
