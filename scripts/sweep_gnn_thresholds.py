#!/usr/bin/env python
"""Sweep a GNN probability threshold through solver-in-the-loop benchmarks."""

import argparse
import json
import time
from pathlib import Path

from tci.evaluate import run_benchmark
from tci.indicators.learned import GNNIndicator
from tci.models import GNNDetector
from tci.thresholds import EulerFinalStageRecorder, validate_thresholds


def sweep(model, checkpoint, problems, thresholds, N):
    validate_thresholds(thresholds)
    expected_N = int(model.hparams["in_dim"]) - 1
    if N != expected_N:
        raise ValueError(
            f"checkpoint expects N={expected_N}, but benchmark requested N={N}"
        )

    rows = []
    for problem in problems:
        for threshold in thresholds:
            recorder = EulerFinalStageRecorder(
                GNNIndicator(model=model, threshold=threshold)
            )
            started = time.perf_counter()
            try:
                metrics, artifacts = run_benchmark(problem, recorder, N=N)
            except RuntimeError as exc:
                if not str(exc).startswith("solve diverged:"):
                    raise
                rows.append(
                    {
                        "problem": problem,
                        "threshold": threshold,
                        "status": "diverged",
                        "N": N,
                        "checkpoint": str(checkpoint),
                        "flagged_pct": recorder.flagged_pct,
                        "flagged_pct_scope": "partial",
                        "completed_steps": recorder.completed_steps,
                        "runtime_s": time.perf_counter() - started,
                        "l1_error": None,
                        "l2_error": None,
                        "total_variation": None,
                        "reason": str(exc),
                    }
                )
                continue
            rows.append(
                {
                    "problem": problem,
                    "threshold": threshold,
                    "status": "ok",
                    "N": N,
                    "checkpoint": str(checkpoint),
                    "flagged_pct_scope": "full",
                    "completed_steps": len(artifacts["history"]),
                    **metrics,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2, 0.3]
    )
    parser.add_argument(
        "--problems", nargs="+", choices=["sod", "shu_osher"], default=["sod", "shu_osher"]
    )
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model = GNNDetector.load(args.model)
    inferred_N = int(model.hparams["in_dim"]) - 1
    rows = sweep(
        model,
        args.model,
        args.problems,
        args.thresholds,
        inferred_N if args.N is None else args.N,
    )
    text = json.dumps(rows, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
