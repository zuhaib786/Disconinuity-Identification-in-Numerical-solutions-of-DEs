from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tci.data import generate_piecewise_fourier
from tci.evaluation import evaluate_binary


def _generate(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seed_sequence = np.random.SeedSequence(args.seed)
    sample_seeds = seed_sequence.spawn(args.num_samples)
    records: list[dict[str, object]] = []

    for index, sample_seed in enumerate(sample_seeds):
        example = generate_piecewise_fourier(
            n_cells=args.n_cells,
            max_jumps=args.max_jumps,
            n_modes=args.n_modes,
            rng=np.random.default_rng(sample_seed),
        )
        filename = f"sample_{index:06d}.npz"
        np.savez_compressed(
            output / filename,
            vertices=example.vertices,
            values=example.values,
            labels=example.labels,
            jump_locations=example.jump_locations,
        )
        records.append(
            {
                "file": filename,
                "sample": index,
                "split": "unassigned",
                "n_jumps": int(example.labels.sum()),
            }
        )

    manifest = {
        "generator": "piecewise_fourier_v1",
        "seed": args.seed,
        "num_samples": args.num_samples,
        "n_cells": args.n_cells,
        "max_jumps": args.max_jumps,
        "n_modes": args.n_modes,
        "samples": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"generated {args.num_samples} samples in {output}")
    return 0


def _load_array(path: str, key: str | None) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if key is None:
            raise ValueError(f"--{Path(path).stem}-key is required for an NPZ input")
        return loaded[key]
    return loaded


def _evaluate(args: argparse.Namespace) -> int:
    truth = _load_array(args.labels, args.labels_key)
    scores = _load_array(args.scores, args.scores_key)
    result = evaluate_binary(truth, scores, args.threshold)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tci")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="generate deterministic data")
    generate.add_argument("--output", required=True)
    generate.add_argument("--num-samples", type=int, default=100)
    generate.add_argument("--n-cells", type=int, default=100)
    generate.add_argument("--max-jumps", type=int, default=5)
    generate.add_argument("--n-modes", type=int, default=15)
    generate.add_argument("--seed", type=int, default=0)
    generate.set_defaults(handler=_generate)

    evaluate = commands.add_parser("evaluate", help="evaluate saved predictions")
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--scores", required=True)
    evaluate.add_argument("--labels-key")
    evaluate.add_argument("--scores-key")
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.set_defaults(handler=_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))

