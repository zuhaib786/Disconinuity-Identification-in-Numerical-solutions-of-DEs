#!/usr/bin/env python
"""Evaluate one 2D GNN checkpoint across unseen triangular mesh sizes."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tci.data.generate2d import generate_exact_2d_samples
from tci.data.graphs import sample2d_to_data
from tci.models import GNNDetector
from tci.train import label_metrics


def evaluate(model, n_samples, sizes, threshold, seed):
    rows = []
    for index, size in enumerate(sizes):
        samples = generate_exact_2d_samples(
            n_samples,
            n_interior_range=(size, size),
            boundary_divisions=(8, 8),
            seed=seed + index,
        )
        probabilities, labels = [], []
        with torch.no_grad():
            for sample in samples:
                data = sample2d_to_data(sample)
                probabilities.append(
                    torch.sigmoid(model(data.x, data.edge_index)).numpy()
                )
                labels.append(np.asarray(data.y).astype(bool))
        y_probability = np.concatenate(probabilities)
        y_true = np.concatenate(labels)
        rows.append(
            {
                "n_interior": size,
                "mean_cells": float(np.mean([sample.mesh.K for sample in samples])),
                "n_samples": n_samples,
                "threshold": threshold,
                "positive_pct": 100.0 * float(np.mean(y_true)),
                "flagged_pct": 100.0 * float(np.mean(y_probability > threshold)),
                **label_metrics(y_true, y_probability, threshold),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 300])
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model = GNNDetector.load(args.model)
    rows = evaluate(model, args.n_samples, args.sizes, args.threshold, args.seed)
    text = json.dumps(rows, indent=2)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
