#!/usr/bin/env python
"""Train the controlled five-seed Phase 3 invariant feature ablations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from tci.data.graphs import FEATURE_SCHEMAS
from tci.models import GNNDetector
from tci.train import TRAINING_PROTOCOL_VERSION, train


PHASE3_REPRESENTATIONS = (
    "ordered-global-v1",
    "invariant-node-v2",
    "invariant-edge-v2",
    "invariant-local-v2",
)
REPRESENTATIONS = tuple(FEATURE_SCHEMAS)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def run_dir_for(representation, seed, output_root, ordered_root):
    if representation == "ordered-global-v1":
        return ordered_root / f"seed{seed}"
    return output_root / representation / f"seed{seed}"


def validate_run(run_dir, representation, seed):
    paths = {name: run_dir / name for name in ("model.pt", "config.yaml", "metrics.json")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("incomplete feature-ablation run: " + ", ".join(missing))
    config = yaml.safe_load(paths["config.yaml"].read_text())
    history = json.loads(paths["metrics.json"].read_text())
    final = history[-1]
    best = next(row for row in history if row["is_best"])
    model = GNNDetector.load(paths["model.pt"])
    metadata = model.checkpoint_metadata
    spec = FEATURE_SCHEMAS[representation]
    assert config["model"]["feature_schema"] == representation
    assert config["model"]["edge_dim"] == spec["edge_dim"]
    assert config["train"]["seed"] == seed
    assert model.hparams["in_dim"] == spec["node_dim"]
    assert model.hparams.get("edge_dim") == spec["edge_dim"]
    assert metadata["feature_schema"] == representation
    assert metadata.get("label_policy", {}).get("training_hops", 0) == config[
        "train"
    ].get("label_halo", 0)
    assert metadata["best_epoch"] == final["best_epoch"]
    assert metadata["training_protocol_version"] == TRAINING_PROTOCOL_VERSION

    x = torch.linspace(0.0, 1.0, steps=4 * spec["node_dim"]).reshape(
        4, spec["node_dim"]
    )
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    edge_attr = (
        None
        if spec["edge_dim"] is None
        else torch.linspace(0.0, 1.0, steps=6 * spec["edge_dim"]).reshape(
            6, spec["edge_dim"]
        )
    )
    with torch.no_grad():
        before = model(x, edge_index, edge_attr)
        after = GNNDetector.load(paths["model.pt"])(x, edge_index, edge_attr)
    if not torch.equal(before, after):
        raise AssertionError(f"save/load inference mismatch in {run_dir}")
    metrics = {
        key: best[key]
        for key in ("accuracy", "precision", "recall", "f1", "pr_auc", "ece")
    }
    return {
        "representation": representation,
        "train_seed": seed,
        "run_dir": str(run_dir),
        "data_id": final["data_id"],
        "split_id": final["split_id"],
        "best_epoch": final["best_epoch"],
        "final_epoch": final["final_epoch"],
        "stopping_reason": final["stopping_reason"],
        "best_metrics": metrics,
        "checkpoint_metadata": metadata,
        "label_policy": metadata.get("label_policy", {"training_hops": 0}),
        "save_load_inference": "exact",
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }


def summarize(rows, representations):
    output = []
    for representation in representations:
        group = [row for row in rows if row["representation"] == representation]
        output.append(
            {
                "representation": representation,
                "row_count": len(group),
                "metrics": {
                    metric: {
                        "mean": statistics.mean(
                            row["best_metrics"][metric] for row in group
                        ),
                        "sample_std": statistics.stdev(
                            row["best_metrics"][metric] for row in group
                        ),
                    }
                    for metric in ("accuracy", "precision", "recall", "f1", "pr_auc", "ece")
                },
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gnn2d-feature-ablation.yaml")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=list(PHASE3_REPRESENTATIONS),
    )
    parser.add_argument(
        "--ordered-root", type=Path, default=Path("runs/gnn2d-corrected-baseline")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/gnn2d-feature-ablation")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("runs/paper/phase3-training-summary.json")
    )
    args = parser.parse_args()

    base_config = yaml.safe_load(args.config.read_text()) or {}
    rows = []
    for representation in args.representations:
        spec = FEATURE_SCHEMAS[representation]
        for seed in args.seeds:
            run_dir = run_dir_for(
                representation, seed, args.output_root, args.ordered_root
            )
            existing = all(
                (run_dir / name).exists()
                for name in ("model.pt", "config.yaml", "metrics.json")
            )
            if existing:
                print(f"validating {representation} seed {seed}: {run_dir}", flush=True)
            else:
                if representation == "ordered-global-v1":
                    raise FileNotFoundError(f"missing Phase 2 corrected baseline {run_dir}")
                if run_dir.exists() and any(run_dir.iterdir()):
                    raise FileExistsError(f"refusing to overwrite incomplete run {run_dir}")
                config = copy.deepcopy(base_config)
                config["model"]["feature_schema"] = representation
                config["model"]["edge_dim"] = spec["edge_dim"]
                config["train"]["seed"] = seed
                print(f"training {representation} seed {seed}: {run_dir}", flush=True)
                train(config, run_dir)
            rows.append(validate_run(run_dir, representation, seed))

    data_ids = {row["data_id"] for row in rows}
    split_ids = {row["split_id"] for row in rows}
    if len(data_ids) != 1 or len(split_ids) != 1:
        raise AssertionError("feature ablations do not share fixed data/split IDs")
    payload = {
        "schema_version": 1,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "representations": args.representations,
        "seeds": args.seeds,
        "data_id": next(iter(data_ids)),
        "split_id": next(iter(split_ids)),
        "offline_summary": summarize(rows, args.representations),
        "rows": rows,
    }
    atomic_json(args.summary, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
