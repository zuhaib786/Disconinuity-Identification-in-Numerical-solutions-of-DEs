#!/usr/bin/env python
"""Train and validate the five Phase 2 corrected ordered/global baselines."""

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

from tci.models import GNNDetector
from tci.train import TRAINING_PROTOCOL_VERSION, train


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def validate_run(run_dir, expected_seed):
    paths = {name: run_dir / name for name in ("model.pt", "config.yaml", "metrics.json")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("incomplete corrected-baseline run: " + ", ".join(missing))
    config = yaml.safe_load(paths["config.yaml"].read_text())
    history = json.loads(paths["metrics.json"].read_text())
    final = history[-1]
    best = next(row for row in history if row["is_best"])
    model = GNNDetector.load(paths["model.pt"])
    metadata = model.checkpoint_metadata
    assert config["train"]["seed"] == expected_seed
    assert final["train_seed"] == expected_seed
    assert metadata["train_seed"] == expected_seed
    assert metadata["best_epoch"] == final["best_epoch"]
    assert metadata["final_epoch"] == final["final_epoch"]
    assert metadata["training_protocol_version"] == TRAINING_PROTOCOL_VERSION

    in_dim = int(model.hparams["in_dim"])
    x = torch.linspace(0.0, 1.0, steps=4 * in_dim).reshape(4, in_dim)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    with torch.no_grad():
        before = model(x, edge_index)
        reloaded = GNNDetector.load(paths["model.pt"])
        after = reloaded(x, edge_index)
    if not torch.equal(before, after):
        raise AssertionError(f"save/load inference mismatch in {run_dir}")

    return {
        "train_seed": expected_seed,
        "run_dir": str(run_dir),
        "data_id": final["data_id"],
        "split_id": final["split_id"],
        "best_epoch": final["best_epoch"],
        "final_epoch": final["final_epoch"],
        "stopping_reason": final["stopping_reason"],
        "selection_metric": final["selection_metric"],
        "best_selection_value": final["best_selection_value"],
        "best_metrics": {
            key: best[key]
            for key in ("accuracy", "precision", "recall", "f1", "pr_auc", "ece")
        },
        "final_metrics": {
            key: final[key]
            for key in ("accuracy", "precision", "recall", "f1", "pr_auc", "ece")
        },
        "checkpoint_metadata": metadata,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "save_load_inference": "exact",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gnn2d-corrected-baseline.yaml"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--output-root", type=Path, default=Path("runs/gnn2d-corrected-baseline")
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("runs/paper/phase2-corrected-baseline-summary.json"),
    )
    args = parser.parse_args()

    base_config = yaml.safe_load(args.config.read_text()) or {}
    if "seed" not in base_config.get("data", {}) or "split_seed" not in base_config.get("data", {}):
        raise ValueError("corrected baseline config must declare data.seed and data.split_seed")
    rows = []
    for seed in args.seeds:
        run_dir = args.output_root / f"seed{seed}"
        existing = all((run_dir / name).exists() for name in ("model.pt", "config.yaml", "metrics.json"))
        if existing:
            print(f"validating existing seed {seed}: {run_dir}", flush=True)
        else:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(f"refusing to overwrite incomplete run {run_dir}")
            config = copy.deepcopy(base_config)
            config.setdefault("train", {})["seed"] = seed
            print(f"training seed {seed}: {run_dir}", flush=True)
            train(config, run_dir)
        rows.append(validate_run(run_dir, seed))

    data_ids = {row["data_id"] for row in rows}
    split_ids = {row["split_id"] for row in rows}
    if len(data_ids) != 1 or len(split_ids) != 1:
        raise AssertionError("corrected baselines do not share one data_id and split_id")
    payload = {
        "schema_version": 1,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "seeds": args.seeds,
        "data_id": next(iter(data_ids)),
        "split_id": next(iter(split_ids)),
        "gate": {
            "five_models": len(rows) == 5 and sorted(args.seeds) == [0, 1, 2, 3, 4],
            "identical_data_and_split": True,
            "best_epochs_restored": all(
                row["checkpoint_metadata"]["best_epoch"] == row["best_epoch"]
                for row in rows
            ),
            "save_load_inference_reproduced": all(
                row["save_load_inference"] == "exact" for row in rows
            ),
        },
        "best_metric_summary": {
            metric: {
                "mean": statistics.mean(row["best_metrics"][metric] for row in rows),
                "sample_std": statistics.stdev(
                    row["best_metrics"][metric] for row in rows
                ),
            }
            for metric in ("accuracy", "precision", "recall", "f1", "pr_auc", "ece")
        },
        "rows": rows,
    }
    atomic_json(args.summary, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
