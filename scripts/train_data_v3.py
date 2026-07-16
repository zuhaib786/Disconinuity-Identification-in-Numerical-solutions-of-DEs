#!/usr/bin/env python
"""Train the controlled Phase 6 `data-v3` ladder (plan 6.3).

Only the training data changes across v3-A/B/C.  The representation
(`invariant-node-v2`), GAT capacity, optimizer, epochs, early stopping, and the
five training seeds are held at the frozen Phase 4 primary.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tci.data.generate2d_v3 import load_or_generate, resolve_spec, spec_id
from tci.data.graphs import FEATURE_SCHEMAS
from tci.models import GNNDetector
from tci.phase6 import LADDER, OFFLINE_METRICS, mean_and_sample_std
from tci.train import TRAINING_PROTOCOL_VERSION, resolve_config, train

REPRESENTATION = "invariant-node-v2"
CONFIGS = {step: Path(f"configs/gnn2d-{step.replace('v3-', 'data-v3-')}.yaml") for step in LADDER}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def dataset_report(config):
    """Freeze what the ladder step actually generated, before any training."""
    data = resolve_config(config)["data"]
    spec = dict(data["data_v3"])
    spec["n_samples"] = data["n_samples"]
    spec["seed"] = data["seed"]
    cache_dir = spec.pop("cache_dir", "runs/data-v3")
    samples = load_or_generate(spec, cache_dir=cache_dir)
    resolved = resolve_spec(spec)

    cells = sum(sample.mesh.K for sample in samples)
    positives = sum(int(sample.labels.sum()) for sample in samples)
    geometric = sum(int(sample.aux_labels.sum()) for sample in samples)
    agree = sum(
        int((sample.labels & sample.aux_labels).sum()) for sample in samples
    )
    components = {}
    for sample in samples:
        components[sample.curve] = components.get(sample.curve, 0) + 1
    families = {}
    for sample in samples:
        family = (sample.parameters.get("mesh") or {}).get("family", "unknown")
        families[family] = families.get(family, 0) + 1
    return {
        "spec_id": spec_id(resolved),
        "spec": resolved,
        "samples": len(samples),
        "cells": cells,
        "trajectories": len({s.trajectory_id for s in samples if s.trajectory_id >= 0}),
        "component_counts": components,
        "mesh_family_counts": families,
        "positive_rate": positives / cells,
        "geometric_cut_rate": geometric / cells,
        "new_labels_within_geometric_cut": agree / positives if positives else None,
        "mean_cells_per_sample": cells / len(samples),
    }


def validate_run(run_dir, step, seed):
    paths = {name: run_dir / name for name in ("model.pt", "config.yaml", "metrics.json")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("incomplete data-v3 run: " + ", ".join(missing))
    config = yaml.safe_load(paths["config.yaml"].read_text())
    history = json.loads(paths["metrics.json"].read_text())
    final = history[-1]
    best = next(row for row in history if row["is_best"])
    model = GNNDetector.load(paths["model.pt"])
    spec = FEATURE_SCHEMAS[REPRESENTATION]

    assert config["model"]["feature_schema"] == REPRESENTATION
    assert config["data"]["mode"] == "data-v3"
    assert config["data"]["data_v3"]["ladder"] == step
    assert config["train"]["seed"] == seed
    assert model.hparams["in_dim"] == spec["node_dim"]
    assert model.checkpoint_metadata["training_protocol_version"] == TRAINING_PROTOCOL_VERSION
    assert model.checkpoint_metadata["best_epoch"] == final["best_epoch"]

    return {
        "ladder": step,
        "representation": REPRESENTATION,
        "train_seed": seed,
        "run_dir": str(run_dir),
        "data_id": final["data_id"],
        "split_id": final["split_id"],
        "data_v3": final["data_v3"],
        "best_epoch": final["best_epoch"],
        "final_epoch": final["final_epoch"],
        "stopping_reason": final["stopping_reason"],
        "best_metrics": {key: best[key] for key in OFFLINE_METRICS},
        "checkpoint_metadata": model.checkpoint_metadata,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", choices=LADDER, default=list(LADDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output-root", type=Path, default=Path("runs/gnn2d-data-v3"))
    parser.add_argument("--summary", type=Path, default=Path("runs/paper/phase6-training-summary.json"))
    parser.add_argument(
        "--datasets-only",
        action="store_true",
        help="generate and cache the ladder datasets without training",
    )
    args = parser.parse_args()

    datasets = {}
    rows = []
    for step in args.steps:
        base_config = yaml.safe_load(CONFIGS[step].read_text())
        datasets[step] = dataset_report(base_config)
        print(
            f"{step}: {datasets[step]['samples']} samples, "
            f"positive rate {datasets[step]['positive_rate']:.4f}, "
            f"data spec {datasets[step]['spec_id'][:12]}",
            flush=True,
        )
        if args.datasets_only:
            continue
        for seed in args.seeds:
            run_dir = args.output_root / step / f"seed{seed}"
            if all((run_dir / name).exists() for name in ("model.pt", "config.yaml", "metrics.json")):
                print(f"validating {step} seed {seed}: {run_dir}", flush=True)
            else:
                if run_dir.exists() and any(run_dir.iterdir()):
                    raise FileExistsError(f"refusing to overwrite incomplete run {run_dir}")
                config = copy.deepcopy(base_config)
                config["train"]["seed"] = seed
                print(f"training {step} seed {seed}: {run_dir}", flush=True)
                train(config, run_dir)
            rows.append(validate_run(run_dir, step, seed))

    if args.datasets_only:
        print(json.dumps(datasets, indent=2, default=str))
        return

    for step in args.steps:
        step_rows = [row for row in rows if row["ladder"] == step]
        identifiers = {(row["data_id"], row["split_id"]) for row in step_rows}
        if len(identifiers) != 1:
            raise AssertionError(f"{step} seeds do not share one data/split ID")

    payload = {
        "schema_version": 1,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "phase 6 data-v3 controlled ladder",
        "representation": REPRESENTATION,
        "steps": args.steps,
        "seeds": args.seeds,
        "configs": {step: str(path) for step, path in CONFIGS.items() if step in args.steps},
        "datasets": datasets,
        "offline_summary": [
            {
                "ladder": step,
                "row_count": len([row for row in rows if row["ladder"] == step]),
                "metrics": {
                    metric: mean_and_sample_std(
                        [row["best_metrics"][metric] for row in rows if row["ladder"] == step]
                    )
                    for metric in OFFLINE_METRICS
                },
            }
            for step in args.steps
        ],
        "rows": rows,
    }
    atomic_json(args.summary, payload)
    print(json.dumps(payload["offline_summary"], indent=2))


if __name__ == "__main__":
    main()
