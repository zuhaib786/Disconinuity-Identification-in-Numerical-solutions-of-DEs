#!/usr/bin/env python
"""Calibrate and evaluate the controlled Phase 3 feature representations."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from tci.data.generate2d import generate_exact_2d_samples
from tci.data.graphs import FEATURE_SCHEMAS, TriangleFeatureBuilder
from tci.evaluate2d import run_slotted_rotation
from tci.feature_evaluation import select_thresholds
from tci.indicators.learned import GNN2DIndicator
from tci.mesh import TriangleMesh
from tci.models import GNNDetector


METRICS = (
    "l1_error",
    "l2_error",
    "total_variation",
    "undershoot",
    "overshoot",
    "mass_error",
    "flagged_pct",
    "runtime_s",
)
PHASE3_REPRESENTATIONS = (
    "ordered-global-v1",
    "invariant-node-v2",
    "invariant-edge-v2",
    "invariant-local-v2",
)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def load_rows(path, configuration):
    if path.exists():
        payload = json.loads(path.read_text())
        existing = payload["configuration"]
        if existing != configuration:
            existing_base = {
                key: value for key, value in existing.items() if key != "selected_thresholds"
            }
            requested_base = {
                key: value for key, value in configuration.items() if key != "selected_thresholds"
            }
            requested_thresholds = configuration.get("selected_thresholds", {})
            existing_thresholds = existing.get("selected_thresholds", {})
            compatible_subset = (
                existing.get("kind") == "heldout"
                and existing_base == requested_base
                and all(
                    existing_thresholds.get(schema) == threshold
                    for schema, threshold in requested_thresholds.items()
                )
            )
            if not compatible_subset:
                raise ValueError(f"existing {path} has incompatible configuration")
            payload["configuration"] = configuration
            atomic_json(path, payload)
        return payload
    return {"schema_version": 1, "configuration": configuration, "rows": []}


def model_rows(training_summary):
    summary = json.loads(training_summary.read_text())
    grouped = {}
    representations = tuple(summary.get("representations", PHASE3_REPRESENTATIONS))
    unknown = set(representations) - set(FEATURE_SCHEMAS)
    if unknown:
        raise ValueError(f"training summary contains unknown schemas: {sorted(unknown)}")
    for representation in representations:
        rows = [
            row
            for row in summary["rows"]
            if row["representation"] == representation
        ]
        if sorted(row["train_seed"] for row in rows) != [0, 1, 2, 3, 4]:
            raise ValueError(f"{representation} does not have seeds 0..4")
        grouped[representation] = rows
    return summary, grouped


def evaluate_case(case):
    representation, train_seed, model_path, mesh, resolution, threshold, max_seconds = case
    row = {
        "representation": representation,
        "train_seed": train_seed,
        "model": model_path,
        "mesh": mesh,
        "mesh_seed": 0,
        "resolution": resolution,
        "threshold": threshold,
    }
    started = time.perf_counter()
    try:
        indicator = GNN2DIndicator(model_path=model_path, threshold=threshold)
        metrics, _ = run_slotted_rotation(
            indicator,
            n=resolution,
            mesh_type=mesh,
            seed=0,
            max_seconds=max_seconds,
        )
        row.update(status="ok", metrics=metrics)
    except TimeoutError as exc:
        row.update(status="timeout", reason=str(exc))
    except (RuntimeError, ValueError) as exc:
        row.update(status="failed", reason=str(exc))
    row["wall_time_s"] = time.perf_counter() - started
    return row


def run_grid(
    payload,
    output,
    grouped_models,
    thresholds,
    meshes,
    resolutions,
    max_seconds,
    workers,
):
    completed = {
        (
            row["representation"],
            row["train_seed"],
            row["mesh"],
            row["resolution"],
            row["threshold"],
        )
        for row in payload["rows"]
    }
    tasks = []
    for representation, models in grouped_models.items():
        for threshold in thresholds:
            for model_row in models:
                for mesh in meshes:
                    for resolution in resolutions:
                        key = (
                            representation,
                            model_row["train_seed"],
                            mesh,
                            resolution,
                            threshold,
                        )
                        if key in completed:
                            continue
                        tasks.append(
                            (
                                representation,
                                model_row["train_seed"],
                                str(Path(model_row["run_dir"]) / "model.pt"),
                                mesh,
                                resolution,
                                threshold,
                                max_seconds,
                            )
                        )
    target_total = len(payload["rows"]) + len(tasks)
    print(
        f"running {len(tasks)} pending rows with {workers} workers "
        f"({len(payload['rows'])}/{target_total} already complete)",
        flush=True,
    )
    def record(row):
        key = (
            row["representation"],
            row["train_seed"],
            row["mesh"],
            row["resolution"],
            row["threshold"],
        )
        payload["rows"].append(row)
        completed.add(key)
        atomic_json(output, payload)
        print(
            f"finished {len(payload['rows'])}/{target_total}: {key} "
            f"status={row['status']}",
            flush=True,
        )

    if workers == 1:
        for task in tasks:
            record(evaluate_case(task))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(evaluate_case, task): task for task in tasks}
            for future in concurrent.futures.as_completed(futures):
                record(future.result())
    return payload


def aggregate_heldout(rows, representations):
    groups = []
    for representation in representations:
        for mesh in ("structured", "delaunay"):
            for resolution in (8, 12, 16):
                selected = [
                    row
                    for row in rows
                    if row["representation"] == representation
                    and row["mesh"] == mesh
                    and row["resolution"] == resolution
                    and row["status"] == "ok"
                ]
                metric_summary = {}
                for metric in METRICS:
                    values = [row["metrics"][metric] for row in selected]
                    metric_summary[metric] = {
                        "mean": statistics.mean(values) if values else None,
                        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                    }
                groups.append(
                    {
                        "representation": representation,
                        "mesh": mesh,
                        "resolution": resolution,
                        "row_count": len(selected),
                        "metrics": metric_summary,
                    }
                )
    return groups


def permutation_stress(grouped_models):
    sample = generate_exact_2d_samples(
        1, n_interior_range=(20, 20), boundary_divisions=(4, 4), seed=2718
    )[0]
    shifts = np.random.default_rng(19).integers(0, 3, size=sample.mesh.K)
    orders = np.stack([np.roll(np.arange(3), -shift) for shift in shifts])
    permuted_cells = np.take_along_axis(sample.mesh.cells, orders, axis=1)
    permuted_u = np.empty_like(sample.u)
    for cell, order in enumerate(orders):
        permuted_u[:, cell] = sample.u[order, cell]
    permuted_mesh = TriangleMesh(sample.mesh.points, permuted_cells)
    rows = []
    for representation, models in grouped_models.items():
        original_builder = TriangleFeatureBuilder(sample.mesh, representation)
        permuted_builder = TriangleFeatureBuilder(permuted_mesh, representation)
        original_x, original_edge_attr = original_builder.build(sample.u)
        permuted_x, permuted_edge_attr = permuted_builder.build(permuted_u)
        feature_difference = float(np.max(np.abs(original_x - permuted_x)))
        edge_difference = (
            None
            if original_edge_attr is None
            else float(np.max(np.abs(original_edge_attr - permuted_edge_attr)))
        )
        edge_index = torch.from_numpy(original_builder.edge_index)
        for model_row in models:
            model = GNNDetector.load(Path(model_row["run_dir"]) / "model.pt")
            with torch.no_grad():
                original_logits = model(
                    torch.from_numpy(original_x),
                    edge_index,
                    None
                    if original_edge_attr is None
                    else torch.from_numpy(original_edge_attr),
                )
                permuted_logits = model(
                    torch.from_numpy(permuted_x),
                    edge_index,
                    None
                    if permuted_edge_attr is None
                    else torch.from_numpy(permuted_edge_attr),
                )
            logit_difference = float(torch.max(torch.abs(original_logits - permuted_logits)))
            expected_invariant = representation != "ordered-global-v1"
            rows.append(
                {
                    "representation": representation,
                    "train_seed": model_row["train_seed"],
                    "max_node_feature_difference": feature_difference,
                    "max_edge_feature_difference": edge_difference,
                    "max_logit_difference": logit_difference,
                    "expected_invariant": expected_invariant,
                    "passes": (not expected_invariant)
                    or (
                        feature_difference <= 1e-7
                        and (edge_difference is None or edge_difference <= 1e-7)
                        and logit_difference <= 1e-7
                    ),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-summary", type=Path, default=Path("runs/paper/phase3-training-summary.json")
    )
    parser.add_argument(
        "--calibration-output", type=Path, default=Path("runs/paper/phase3-calibration-rows.json")
    )
    parser.add_argument(
        "--heldout-output", type=Path, default=Path("runs/paper/phase3-heldout-rows.json")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("runs/paper/phase3-controlled-table.json")
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2, 0.3]
    )
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--safety-tolerance", type=float, default=1e-2)
    args = parser.parse_args()

    training, grouped = model_rows(args.training_summary)
    calibration_config = {
        "kind": "calibration",
        "meshes": ["structured", "delaunay"],
        "resolutions": [10, 14],
        "thresholds": args.thresholds,
        "mesh_seed": 0,
        "max_seconds": args.max_seconds,
    }
    calibration = load_rows(args.calibration_output, calibration_config)
    run_grid(
        calibration,
        args.calibration_output,
        grouped,
        args.thresholds,
        calibration_config["meshes"],
        calibration_config["resolutions"],
        args.max_seconds,
        args.workers,
    )
    selections = select_thresholds(
        calibration["rows"], args.thresholds, args.safety_tolerance
    )

    heldout_thresholds = {
        row["representation"]: row["selected_threshold"] for row in selections
    }
    heldout_config = {
        "kind": "heldout",
        "meshes": ["structured", "delaunay"],
        "resolutions": [8, 12, 16],
        "selected_thresholds": heldout_thresholds,
        "mesh_seed": 0,
        "max_seconds": args.max_seconds,
    }
    heldout = load_rows(args.heldout_output, heldout_config)
    for representation, models in grouped.items():
        run_grid(
            heldout,
            args.heldout_output,
            {representation: models},
            [heldout_thresholds[representation]],
            heldout_config["meshes"],
            heldout_config["resolutions"],
            args.max_seconds,
            args.workers,
        )

    stress = permutation_stress(grouped)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_summary": str(args.training_summary),
        "data_id": training["data_id"],
        "split_id": training["split_id"],
        "offline_summary": training["offline_summary"],
        "threshold_selection": selections,
        "heldout_summary": aggregate_heldout(heldout["rows"], tuple(grouped)),
        "permutation_stress": stress,
        "gate": {
            "training_rows": len(training["rows"]),
            "calibration_rows": len(calibration["rows"]),
            "heldout_rows": len(heldout["rows"]),
            "all_invariant_stress_tests_pass": all(
                row["passes"] for row in stress if row["expected_invariant"]
            ),
        },
    }
    atomic_json(args.summary, payload)
    print(json.dumps(payload["gate"], indent=2))


if __name__ == "__main__":
    main()
