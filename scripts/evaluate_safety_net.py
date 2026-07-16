#!/usr/bin/env python
"""Evaluate the frozen primary GNN OR KXRCF safety net (P5-SAFETY-OR-001)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from tci.evaluate2d import run_slotted_rotation
from tci.indicators.base import OrIndicator
from tci.indicators.classical2d import KXRCFIndicator2D
from tci.indicators.learned import GNN2DIndicator
from tci.safety_evaluation import METRICS, aggregate, assess_safety_net


EXPERIMENT_ID = "P5-SAFETY-OR-001"


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def load_rows(path, configuration):
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("configuration") != configuration:
            raise ValueError(f"existing {path} has incompatible configuration")
        return payload
    return {"schema_version": 1, "configuration": configuration, "rows": []}


def primary_models(summary_path):
    summary = json.loads(summary_path.read_text())
    if summary["primary"]["representation"] != "invariant-node-v2":
        raise ValueError("P5-SAFETY-OR-001 requires the frozen invariant-node-v2 primary")
    rows = summary["primary_checkpoints"]
    if [row["train_seed"] for row in rows] != list(range(5)):
        raise ValueError("primary summary must retain training seeds 0--4")
    return summary, rows


def build_indicator(method, model_path, gnn_threshold, kxrcf_threshold):
    classical = KXRCFIndicator2D(threshold=kxrcf_threshold)
    if method == "kxrcf":
        return classical
    learned = GNN2DIndicator(model_path=model_path, threshold=gnn_threshold)
    if method == "gnn-or-kxrcf":
        return OrIndicator(learned, classical)
    raise ValueError(f"unknown method {method!r}")


def evaluate_case(case):
    method, train_seed, model_path, mesh, resolution, gnn_threshold, kxrcf_threshold, max_seconds = case
    row = {
        "experiment_id": EXPERIMENT_ID,
        "method": method,
        "train_seed": train_seed,
        "model": model_path,
        "mesh": mesh,
        "mesh_seed": 0,
        "resolution": resolution,
        "gnn_threshold": gnn_threshold if method == "gnn-or-kxrcf" else None,
        "kxrcf_threshold": kxrcf_threshold,
    }
    started = time.perf_counter()
    try:
        indicator = build_indicator(method, model_path, gnn_threshold, kxrcf_threshold)
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


def run_grid(payload, output, models, meshes, resolutions, thresholds, max_seconds):
    gnn_threshold, kxrcf_threshold = thresholds
    completed = {
        (row["method"], row["train_seed"], row["mesh"], row["resolution"])
        for row in payload["rows"]
    }
    tasks = []
    for mesh in meshes:
        for resolution in resolutions:
            key = ("kxrcf", None, mesh, resolution)
            if key not in completed:
                tasks.append(
                    ("kxrcf", None, None, mesh, resolution, gnn_threshold, kxrcf_threshold, max_seconds)
                )
    for model in models:
        model_path = model["artifacts"]["model.pt"]["path"]
        for mesh in meshes:
            for resolution in resolutions:
                key = ("gnn-or-kxrcf", model["train_seed"], mesh, resolution)
                if key not in completed:
                    tasks.append(
                        (
                            "gnn-or-kxrcf",
                            model["train_seed"],
                            model_path,
                            mesh,
                            resolution,
                            gnn_threshold,
                            kxrcf_threshold,
                            max_seconds,
                        )
                    )
    target = len(payload["rows"]) + len(tasks)
    print(f"running {len(tasks)} pending rows ({len(payload['rows'])}/{target} complete)", flush=True)
    for task in tasks:
        row = evaluate_case(task)
        payload["rows"].append(row)
        atomic_json(output, payload)
        print(
            f"finished {len(payload['rows'])}/{target}: "
            f"{row['method']} seed={row['train_seed']} {row['mesh']} n={row['resolution']} "
            f"status={row['status']}",
            flush=True,
        )
    return payload


def grouped_summary(rows, primary_rows):
    groups = []
    for method in ("primary-gnn", "kxrcf", "gnn-or-kxrcf"):
        source = primary_rows if method == "primary-gnn" else rows
        for mesh in ("structured", "delaunay"):
            for resolution in (8, 12, 16):
                selected = [
                    row
                    for row in source
                    if (method == "primary-gnn" or row["method"] == method)
                    and row["mesh"] == mesh
                    and int(row["resolution"]) == resolution
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
                        "method": method,
                        "mesh": mesh,
                        "resolution": resolution,
                        "row_count": len(selected),
                        "metrics": metric_summary,
                    }
                )
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary-summary", type=Path, default=Path("runs/paper/phase4-primary-summary.json")
    )
    parser.add_argument(
        "--phase3-calibration", type=Path, default=Path("runs/paper/phase3-calibration-rows.json")
    )
    parser.add_argument(
        "--phase3-heldout", type=Path, default=Path("runs/paper/phase3-heldout-rows.json")
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=Path("runs/feature-review/p5-safety-or-calibration.json"),
    )
    parser.add_argument(
        "--heldout-output",
        type=Path,
        default=Path("runs/feature-review/p5-safety-or-heldout.json"),
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("runs/feature-review/p5-safety-or-summary.json")
    )
    parser.add_argument("--gnn-threshold", type=float, default=0.02)
    parser.add_argument("--kxrcf-threshold", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    args = parser.parse_args()

    for path in (args.primary_summary, args.phase3_calibration, args.phase3_heldout):
        if not path.exists():
            raise FileNotFoundError(path)
    primary, models = primary_models(args.primary_summary)
    calibration_config = {
        "experiment_id": EXPERIMENT_ID,
        "kind": "calibration",
        "meshes": ["structured", "delaunay"],
        "resolutions": [10, 14],
        "gnn_threshold": args.gnn_threshold,
        "kxrcf_threshold": args.kxrcf_threshold,
        "mesh_seed": 0,
        "max_seconds": args.max_seconds,
        "threshold_tuning": False,
    }
    heldout_config = {
        "experiment_id": EXPERIMENT_ID,
        "kind": "heldout",
        "meshes": ["structured", "delaunay"],
        "resolutions": [8, 12, 16],
        "gnn_threshold": args.gnn_threshold,
        "kxrcf_threshold": args.kxrcf_threshold,
        "mesh_seed": 0,
        "max_seconds": args.max_seconds,
        "threshold_tuning": False,
    }
    calibration = load_rows(args.calibration_output, calibration_config)
    heldout = load_rows(args.heldout_output, heldout_config)
    run_grid(
        calibration,
        args.calibration_output,
        models,
        calibration_config["meshes"],
        calibration_config["resolutions"],
        (args.gnn_threshold, args.kxrcf_threshold),
        args.max_seconds,
    )
    run_grid(
        heldout,
        args.heldout_output,
        models,
        heldout_config["meshes"],
        heldout_config["resolutions"],
        (args.gnn_threshold, args.kxrcf_threshold),
        args.max_seconds,
    )

    phase3_calibration = json.loads(args.phase3_calibration.read_text())["rows"]
    phase3_heldout = json.loads(args.phase3_heldout.read_text())["rows"]
    primary_calibration = [
        row for row in phase3_calibration if row["representation"] == "invariant-node-v2" and row["threshold"] == args.gnn_threshold
    ]
    primary_heldout = [
        row for row in phase3_heldout if row["representation"] == "invariant-node-v2" and row["threshold"] == args.gnn_threshold
    ]
    union_calibration = [row for row in calibration["rows"] if row["method"] == "gnn-or-kxrcf"]
    union_heldout = [row for row in heldout["rows"] if row["method"] == "gnn-or-kxrcf"]
    kx_calibration = [row for row in calibration["rows"] if row["method"] == "kxrcf"]
    kx_heldout = [row for row in heldout["rows"] if row["method"] == "kxrcf"]
    assessment = assess_safety_net(union_calibration, union_heldout, primary_heldout)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "experiment_id": EXPERIMENT_ID,
        "scientific_change": "boolean union with KXRCF; GNN checkpoints/features and both thresholds frozen",
        "primary_summary": str(args.primary_summary),
        "data_id": primary["data_id"],
        "split_id": primary["split_id"],
        "thresholds": {"gnn": args.gnn_threshold, "kxrcf": args.kxrcf_threshold},
        "sources": {
            "primary_calibration": str(args.phase3_calibration),
            "primary_heldout": str(args.phase3_heldout),
            "union_calibration": str(args.calibration_output),
            "union_heldout": str(args.heldout_output),
        },
        "overall": {
            "primary_calibration": aggregate(primary_calibration),
            "kxrcf_calibration": aggregate(kx_calibration),
            "union_calibration": assessment["calibration_union"],
            "primary_heldout": assessment["heldout_primary"],
            "kxrcf_heldout": aggregate(kx_heldout),
            "union_heldout": assessment["heldout_union"],
        },
        "heldout_by_mesh_resolution": grouped_summary(heldout["rows"], primary_heldout),
        "acceptance": {"criteria": assessment["criteria"], "passes": assessment["passes"]},
        "gate": {
            "primary_calibration_rows_reused": len(primary_calibration),
            "primary_heldout_rows_reused": len(primary_heldout),
            "kxrcf_calibration_rows": len(kx_calibration),
            "union_calibration_rows": len(union_calibration),
            "kxrcf_heldout_rows": len(kx_heldout),
            "union_heldout_rows": len(union_heldout),
            "all_new_rows_successful": all(
                row["status"] == "ok" for row in calibration["rows"] + heldout["rows"]
            ),
        },
    }
    atomic_json(args.summary, payload)
    print(json.dumps({"gate": payload["gate"], "acceptance": payload["acceptance"]}, indent=2))


if __name__ == "__main__":
    main()
