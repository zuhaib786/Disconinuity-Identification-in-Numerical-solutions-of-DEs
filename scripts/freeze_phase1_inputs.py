#!/usr/bin/env python
"""Freeze Phase 1 profile, rotation-field, and seed-level summary inputs.

Scientific solves are incremental and bounded.  This script only writes
machine-readable JSON/NPZ under ``runs/paper``; it does not create figures or
edit the manuscript.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from tci.evaluate import run_benchmark
from tci.evaluate2d import run_slotted_rotation
from tci.indicators.classical import KXRCFIndicator, MinmodIndicator
from tci.indicators.classical2d import KXRCFIndicator2D, MinmodIndicator2D
from tci.indicators.learned import (
    GNN2DIndicator,
    GNNIndicator,
    MLP2DIndicator,
    MLPIndicator,
)
from tci.indicators.pa import PAIndicator


METHODS_1D = ("unlimited", "minmod", "kxrcf", "pa", "mlp", "gnn")
METHODS_2D = ("unlimited", "minmod", "kxrcf", "mlp", "gnn")
PROBLEMS_1D = ("box", "sod", "shu_osher")


class DeadlineIndicator:
    """Delegate an indicator while enforcing a per-solve wall-clock bound."""

    def __init__(self, indicator, max_seconds):
        self.indicator = indicator
        self.deadline = time.perf_counter() + float(max_seconds)

    def flag(self, solver, u):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("1D solve exceeded its wall-clock limit")
        if self.indicator is None:
            return np.zeros(solver.K, dtype=bool)
        return self.indicator.flag(solver, u)


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **fields)
    os.replace(temporary, path)


def slug(value):
    return value.replace("-", "_").replace(".", "p")


def build_1d_indicator(method, args):
    if method == "unlimited":
        return None, None, None
    if method == "minmod":
        return MinmodIndicator(), None, None
    if method == "kxrcf":
        return KXRCFIndicator(threshold=args.kxrcf_threshold), None, args.kxrcf_threshold
    if method == "pa":
        return PAIndicator(threshold=args.pa_threshold), None, args.pa_threshold
    if method == "mlp":
        return MLPIndicator(args.mlp1d_model, threshold=args.mlp_threshold), args.mlp1d_model, args.mlp_threshold
    return GNNIndicator(args.gnn1d_model, threshold=args.gnn_threshold), args.gnn1d_model, args.gnn_threshold


def build_2d_indicator(method, args):
    if method == "unlimited":
        return None, None, None
    if method == "minmod":
        return MinmodIndicator2D(), None, None
    if method == "kxrcf":
        return KXRCFIndicator2D(threshold=args.kxrcf_threshold), None, args.kxrcf_threshold
    if method == "mlp":
        return MLP2DIndicator(args.mlp2d_model, threshold=args.mlp_threshold), args.mlp2d_model, args.mlp_threshold
    return GNN2DIndicator(args.gnn2d_model, threshold=args.gnn_threshold), args.gnn2d_model, args.gnn_threshold


def run_profile(problem, method, args):
    base_indicator, model_path, threshold = build_1d_indicator(method, args)
    # A false-only wrapper makes the unlimited run deadline-aware without
    # changing its field: the limiter immediately returns when no cell flags.
    indicator = DeadlineIndicator(base_indicator, args.max_seconds)
    metrics, artifacts = run_benchmark(
        problem,
        indicator,
        N=1,
        K=args.shu_k if problem == "shu_osher" else None,
    )
    solver = artifacts["solver"]
    state = artifacts["u"]
    density = state[:, :, 0] if state.ndim == 3 else state
    means = solver.cell_means(density)
    centers = 0.5 * (solver.VX[:-1] + solver.VX[1:])
    history = artifacts["history"]
    final_flags = history[-1][1] if history else np.zeros(solver.K, dtype=bool)
    key = f"{slug(problem)}_{slug(method)}"
    fields = {
        f"{key}_centers": centers,
        f"{key}_nodal_coordinates": solver.x,
        f"{key}_nodal_values": density,
        f"{key}_cell_means": means,
        f"{key}_reference_cell_means": np.asarray(artifacts["v_exact"]),
        f"{key}_final_flags": final_flags,
    }
    metrics = dict(metrics)
    metrics["flagged_pct"] = 100.0 * float(np.mean([flags.mean() for _, flags in history]))
    row = {
        "kind": "profile_1d",
        "problem": problem,
        "method": method,
        "polynomial_degree": 1,
        "cells": solver.K,
        "model_path": model_path,
        "threshold": threshold,
        "status": "ok",
        "metrics": metrics,
        "field_key": key,
    }
    return row, fields


def run_rotation(mesh_type, method, args):
    indicator, model_path, threshold = build_2d_indicator(method, args)
    metrics, artifacts = run_slotted_rotation(
        indicator,
        n=args.rotation_n,
        mesh_type=mesh_type,
        seed=args.mesh_seed,
        max_seconds=args.max_seconds,
    )
    solver = artifacts["solver"]
    history = artifacts["history"]
    final_flags = history[-1][1] if history else np.zeros(solver.K, dtype=bool)
    key = f"rotation_{slug(mesh_type)}_{slug(method)}_n{args.rotation_n}"
    fields = {
        f"{key}_points": solver.mesh.points,
        f"{key}_cells": solver.mesh.cells,
        f"{key}_centroids": solver.mesh.centroids,
        f"{key}_initial_cell_means": np.asarray(artifacts["v_exact"]),
        f"{key}_final_nodal_values": artifacts["u"],
        f"{key}_final_cell_means": solver.cell_means(artifacts["u"]),
        f"{key}_final_flags": final_flags,
    }
    row = {
        "kind": "rotation_2d",
        "mesh": mesh_type,
        "resolution": args.rotation_n,
        "mesh_seed": args.mesh_seed,
        "method": method,
        "model_path": model_path,
        "threshold": threshold,
        "status": "ok",
        "metrics": metrics,
        "field_key": key,
    }
    return row, fields


def sample_summary(rows):
    metric_names = (
        "l1_error",
        "total_variation",
        "undershoot",
        "overshoot",
        "mass_error",
        "flagged_pct",
        "runtime_s",
    )
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[(row["mesh"], int(row["n"]))].append(row)
    summaries = []
    for (mesh, resolution), group in sorted(grouped.items()):
        item = {
            "mesh": mesh,
            "resolution": resolution,
            "row_count": len(group),
            "models": [row["model"] for row in group],
            "metrics": {},
        }
        for name in metric_names:
            values = np.asarray([row["metrics"][name] for row in group], dtype=float)
            item["metrics"][name] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            }
        summaries.append(item)
    return summaries


def validate_grid(rows, expected_count, name):
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"{name} must contain {expected_count} rows, got {len(rows)}")
    keys = [(row.get("model"), row.get("mesh"), row.get("n")) for row in rows]
    if len(set(keys)) != expected_count:
        raise ValueError(f"{name} contains duplicate or incomplete model/mesh/n keys")


def freeze_seed_level_summary(args):
    sources = {
        "rotation_grid_tau_0.1": args.grid_tau_0_1,
        "rotation_grid_tau_0.05": args.grid_tau_0_05,
        "rotation_threshold_sweep_seed0": args.threshold_sweep,
    }
    missing = [str(path) for path in sources.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing summary source(s): " + ", ".join(missing))
    grid_01 = json.loads(args.grid_tau_0_1.read_text())
    grid_005 = json.loads(args.grid_tau_0_05.read_text())
    sweep = json.loads(args.threshold_sweep.read_text())
    validate_grid(grid_01, 30, "threshold-0.1 grid")
    validate_grid(grid_005, 30, "threshold-0.05 grid")
    if not isinstance(sweep, list) or len(sweep) != 10:
        raise ValueError(f"seed-zero threshold sweep must contain 10 rows, got {len(sweep)}")
    payload = {
        "schema_version": 1,
        "experiment": "Phase 1 threshold and resolution summaries",
        "aggregation": "arithmetic mean and sample standard deviation (ddof=1)",
        "sources": {name: str(path) for name, path in sources.items()},
        "rotation_grid_tau_0.1": {
            "threshold": 0.1,
            "row_count": len(grid_01),
            "seed_level_rows": grid_01,
            "groups": sample_summary(grid_01),
        },
        "rotation_grid_tau_0.05": {
            "threshold": 0.05,
            "row_count": len(grid_005),
            "seed_level_rows": grid_005,
            "groups": sample_summary(grid_005),
        },
        "rotation_threshold_sweep_seed0": {
            "row_count": len(sweep),
            "seed_level_rows": sweep,
        },
    }
    atomic_json(args.summary_output, payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/paper/phase1-visualization-inputs.json"))
    parser.add_argument(
        "--fields-output",
        type=Path,
        default=Path("runs/paper/phase1-visualization-inputs.npz"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("runs/paper/phase1-threshold-resolution-summary.json"),
    )
    parser.add_argument("--problems-1d", nargs="+", choices=PROBLEMS_1D, default=list(PROBLEMS_1D))
    parser.add_argument("--methods-1d", nargs="+", choices=METHODS_1D, default=list(METHODS_1D))
    parser.add_argument("--meshes-2d", nargs="+", choices=("structured", "delaunay"), default=["structured", "delaunay"])
    parser.add_argument("--methods-2d", nargs="+", choices=METHODS_2D, default=list(METHODS_2D))
    parser.add_argument("--rotation-n", type=int, default=12)
    parser.add_argument("--shu-k", type=int, default=300)
    parser.add_argument("--mesh-seed", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--gnn1d-model", default="runs/gnn1d/model.pt")
    parser.add_argument("--mlp1d-model", default="runs/mlp1d/model.pt")
    parser.add_argument("--gnn2d-model", default="runs/gnn2d-exact-seed0/model.pt")
    parser.add_argument("--mlp2d-model", default="runs/mlp2d-exact/model.pt")
    parser.add_argument("--gnn-threshold", type=float, default=0.05)
    parser.add_argument("--mlp-threshold", type=float, default=0.5)
    parser.add_argument("--kxrcf-threshold", type=float, default=1.0)
    parser.add_argument("--pa-threshold", type=float, default=1.0)
    parser.add_argument("--grid-tau-0-1", type=Path, default=Path("runs/paper/rotation-grid-tau-0.1.json"))
    parser.add_argument("--grid-tau-0-05", type=Path, default=Path("runs/paper/rotation-grid-tau-0.05.json"))
    parser.add_argument("--threshold-sweep", type=Path, default=Path("runs/paper/rotation-threshold-sweep-seed0.json"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.loads(args.output.read_text())
        if args.output.exists()
        else {
            "schema_version": 1,
            "experiment": "Phase 1 deterministic paper visualization inputs",
            "configuration": {
                "profile_polynomial_degree": 1,
                "rotation_resolution": args.rotation_n,
                "rotation_mesh_seed": args.mesh_seed,
                "gnn_threshold": args.gnn_threshold,
                "max_seconds_per_run": args.max_seconds,
                "flag_mask_scope": "last RK stage of the final completed time step",
            },
            "rows": [],
        }
    )
    fields = {}
    if args.fields_output.exists():
        with np.load(args.fields_output, allow_pickle=False) as archive:
            fields.update({key: archive[key] for key in archive.files})
    completed = {
        (row["kind"], row.get("problem", row.get("mesh")), row["method"])
        for row in payload["rows"]
    }
    requested = [
        ("profile_1d", problem, method)
        for problem in args.problems_1d
        for method in args.methods_1d
    ] + [
        ("rotation_2d", mesh, method)
        for mesh in args.meshes_2d
        for method in args.methods_2d
    ]
    print(f"Loaded {len(completed)}/{len(requested)} requested rows", flush=True)
    for kind, case, method in requested:
        key = (kind, case, method)
        if key in completed:
            print(f"Skipping completed row {key}", flush=True)
            continue
        print(f"Starting {key}", flush=True)
        try:
            row, row_fields = (
                run_profile(case, method, args)
                if kind == "profile_1d"
                else run_rotation(case, method, args)
            )
            fields.update(row_fields)
        except (RuntimeError, TimeoutError, ValueError) as exc:
            row = {
                "kind": kind,
                "problem" if kind == "profile_1d" else "mesh": case,
                "method": method,
                "status": "timeout" if isinstance(exc, TimeoutError) else "failed",
                "reason": str(exc),
            }
        payload["rows"].append(row)
        completed.add(key)
        atomic_json(args.output, payload)
        atomic_npz(args.fields_output, fields)
        print(f"Finished {key}: {row['status']}", flush=True)
    freeze_seed_level_summary(args)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
