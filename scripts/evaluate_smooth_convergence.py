#!/usr/bin/env python
"""Bounded smooth-advection convergence study with frozen final fields.

The output JSON is written after every run and can be resumed.  A companion
NPZ stores the numerical and analytic final fields used to compute the rows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from tci.indicators.classical import MinmodIndicator
from tci.indicators.classical2d import MinmodIndicator2D
from tci.indicators.learned import (
    GNN2DIndicator,
    GNNIndicator,
    MLP2DIndicator,
    MLPIndicator,
)
from tci.mesh import rectangular_mesh
from tci.solvers.dg1d import DG1D
from tci.solvers.dg2d import AdvectionDG2D


SCHEMA_VERSION = 1
METHODS = ("unlimited", "minmod", "mlp", "gnn-tau-0.05", "gnn-tau-0.1")


class DeadlineIndicator:
    """Check a wall-clock deadline before each delegated indicator call."""

    def __init__(self, indicator, max_seconds):
        self.indicator = indicator
        self.deadline = time.perf_counter() + float(max_seconds)

    def flag(self, solver, u):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("indicator solve exceeded its wall-clock limit")
        return self.indicator.flag(solver, u)


def smooth_1d(x):
    return 0.5 + 0.25 * np.sin(2.0 * np.pi * x)


def smooth_2d(x, y):
    return 0.5 + 0.2 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)


def translated_1d(x, speed, final_time):
    return smooth_1d(np.mod(x - speed * final_time, 1.0))


def translated_2d(x, y, velocity, final_time):
    return smooth_2d(
        np.mod(x - velocity[0] * final_time, 1.0),
        np.mod(y - velocity[1] * final_time, 1.0),
    )


def exact_cell_means_1d(solver, speed, final_time, quadrature_order=12):
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    centers = 0.5 * (solver.VX[:-1] + solver.VX[1:])
    x = centers[:, None] + 0.5 * solver.h[:, None] * nodes[None, :]
    values = translated_1d(x, speed, final_time)
    return 0.5 * np.sum(values * weights[None, :], axis=1)


def errors_1d(solver, u, exact_means):
    error = solver.cell_means(u) - exact_means
    length = float(np.sum(solver.h))
    return (
        float(np.sum(solver.h * np.abs(error)) / length),
        float(np.sqrt(np.sum(solver.h * error**2) / length)),
    )


def errors_2d(solver, u, velocity, final_time, quadrature_order=8):
    """Integrate error against the analytic solution by a Duffy quadrature."""

    gauss, gauss_weights = np.polynomial.legendre.leggauss(quadrature_order)
    r_values = 0.5 * (gauss + 1.0)
    r_weights = 0.5 * gauss_weights
    vertices = solver.mesh.points[solver.mesh.cells]
    l1_integral = 0.0
    l2_integral = 0.0
    exact_mean_integral = np.zeros(solver.K)
    for r, wr in zip(r_values, r_weights):
        for s, ws in zip(r_values, r_weights):
            barycentric = np.array([1.0 - r, r * (1.0 - s), r * s])
            xy = np.einsum("i,kid->kd", barycentric, vertices)
            numerical = np.einsum("i,ik->k", barycentric, u)
            exact = translated_2d(xy[:, 0], xy[:, 1], velocity, final_time)
            physical_weights = 2.0 * solver.mesh.areas * r * wr * ws
            error = numerical - exact
            l1_integral += float(np.sum(physical_weights * np.abs(error)))
            l2_integral += float(np.sum(physical_weights * error**2))
            exact_mean_integral += physical_weights * exact
    total_area = float(np.sum(solver.mesh.areas))
    exact_means = exact_mean_integral / solver.mesh.areas
    return (
        l1_integral / total_area,
        math.sqrt(l2_integral / total_area),
        exact_means,
    )


def flag_metrics(history, cell_count):
    if not history:
        return 0.0, 0.0, np.zeros(cell_count, dtype=bool)
    fractions = np.asarray([np.mean(flags) for _, flags in history], dtype=float)
    return float(np.mean(fractions)), float(np.max(fractions)), history[-1][1].copy()


def indicator_for(method, dimension, args):
    if method == "unlimited":
        return None, None, None
    if dimension == "1d":
        if method == "minmod":
            return MinmodIndicator(), None, None
        if method == "mlp":
            return MLPIndicator(model_path=args.mlp1d_model), args.mlp1d_model, 0.5
        threshold = 0.05 if method.endswith("0.05") else 0.1
        return GNNIndicator(model_path=args.gnn1d_model, threshold=threshold), args.gnn1d_model, threshold
    if method == "minmod":
        return MinmodIndicator2D(), None, None
    if method == "mlp":
        return MLP2DIndicator(model_path=args.mlp2d_model), args.mlp2d_model, 0.5
    threshold = 0.05 if method.endswith("0.05") else 0.1
    return GNN2DIndicator(model_path=args.gnn2d_model, threshold=threshold), args.gnn2d_model, threshold


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **fields)
    os.replace(temporary, path)


def update_slopes(payload):
    fits = []
    for dimension in ("1d", "2d"):
        for method in METHODS:
            rows = sorted(
                (
                    row
                    for row in payload["rows"]
                    if row["dimension"] == dimension
                    and row["method"] == method
                    and row["status"] == "ok"
                ),
                key=lambda row: row["resolution"],
            )
            for index, row in enumerate(rows):
                metrics = row["metrics"]
                metrics["observed_l1_slope"] = None
                metrics["observed_l2_slope"] = None
                if index:
                    previous = rows[index - 1]
                    ratio = row["resolution"] / previous["resolution"]
                    for norm in ("l1", "l2"):
                        current_error = metrics[f"{norm}_error"]
                        previous_error = previous["metrics"][f"{norm}_error"]
                        if current_error > 0.0 and previous_error > 0.0:
                            metrics[f"observed_{norm}_slope"] = math.log(
                                previous_error / current_error
                            ) / math.log(ratio)
            fit = {
                "dimension": dimension,
                "method": method,
                "row_count": len(rows),
                "l1_slope": None,
                "l2_slope": None,
                "preserves_p1_rate": None,
            }
            if len(rows) >= 2:
                resolutions = np.asarray([row["resolution"] for row in rows], dtype=float)
                for norm in ("l1", "l2"):
                    errors = np.asarray(
                        [row["metrics"][f"{norm}_error"] for row in rows], dtype=float
                    )
                    if np.all(errors > 0.0):
                        fit[f"{norm}_slope"] = float(
                            -np.polyfit(np.log(resolutions), np.log(errors), 1)[0]
                        )
                if method.startswith("gnn") and fit["l2_slope"] is not None:
                    fit["preserves_p1_rate"] = fit["l2_slope"] >= 1.8
            fits.append(fit)
    payload["fits"] = fits


def run_1d(method, resolution, args):
    speed = float(args.speed_1d)
    solver = DG1D(0.0, 1.0, K=resolution, N=1, bc="periodic")
    u0 = solver.project(smooth_1d)
    initial_mass = float(np.sum(solver.h * solver.cell_means(u0)))
    indicator, model_path, threshold = indicator_for(method, "1d", args)
    if indicator is not None:
        indicator = DeadlineIndicator(indicator, args.max_seconds)
    started = time.perf_counter()
    u, history = solver.advect(
        u0,
        speed,
        args.final_time,
        indicator=indicator,
        cfl=args.cfl_1d,
        record_flags=True,
    )
    runtime = time.perf_counter() - started
    exact_means = exact_cell_means_1d(solver, speed, args.final_time)
    l1_error, l2_error = errors_1d(solver, u, exact_means)
    mean_flags, max_flags, final_flags = flag_metrics(history, solver.K)
    final_mass = float(np.sum(solver.h * solver.cell_means(u)))
    key = f"d1_{method.replace('-', '_').replace('.', 'p')}_k{resolution}"
    fields = {
        f"{key}_x": solver.x,
        f"{key}_u": u,
        f"{key}_exact_nodal": translated_1d(solver.x, speed, args.final_time),
        f"{key}_cell_means": solver.cell_means(u),
        f"{key}_exact_cell_means": exact_means,
        f"{key}_final_flags": final_flags,
    }
    row = {
        "dimension": "1d",
        "method": method,
        "resolution": resolution,
        "polynomial_degree": 1,
        "h": 1.0 / resolution,
        "model_path": model_path,
        "threshold": threshold,
        "status": "ok",
        "metrics": {
            "l1_error": l1_error,
            "l2_error": l2_error,
            "mean_flag_fraction": mean_flags,
            "max_flag_fraction": max_flags,
            "mass_error": abs(final_mass - initial_mass),
            "runtime_s": runtime,
        },
        "field_key": key,
    }
    return row, fields


def run_2d(method, resolution, args):
    velocity = np.asarray(args.velocity_2d, dtype=float)
    mesh = rectangular_mesh(nx=resolution, ny=resolution)
    solver = AdvectionDG2D(mesh, velocity=velocity, periodic=(True, True))
    u0 = solver.project(smooth_2d)
    initial_mass = solver.integral(u0)
    indicator, model_path, threshold = indicator_for(method, "2d", args)
    started = time.perf_counter()
    u, history = solver.solve(
        u0,
        args.final_time,
        cfl=args.cfl_2d,
        indicator=indicator,
        record_flags=True,
        max_seconds=args.max_seconds,
    )
    runtime = time.perf_counter() - started
    l1_error, l2_error, exact_means = errors_2d(
        solver, u, velocity, args.final_time
    )
    mean_flags, max_flags, final_flags = flag_metrics(history, solver.K)
    key = f"d2_{method.replace('-', '_').replace('.', 'p')}_n{resolution}"
    exact_nodal = translated_2d(
        solver.nodes[:, :, 0], solver.nodes[:, :, 1], velocity, args.final_time
    )
    fields = {
        f"{key}_points": mesh.points,
        f"{key}_cells": mesh.cells,
        f"{key}_u": u,
        f"{key}_exact_nodal": exact_nodal,
        f"{key}_cell_means": solver.cell_means(u),
        f"{key}_exact_cell_means": exact_means,
        f"{key}_final_flags": final_flags,
    }
    row = {
        "dimension": "2d",
        "method": method,
        "resolution": resolution,
        "cells": solver.K,
        "polynomial_degree": 1,
        "h": 1.0 / resolution,
        "mesh": "structured",
        "model_path": model_path,
        "threshold": threshold,
        "status": "ok",
        "metrics": {
            "l1_error": l1_error,
            "l2_error": l2_error,
            "mean_flag_fraction": mean_flags,
            "max_flag_fraction": max_flags,
            "mass_error": abs(solver.integral(u) - initial_mass),
            "runtime_s": runtime,
        },
        "field_key": key,
    }
    return row, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/paper/smooth-convergence.json"))
    parser.add_argument(
        "--fields-output",
        type=Path,
        default=Path("runs/paper/smooth-convergence.fields.npz"),
    )
    parser.add_argument("--resolutions-1d", nargs="+", type=int, default=[50, 100, 200, 400])
    parser.add_argument("--resolutions-2d", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--final-time", type=float, default=0.25)
    parser.add_argument("--speed-1d", type=float, default=1.0)
    parser.add_argument("--velocity-2d", nargs=2, type=float, default=[1.0, 0.5])
    parser.add_argument("--cfl-1d", type=float, default=0.375)
    parser.add_argument("--cfl-2d", type=float, default=0.15)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--gnn1d-model", default="runs/gnn1d/model.pt")
    parser.add_argument("--mlp1d-model", default="runs/mlp1d/model.pt")
    parser.add_argument("--gnn2d-model", default="runs/gnn2d-exact-seed0/model.pt")
    parser.add_argument("--mlp2d-model", default="runs/mlp2d-exact/model.pt")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        payload = json.loads(args.output.read_text())
    else:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "experiment": "active-indicator smooth periodic advection convergence",
            "configuration": {
                "initial_1d": "0.5 + 0.25 sin(2 pi x)",
                "initial_2d": "0.5 + 0.2 sin(2 pi x) cos(2 pi y)",
                "analytic_reference": "periodic translation of the initial condition",
                "final_time": args.final_time,
                "speed_1d": args.speed_1d,
                "velocity_2d": args.velocity_2d,
                "cfl_1d": args.cfl_1d,
                "cfl_2d": args.cfl_2d,
                "max_seconds_per_run": args.max_seconds,
                "l1_l2_scope": "domain-normalized analytic field error",
                "flag_scope": "last RK stage recorded once per time step",
            },
            "rows": [],
            "fits": [],
        }
    fields = {}
    if args.fields_output.exists():
        with np.load(args.fields_output, allow_pickle=False) as archive:
            fields.update({key: archive[key] for key in archive.files})
    completed = {
        (row["dimension"], row["method"], row["resolution"])
        for row in payload["rows"]
    }
    requested = [
        ("1d", method, resolution)
        for method in args.methods
        for resolution in args.resolutions_1d
    ] + [
        ("2d", method, resolution)
        for method in args.methods
        for resolution in args.resolutions_2d
    ]
    print(f"Loaded {len(completed)}/{len(requested)} requested rows", flush=True)
    for dimension, method, resolution in requested:
        key = (dimension, method, resolution)
        if key in completed:
            print(f"Skipping completed row {key}", flush=True)
            continue
        print(f"Starting {key}", flush=True)
        try:
            row, row_fields = (
                run_1d(method, resolution, args)
                if dimension == "1d"
                else run_2d(method, resolution, args)
            )
            fields.update(row_fields)
        except (RuntimeError, TimeoutError) as exc:
            row = {
                "dimension": dimension,
                "method": method,
                "resolution": resolution,
                "polynomial_degree": 1,
                "status": "timeout" if isinstance(exc, TimeoutError) else "failed",
                "reason": str(exc),
            }
        payload["rows"].append(row)
        completed.add(key)
        update_slopes(payload)
        atomic_json(args.output, payload)
        atomic_npz(args.fields_output, fields)
        print(f"Finished {key}: {row['status']}", flush=True)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
