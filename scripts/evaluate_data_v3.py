#!/usr/bin/env python
"""Evaluate the Phase 6 `data-v3` ladder against the frozen Phase 4 primary.

Stages, each resumable and written atomically after every row:

1. calibration on structured/Delaunay ``n={10,14}`` and the Phase 3 candidate
   thresholds, then one frozen threshold per ladder step by the Phase 3 rule;
2. the held-out rotation grid on structured/Delaunay ``n={8,12,16}``, which is
   the full-period slotted disk -- corners are out of family for the old data
   and in family for `data-v3`;
3. offline metrics on both the old exact held-out set and the new v3 held-out
   set (old-data metrics may drop; that is reported, not hidden);
4. the Phase 1.2 smooth-convergence preservation check at the frozen threshold;
5. the smooth-extremum false-positive rate; and
6. the predeclared plan 6.4 adopt/reject decision.

The Phase 4 primary is the named baseline throughout.  Its held-out rows are
reused from the frozen Phase 3 artifact rather than recomputed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# The script's own directory is on sys.path, so the frozen Phase 1.2 smooth-
# convergence definitions are reused here instead of being restated.
from evaluate_smooth_convergence import errors_2d, smooth_2d

from tci.data.generate2d import generate_exact_2d_samples
from tci.data.generate2d_v3 import load_or_generate, smooth_negative_samples
from tci.data.graphs import TriangleFeatureBuilder
from tci.evaluate2d import run_slotted_rotation
from tci.feature_evaluation import select_threshold
from tci.indicators.learned import GNN2DIndicator
from tci.mesh import rectangular_mesh
from tci.models import GNNDetector
from tci.phase6 import LADDER, aggregate_downstream, assess_data_v3, summarize_offline
from tci.solvers.dg2d import AdvectionDG2D
from tci.train import label_metrics

BASELINE = "phase4-primary"
BASELINE_THRESHOLD = 0.02
BASELINE_RUNS = Path("runs/gnn2d-feature-ablation/invariant-node-v2")
CONVERGENCE_RESOLUTIONS = (8, 16, 32)
CONVERGENCE_VELOCITY = np.array([1.0, 0.5])
CONVERGENCE_TIME = 0.25
CONVERGENCE_CFL = 0.15


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def load_rows(path, configuration):
    if path.exists():
        payload = json.loads(path.read_text())
        if payload["configuration"] != configuration:
            raise ValueError(f"existing {path} has an incompatible configuration")
        return payload
    return {"schema_version": 1, "configuration": configuration, "rows": []}


def model_index(training_summary, steps, seeds):
    """Map every ladder step and the baseline to its five checkpoints."""
    summary = json.loads(training_summary.read_text())
    models = {}
    for step in steps:
        rows = [row for row in summary["rows"] if row["ladder"] == step]
        if sorted(row["train_seed"] for row in rows) != sorted(seeds):
            raise ValueError(f"{step} does not have training seeds {seeds}")
        models[step] = {
            row["train_seed"]: str(Path(row["run_dir"]) / "model.pt") for row in rows
        }
    models[BASELINE] = {
        seed: str(BASELINE_RUNS / f"seed{seed}" / "model.pt") for seed in seeds
    }
    for path in models[BASELINE].values():
        if not Path(path).exists():
            raise FileNotFoundError(f"missing frozen Phase 4 primary checkpoint {path}")
    return summary, models


def rotation_case(case):
    group, train_seed, model_path, mesh, resolution, threshold, max_seconds = case
    row = {
        "group": group,
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
            indicator, n=resolution, mesh_type=mesh, seed=0, max_seconds=max_seconds
        )
        row.update(status="ok", metrics=metrics)
    except TimeoutError as exc:
        row.update(status="timeout", reason=str(exc))
    except (RuntimeError, ValueError) as exc:
        row.update(status="failed", reason=str(exc))
    row["wall_time_s"] = time.perf_counter() - started
    return row


def run_rotation_grid(payload, output, tasks, workers):
    completed = {
        (row["group"], row["train_seed"], row["mesh"], row["resolution"], row["threshold"])
        for row in payload["rows"]
    }
    pending = [
        task
        for task in tasks
        if (task[0], task[1], task[3], task[4], task[5]) not in completed
    ]
    total = len(payload["rows"]) + len(pending)
    print(f"rotation: {len(pending)} pending of {total} rows ({workers} workers)", flush=True)

    def record(row):
        payload["rows"].append(row)
        atomic_json(output, payload)
        print(
            f"  [{len(payload['rows'])}/{total}] {row['group']} seed{row['train_seed']} "
            f"{row['mesh']} n={row['resolution']} tau={row['threshold']} -> {row['status']}",
            flush=True,
        )

    if workers == 1:
        for task in pending:
            record(rotation_case(task))
    else:
        # Processes, not threads: a rotation is small-array NumPy driven from
        # Python, so worker threads serialize on the GIL instead of scaling.
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for future in concurrent.futures.as_completed(
                [executor.submit(rotation_case, task) for task in pending]
            ):
                record(future.result())
    return payload


def probabilities(model_path, samples):
    """Sigmoid probabilities and labels pooled over an evaluation set."""
    model = GNNDetector.load(model_path).eval()
    schema = model.checkpoint_metadata.get("feature_schema", "ordered-global-v1")
    probability, truth = [], []
    for sample in samples:
        features, edge_attr = TriangleFeatureBuilder(sample.mesh, schema).build(sample.u)
        edge_index = torch.from_numpy(sample.mesh.graph_edge_index()).long()
        with torch.no_grad():
            logits = model(
                torch.from_numpy(features),
                edge_index,
                None if edge_attr is None else torch.from_numpy(edge_attr),
            )
        probability.append(torch.sigmoid(logits).numpy())
        truth.append(np.asarray(sample.labels, dtype=bool))
    return np.concatenate(probability), np.concatenate(truth)


def offline_rows(models, thresholds, evaluation_sets):
    rows = []
    for group, checkpoints in models.items():
        for train_seed, model_path in sorted(checkpoints.items()):
            for set_name, samples in evaluation_sets.items():
                probability, truth = probabilities(model_path, samples)
                rows.append(
                    {
                        "group": group,
                        "train_seed": train_seed,
                        "evaluation_set": set_name,
                        "threshold": thresholds[group],
                        "samples": len(samples),
                        "cells": int(truth.size),
                        "positive_rate": float(np.mean(truth)),
                        "metrics": label_metrics(truth, probability, thresholds[group]),
                    }
                )
                print(
                    f"  offline {group} seed{train_seed} on {set_name}: "
                    f"F1 {rows[-1]['metrics']['f1']:.3f} "
                    f"PR-AUC {rows[-1]['metrics']['pr_auc']:.3f}",
                    flush=True,
                )
    return rows


def smooth_convergence(model_path, threshold, max_seconds):
    """The Phase 1.2 preservation check for one checkpoint at its threshold."""
    errors, flags = [], []
    for resolution in CONVERGENCE_RESOLUTIONS:
        mesh = rectangular_mesh(nx=resolution, ny=resolution)
        solver = AdvectionDG2D(mesh, velocity=CONVERGENCE_VELOCITY, periodic=(True, True))
        indicator = GNN2DIndicator(model_path=model_path, threshold=threshold)
        u, history = solver.solve(
            solver.project(smooth_2d),
            CONVERGENCE_TIME,
            cfl=CONVERGENCE_CFL,
            indicator=indicator,
            record_flags=True,
            max_seconds=max_seconds,
        )
        _, l2_error, _ = errors_2d(solver, u, CONVERGENCE_VELOCITY, CONVERGENCE_TIME)
        errors.append(l2_error)
        flags.append(float(np.mean([np.mean(flag) for _, flag in history])) if history else 0.0)
    resolutions = np.asarray(CONVERGENCE_RESOLUTIONS, dtype=float)
    slope = float(-np.polyfit(np.log(resolutions), np.log(np.asarray(errors)), 1)[0])
    return {
        "resolutions": list(CONVERGENCE_RESOLUTIONS),
        "l2_errors": errors,
        "mean_flag_fraction": flags,
        "l2_slope": slope,
        "preserves_p1_rate": slope >= 1.8,
    }




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, default=Path("runs/paper/phase6-training-summary.json"))
    parser.add_argument("--baseline-heldout", type=Path, default=Path("runs/paper/phase3-heldout-rows.json"))
    parser.add_argument("--calibration-output", type=Path, default=Path("runs/paper/phase6-calibration-rows.json"))
    parser.add_argument("--heldout-output", type=Path, default=Path("runs/paper/phase6-heldout-rows.json"))
    parser.add_argument("--offline-output", type=Path, default=Path("runs/paper/phase6-offline-rows.json"))
    parser.add_argument("--convergence-output", type=Path, default=Path("runs/paper/phase6-convergence-rows.json"))
    parser.add_argument("--extrema-output", type=Path, default=Path("runs/paper/phase6-extrema-rows.json"))
    parser.add_argument("--fields-output", type=Path, default=Path("runs/paper/phase6-figure-inputs.npz"))
    parser.add_argument("--decision-output", type=Path, default=Path("runs/paper/phase6-decision.json"))
    parser.add_argument("--steps", nargs="+", choices=LADDER, default=list(LADDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.02, 0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--safety-tolerance", type=float, default=1e-2)
    parser.add_argument("--max-seconds", type=float, default=240.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--offline-samples", type=int, default=200)
    parser.add_argument("--offline-seed", type=int, default=987654321)
    parser.add_argument("--extrema-samples", type=int, default=40)
    args = parser.parse_args()

    training, models = model_index(args.training_summary, args.steps, args.seeds)

    # ---- 1. calibration and threshold freezing -----------------------------
    calibration_config = {
        "kind": "calibration",
        "meshes": ["structured", "delaunay"],
        "resolutions": [10, 14],
        "thresholds": args.thresholds,
        "mesh_seed": 0,
        "max_seconds": args.max_seconds,
        "rule": "phase 3 select_threshold",
    }
    calibration = load_rows(args.calibration_output, calibration_config)
    tasks = [
        (step, seed, models[step][seed], mesh, resolution, threshold, args.max_seconds)
        for step in args.steps
        for seed in args.seeds
        for threshold in args.thresholds
        for mesh in calibration_config["meshes"]
        for resolution in calibration_config["resolutions"]
    ]
    run_rotation_grid(calibration, args.calibration_output, tasks, args.workers)

    selections = {}
    for step in args.steps:
        rows = [row for row in calibration["rows"] if row["group"] == step]
        selections[step] = {
            "ladder": step,
            **select_threshold(rows, args.thresholds, args.safety_tolerance),
        }
        print(
            f"{step}: frozen threshold {selections[step]['selected_threshold']} "
            f"({selections[step]['safety_constraint']})",
            flush=True,
        )
    thresholds = {step: selections[step]["selected_threshold"] for step in args.steps}
    thresholds[BASELINE] = BASELINE_THRESHOLD

    # ---- 2. held-out rotation grid (the full-period slotted disk) ----------
    heldout_config = {
        "kind": "heldout",
        "meshes": ["structured", "delaunay"],
        "resolutions": [8, 12, 16],
        "selected_thresholds": thresholds,
        "mesh_seed": 0,
        "max_seconds": args.max_seconds,
    }
    heldout = load_rows(args.heldout_output, heldout_config)
    tasks = [
        (step, seed, models[step][seed], mesh, resolution, thresholds[step], args.max_seconds)
        for step in args.steps
        for seed in args.seeds
        for mesh in heldout_config["meshes"]
        for resolution in heldout_config["resolutions"]
    ]
    run_rotation_grid(heldout, args.heldout_output, tasks, args.workers)

    baseline_rows = [
        {**row, "group": BASELINE}
        for row in json.loads(args.baseline_heldout.read_text())["rows"]
        if row["representation"] == "invariant-node-v2"
    ]

    # ---- 3. offline metrics on the old and the new held-out sets -----------
    offline_config = {
        "kind": "offline",
        "old_exact_set": {
            "generator": "generate_exact_2d_samples",
            "samples": args.offline_samples,
            "seed": args.offline_seed,
            "labels": "legacy geometric cut labels",
        },
        "v3_set": {
            "generator": "data-v3 (v3-c mixture)",
            "samples": args.offline_samples,
            "seed": args.offline_seed,
            "labels": "phase 6 reference label rule",
        },
        "thresholds": thresholds,
    }
    offline = load_rows(args.offline_output, offline_config)
    if not offline["rows"]:
        print("building the held-out offline evaluation sets", flush=True)
        evaluation_sets = {
            "old_exact": generate_exact_2d_samples(
                args.offline_samples, seed=args.offline_seed
            ),
            "data_v3": load_or_generate(
                {"ladder": "v3-c", "n_samples": args.offline_samples, "seed": args.offline_seed}
            ),
        }
        offline["rows"] = offline_rows(models, thresholds, evaluation_sets)
        atomic_json(args.offline_output, offline)

    # ---- 4. smooth-convergence preservation check --------------------------
    convergence_config = {
        "kind": "smooth-convergence",
        "resolutions": list(CONVERGENCE_RESOLUTIONS),
        "velocity": CONVERGENCE_VELOCITY.tolist(),
        "final_time": CONVERGENCE_TIME,
        "cfl": CONVERGENCE_CFL,
        "thresholds": thresholds,
        "check": "fitted L2 slope >= 1.8 (plan 1.2)",
    }
    convergence = load_rows(args.convergence_output, convergence_config)
    done = {(row["group"], row["train_seed"]) for row in convergence["rows"]}
    for group, checkpoints in models.items():
        for seed, model_path in sorted(checkpoints.items()):
            if (group, seed) in done:
                continue
            row = {
                "group": group,
                "train_seed": seed,
                "model": model_path,
                "threshold": thresholds[group],
                **smooth_convergence(model_path, thresholds[group], args.max_seconds),
            }
            convergence["rows"].append(row)
            atomic_json(args.convergence_output, convergence)
            print(
                f"  convergence {group} seed{seed}: fitted L2 slope {row['l2_slope']:.2f}",
                flush=True,
            )

    # ---- 5. smooth-extremum false positives --------------------------------
    extrema_config = {
        "kind": "smooth-extremum",
        "samples": args.extrema_samples,
        "seed": args.offline_seed + 1,
        "thresholds": thresholds,
        "note": "component-3 fields with no interface; flagged cells are false positives",
    }
    extrema = load_rows(args.extrema_output, extrema_config)
    if not extrema["rows"]:
        batch = smooth_negative_samples(args.extrema_samples, args.offline_seed + 1)
        for group, checkpoints in models.items():
            for seed, model_path in sorted(checkpoints.items()):
                probability, truth = probabilities(model_path, batch)
                flagged = probability > thresholds[group]
                extrema["rows"].append(
                    {
                        "group": group,
                        "train_seed": seed,
                        "threshold": thresholds[group],
                        "cells": int(truth.size),
                        "label_positive_rate": float(np.mean(truth)),
                        "flag_rate": float(np.mean(flagged)),
                        "false_positive_rate": float(np.mean(flagged & ~truth)),
                    }
                )
                print(
                    f"  extrema {group} seed{seed}: FP rate "
                    f"{extrema['rows'][-1]['false_positive_rate']:.4f}",
                    flush=True,
                )
        atomic_json(args.extrema_output, extrema)

    # ---- 6. figure inputs and the predeclared decision ----------------------
    fields = {}
    for group in list(args.steps) + [BASELINE]:
        for mesh in ("structured", "delaunay"):
            indicator = GNN2DIndicator(model_path=models[group][0], threshold=thresholds[group])
            metrics, extra = run_slotted_rotation(
                indicator, n=12, mesh_type=mesh, seed=0, max_seconds=args.max_seconds
            )
            key = f"{group.replace('-', '_')}_{mesh}_n12"
            solver = extra["solver"]
            fields[f"{key}_points"] = solver.mesh.points
            fields[f"{key}_cells"] = solver.mesh.cells
            fields[f"{key}_cell_means"] = solver.cell_means(extra["u"])
            fields[f"{key}_exact_cell_means"] = extra["v_exact"]
            fields[f"{key}_final_flags"] = extra["history"][-1][1]
            print(f"  fields {key}: flagged {metrics['flagged_pct']:.1f}%", flush=True)
    args.fields_output.parent.mkdir(parents=True, exist_ok=True)
    with args.fields_output.open("wb") as handle:
        np.savez_compressed(handle, **fields)

    baseline_aggregate = aggregate_downstream(baseline_rows)
    assessments = []
    for step in args.steps:
        step_rows = [row for row in heldout["rows"] if row["group"] == step]
        candidate = aggregate_downstream(step_rows)
        convergence_fits = [
            {"dimension": f"2d-seed{row['train_seed']}", "l2_slope": row["l2_slope"]}
            for row in convergence["rows"]
            if row["group"] == step
        ]
        extrema_summary = {
            "candidate_false_positive_rate": float(
                np.mean([row["false_positive_rate"] for row in extrema["rows"] if row["group"] == step])
            ),
            "baseline_false_positive_rate": float(
                np.mean([row["false_positive_rate"] for row in extrema["rows"] if row["group"] == BASELINE])
            ),
        }
        assessments.append(
            {
                "ladder": step,
                "threshold": thresholds[step],
                "threshold_selection": selections[step],
                "heldout": candidate,
                **assess_data_v3(
                    candidate, baseline_aggregate, {"fits": convergence_fits}, extrema_summary
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "phase 6 data-v3 controlled ladder evaluation",
        "training_summary": str(args.training_summary),
        "baseline": {
            "name": BASELINE,
            "representation": "invariant-node-v2",
            "threshold": BASELINE_THRESHOLD,
            "source": str(args.baseline_heldout),
            "heldout": baseline_aggregate,
        },
        "notes": {
            "heldout_grid": (
                "the held-out rotation grid is the full-period slotted disk; its "
                "corners are out of family for the old exact training data and in "
                "family for data-v3, so it is the direct test of geometry drift"
            ),
            "offline_thresholds": (
                "P/R/F1 are reported at each group's frozen operating threshold; "
                "PR-AUC and ECE are threshold-free"
            ),
            "baseline_rows": "reused from the frozen Phase 3 held-out artifact, not recomputed",
        },
        "datasets": training["datasets"],
        "offline_summary": summarize_offline(offline["rows"], list(args.steps) + [BASELINE]),
        "offline": offline["rows"],
        "convergence": convergence["rows"],
        "extrema": extrema["rows"],
        "assessments": assessments,
        "decision": {
            step["ladder"]: {"decision": step["decision"], "criteria": step["criteria"]}
            for step in assessments
        },
    }
    atomic_json(args.decision_output, payload)
    print(json.dumps(payload["decision"], indent=2))


if __name__ == "__main__":
    main()
