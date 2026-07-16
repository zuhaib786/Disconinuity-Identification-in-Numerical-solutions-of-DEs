#!/usr/bin/env python
"""Freeze Phase 4 evidence without training models or running the PDE solver."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tci.phase4 import (
    REPRESENTATIONS,
    aggregate_heldout,
    build_figure_arrays,
    selected_candidate,
    upsert_by_key,
    validate_phase3,
)


PRIMARY = "invariant-node-v2"


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def artifact(path):
    return {"path": str(path), "sha256": sha256(path)}


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def selection_record(controlled, heldout_overall):
    offline = {row["representation"]: row for row in controlled["offline_summary"]}
    selections = {row["representation"]: row for row in controlled["threshold_selection"]}
    downstream = {row["representation"]: row for row in heldout_overall}
    primary_selection = selections[PRIMARY]
    primary_downstream = downstream[PRIMARY]["metrics"]
    edge_downstream = downstream["invariant-edge-v2"]["metrics"]
    return {
        "representation": PRIMARY,
        "selection_unit": "five-checkpoint family; all training seeds 0--4 are retained",
        "threshold": primary_selection["selected_threshold"],
        "calibration_safety_constraint": primary_selection["safety_constraint"],
        "calibration_at_selected_threshold": selected_candidate(primary_selection),
        "decision": "primary representation selected for the Phase 4 evidence freeze",
        "rationale": [
            "It is exactly invariant to local triangle vertex permutations.",
            "It provides the large offline and selectivity gain over ordered-global-v1.",
            "Its held-out maximum undershoot remains below 1e-2 at the frozen threshold.",
            "Directed edge attributes add only marginal predictive/selectivity gains while increasing mean runtime.",
            "Local robust scaling is excluded because its calibration safety failure persists on held-out meshes.",
        ],
        "quantitative_basis": {
            "offline_f1": offline[PRIMARY]["metrics"]["f1"],
            "offline_pr_auc": offline[PRIMARY]["metrics"]["pr_auc"],
            "heldout_flagged_pct": primary_downstream["flagged_pct"],
            "heldout_l2_error": primary_downstream["l2_error"],
            "heldout_max_undershoot": primary_downstream["undershoot"]["maximum"],
            "mean_runtime_s": primary_downstream["runtime_s"]["mean"],
            "edge_mean_runtime_s": edge_downstream["runtime_s"]["mean"],
        },
        "limitations": [
            "The primary representation narrowly fails the predeclared calibration safety constraint: maximum calibration undershoot is 0.01447.",
            "The held-out maximum undershoot is below 1e-2 but the representation is not described as universally safety-qualified.",
            "No Euler-aware learned-indicator claim follows from these scalar-advection experiments.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=Path("runs/paper/phase3-training-summary.json"))
    parser.add_argument("--calibration", type=Path, default=Path("runs/paper/phase3-calibration-rows.json"))
    parser.add_argument("--heldout", type=Path, default=Path("runs/paper/phase3-heldout-rows.json"))
    parser.add_argument("--controlled", type=Path, default=Path("runs/paper/phase3-controlled-table.json"))
    parser.add_argument("--current-v1", type=Path, default=Path("runs/paper/current-v1-summary.json"))
    parser.add_argument("--manifest", type=Path, default=Path("runs/paper/manifest.json"))
    parser.add_argument("--historical-output", type=Path, default=Path("runs/paper/historical-v1-summary.json"))
    parser.add_argument("--tables-output", type=Path, default=Path("runs/paper/phase4-controlled-tables.json"))
    parser.add_argument("--figures-output", type=Path, default=Path("runs/paper/phase4-figure-inputs.npz"))
    parser.add_argument("--summary-output", type=Path, default=Path("runs/paper/phase4-primary-summary.json"))
    args = parser.parse_args()

    inputs = (
        args.training,
        args.calibration,
        args.heldout,
        args.controlled,
        args.current_v1,
        args.manifest,
    )
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("missing Phase 4 input(s): " + ", ".join(missing))

    training = read_json(args.training)
    calibration = read_json(args.calibration)
    heldout = read_json(args.heldout)
    controlled = read_json(args.controlled)
    validate_phase3(training, calibration, heldout, controlled)
    heldout_overall = aggregate_heldout(heldout["rows"])
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    historical = read_json(args.current_v1)
    historical.update(
        {
            "evidence_label": "historical-v1",
            "role": "historical baseline; not the Phase 4 primary representation",
            "source_artifact": str(args.current_v1),
            "source_sha256": sha256(args.current_v1),
            "frozen_at_utc": now,
        }
    )
    atomic_json(args.historical_output, historical)

    source_artifacts = {
        "training": artifact(args.training),
        "calibration": artifact(args.calibration),
        "heldout": artifact(args.heldout),
        "controlled": artifact(args.controlled),
    }
    tables = {
        "schema_version": 1,
        "generated_at_utc": now,
        "primary_representation": PRIMARY,
        "data_id": controlled["data_id"],
        "split_id": controlled["split_id"],
        "aggregation": "arithmetic mean and sample standard deviation (ddof=1)",
        "source_artifacts": source_artifacts,
        "offline_comparison": controlled["offline_summary"],
        "threshold_selection": controlled["threshold_selection"],
        "heldout_by_mesh_resolution": controlled["heldout_summary"],
        "heldout_overall": heldout_overall,
        "permutation_stress": controlled["permutation_stress"],
    }
    atomic_json(args.tables_output, tables)
    atomic_npz(args.figures_output, build_figure_arrays(controlled, heldout_overall))

    checkpoints = []
    for row in training["rows"]:
        if row["representation"] == PRIMARY:
            checkpoints.append(
                {
                    "train_seed": row["train_seed"],
                    "best_epoch": row["best_epoch"],
                    "run_dir": row["run_dir"],
                    "artifacts": row["artifacts"],
                }
            )
    checkpoints.sort(key=lambda row: row["train_seed"])
    if [row["train_seed"] for row in checkpoints] != list(range(5)):
        raise ValueError("primary family does not contain training seeds 0--4")

    summary = {
        "schema_version": 1,
        "generated_at_utc": now,
        "primary": selection_record(controlled, heldout_overall),
        "data_id": controlled["data_id"],
        "split_id": controlled["split_id"],
        "primary_checkpoints": checkpoints,
        "evidence_sets": {
            "historical_v1": artifact(args.historical_output),
            "phase4_tables": artifact(args.tables_output),
            "phase4_figure_inputs": artifact(args.figures_output),
        },
        "gate": {
            **controlled["gate"],
            "primary_checkpoint_count": len(checkpoints),
            "scientific_experiments_rerun": 0,
            "paper_modified": False,
        },
    }
    atomic_json(args.summary_output, summary)

    manifest = read_json(args.manifest)
    manifest["schema_version"] = max(2, int(manifest.get("schema_version", 1)))
    manifest["generated_at_utc"] = now
    manifest["phase4_freeze"] = {
        "git_commit": git_commit(),
        "command": "python scripts/freeze_phase4_evidence.py",
        "primary_representation": PRIMARY,
        "primary_threshold": 0.02,
        "data_id": controlled["data_id"],
        "split_id": controlled["split_id"],
        "scientific_experiments_rerun": 0,
    }
    manifest["evidence_sets"] = {
        "historical_v1": artifact(args.historical_output),
        "phase4_primary": artifact(args.summary_output),
        "phase4_tables": artifact(args.tables_output),
        "phase4_figure_inputs": artifact(args.figures_output),
    }

    phase3_experiment = {
        "id": "gnn2d-phase3-controlled-feature-ablation",
        "artifact_path": [str(path) for path in (args.training, args.calibration, args.heldout, args.controlled)],
        "sha256": [sha256(path) for path in (args.training, args.calibration, args.heldout, args.controlled)],
        "experiment_name": "five-seed invariant feature ablation with calibrated downstream evaluation",
        "git_commit": git_commit(),
        "command": [
            "python scripts/train_feature_ablations.py",
            "python scripts/evaluate_feature_ablations.py",
        ],
        "configuration": "configs/gnn2d-feature-ablation.yaml",
        "data_seed": 0,
        "training_seed": list(range(5)),
        "threshold": [0.02, 0.05, 0.1, 0.2, 0.3],
        "mesh": ["structured", "delaunay"],
        "resolution": {"calibration": [10, 14], "heldout": [8, 12, 16]},
        "status": "complete (20 training, 400 calibration, 120 held-out rows)",
        "creation_time": controlled.get("generated_at_utc"),
    }
    primary_experiment = {
        "id": "gnn2d-phase4-primary-invariant-node-v2",
        "artifact_path": [str(args.summary_output), str(args.tables_output), str(args.figures_output)],
        "sha256": [sha256(args.summary_output), sha256(args.tables_output), sha256(args.figures_output)],
        "experiment_name": "Phase 4 frozen primary five-seed invariant-node-v2 evidence",
        "git_commit": git_commit(),
        "command": "python scripts/freeze_phase4_evidence.py",
        "configuration": "configs/gnn2d-feature-ablation.yaml",
        "data_seed": 0,
        "training_seed": list(range(5)),
        "threshold": 0.02,
        "mesh": ["structured", "delaunay"],
        "resolution": [8, 12, 16],
        "status": "frozen; calibration safety constraint narrowly failed and is recorded",
        "creation_time": now,
    }
    historical_experiment = {
        "id": "historical-v1-baseline",
        "artifact_path": str(args.historical_output),
        "sha256": sha256(args.historical_output),
        "experiment_name": "audited pre-Phase-2/3 historical evidence baseline",
        "git_commit": manifest.get("workspace", {}).get("git_commit"),
        "command": None,
        "configuration": None,
        "data_seed": None,
        "training_seed": None,
        "threshold": None,
        "mesh": None,
        "resolution": None,
        "status": "frozen and retained for historical comparison",
        "creation_time": historical.get("generated_at_utc"),
    }
    for experiment in (historical_experiment, phase3_experiment, primary_experiment):
        manifest["experiments"] = upsert_by_key(manifest["experiments"], experiment, "id")

    maps = (
        {
            "document_section": "future manuscript historical-v1 baseline",
            "experiment_ids": ["historical-v1-baseline"],
            "status": "frozen",
        },
        {
            "document_section": "future manuscript Phase 3 controlled representation comparison",
            "experiment_ids": ["gnn2d-phase3-controlled-feature-ablation"],
            "status": "verified and frozen",
        },
        {
            "document_section": "future manuscript Phase 4 primary representation",
            "experiment_ids": ["gnn2d-phase4-primary-invariant-node-v2"],
            "status": "verified and frozen; limitations retained",
        },
    )
    for row in maps:
        manifest["paper_input_map"] = upsert_by_key(
            manifest["paper_input_map"], row, "document_section"
        )
    atomic_json(args.manifest, manifest)

    print(
        json.dumps(
            {
                "primary": PRIMARY,
                "primary_checkpoints": len(checkpoints),
                "tables": str(args.tables_output),
                "figure_inputs": str(args.figures_output),
                "summary": str(args.summary_output),
                "manifest": str(args.manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
