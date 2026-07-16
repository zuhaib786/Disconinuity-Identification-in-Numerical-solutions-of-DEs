"""Helpers for freezing Phase 4 evidence from completed Phase 3 artifacts."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


REPRESENTATIONS = (
    "ordered-global-v1",
    "invariant-node-v2",
    "invariant-edge-v2",
    "invariant-local-v2",
)
MESHES = ("structured", "delaunay")
RESOLUTIONS = (8, 12, 16)
OFFLINE_METRICS = ("accuracy", "precision", "recall", "f1", "pr_auc", "ece")
DOWNSTREAM_METRICS = (
    "l1_error",
    "l2_error",
    "total_variation",
    "undershoot",
    "overshoot",
    "mass_error",
    "flagged_pct",
    "runtime_s",
)


def mean_and_sample_std(values):
    """Return JSON-safe mean and sample standard deviation."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot aggregate an empty sequence")
    return {
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if values.size > 1 else None,
    }


def aggregate_heldout(rows):
    """Aggregate successful held-out rows by representation."""

    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") != "ok":
            continue
        grouped[row["representation"]].append(row)

    result = []
    for representation in REPRESENTATIONS:
        group = grouped[representation]
        if not group:
            raise ValueError(f"no successful held-out rows for {representation}")
        metrics = {
            name: mean_and_sample_std([row["metrics"][name] for row in group])
            for name in DOWNSTREAM_METRICS
        }
        metrics["undershoot"]["maximum"] = float(
            max(row["metrics"]["undershoot"] for row in group)
        )
        metrics["overshoot"]["maximum"] = float(
            max(row["metrics"]["overshoot"] for row in group)
        )
        result.append(
            {
                "representation": representation,
                "row_count": len(group),
                "metrics": metrics,
            }
        )
    return result


def validate_phase3(training, calibration, heldout, controlled):
    """Validate the completed controlled experiment before freezing it."""

    expected = {
        "training_rows": 20,
        "calibration_rows": 400,
        "heldout_rows": 120,
    }
    if controlled.get("gate", {}) != {
        **expected,
        "all_invariant_stress_tests_pass": True,
    }:
        raise ValueError("Phase 3 controlled-table gate is incomplete")

    row_sets = (
        ("training", training.get("rows"), expected["training_rows"]),
        ("calibration", calibration.get("rows"), expected["calibration_rows"]),
        ("heldout", heldout.get("rows"), expected["heldout_rows"]),
    )
    for name, rows, count in row_sets:
        if not isinstance(rows, list) or len(rows) != count:
            raise ValueError(f"{name} must contain exactly {count} rows")
        if name != "training" and any(row.get("status") != "ok" for row in rows):
            raise ValueError(f"{name} contains an unsuccessful row")

    pairs = {
        (row.get("representation"), row.get("train_seed"))
        for row in training["rows"]
    }
    expected_pairs = {(representation, seed) for representation in REPRESENTATIONS for seed in range(5)}
    if pairs != expected_pairs:
        raise ValueError("training rows do not cover four representations and seeds 0--4")

    data_ids = {training.get("data_id"), controlled.get("data_id")}
    split_ids = {training.get("split_id"), controlled.get("split_id")}
    data_ids.update(row.get("data_id") for row in training["rows"])
    split_ids.update(row.get("split_id") for row in training["rows"])
    if None in data_ids or len(data_ids) != 1:
        raise ValueError("Phase 3 training rows do not share one data ID")
    if None in split_ids or len(split_ids) != 1:
        raise ValueError("Phase 3 training rows do not share one split ID")

    invariant_stress = [
        row
        for row in controlled.get("permutation_stress", [])
        if row.get("expected_invariant")
    ]
    if len(invariant_stress) != 15 or not all(row.get("passes") for row in invariant_stress):
        raise ValueError("invariant permutation stress test is incomplete")


def selected_candidate(selection):
    """Return the calibration row corresponding to the frozen threshold."""

    selected = selection["selected_threshold"]
    matches = [row for row in selection["candidates"] if row["threshold"] == selected]
    if len(matches) != 1:
        raise ValueError("selected threshold does not identify one calibration candidate")
    return matches[0]


def build_figure_arrays(controlled, heldout_overall):
    """Build dense arrays for Phase 4 tables and plots without rerunning solves."""

    offline = {row["representation"]: row for row in controlled["offline_summary"]}
    selections = {row["representation"]: row for row in controlled["threshold_selection"]}
    grouped = {
        (row["representation"], row["mesh"], int(row["resolution"])): row
        for row in controlled["heldout_summary"]
    }
    overall = {row["representation"]: row for row in heldout_overall}

    arrays = {
        "representation": np.asarray(REPRESENTATIONS),
        "offline_metric": np.asarray(OFFLINE_METRICS),
        "offline_mean": np.asarray(
            [
                [offline[representation]["metrics"][metric]["mean"] for metric in OFFLINE_METRICS]
                for representation in REPRESENTATIONS
            ],
            dtype=float,
        ),
        "offline_sample_std": np.asarray(
            [
                [
                    offline[representation]["metrics"][metric]["sample_std"]
                    for metric in OFFLINE_METRICS
                ]
                for representation in REPRESENTATIONS
            ],
            dtype=float,
        ),
        "calibration_threshold": np.asarray(
            [row["threshold"] for row in selections[REPRESENTATIONS[0]]["candidates"]],
            dtype=float,
        ),
        "calibration_max_undershoot": np.asarray(
            [
                [row["max_undershoot"] for row in selections[representation]["candidates"]]
                for representation in REPRESENTATIONS
            ],
            dtype=float,
        ),
        "calibration_max_overshoot": np.asarray(
            [
                [row["max_overshoot"] for row in selections[representation]["candidates"]]
                for representation in REPRESENTATIONS
            ],
            dtype=float,
        ),
        "calibration_qualifies": np.asarray(
            [
                [row["qualifies"] for row in selections[representation]["candidates"]]
                for representation in REPRESENTATIONS
            ],
            dtype=bool,
        ),
        "mesh": np.asarray(MESHES),
        "resolution": np.asarray(RESOLUTIONS, dtype=int),
        "downstream_metric": np.asarray(DOWNSTREAM_METRICS),
        "heldout_mean": np.asarray(
            [
                [
                    [
                        [
                            grouped[(representation, mesh, resolution)]["metrics"][metric]["mean"]
                            for metric in DOWNSTREAM_METRICS
                        ]
                        for resolution in RESOLUTIONS
                    ]
                    for mesh in MESHES
                ]
                for representation in REPRESENTATIONS
            ],
            dtype=float,
        ),
        "heldout_sample_std": np.asarray(
            [
                [
                    [
                        [
                            grouped[(representation, mesh, resolution)]["metrics"][metric][
                                "sample_std"
                            ]
                            for metric in DOWNSTREAM_METRICS
                        ]
                        for resolution in RESOLUTIONS
                    ]
                    for mesh in MESHES
                ]
                for representation in REPRESENTATIONS
            ],
            dtype=float,
        ),
        "heldout_overall_mean": np.asarray(
            [
                [overall[representation]["metrics"][metric]["mean"] for metric in DOWNSTREAM_METRICS]
                for representation in REPRESENTATIONS
            ],
            dtype=float,
        ),
        "heldout_overall_sample_std": np.asarray(
            [
                [
                    overall[representation]["metrics"][metric]["sample_std"]
                    for metric in DOWNSTREAM_METRICS
                ]
                for representation in REPRESENTATIONS
            ],
            dtype=float,
        ),
    }
    return arrays


def upsert_by_key(rows, new_row, key):
    """Replace one manifest row by key while preserving the surrounding order."""

    rows = list(rows)
    for index, row in enumerate(rows):
        if row.get(key) == new_row[key]:
            rows[index] = new_row
            return rows
    rows.append(new_row)
    return rows
