"""Aggregation and acceptance checks for Phase 5 safety-net experiments."""

from __future__ import annotations

import statistics


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


def aggregate(rows):
    successful = [row for row in rows if row.get("status") == "ok"]
    metrics = {}
    for name in METRICS:
        values = [row["metrics"][name] for row in successful]
        metrics[name] = {
            "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            "maximum": max(values) if values else None,
        }
    return {"row_count": len(rows), "successful_rows": len(successful), "metrics": metrics}


def assess_safety_net(calibration_rows, heldout_rows, primary_heldout_rows):
    """Evaluate the predeclared P5-SAFETY-OR-001 acceptance criteria."""

    calibration = aggregate(calibration_rows)
    heldout = aggregate(heldout_rows)
    primary = aggregate(primary_heldout_rows)
    criteria = {
        "all_20_calibration_rows_complete": calibration["successful_rows"] == 20,
        "calibration_undershoot_at_most_0.01": (
            calibration["metrics"]["undershoot"]["maximum"] is not None
            and calibration["metrics"]["undershoot"]["maximum"] <= 0.01
        ),
        "calibration_overshoot_at_most_0.01": (
            calibration["metrics"]["overshoot"]["maximum"] is not None
            and calibration["metrics"]["overshoot"]["maximum"] <= 0.01
        ),
        "all_30_heldout_rows_complete": heldout["successful_rows"] == 30,
        "heldout_mean_flagged_pct_below_60": (
            heldout["metrics"]["flagged_pct"]["mean"] is not None
            and heldout["metrics"]["flagged_pct"]["mean"] < 60.0
        ),
        "heldout_mean_l1_within_10pct_of_primary": (
            heldout["metrics"]["l1_error"]["mean"] is not None
            and primary["metrics"]["l1_error"]["mean"] is not None
            and heldout["metrics"]["l1_error"]["mean"]
            <= 1.1 * primary["metrics"]["l1_error"]["mean"]
        ),
        "heldout_mean_l2_within_10pct_of_primary": (
            heldout["metrics"]["l2_error"]["mean"] is not None
            and primary["metrics"]["l2_error"]["mean"] is not None
            and heldout["metrics"]["l2_error"]["mean"]
            <= 1.1 * primary["metrics"]["l2_error"]["mean"]
        ),
    }
    return {
        "criteria": criteria,
        "passes": all(criteria.values()),
        "calibration_union": calibration,
        "heldout_union": heldout,
        "heldout_primary": primary,
    }
