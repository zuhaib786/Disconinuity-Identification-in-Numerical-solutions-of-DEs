"""Threshold selection for controlled ablations (Phase 3 rule).

Phase 6 selects the `data-v3` thresholds with the same rule, so the rule itself
lives in `select_threshold` and both callers share it.
"""

import math

from tci.data.graphs import FEATURE_SCHEMAS

CALIBRATION_ROWS = 20


def select_threshold(rows, candidates, safety_tolerance, expected_rows=CALIBRATION_ROWS):
    """Choose the largest fully complete, bounds-safe candidate threshold.

    A candidate qualifies when all ``expected_rows`` calibration runs complete
    and neither overshoot nor undershoot exceeds ``safety_tolerance``.  If none
    qualifies, take the candidate minimizing the maximum violation and label the
    safety constraint as failed rather than relaxing it.
    """
    threshold_rows = []
    for threshold in candidates:
        selected = [row for row in rows if row["threshold"] == threshold]
        complete = len(selected) == expected_rows and all(
            row["status"] == "ok" for row in selected
        )
        if complete:
            max_undershoot = max(row["metrics"]["undershoot"] for row in selected)
            max_overshoot = max(row["metrics"]["overshoot"] for row in selected)
            violation = max(
                0.0,
                max_undershoot - safety_tolerance,
                max_overshoot - safety_tolerance,
            )
        else:
            max_undershoot = None
            max_overshoot = None
            violation = math.inf
        threshold_rows.append(
            {
                "threshold": threshold,
                "row_count": len(selected),
                "complete": complete,
                "max_undershoot": max_undershoot,
                "max_overshoot": max_overshoot,
                "max_safety_violation": None if math.isinf(violation) else violation,
                "qualifies": complete and violation <= 0.0,
            }
        )

    qualified = [row for row in threshold_rows if row["qualifies"]]
    if qualified:
        selected_threshold = max(row["threshold"] for row in qualified)
        constraint = "satisfied"
    else:
        finite = [row for row in threshold_rows if row["max_safety_violation"] is not None]
        if finite:
            best_violation = min(row["max_safety_violation"] for row in finite)
            selected_threshold = max(
                row["threshold"]
                for row in finite
                if row["max_safety_violation"] == best_violation
            )
        else:
            selected_threshold = max(candidates)
        constraint = "failed"
    return {
        "selected_threshold": selected_threshold,
        "safety_constraint": constraint,
        "candidates": threshold_rows,
    }


def select_thresholds(calibration_rows, candidates, safety_tolerance):
    """Apply `select_threshold` to each feature representation present."""
    present = {row["representation"] for row in calibration_rows}
    unknown = present - set(FEATURE_SCHEMAS)
    if unknown:
        raise ValueError(f"calibration rows contain unknown schemas: {sorted(unknown)}")
    selections = []
    for representation in [name for name in FEATURE_SCHEMAS if name in present]:
        rows = [row for row in calibration_rows if row["representation"] == representation]
        selections.append(
            {
                "representation": representation,
                **select_threshold(rows, candidates, safety_tolerance),
            }
        )
    return selections
