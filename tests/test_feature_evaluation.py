from tci.data.graphs import FEATURE_SCHEMAS
from tci.feature_evaluation import select_thresholds


def calibration_rows(violations_by_threshold):
    rows = []
    for representation in FEATURE_SCHEMAS:
        for threshold, violation in violations_by_threshold.items():
            for seed in range(5):
                for mesh in ("structured", "delaunay"):
                    for resolution in (10, 14):
                        rows.append(
                            {
                                "representation": representation,
                                "threshold": threshold,
                                "status": "ok",
                                "train_seed": seed,
                                "mesh": mesh,
                                "resolution": resolution,
                                "metrics": {
                                    "undershoot": 0.01 + violation,
                                    "overshoot": 0.0,
                                },
                            }
                        )
    return rows


def test_selects_largest_threshold_that_satisfies_safety_constraint():
    candidates = [0.02, 0.05, 0.1, 0.2, 0.3]
    rows = calibration_rows({0.02: 0.0, 0.05: 0.0, 0.1: 0.0, 0.2: 0.01, 0.3: 0.02})
    selected = select_thresholds(rows, candidates, safety_tolerance=0.01)
    assert all(row["selected_threshold"] == 0.1 for row in selected)
    assert all(row["safety_constraint"] == "satisfied" for row in selected)


def test_failed_safety_constraint_uses_smallest_maximum_violation():
    candidates = [0.02, 0.05, 0.1, 0.2, 0.3]
    rows = calibration_rows({0.02: 0.003, 0.05: 0.002, 0.1: 0.005, 0.2: 0.01, 0.3: 0.02})
    selected = select_thresholds(rows, candidates, safety_tolerance=0.01)
    assert all(row["selected_threshold"] == 0.05 for row in selected)
    assert all(row["safety_constraint"] == "failed" for row in selected)


def test_single_schema_rows_produce_only_one_selection():
    rows = [
        row
        for row in calibration_rows({0.02: 0.0})
        if row["representation"] == "invariant-extrema-v3"
    ]
    selected = select_thresholds(rows, [0.02], safety_tolerance=0.01)
    assert len(selected) == 1
    assert selected[0]["representation"] == "invariant-extrema-v3"
