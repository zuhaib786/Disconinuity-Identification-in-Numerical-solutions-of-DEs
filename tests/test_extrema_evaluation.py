from tci.extrema_evaluation import assess_extrema


def summary(f1=0.6, pr_auc=0.7, l1=1.0, l2=1.0, flags=35.0):
    return {
        "metrics": {
            "f1": {"mean": f1},
            "pr_auc": {"mean": pr_auc},
            "l1_error": {"mean": l1},
            "l2_error": {"mean": l2},
            "flagged_pct": {"mean": flags},
        }
    }


def selection(undershoot=0.0, overshoot=0.0):
    return {
        "selected_threshold": 0.02,
        "candidates": [
            {
                "threshold": 0.02,
                "max_undershoot": undershoot,
                "max_overshoot": overshoot,
            }
        ],
    }


def test_extrema_accepts_improved_safe_selective_result():
    result = assess_extrema(
        summary(), selection(), summary(), summary(f1=0.5, pr_auc=0.6), summary()
    )
    assert result["passes"]


def test_extrema_rejects_calibration_violation():
    result = assess_extrema(
        summary(), selection(undershoot=0.011), summary(), summary(f1=0.5, pr_auc=0.6), summary()
    )
    assert not result["passes"]
    assert not result["criteria"]["calibration_undershoot_at_most_0.01"]
