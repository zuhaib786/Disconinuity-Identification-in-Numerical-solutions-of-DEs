from tci.halo_evaluation import assess_label_halo


def offline(recall=0.85, pr_auc=0.65):
    return {"metrics": {"recall": {"mean": recall}, "pr_auc": {"mean": pr_auc}}}


def heldout(l1=1.0, l2=1.0, flags=45.0, successful=30):
    return {
        "successful_rows": successful,
        "metrics": {
            "l1_error": {"mean": l1},
            "l2_error": {"mean": l2},
            "flagged_pct": {"mean": flags},
        },
    }


def selection(undershoot=0.0, overshoot=0.0, row_count=20, complete=True):
    candidates = []
    for threshold in (0.02, 0.05, 0.1, 0.2, 0.3):
        candidates.append(
            {
                "threshold": threshold,
                "row_count": row_count,
                "complete": complete,
                "max_undershoot": undershoot,
                "max_overshoot": overshoot,
            }
        )
    return {"selected_threshold": 0.02, "candidates": candidates}


def test_label_halo_accepts_safe_recall_gain_with_bounded_selectivity():
    result = assess_label_halo(
        offline(), selection(), heldout(), offline(recall=0.75, pr_auc=0.7), heldout()
    )
    assert result["passes"]


def test_label_halo_rejects_low_pr_auc_and_calibration_violation():
    result = assess_label_halo(
        offline(pr_auc=0.62),
        selection(undershoot=0.011),
        heldout(),
        offline(recall=0.75, pr_auc=0.7),
        heldout(),
    )
    assert not result["passes"]
    assert not result["criteria"]["calibration_undershoot_at_most_0.01"]
    assert not result["criteria"]["validation_pr_auc_at_least_90pct_of_primary"]


def test_label_halo_requires_complete_calibration_and_heldout_tables():
    result = assess_label_halo(
        offline(),
        selection(row_count=19),
        heldout(successful=29),
        offline(recall=0.75, pr_auc=0.7),
        heldout(),
    )
    assert not result["criteria"]["all_100_calibration_rows_complete"]
    assert not result["criteria"]["all_30_heldout_rows_complete"]
