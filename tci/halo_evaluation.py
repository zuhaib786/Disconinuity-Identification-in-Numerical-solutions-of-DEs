"""Acceptance checks for the P5-LABEL-SAFETY-001 label-halo experiment."""

from __future__ import annotations


def assess_label_halo(
    offline,
    threshold_selection,
    heldout,
    primary_offline,
    primary_heldout,
):
    """Evaluate the one-cell label-halo criteria frozen before training."""

    selected = threshold_selection["selected_threshold"]
    candidate = next(
        row for row in threshold_selection["candidates"] if row["threshold"] == selected
    )
    metrics = heldout["metrics"]
    primary_metrics = primary_heldout["metrics"]
    criteria = {
        "all_100_calibration_rows_complete": candidate["complete"]
        and candidate["row_count"] == 20
        and all(row["complete"] and row["row_count"] == 20 for row in threshold_selection["candidates"]),
        "calibration_undershoot_at_most_0.01": candidate["max_undershoot"] <= 0.01,
        "calibration_overshoot_at_most_0.01": candidate["max_overshoot"] <= 0.01,
        "all_30_heldout_rows_complete": heldout["successful_rows"] == 30,
        "heldout_mean_flagged_pct_at_most_50": metrics["flagged_pct"]["mean"] <= 50.0,
        "heldout_mean_l1_within_10pct_of_primary": metrics["l1_error"]["mean"]
        <= 1.1 * primary_metrics["l1_error"]["mean"],
        "heldout_mean_l2_within_10pct_of_primary": metrics["l2_error"]["mean"]
        <= 1.1 * primary_metrics["l2_error"]["mean"],
        "validation_recall_above_primary": offline["metrics"]["recall"]["mean"]
        > primary_offline["metrics"]["recall"]["mean"],
        "validation_pr_auc_at_least_90pct_of_primary": offline["metrics"]["pr_auc"]["mean"]
        >= 0.9 * primary_offline["metrics"]["pr_auc"]["mean"],
    }
    return {"criteria": criteria, "passes": all(criteria.values())}
