"""Acceptance checks for the P5-LOCAL-EXTREMA-001 feature experiment."""

from __future__ import annotations


def assess_extrema(offline, threshold_selection, heldout, primary_offline, primary_heldout):
    selected = threshold_selection["selected_threshold"]
    candidate = next(
        row for row in threshold_selection["candidates"] if row["threshold"] == selected
    )
    metrics = heldout["metrics"]
    primary_metrics = primary_heldout["metrics"]
    criteria = {
        "offline_f1_above_primary": offline["metrics"]["f1"]["mean"]
        > primary_offline["metrics"]["f1"]["mean"],
        "offline_pr_auc_above_primary": offline["metrics"]["pr_auc"]["mean"]
        > primary_offline["metrics"]["pr_auc"]["mean"],
        "calibration_undershoot_at_most_0.01": candidate["max_undershoot"] <= 0.01,
        "calibration_overshoot_at_most_0.01": candidate["max_overshoot"] <= 0.01,
        "heldout_mean_flagged_pct_at_most_40": metrics["flagged_pct"]["mean"] <= 40.0,
        "heldout_mean_l1_within_10pct_of_primary": metrics["l1_error"]["mean"]
        <= 1.1 * primary_metrics["l1_error"]["mean"],
        "heldout_mean_l2_within_10pct_of_primary": metrics["l2_error"]["mean"]
        <= 1.1 * primary_metrics["l2_error"]["mean"],
    }
    return {"criteria": criteria, "passes": all(criteria.values())}
