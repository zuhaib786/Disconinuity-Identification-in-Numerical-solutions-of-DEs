"""Label calibration, aggregation, and acceptance rules for Phase 6 (`data-v3`).

Two decisions in this phase must be made once and then frozen:

* the label constants ``alpha`` and ``gamma`` of plan 6.1, chosen on a held-out
  calibration batch before any `data-v3` model is trained; and
* the adopt/reject decision of plan 6.4, evaluated against the predeclared
  criteria with the frozen Phase 4 primary as the named baseline.
"""

from __future__ import annotations

import math
import statistics

import numpy as np

LADDER = ("v3-a", "v3-b", "v3-c")
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

# Predeclared label-calibration rule (plan 6.1).  A cell is a *core* cell of a
# reference discontinuity when the curve genuinely bisects it; a corner clip is
# deliberately not one, because the new rule is meant to be amplitude- and
# resolution-aware rather than scale-free like the legacy cut labels.
#
# Labelling a smooth cell troubled is the corrupting error -- it is exactly the
# label noise the phase exists to remove -- so it is the hard constraint, and
# recall of the bisected cells is the objective under it.
CORE_AREA_FRACTION = (0.25, 0.75)
SMOOTH_FALSE_POSITIVE_CEILING = 1e-3
RECALL_TOLERANCE = 0.01
ALPHA_CANDIDATES = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
GAMMA_CANDIDATES = (2.0, 3.0, 4.0)


def mean_and_sample_std(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot aggregate an empty sequence")
    return {
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if values.size > 1 else None,
    }


def select_label_constants(
    candidates,
    smooth_ceiling=SMOOTH_FALSE_POSITIVE_CEILING,
    recall_tolerance=RECALL_TOLERANCE,
):
    """Choose ``(alpha, gamma)`` from a measured calibration grid.

    Among the candidates whose smooth false-positive rate stays within
    ``smooth_ceiling``, take the one that recovers the most cells a reference
    discontinuity bisects; among candidates within ``recall_tolerance`` of that
    best recall, prefer the more amplitude-selective one (largest ``alpha``,
    then smallest ``gamma``).  If the ceiling excludes every candidate, fall
    back to the safest one and mark the calibration as failed rather than
    relaxing the constraint silently.
    """
    feasible = [
        row for row in candidates if row["smooth_false_positive_rate"] <= smooth_ceiling
    ]
    scored = [{**row, "feasible": row in feasible} for row in candidates]
    if feasible:
        best_recall = max(row["core_recall"] for row in feasible)
        near_best = [
            row for row in feasible if row["core_recall"] >= best_recall - recall_tolerance
        ]
        best = max(near_best, key=lambda row: (row["alpha"], -row["gamma"]))
        status = "satisfied"
    else:
        best = min(candidates, key=lambda row: row["smooth_false_positive_rate"])
        status = "failed"
    return {
        "alpha": best["alpha"],
        "gamma": best["gamma"],
        "constraint": status,
        "rule": {
            "core_area_fraction": list(CORE_AREA_FRACTION),
            "smooth_false_positive_ceiling": smooth_ceiling,
            "recall_tolerance": recall_tolerance,
            "objective": (
                "maximize recall on bisected core cells subject to the smooth "
                "false-positive ceiling; break ties toward the largest alpha"
            ),
        },
        "selected_candidate": best,
        "candidates": scored,
    }


def aggregate_downstream(rows, meshes=MESHES, resolutions=RESOLUTIONS):
    """Per-(mesh, resolution) and overall mean +/- sample SD over seeds."""
    successful = [row for row in rows if row.get("status") == "ok"]
    groups = []
    for mesh in meshes:
        for resolution in resolutions:
            selected = [
                row
                for row in successful
                if row["mesh"] == mesh and row["resolution"] == resolution
            ]
            if not selected:
                continue
            groups.append(
                {
                    "mesh": mesh,
                    "resolution": resolution,
                    "row_count": len(selected),
                    "metrics": {
                        name: mean_and_sample_std([row["metrics"][name] for row in selected])
                        for name in DOWNSTREAM_METRICS
                    },
                }
            )
    overall = {
        "row_count": len(rows),
        "successful_rows": len(successful),
        "metrics": {
            name: {
                **mean_and_sample_std([row["metrics"][name] for row in successful]),
                "maximum": max(row["metrics"][name] for row in successful),
            }
            for name in DOWNSTREAM_METRICS
        }
        if successful
        else {},
    }
    return {"groups": groups, "overall": overall}


def summarize_offline(rows, groups, evaluation_sets=("old_exact", "data_v3")):
    """Mean +/- sample SD of the offline metrics per group and evaluation set."""
    summary = []
    for group in groups:
        for evaluation_set in evaluation_sets:
            selected = [
                row
                for row in rows
                if row["group"] == group and row["evaluation_set"] == evaluation_set
            ]
            if not selected:
                continue
            summary.append(
                {
                    "group": group,
                    "evaluation_set": evaluation_set,
                    "row_count": len(selected),
                    "label_positive_rate": selected[0]["positive_rate"],
                    "metrics": {
                        metric: mean_and_sample_std(
                            [row["metrics"][metric] for row in selected]
                        )
                        for metric in OFFLINE_METRICS
                    },
                }
            )
    return summary


def resolution_mean(groups, metric, resolution):
    values = [
        group["metrics"][metric]["mean"]
        for group in groups
        if group["resolution"] == resolution
    ]
    if not values:
        raise ValueError(f"no aggregated rows at resolution {resolution}")
    return statistics.mean(values)


def over_flagging_gap(groups):
    """Excess mean flagging on the coarsest mesh relative to the finest.

    The verified findings record small-mesh over-flagging and reduced flagging
    at fine resolution; this single number is the gap criterion 2 must shrink.
    """
    return resolution_mean(groups, "flagged_pct", min(RESOLUTIONS)) - resolution_mean(
        groups, "flagged_pct", max(RESOLUTIONS)
    )


def _relative_reduction(baseline, candidate):
    if baseline <= 0.0:
        return 0.0
    return (baseline - candidate) / baseline


def assess_data_v3(candidate, baseline, convergence, extrema):
    """Evaluate the predeclared plan 6.4 acceptance criteria for one ladder step.

    ``candidate`` and ``baseline`` are ``aggregate_downstream`` payloads;
    ``convergence`` carries the fitted 1D/2D L2 slopes of the Phase 1.2 check;
    ``extrema`` carries the smooth-extremum false-positive rates.
    """
    candidate_metrics = candidate["overall"]["metrics"]
    baseline_metrics = baseline["overall"]["metrics"]

    undershoot_reduction = _relative_reduction(
        baseline_metrics["undershoot"]["maximum"], candidate_metrics["undershoot"]["maximum"]
    )
    flag_reduction = _relative_reduction(
        baseline_metrics["flagged_pct"]["mean"], candidate_metrics["flagged_pct"]["mean"]
    )
    l2_growth = (
        candidate_metrics["l2_error"]["mean"] / baseline_metrics["l2_error"]["mean"] - 1.0
        if baseline_metrics["l2_error"]["mean"] > 0.0
        else math.inf
    )
    candidate_gap = over_flagging_gap(candidate["groups"])
    baseline_gap = over_flagging_gap(baseline["groups"])
    slopes = [
        fit["l2_slope"]
        for fit in convergence["fits"]
        if fit["l2_slope"] is not None
    ]

    criteria = {
        "1_selectivity_or_safety_gain_at_bounded_l2_cost": bool(
            max(undershoot_reduction, flag_reduction) >= 0.25 and l2_growth <= 0.10
        ),
        "2_small_mesh_over_flagging_gap_reduced": bool(candidate_gap < baseline_gap),
        "3_smooth_convergence_preserved": bool(
            slopes and min(slopes) >= 1.8
        ),
        "4_smooth_extremum_false_positives_not_increased": bool(
            extrema["candidate_false_positive_rate"] <= extrema["baseline_false_positive_rate"]
        ),
    }
    return {
        "criteria": criteria,
        "decision": "adopt" if all(criteria.values()) else "reject",
        "measurements": {
            "heldout_worst_undershoot": {
                "baseline": baseline_metrics["undershoot"]["maximum"],
                "candidate": candidate_metrics["undershoot"]["maximum"],
                "relative_reduction": undershoot_reduction,
            },
            "heldout_mean_flagged_pct": {
                "baseline": baseline_metrics["flagged_pct"]["mean"],
                "candidate": candidate_metrics["flagged_pct"]["mean"],
                "relative_reduction": flag_reduction,
            },
            "heldout_mean_l2_error": {
                "baseline": baseline_metrics["l2_error"]["mean"],
                "candidate": candidate_metrics["l2_error"]["mean"],
                "relative_increase": l2_growth,
            },
            "small_mesh_over_flagging_gap": {
                "definition": "mean flagged_pct at n=8 minus mean flagged_pct at n=16",
                "baseline": baseline_gap,
                "candidate": candidate_gap,
            },
            "fitted_l2_slopes": {fit["dimension"]: fit["l2_slope"] for fit in convergence["fits"]},
            "smooth_extremum_false_positive_rate": extrema,
        },
    }
