import numpy as np

from tci.phase4 import aggregate_heldout, mean_and_sample_std, upsert_by_key


def test_mean_and_sample_std_uses_ddof_one():
    result = mean_and_sample_std([1.0, 2.0, 3.0])
    assert result == {"mean": 2.0, "sample_std": 1.0}


def test_aggregate_heldout_retains_maximum_bounds_violation():
    rows = []
    metric_names = (
        "l1_error",
        "l2_error",
        "total_variation",
        "undershoot",
        "overshoot",
        "mass_error",
        "flagged_pct",
        "runtime_s",
    )
    representations = (
        "ordered-global-v1",
        "invariant-node-v2",
        "invariant-edge-v2",
        "invariant-local-v2",
    )
    for representation in representations:
        for seed, value in enumerate((0.0, 2.0)):
            rows.append(
                {
                    "representation": representation,
                    "train_seed": seed,
                    "status": "ok",
                    "metrics": {name: value for name in metric_names},
                }
            )
    result = aggregate_heldout(rows)
    assert [row["representation"] for row in result] == list(representations)
    assert result[0]["metrics"]["l1_error"] == {"mean": 1.0, "sample_std": np.sqrt(2.0)}
    assert result[0]["metrics"]["undershoot"]["maximum"] == 2.0
    assert result[0]["metrics"]["overshoot"]["maximum"] == 2.0


def test_upsert_by_key_replaces_without_reordering():
    rows = [{"id": "a", "value": 1}, {"id": "b", "value": 2}]
    assert upsert_by_key(rows, {"id": "a", "value": 3}, "id") == [
        {"id": "a", "value": 3},
        {"id": "b", "value": 2},
    ]
    assert upsert_by_key(rows, {"id": "c", "value": 3}, "id")[-1]["id"] == "c"
