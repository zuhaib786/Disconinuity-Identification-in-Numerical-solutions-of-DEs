from tci.feature_review import DISPOSITIONS, SUGGESTIONS, validate_suggestions


def test_feature_review_has_complete_unique_dispositions():
    assert validate_suggestions()
    assert len(SUGGESTIONS) >= 25
    assert {row["disposition"] for row in SUGGESTIONS} <= DISPOSITIONS


def test_every_test_disposition_has_scheduled_experiment():
    tested = [row for row in SUGGESTIONS if row["disposition"] == "test"]
    assert tested
    assert all(row["scheduled_experiment_id"] for row in tested)
    assert any(
        row["scheduled_experiment_id"] == "P5-LABEL-SAFETY-002" for row in tested
    )
