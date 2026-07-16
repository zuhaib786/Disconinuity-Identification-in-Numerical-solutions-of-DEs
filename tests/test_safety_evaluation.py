from tci.safety_evaluation import assess_safety_net


def row(l1=1.0, l2=1.0, under=0.0, over=0.0, flags=40.0):
    return {
        "status": "ok",
        "metrics": {
            "l1_error": l1,
            "l2_error": l2,
            "total_variation": 1.0,
            "undershoot": under,
            "overshoot": over,
            "mass_error": 0.0,
            "flagged_pct": flags,
            "runtime_s": 1.0,
        },
    }


def test_safety_net_accepts_complete_safe_selective_result():
    result = assess_safety_net([row()] * 20, [row()] * 30, [row()] * 30)
    assert result["passes"]
    assert all(result["criteria"].values())


def test_safety_net_rejects_one_large_calibration_undershoot():
    calibration = [row()] * 19 + [row(under=0.011)]
    result = assess_safety_net(calibration, [row()] * 30, [row()] * 30)
    assert not result["passes"]
    assert not result["criteria"]["calibration_undershoot_at_most_0.01"]
