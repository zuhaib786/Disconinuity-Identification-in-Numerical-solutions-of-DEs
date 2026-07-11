import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from tci.thresholds import EulerFinalStageRecorder, validate_thresholds


class FakeIndicator:
    def flag(self, solver, u):
        return np.array([True, False, False, True])


class FakeSolver:
    K = 4


def test_threshold_validation():
    validate_thresholds([0.0, 0.1, 1.0])
    for invalid in ([], [-0.1], [1.1], [np.nan]):
        with pytest.raises(ValueError):
            validate_thresholds(invalid)


def test_euler_recorder_counts_only_completed_rk_steps():
    recorder = EulerFinalStageRecorder(FakeIndicator())
    for _ in range(5):
        recorder.flag(FakeSolver(), None)
    assert recorder.completed_steps == 1
    assert recorder.flagged_pct == 50.0
    recorder.flag(FakeSolver(), None)
    assert recorder.completed_steps == 2
