"""Utilities for solver-in-the-loop probability-threshold experiments."""

import numpy as np


def validate_thresholds(thresholds):
    if not thresholds:
        raise ValueError("at least one threshold is required")
    for threshold in thresholds:
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"invalid probability threshold: {threshold}")


class EulerFinalStageRecorder:
    """Retain final-stage flags so divergent Euler runs have partial cost data."""

    def __init__(self, indicator):
        self.indicator = indicator
        self.calls = 0
        self.final_stage_flags = []

    def flag(self, solver, u):
        flags = np.asarray(self.indicator.flag(solver, u), dtype=bool)
        if flags.shape != (solver.K,):
            raise ValueError(
                f"indicator returned shape {flags.shape}; expected {(solver.K,)}"
            )
        self.calls += 1
        if self.calls % 3 == 0:
            self.final_stage_flags.append(flags.copy())
        return flags

    @property
    def completed_steps(self):
        return len(self.final_stage_flags)

    @property
    def flagged_pct(self):
        if not self.final_stage_flags:
            return None
        return 100.0 * float(np.mean([f.mean() for f in self.final_stage_flags]))
