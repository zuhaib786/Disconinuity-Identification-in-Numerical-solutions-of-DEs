"""Canonical binary-classification metrics used by every experiment.

The original thesis notebook interchanged false positives and false negatives in
one evaluation path. Keeping the definitions in one tested module prevents the
reported precision and recall from silently changing between experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


def _binary_vector(values: Iterable[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if not np.all(np.isin(array, (0, 1))):
        raise ValueError(f"{name} must contain only binary values 0 and 1")
    return array.astype(np.uint8, copy=False)


@dataclass(frozen=True)
class BinaryConfusionMatrix:
    """Counts with the conventional layout: ``[[tn, fp], [fn, tp]]``."""

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    @classmethod
    def from_predictions(
        cls,
        truth: Iterable[int] | np.ndarray,
        prediction: Iterable[int] | np.ndarray,
    ) -> "BinaryConfusionMatrix":
        y_true = _binary_vector(truth, "truth")
        y_pred = _binary_vector(prediction, "prediction")
        if y_true.shape != y_pred.shape:
            raise ValueError(
                f"truth and prediction must have the same size; got "
                f"{y_true.size} and {y_pred.size}"
            )

        return cls(
            true_negative=int(np.sum((y_true == 0) & (y_pred == 0))),
            false_positive=int(np.sum((y_true == 0) & (y_pred == 1))),
            false_negative=int(np.sum((y_true == 1) & (y_pred == 0))),
            true_positive=int(np.sum((y_true == 1) & (y_pred == 1))),
        )

    @property
    def total(self) -> int:
        return sum(asdict(self).values())

    @staticmethod
    def _ratio(numerator: int | float, denominator: int | float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    @property
    def accuracy(self) -> float:
        return self._ratio(self.true_positive + self.true_negative, self.total)

    @property
    def precision(self) -> float:
        return self._ratio(
            self.true_positive, self.true_positive + self.false_positive
        )

    @property
    def recall(self) -> float:
        return self._ratio(
            self.true_positive, self.true_positive + self.false_negative
        )

    @property
    def specificity(self) -> float:
        return self._ratio(
            self.true_negative, self.true_negative + self.false_positive
        )

    @property
    def f1(self) -> float:
        return self._ratio(
            2 * self.precision * self.recall, self.precision + self.recall
        )

    @property
    def balanced_accuracy(self) -> float:
        return 0.5 * (self.recall + self.specificity)

    def metrics(self) -> dict[str, float | int]:
        return {
            **asdict(self),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
        }


def evaluate_binary(
    truth: Iterable[int] | np.ndarray,
    scores: Iterable[float] | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Threshold positive-class scores and return canonical binary metrics."""

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    probabilities = np.asarray(scores, dtype=float).reshape(-1)
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("scores must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("scores must lie in [0, 1]")

    prediction = (probabilities >= threshold).astype(np.uint8)
    metrics = BinaryConfusionMatrix.from_predictions(truth, prediction).metrics()
    return {"threshold": float(threshold), **metrics}


def threshold_sweep(
    truth: Iterable[int] | np.ndarray,
    scores: Iterable[float] | np.ndarray,
    thresholds: Iterable[float] | np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    """Evaluate operating points without selecting a threshold on test data."""

    values = np.linspace(0.0, 1.0, 101) if thresholds is None else thresholds
    return [evaluate_binary(truth, scores, float(value)) for value in values]

