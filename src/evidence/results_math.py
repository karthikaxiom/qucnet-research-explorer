"""Pure, dependency-free aggregate metric calculations for metric sensitivity analysis.

This module does not access research data or train classifiers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BinaryConfusion:
    tn: int
    fp: int
    fn: int
    tp: int

    def __post_init__(self):
        values = (self.tn, self.fp, self.fn, self.tp)
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError("Confusion counts must be nonnegative integers")
        if sum(values) == 0:
            raise ValueError("A confusion matrix needs at least one sample")

    @property
    def n(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    @staticmethod
    def _fraction(num: int, denom: int) -> float | None:
        return num / denom if denom else None

    def metrics(self) -> dict[str, float | None]:
        precision = self._fraction(self.tp, self.tp + self.fp)
        recall = self._fraction(self.tp, self.tp + self.fn)
        specificity = self._fraction(self.tn, self.tn + self.fp)
        f1 = (2 * precision * recall / (precision + recall)
              if precision is not None and recall is not None and precision + recall else None)
        return {
            "accuracy": (self.tp + self.tn) / self.n,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        }


def multiclass_summary(matrix: list[list[int]]) -> dict:
    """Aggregate confusion-matrix accuracy and class recall, with no patient records."""
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        raise ValueError("Expected a nonempty square matrix")
    if any(type(v) is not int or v < 0 for row in matrix for v in row):
        raise ValueError("Expected nonnegative integer counts")
    total = sum(sum(row) for row in matrix)
    if total == 0:
        raise ValueError("Confusion matrix must contain cases")
    recalls = [matrix[i][i] / sum(matrix[i]) if sum(matrix[i]) else None for i in range(size)]
    return {"n": total, "correct": sum(matrix[i][i] for i in range(size)),
            "accuracy": sum(matrix[i][i] for i in range(size)) / total,
            "class_recall": recalls}
