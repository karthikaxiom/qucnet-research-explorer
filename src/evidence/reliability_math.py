"""Audit-only arithmetic for the QuCNet reliability workbench.

Pure calculations on archived *aggregate* matrices. No predictions, patients,
medical data, models, training or file I/O.
"""
from __future__ import annotations

from math import isclose

LABELS = ("2", "3", "4a", "4b", "4c", "5")


def classwise_from_confusion(matrix: list[list[int]]) -> dict:
    """Return support, diagonal correct, and sensitivity for six classes.

    All labels use the source's fixed TI-RADS order. Missing support is
    explicitly None; it must not be shown as zero recall.
    """
    if not isinstance(matrix, list) or len(matrix) != 6:
        raise ValueError("Expected six rows in confusion matrix")
    if any(not isinstance(row, list) or len(row) != 6 for row in matrix):
        raise ValueError("Expected a six-by-six confusion matrix")
    if any(type(value) is not int or value < 0 for row in matrix for value in row):
        raise ValueError("Matrix counts must be nonnegative integers")
    support = [sum(row) for row in matrix]
    n = sum(support)
    if n == 0:
        raise ValueError("Empty confusion matrix")
    correct_by_class = [matrix[i][i] for i in range(6)]
    recalls = [correct_by_class[i] / support[i] if support[i] else None for i in range(6)]
    return {
        "n": n,
        "correct": sum(correct_by_class),
        "accuracy": sum(correct_by_class) / n,
        "support": support,
        "correct_by_class": correct_by_class,
        "recall": recalls,
    }


def development_scores(data: dict, study: str) -> list[dict]:
    """Select comparable *case-level development OOF* score rows.

    Never include validation-selected binary, historical image or ROI-test
    results alongside these experimental arms.
    """
    if study not in ("fusion", "preprocessing"):
        raise ValueError("Unsupported comparison")
    source = data[study]
    if source["case_count"] != 203:
        raise ValueError("Unexpected development-case denominator")
    result = []
    for name, vals in source["models"].items():
        accuracy = vals["accuracy"]
        balanced = vals["balanced_accuracy"]
        macro_f1 = vals["macro_f1"]
        for value in (accuracy, balanced, macro_f1):
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError("Invalid aggregate score")
        result.append({"name": name, "accuracy": accuracy,
                       "balanced_accuracy": balanced, "macro_f1": macro_f1})
    return result


def validate_bn_conditions(data: dict) -> dict:
    """Enforce the archived, validation-only recalibration denominator."""
    if data["partition"] != "historical_validation":
        raise ValueError("Not the archived validation-only BN diagnostic")
    if data["learned_weights_updated"] is not False:
        raise ValueError("This audit must not be described as retraining")
    conditions = data["conditions"]
    if set(conditions) != {"original", "recalibrated"}:
        raise ValueError("Unexpected BatchNorm comparisons")
    original = conditions["original"]
    adjusted = conditions["recalibrated"]
    if original["samples"] != 58 or adjusted["samples"] != 58:
        raise ValueError("BatchNorm denominator mismatch")
    if not isclose(original["accuracy"], 2/58, abs_tol=1e-10):
        raise ValueError("Original BN score differs from the archive")
    if not isclose(adjusted["accuracy"], 11/58, abs_tol=1e-10):
        raise ValueError("Recalibrated BN score differs from the archive")
    if not isclose(original["majority_reference"], 17/58, abs_tol=1e-10):
        raise ValueError("Validation majority reference mismatch")
    if not isclose(adjusted["majority_reference"], 17/58, abs_tol=1e-10):
        raise ValueError("Validation majority reference mismatch")
    return {"original": original["accuracy"],
            "recalibrated": adjusted["accuracy"],
            "majority": original["majority_reference"], "n": 58}
