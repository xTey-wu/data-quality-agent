"""Run-level selection rules shared by the API planner and the MCP service."""

import math
from numbers import Real


def validate_fpr_limit(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
        raise ValueError("validation_false_positive_rate must be a finite number between 0 and 1.")
    return float(value)


def selection_objective(limit: float | None) -> dict:
    if limit is None:
        return {"metric": "macro_f1", "tie_break": "earlier_candidate"}
    return {"metric": "minority_recall", "validation_false_positive_rate_limit": limit,
            "tie_break": "fewer_false_positives_then_earlier_candidate"}


def validation_rank(metrics: dict | None, limit: float | None) -> tuple | None:
    if not metrics:
        return None
    if limit is None:
        score = metrics.get("macro_f1")
        if type(score) in (float, int) and math.isfinite(score) and 0 <= score <= 1:
            return (score,)
        return None
    try:
        (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    except (KeyError, TypeError, ValueError):
        return None
    if any(type(count) is not int or count < 0 for count in (tn, fp, fn, tp)):
        return None
    if tn + fp == 0 or fn + tp == 0 or fp / (tn + fp) > limit:
        return None
    return (tp / (fn + tp), -fp)
