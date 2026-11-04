"""Comparable classification experiments with an untouched final test split."""

from __future__ import annotations

import json
import math
from numbers import Real
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .quality import Cleaner, MAX_LEGAL_CLIP_FRACTION, profile, validate_plan
from .schema import (CATEGORICAL, MISSING_INDICATOR_COLUMNS, MODEL_FEATURES,
                     MODEL_NUMERIC, ROW_ID, TARGET)


THRESHOLD_GRID = tuple(index / 20 for index in range(1, 20))


def _validate_threshold(value: float, policy: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("decision_threshold must be a finite number between 0 and 1.")
    try:
        threshold = float(value)
    except (ValueError, OverflowError):
        raise ValueError("decision_threshold must be a finite number between 0 and 1.") from None
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("decision_threshold must be a finite number between 0 and 1.")
    if policy == "fixed" and threshold != 0.5:
        raise ValueError("The fixed threshold policy only permits decision_threshold=0.5.")
    return threshold


class ThresholdClassifier:
    """A fitted classifier whose persisted predict method uses its locked threshold."""

    def __init__(self, base_model: Pipeline, decision_threshold: float):
        self.base_model = base_model
        self.decision_threshold = _validate_threshold(decision_threshold, "validation_macro_f1")

    @property
    def classes_(self):
        return self.base_model.classes_

    @property
    def feature_names_in_(self):
        return self.base_model.feature_names_in_

    @property
    def n_features_in_(self):
        return self.base_model.n_features_in_

    def predict_proba(self, inputs):
        return self.base_model.predict_proba(inputs)

    def predict(self, inputs):
        # Preserve LogisticRegression's existing decision boundary, including exact ties.
        if self.decision_threshold == 0.5:
            return self.base_model.predict(inputs)
        return (self.predict_proba(inputs)[:, 1] > self.decision_threshold).astype(int)


def _prediction_features(plan: dict) -> list[str]:
    return MODEL_FEATURES + (MISSING_INDICATOR_COLUMNS if plan["add_missing_indicators"] else [])


def _labels(df: pd.DataFrame) -> np.ndarray:
    if not set(df[TARGET]).issubset({"yes", "no"}):
        raise ValueError("Unexpected labels: only yes/no labels are allowed and labels are never changed.")
    return df[TARGET].map({"no": 0, "yes": 1}).to_numpy(dtype=int)


def _model(plan: dict, seed: int) -> Pipeline:
    numeric = MODEL_NUMERIC + (MISSING_INDICATOR_COLUMNS if plan["add_missing_indicators"] else [])
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="drop")
    return Pipeline([
        ("features", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=1500, class_weight="balanced" if plan["class_weight"] == "balanced" else None,
            random_state=seed)),
    ])


def _select_threshold(model: Pipeline, validation: pd.DataFrame, plan: dict,
                      prediction_features: list[str]) -> tuple[float, dict]:
    policy = plan["threshold_policy"]
    selection = {"policy": policy, "candidates_evaluated": 0,
                 "fit_scope": "none; fixed threshold", "decision_threshold": 0.5,
                 "positive_prediction_rule": "probability > decision_threshold"}
    if policy == "fixed":
        return 0.5, selection
    truth = _labels(validation)
    scores = model.predict_proba(validation[prediction_features])[:, 1]
    ranked = [(float(f1_score(truth, scores > threshold, average="macro", zero_division=0)), threshold)
              for threshold in THRESHOLD_GRID]
    score, threshold = max(ranked, key=lambda item: (item[0], -round(abs(item[1] - 0.5), 12), item[1]))
    selection.update({"fit_scope": "validation only", "decision_threshold": threshold,
                      "candidates_evaluated": len(THRESHOLD_GRID), "grid": list(THRESHOLD_GRID),
                      "objective": "macro_f1", "validation_macro_f1": score,
                      "tie_break": "closest to 0.5, then higher threshold"})
    return threshold, selection


def _metrics(model: ThresholdClassifier, df: pd.DataFrame, prediction_features: list[str]) -> dict:
    truth = _labels(df)
    predicted = model.predict(df[prediction_features])
    scores = model.predict_proba(df[prediction_features])[:, 1]
    return {
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "precision": float(precision_score(truth, predicted, pos_label=1, zero_division=0)),
        "positive_f1": float(f1_score(truth, predicted, pos_label=1, zero_division=0)),
        "minority_recall": float(recall_score(truth, predicted, pos_label=1, zero_division=0)),
        "average_precision": float(average_precision_score(truth, scores)) if np.any(truth == 1) else None,
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": ["no", "yes"],
        "rows": len(df), "positive_class": "yes", "decision_threshold": model.decision_threshold,
    }


def _guardrails(original: pd.DataFrame, cleaned: pd.DataFrame, changes: dict, after: dict) -> dict:
    reasons = []
    if after["missing_cells"]:
        reasons.append("Missing feature cells remain after preprocessing.")
    if after["numeric_parse_errors"]:
        reasons.append("Unparseable numeric cells remain.")
    if after["numeric_range_violations"]:
        reasons.append("Schema-invalid numeric values remain.")
    if changes.get("clipped_legal_fraction", 0) > MAX_LEGAL_CLIP_FRACTION:
        reasons.append(f"IQR clipping changed more than {MAX_LEGAL_CLIP_FRACTION:.0%} of legal numeric cells; reject and roll back.")
    if len(cleaned) < len(original) * 0.8:
        reasons.append("Preprocessing removed more than 20% of training rows.")
    if cleaned[TARGET].nunique() < 2:
        reasons.append("Both classes must remain in training.")
    if ROW_ID in original and ROW_ID in cleaned:
        labels = original.set_index(ROW_ID)[TARGET]
        retained = cleaned.set_index(ROW_ID)[TARGET]
        if not retained.index.isin(labels.index).all() or not retained.equals(labels.loc[retained.index]):
            reasons.append("Labels or row identity changed.")
    return {"passed": not reasons, "reasons": reasons,
            "max_legal_clip_fraction": MAX_LEGAL_CLIP_FRACTION,
            "maximum_training_row_removal_fraction": 0.2}


def evaluate_plan(train_df: pd.DataFrame, val_df: pd.DataFrame, plan: dict, seed: int = 42) -> dict:
    result = {"ok": False, "plan": plan, "validation": None, "quality_before": None,
              "quality_after": None, "guardrails": {"passed": False, "reasons": []},
              "changes": {}, "error": None, "decision_threshold": None, "threshold_selection": None}
    try:
        result["plan"] = validate_plan(plan)
        result["quality_before"] = profile(train_df)
        cleaner = Cleaner(result["plan"])
        cleaned = cleaner.fit_transform(train_df)
        changes = dict(cleaner.last_changes_)
        result["changes"] = changes
        result["quality_after"] = profile(cleaned)
        result["guardrails"] = _guardrails(train_df, cleaned, changes, result["quality_after"])
        if not result["guardrails"]["passed"]:
            result["error"] = "; ".join(result["guardrails"]["reasons"])
            return result
        validation = cleaner.transform(val_df)
        model = _model(result["plan"], seed)
        features = _prediction_features(result["plan"])
        model.fit(cleaned[features], _labels(cleaned))
        threshold, selection = _select_threshold(model, validation, result["plan"], features)
        result.update({"decision_threshold": threshold, "threshold_selection": selection,
                       "prediction_features": features,
                       "derived_features": MISSING_INDICATOR_COLUMNS if result["plan"]["add_missing_indicators"] else []})
        result["validation"] = _metrics(ThresholdClassifier(model, threshold), validation, features)
        result["ok"] = True
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["guardrails"]["reasons"] = [result["error"]]
    return result


def finalize_plan(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                  plan: dict, output_dir: str | Path, seed: int = 42, *,
                  decision_threshold: float | None = None) -> dict:
    """Lock the selected plan, fit only training, and evaluate test once per output directory."""
    output_dir = Path(output_dir)
    marker = output_dir / "test_metrics.json"
    if marker.exists():
        raise ValueError("This output directory already contains a final test evaluation. Reuse that result; do not tune against test metrics.")
    plan = validate_plan(plan)
    if decision_threshold is not None:
        decision_threshold = _validate_threshold(decision_threshold, plan["threshold_policy"])
    cleaner = Cleaner(plan)
    training = cleaner.fit_transform(train_df)
    changes = dict(cleaner.last_changes_)
    quality_after = profile(training)
    guardrails = _guardrails(train_df, training, changes, quality_after)
    if not guardrails["passed"]:
        raise ValueError("Final plan failed guardrails: " + "; ".join(guardrails["reasons"]))
    validation = cleaner.transform(val_df)
    features = _prediction_features(plan)
    derived_features = MISSING_INDICATOR_COLUMNS if plan["add_missing_indicators"] else []
    base_model = _model(plan, seed)
    base_model.fit(training[features], _labels(training))
    if decision_threshold is None:
        decision_threshold, selection = _select_threshold(base_model, validation, plan, features)
    else:
        selection = {"policy": plan["threshold_policy"], "fit_scope": "none; caller-provided locked threshold",
                     "candidates_evaluated": 0, "decision_threshold": decision_threshold,
                     "positive_prediction_rule": "probability > decision_threshold", "source": "caller_locked"}
    model = ThresholdClassifier(base_model, decision_threshold)
    validation_metrics = _metrics(model, validation, features)
    # The threshold and all fitted preprocessing are fixed before handling the test split.
    test = cleaner.transform(test_df)
    result = {"plan": plan, "validation": validation_metrics, "test": _metrics(model, test, features),
              "guardrails": guardrails, "changes": changes,
              "fit_scope": "training only; validation and test transformed with frozen statistics",
              "prediction_features": features, "derived_features": derived_features,
              "decision_threshold": decision_threshold, "threshold_selection": selection,
              "test_evaluations": 1}
    result["artifacts"] = {name: str(output_dir / filename) for name, filename in {
        "train_processed": "train_processed.csv", "validation_processed": "validation_processed.csv",
        "test_processed": "test_processed.csv", "preprocessing": "preprocessing.json",
        "model": "model.joblib", "test_metrics": "test_metrics.json",
    }.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in [("train", training), ("validation", validation), ("test", test)]:
        table.to_csv(output_dir / f"{name}_processed.csv", index=False)
    (output_dir / "preprocessing.json").write_text(
        json.dumps({**cleaner.description(), "prediction_features": features, "derived_features": derived_features,
                    "decision_threshold": decision_threshold, "threshold_selection": selection},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"cleaner": cleaner, "model": model, "base_model": base_model, "plan": plan,
                 "prediction_features": features, "derived_features": derived_features,
                 "decision_threshold": decision_threshold, "threshold_selection": selection,
                 "target_mapping": {"no": 0, "yes": 1}},
                output_dir / "model.joblib")
    marker.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
