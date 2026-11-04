"""Comparable classification experiments with an untouched final test split."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .quality import Cleaner, MAX_LEGAL_CLIP_FRACTION, profile, validate_plan
from .schema import CATEGORICAL, MODEL_FEATURES, MODEL_NUMERIC, ROW_ID, TARGET


def _labels(df: pd.DataFrame) -> np.ndarray:
    if not set(df[TARGET]).issubset({"yes", "no"}):
        raise ValueError("Unexpected labels: only yes/no labels are allowed and labels are never changed.")
    return df[TARGET].map({"no": 0, "yes": 1}).to_numpy(dtype=int)


def _model(plan: dict, seed: int) -> Pipeline:
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), MODEL_NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="drop")
    return Pipeline([
        ("features", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=1500, class_weight="balanced" if plan["class_weight"] == "balanced" else None,
            random_state=seed)),
    ])


def _metrics(model: Pipeline, df: pd.DataFrame) -> dict:
    truth = _labels(df)
    predicted = model.predict(df[MODEL_FEATURES])
    scores = model.predict_proba(df[MODEL_FEATURES])[:, 1]
    return {
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "minority_recall": float(recall_score(truth, predicted, pos_label=1, zero_division=0)),
        "average_precision": float(average_precision_score(truth, scores)) if np.any(truth == 1) else None,
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": ["no", "yes"],
        "rows": len(df), "positive_class": "yes",
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
              "changes": {}, "error": None}
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
        model.fit(cleaned[MODEL_FEATURES], _labels(cleaned))
        result["validation"] = _metrics(model, validation)
        result["ok"] = True
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["guardrails"]["reasons"] = [result["error"]]
    return result


def finalize_plan(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                  plan: dict, output_dir: str | Path, seed: int = 42) -> dict:
    """Lock the selected plan, fit only training, and evaluate test once per output directory."""
    output_dir = Path(output_dir)
    marker = output_dir / "test_metrics.json"
    if marker.exists():
        raise ValueError("This output directory already contains a final test evaluation. Reuse that result; do not tune against test metrics.")
    plan = validate_plan(plan)
    cleaner = Cleaner(plan)
    training = cleaner.fit_transform(train_df)
    changes = dict(cleaner.last_changes_)
    quality_after = profile(training)
    guardrails = _guardrails(train_df, training, changes, quality_after)
    if not guardrails["passed"]:
        raise ValueError("Final plan failed guardrails: " + "; ".join(guardrails["reasons"]))
    validation = cleaner.transform(val_df)
    test = cleaner.transform(test_df)
    model = _model(plan, seed)
    model.fit(training[MODEL_FEATURES], _labels(training))
    result = {"plan": plan, "validation": _metrics(model, validation), "test": _metrics(model, test),
              "guardrails": guardrails, "changes": changes,
              "fit_scope": "training only; validation and test transformed with frozen statistics",
              "prediction_features": MODEL_FEATURES, "test_evaluations": 1}
    result["artifacts"] = {name: str(output_dir / filename) for name, filename in {
        "train_processed": "train_processed.csv", "validation_processed": "validation_processed.csv",
        "test_processed": "test_processed.csv", "preprocessing": "preprocessing.json",
        "model": "model.joblib", "test_metrics": "test_metrics.json",
    }.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in [("train", training), ("validation", validation), ("test", test)]:
        table.to_csv(output_dir / f"{name}_processed.csv", index=False)
    (output_dir / "preprocessing.json").write_text(
        json.dumps(cleaner.description(), ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump({"cleaner": cleaner, "model": model, "plan": plan,
                 "prediction_features": MODEL_FEATURES, "target_mapping": {"no": 0, "yes": 1}},
                output_dir / "model.joblib")
    marker.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
