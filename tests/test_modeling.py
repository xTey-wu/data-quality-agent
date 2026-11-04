import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from dqagent import modeling
from dqagent.quality import Cleaner
from dqagent.schema import (DEFAULT_PLAN, MISSING_INDICATOR_COLUMNS, MODEL_FEATURES,
                            MODEL_NUMERIC, ROW_ID)


def table(rows=100, *, offset=0):
    """Two risk groups with positive rates below the fixed classification boundary."""
    index = np.arange(rows)
    higher_risk = index % 2 == 0
    labels = np.where(higher_risk & (index % 10 < 4), "yes", "no")
    return pd.DataFrame({
        ROW_ID: [f"row-{offset + i:09d}" for i in index],
        "age": np.where(higher_risk, "60", "25"), "job": "admin.",
        "marital": "single", "education": "secondary", "default": "no",
        "balance": "100", "housing": "yes", "loan": "no", "contact": "cellular",
        "day": "10", "month": "may", "duration": "100", "campaign": "1",
        "pdays": "-1", "previous": "0", "poutcome": "unknown", "y": labels,
    })


def plan(**options):
    return {**DEFAULT_PLAN, "deduplicate": False, "class_weight": "none", **options}


def test_validation_threshold_improves_macro_f1_and_reports_consistent_metrics():
    training, validation = table(200), table(100, offset=1000)
    fixed = modeling.evaluate_plan(training, validation, plan())
    adaptive = modeling.evaluate_plan(training, validation, plan(threshold_policy="validation_macro_f1"))
    assert fixed["ok"] and adaptive["ok"]
    assert adaptive["validation"]["macro_f1"] > fixed["validation"]["macro_f1"]
    assert adaptive["decision_threshold"] < 0.5
    selection = adaptive["threshold_selection"]
    assert selection["fit_scope"] == "validation only"
    assert 1 < selection["candidates_evaluated"] <= 101
    assert adaptive["validation"]["decision_threshold"] == adaptive["decision_threshold"]
    tn, fp, fn, tp = np.array(adaptive["validation"]["confusion_matrix"]).ravel()
    assert adaptive["validation"]["precision"] == pytest.approx(tp / (tp + fp))
    assert adaptive["validation"]["minority_recall"] == pytest.approx(tp / (tp + fn))
    assert adaptive["validation"]["positive_f1"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert tn + fp + fn + tp == len(validation)


def test_threshold_ties_prefer_closest_to_half_then_higher_threshold():
    class EqualErrors:
        def predict_proba(self, inputs):
            return np.array([[0.52, 0.48], [0.48, 0.52]])

    validation = table(2)
    validation["y"] = ["yes", "no"]
    threshold, _ = modeling._select_threshold(
        EqualErrors(), validation, plan(threshold_policy="validation_macro_f1"), MODEL_FEATURES)
    assert threshold == 0.55


def test_finalization_threshold_and_probabilities_do_not_depend_on_test_data(tmp_path):
    training, validation, test = table(200), table(100, offset=1000), table(40, offset=2000)
    changed_test = test.copy()
    changed_test["y"] = np.where(test["y"] == "yes", "no", "yes")
    changed_test["age"] = "110"
    changed_test["balance"] = "99999999"
    adaptive = plan(threshold_policy="validation_macro_f1")
    first = modeling.finalize_plan(training, validation, test, adaptive, tmp_path / "first")
    second = modeling.finalize_plan(training, validation, changed_test, adaptive, tmp_path / "second")
    assert first["decision_threshold"] == second["decision_threshold"]
    assert first["threshold_selection"] == second["threshold_selection"]
    assert first["validation"] == second["validation"]
    one = joblib.load(tmp_path / "first/model.joblib")
    two = joblib.load(tmp_path / "second/model.joblib")
    clean = one["cleaner"].transform(validation)
    np.testing.assert_array_equal(one["model"].predict_proba(clean), two["model"].predict_proba(clean))


def test_exported_model_predict_uses_locked_threshold_and_preserves_probabilities(tmp_path, monkeypatch):
    training, validation, test = table(200), table(100, offset=1000), table(40, offset=2000)
    adaptive = plan(threshold_policy="validation_macro_f1")
    evaluated = modeling.evaluate_plan(training, validation, adaptive)
    threshold = evaluated["decision_threshold"]
    # Finalization receives the frozen validation threshold and must not select it again.
    monkeypatch.setattr(modeling, "_select_threshold", lambda *args: pytest.fail("locked threshold was reselected"))
    final = modeling.finalize_plan(training, validation, test, adaptive, tmp_path, decision_threshold=threshold)
    bundle = joblib.load(tmp_path / "model.joblib")
    processed = bundle["cleaner"].transform(test)
    expected_scores = bundle["base_model"].predict_proba(processed)
    np.testing.assert_array_equal(bundle["model"].predict_proba(processed), expected_scores)
    predicted = bundle["model"].predict(processed)
    np.testing.assert_array_equal(predicted, expected_scores[:, 1] > threshold)
    assert np.any(predicted != bundle["base_model"].predict(processed))
    truth = (test["y"] == "yes").astype(int)
    assert final["test"]["macro_f1"] == pytest.approx(f1_score(truth, predicted, average="macro"))
    assert final["test"]["precision"] == pytest.approx(precision_score(truth, predicted))
    assert final["test"]["minority_recall"] == pytest.approx(recall_score(truth, predicted))
    for filename in ["preprocessing.json", "test_metrics.json"]:
        record = json.loads((tmp_path / filename).read_text())
        assert record["decision_threshold"] == threshold
        assert record["prediction_features"] == MODEL_FEATURES
    assert bundle["decision_threshold"] == threshold
    assert final["threshold_selection"]["candidates_evaluated"] == 0


def test_preprocessing_and_classifier_fit_only_training_rows(tmp_path, monkeypatch):
    training, validation, test = table(200), table(40, offset=1000), table(30, offset=2000)
    validation["balance"] = "100000"
    test["balance"] = "99999999"
    validation["job"] = "validation-only-category"
    test["job"] = "test-only-category"
    fitted_rows = []
    cleaner_fit = Cleaner.fit

    def tracked_fit(self, frame):
        fitted_rows.append(frame[ROW_ID].tolist())
        return cleaner_fit(self, frame)

    monkeypatch.setattr(Cleaner, "fit", tracked_fit)
    modeling.finalize_plan(training, validation, test, plan(threshold_policy="validation_macro_f1"), tmp_path)
    assert fitted_rows == [training[ROW_ID].tolist()]
    bundle = joblib.load(tmp_path / "model.joblib")
    features = bundle["base_model"].named_steps["features"]
    scaler = features.named_transformers_["numeric"]
    assert scaler.n_samples_seen_ == len(training)
    assert scaler.mean_[MODEL_NUMERIC.index("balance")] == 100
    encoder = features.named_transformers_["categorical"]
    assert not any("validation-only-category" in values or "test-only-category" in values
                   for values in encoder.categories_)
    assert bundle["cleaner"].numeric_fill_["balance"] == 100


def test_legacy_fixed_plan_keeps_pipeline_predictions_metrics_and_feature_set(tmp_path):
    legacy = {key: value for key, value in plan().items()
              if key not in {"add_missing_indicators", "threshold_policy"}}
    training, validation, test = table(200), table(40, offset=1000), table(40, offset=2000)
    evaluated = modeling.evaluate_plan(training, validation, legacy)
    final = modeling.finalize_plan(training, validation, test, legacy, tmp_path)
    bundle = joblib.load(tmp_path / "model.joblib")
    processed = bundle["cleaner"].transform(test)
    expected = bundle["base_model"].predict(processed[MODEL_FEATURES])
    np.testing.assert_array_equal(bundle["model"].predict(processed[MODEL_FEATURES]), expected)
    assert bundle["prediction_features"] == MODEL_FEATURES
    assert not set(MISSING_INDICATOR_COLUMNS).intersection(processed.columns)
    assert evaluated["threshold_selection"]["candidates_evaluated"] == 0
    assert final["threshold_selection"]["candidates_evaluated"] == 0
    assert final["decision_threshold"] == evaluated["decision_threshold"] == 0.5
    assert final["test"]["macro_f1"] == pytest.approx(f1_score(test["y"] == "yes", expected, average="macro"))
    assert final["derived_features"] == []


def test_missing_indicators_enter_model_and_duration_has_no_effect(tmp_path):
    training, validation, test = table(200), table(60, offset=1000), table(40, offset=2000)
    # The only predictive input is whether a field was missing before imputation.
    for frame in (training, validation, test):
        frame["age"] = "40"
        frame.loc[frame["y"] == "yes", "balance"] = ""
    enabled = plan(add_missing_indicators=True, threshold_policy="validation_macro_f1")
    first = modeling.finalize_plan(training, validation, test, enabled, tmp_path / "first")
    altered_training, altered_validation, altered_test = [frame.copy() for frame in (training, validation, test)]
    for frame in (altered_training, altered_validation, altered_test):
        frame["duration"] = np.where(frame["y"] == "yes", "99999999", "unavailable")
    second = modeling.finalize_plan(altered_training, altered_validation, altered_test, enabled, tmp_path / "second")
    assert first["validation"] == second["validation"]
    assert first["test"] == second["test"]
    bundle = joblib.load(tmp_path / "first/model.joblib")
    assert bundle["prediction_features"] == MODEL_FEATURES + MISSING_INDICATOR_COLUMNS
    assert bundle["derived_features"] == MISSING_INDICATOR_COLUMNS
    assert not any("duration" in feature for feature in bundle["model"].feature_names_in_)
    feature_names = bundle["base_model"].named_steps["features"].get_feature_names_out()
    assert "numeric__balance__missing" in feature_names
    index = list(feature_names).index("numeric__balance__missing")
    assert abs(bundle["base_model"].named_steps["classifier"].coef_[0, index]) > 0
    assert first["test"]["macro_f1"] == 1.0
    disabled = modeling.evaluate_plan(training, validation, plan(threshold_policy="validation_macro_f1"))
    assert first["validation"]["macro_f1"] > disabled["validation"]["macro_f1"]
    for name in ("train", "validation", "test"):
        exported = pd.read_csv(tmp_path / "first" / f"{name}_processed.csv")
        assert set(MISSING_INDICATOR_COLUMNS).issubset(exported.columns)
    preprocessing = json.loads((tmp_path / "first/preprocessing.json").read_text())
    assert preprocessing["prediction_features"] == MODEL_FEATURES + MISSING_INDICATOR_COLUMNS


@pytest.mark.parametrize("value", [True, False, np.bool_(True), float("nan"), float("inf"), -0.1, 1.1, "0.5"])
def test_finalization_rejects_invalid_locked_thresholds_before_fitting(tmp_path, monkeypatch, value):
    monkeypatch.setattr(Cleaner, "fit", lambda *args: pytest.fail("invalid threshold reached fitting"))
    with pytest.raises(ValueError, match="decision_threshold"):
        modeling.finalize_plan(table(), table(20), table(20), plan(threshold_policy="validation_macro_f1"),
                               tmp_path, decision_threshold=value)


def test_fixed_policy_rejects_conflicting_locked_threshold(tmp_path):
    with pytest.raises(ValueError, match="fixed"):
        modeling.finalize_plan(table(), table(20), table(20), plan(), tmp_path, decision_threshold=0.4)
