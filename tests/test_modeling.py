import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, precision_score, recall_score

from dqagent import modeling
from dqagent.quality import Cleaner
from dqagent.schema import (DEFAULT_PLAN, MISSING_INDICATOR_COLUMNS, MODEL_FEATURES,
                            MODEL_NUMERIC, OPTIONAL_PLAN_DEFAULTS, ROW_ID)


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
    assert adaptive["validation"]["false_positive_rate"] == pytest.approx(fp / (tn + fp))
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


@pytest.mark.parametrize("fpr_limit", [None, 0.375])
def test_exported_model_predict_uses_locked_threshold_and_preserves_probabilities(tmp_path, monkeypatch, fpr_limit):
    training, validation, test = table(200), table(100, offset=1000), table(40, offset=2000)
    adaptive = plan(threshold_policy="validation_macro_f1")
    evaluated = modeling.evaluate_plan(training, validation, adaptive, validation_false_positive_rate=fpr_limit)
    assert evaluated["ok"], evaluated["error"]
    threshold = evaluated["decision_threshold"]
    if fpr_limit is not None:
        selection = evaluated["threshold_selection"]
        tn, fp, fn, tp = np.array(evaluated["validation"]["confusion_matrix"]).ravel()
        assert selection["fit_scope"] == "validation only"
        assert selection["objective"] == "recall_with_validation_false_positive_rate_limit"
        assert selection["validation_false_positive_rate_limit"] == fpr_limit
        assert selection["validation_false_positive_rate"] == pytest.approx(fp / (tn + fp))
        assert selection["validation_false_positive_rate"] <= fpr_limit
        assert selection["validation_recall"] == pytest.approx(tp / (fn + tp))
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
    assert final["validation"] == evaluated["validation"]


@pytest.mark.parametrize("classifier,estimator", [
    ("logistic_regression", modeling.LogisticRegression),
    ("hist_gradient_boosting", modeling.HistGradientBoostingClassifier),
])
def test_preprocessing_and_classifier_fit_only_training_rows(tmp_path, monkeypatch, classifier, estimator):
    training, validation, test = table(200), table(40, offset=1000), table(30, offset=2000)
    validation["balance"] = "100000"
    test["balance"] = "99999999"
    validation["job"] = "validation-only-category"
    test["job"] = "test-only-category"
    fitted_rows = []
    estimator_inputs = []
    cleaner_fit = Cleaner._fit_training
    estimator_fit = estimator.fit

    def tracked_fit(self, frame):
        fitted_rows.append(frame[ROW_ID].tolist())
        return cleaner_fit(self, frame)

    def tracked_estimator_fit(self, features, labels, **kwargs):
        estimator_inputs.append((features, labels))
        return estimator_fit(self, features, labels, **kwargs)

    monkeypatch.setattr(Cleaner, "_fit_training", tracked_fit)
    monkeypatch.setattr(estimator, "fit", tracked_estimator_fit)
    final = modeling.finalize_plan(training, validation, test, plan(classifier=classifier), tmp_path)
    assert fitted_rows == [training[ROW_ID].tolist()]
    assert len(estimator_inputs) == 1
    fitted_features, fitted_labels = estimator_inputs[0]
    assert fitted_features.shape[0] == len(training)
    np.testing.assert_array_equal(fitted_labels, (training["y"] == "yes").astype(int))
    bundle = joblib.load(tmp_path / "model.joblib")
    assert isinstance(bundle["base_model"].named_steps["classifier"], estimator)
    assert final["plan"]["classifier"] == classifier
    assert final["decision_threshold"] == 0.5
    transformed = bundle["cleaner"].transform(test)[MODEL_FEATURES]
    np.testing.assert_array_equal(bundle["model"].predict(transformed), bundle["base_model"].predict(transformed))
    if classifier == "hist_gradient_boosting":
        assert isinstance(fitted_features, np.ndarray)
        assert bundle["base_model"].named_steps["classifier"].early_stopping is False
    features = bundle["base_model"].named_steps["features"]
    scaler = features.named_transformers_["numeric"]
    assert scaler.n_samples_seen_ == len(training)
    assert scaler.mean_[MODEL_NUMERIC.index("balance")] == 100
    encoder = features.named_transformers_["categorical"]
    assert not any("validation-only-category" in values or "test-only-category" in values
                   for values in encoder.categories_)
    assert bundle["cleaner"].numeric_fill_["balance"] == 100


def test_legacy_fixed_plan_keeps_pipeline_predictions_metrics_and_feature_set(tmp_path):
    legacy = {key: value for key, value in plan().items() if key not in OPTIONAL_PLAN_DEFAULTS}
    training, validation, test = table(200), table(40, offset=1000), table(40, offset=2000)
    evaluated = modeling.evaluate_plan(training, validation, legacy)
    final = modeling.finalize_plan(training, validation, test, legacy, tmp_path)
    bundle = joblib.load(tmp_path / "model.joblib")
    processed = bundle["cleaner"].transform(test)
    expected = bundle["base_model"].predict(processed[MODEL_FEATURES])
    np.testing.assert_array_equal(bundle["model"].predict(processed[MODEL_FEATURES]), expected)
    assert bundle["prediction_features"] == MODEL_FEATURES
    assert final["plan"]["classifier"] == "logistic_regression"
    assert not set(MISSING_INDICATOR_COLUMNS).intersection(processed.columns)
    assert evaluated["threshold_selection"]["candidates_evaluated"] == 0
    assert final["threshold_selection"]["candidates_evaluated"] == 0
    assert final["decision_threshold"] == evaluated["decision_threshold"] == 0.5
    assert final["test"]["macro_f1"] == pytest.approx(f1_score(test["y"] == "yes", expected, average="macro"))
    assert final["derived_features"] == []
    assert final["model_derived_features"] == []


@pytest.mark.parametrize("classifier,unknown_month", [
    ("logistic_regression", 0.0), ("hist_gradient_boosting", np.nan),
])
def test_calendar_feature_is_derived_in_the_exported_model(tmp_path, classifier, unknown_month):
    training, validation, synthetic_holdout = table(200), table(40, offset=1000), table(40, offset=2000)
    training["month"] = np.where(np.arange(len(training)) % 3 == 0, "may", "jul")
    validation.loc[:4, "month"] = ["apr", "feb", "unexpected-month", "", "may"]
    validation.loc[:4, "day"] = ["20", "1", "10", "10", ""]
    calendar_plan = plan(classifier=classifier, add_calendar_feature=True,
                         categorical_imputation="most_frequent", add_missing_indicators=True)
    evaluated = modeling.evaluate_plan(training, validation, calendar_plan)
    assert evaluated["ok"], evaluated["error"]
    final = modeling.finalize_plan(training, validation, synthetic_holdout, calendar_plan, tmp_path)
    bundle = joblib.load(tmp_path / "model.joblib")
    preprocessing = json.loads((tmp_path / "preprocessing.json").read_text())
    saved_metrics = json.loads((tmp_path / "test_metrics.json").read_text())
    for record in (evaluated, final, bundle, preprocessing, saved_metrics):
        assert record["model_derived_features"] == ["calendar_day"]
        assert record["prediction_features"] == MODEL_FEATURES + MISSING_INDICATOR_COLUMNS
        assert record["derived_features"] == MISSING_INDICATOR_COLUMNS

    prepared = bundle["cleaner"].transform(validation.iloc[:5][MODEL_FEATURES])
    assert "calendar_day" not in prepared.columns
    exported = pd.read_csv(tmp_path / "validation_processed.csv")
    assert "calendar_day" not in exported.columns
    preprocessor = bundle["base_model"].named_steps["features"]
    calendar = preprocessor.named_transformers_["calendar"]
    source = prepared[["month", "day"]]
    derived = calendar.named_steps["derive"].transform(source)["calendar_day"]
    np.testing.assert_allclose(derived, [110.0, 32.0, unknown_month, 191.0, 130.0], equal_nan=True)
    assert "calendar__calendar_day" in preprocessor.get_feature_names_out()
    if np.isnan(unknown_month):
        assert np.isnan(calendar.transform(source)[2, 0])
    assert np.isfinite(bundle["model"].predict_proba(prepared)).all()
    assert final["validation"] == evaluated["validation"]


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


@pytest.mark.parametrize("scores,limit,threshold,tp,fp", [
    ([0.95, 0.70, 0.70, 0.10, 0.95, 0.80, 0.70, 0.20], 0.25, 0.70, 2, 1),
    ([0.95, 0.70, 0.70, 0.10, 0.99, 0.99, 0.20, 0.20], 0.25, 0.95, 2, 0),
    ([0.50, 0.10, 0.10, 0.10, 0.75, 0.50, 0.50, 0.10], 0.00, 0.50, 1, 0),
])
def test_fpr_threshold_keeps_probability_ties_together_and_prefers_fewer_false_positives(scores, limit, threshold, tp, fp):
    truth = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array(scores)
    selected, details = modeling._false_positive_rate_threshold(truth, scores, limit)
    assert selected == threshold
    predicted = scores > selected
    assert np.count_nonzero(predicted & (truth == 1)) == tp
    assert np.count_nonzero(predicted & (truth == 0)) == fp
    assert details["validation_false_positive_rate"] == fp / 4
    assert details["validation_recall"] == tp / 4


def test_threshold_at_half_preserves_fixed_predict_and_supports_strict_score_boundary():
    class HalfScores:
        def predict(self, inputs):
            return np.array([1, 1])

        def predict_proba(self, inputs):
            return np.array([[0.50, 0.50], [0.49, 0.51]])

    model = HalfScores()
    fixed = modeling.ThresholdClassifier(model, 0.5)
    strict = modeling.ThresholdClassifier(model, 0.5, use_base_predict_at_half=False)
    np.testing.assert_array_equal(fixed.predict(None), [1, 1])
    np.testing.assert_array_equal(strict.predict(None), [0, 1])


@pytest.mark.parametrize("cap", [True, np.bool_(True), "0.07", float("nan"), -0.1, 1.1])
def test_fpr_cap_rejects_invalid_rates_before_model_fit(cap, monkeypatch):
    monkeypatch.setattr(modeling, "_model", lambda *args: pytest.fail("invalid FPR cap reached model fitting"))
    result = modeling.evaluate_plan(table(), table(), plan(threshold_policy="validation_macro_f1"),
                                    validation_false_positive_rate=cap)
    assert not result["ok"]
    assert "validation_false_positive_rate" in result["error"]


def test_fpr_cap_requires_adaptive_threshold_policy():
    result = modeling.evaluate_plan(table(), table(), plan(), validation_false_positive_rate=0.07)
    assert not result["ok"]
    assert "requires threshold_policy=validation_macro_f1" in result["error"]
