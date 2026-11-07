"""Inspectable quality tools and a training-only fitted cleaner."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from .schema import (CATEGORICAL, DEFAULT_PLAN, FEATURES, MISSING_INDICATOR_COLUMNS,
                     MODEL_FEATURES, MODEL_NUMERIC, NUMERIC, NUMERIC_FALLBACKS,
                     OPTIONAL_PLAN_DEFAULTS, PLAN_OPTIONS, ROW_ID, TARGET)

MISSING_MARKERS = {"", "?", "na", "n/a", "nan", "none", "null"}
MAX_LEGAL_CLIP_FRACTION = 0.02


def validate_plan(plan: dict) -> dict:
    if not isinstance(plan, dict):
        raise ValueError("A cleaning plan must be a JSON object.")
    allowed = set(DEFAULT_PLAN)
    required = allowed - set(OPTIONAL_PLAN_DEFAULTS)
    if not required.issubset(plan) or set(plan) - allowed:
        raise ValueError(f"Plan requires {sorted(required)}; missing={sorted(required - set(plan))}, extra={sorted(set(plan) - allowed)}")
    plan = {**OPTIONAL_PLAN_DEFAULTS, **plan}
    for key in ("name", "reason"):
        if not isinstance(plan[key], str) or not plan[key].strip():
            raise ValueError(f"{key} must be a nonempty string.")
    if len(plan["name"]) > 100 or len(plan["reason"]) > 2000:
        raise ValueError("Plan name/reason exceeds the allowed length.")
    for key, options in PLAN_OPTIONS.items():
        if not isinstance(plan[key], str) or plan[key] not in options:
            raise ValueError(f"Unsupported {key}: choose from {sorted(options)}")
    for key in ("deduplicate", "add_missing_indicators", "add_calendar_feature"):
        if type(plan[key]) is not bool:
            raise ValueError(f"{key} must be a JSON boolean.")
    return dict(plan)


def _text(value) -> str:
    if pd.isna(value):
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip()


def _missing(series: pd.Series) -> pd.Series:
    return series.map(lambda value: _text(value).lower() in MISSING_MARKERS)


def _parse_numeric(series: pd.Series, column: str, parser: str) -> tuple[pd.Series, pd.Series]:
    text = series.map(_text)
    missing = _missing(series)
    if parser == "semantic":
        # Recognized units only: arbitrary trailing text is never silently stripped.
        suffixes = {
            "age": r"(?:years?|岁)", "day": r"(?:days?|天|日)",
            "duration": r"(?:seconds?|secs?|秒)", "campaign": r"(?:times?|次)",
            "pdays": r"(?:days?|天|日)", "previous": r"(?:times?|次)",
            "balance": r"EUR",
        }
        if column == "balance":
            text = text.str.replace(r"^€\s*", "", regex=True)
        text = text.str.replace(suffixes[column] + r"\s*$", "", regex=True, flags=re.IGNORECASE).str.strip()
        # Accept correctly grouped thousands; reject malformed separators such as 12,34.
        grouped = text.str.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?")
        text.loc[grouped] = text.loc[grouped].str.replace(",", "", regex=False)
    result = pd.to_numeric(text.mask(missing), errors="coerce").astype(float)
    invalid_parse = ~missing & (result.isna() | ~np.isfinite(result))
    return result, invalid_parse


def _invalid_numeric(values: pd.Series, column: str) -> pd.Series:
    present = values.notna()
    invalid = present & ~np.isfinite(values)
    if column == "age":
        invalid |= present & ((values < 0) | (values > 120))
    elif column == "day":
        invalid |= present & ((values < 1) | (values > 31))
    elif column == "campaign":
        invalid |= present & (values < 1)
    elif column in {"duration", "previous"}:
        invalid |= present & (values < 0)
    elif column == "pdays":
        invalid |= present & (values < 0) & (values != -1)
    # All source numeric fields are integral counts/amounts. Imputation may be fractional,
    # so integrality is reported through raw formatting rather than treated as invalid.
    return invalid


def _require_columns(df: pd.DataFrame, *, require_target: bool = True,
                     require_duration: bool = True) -> None:
    required = set(MODEL_FEATURES)
    if require_target:
        required.add(TARGET)
    if require_duration:
        required.add("duration")
    absent = required - set(df.columns)
    if absent:
        raise ValueError(f"Missing columns: {sorted(absent)}")
    if ROW_ID in df and (df[ROW_ID].duplicated().any() or df[ROW_ID].isna().any()):
        raise ValueError("_row_id must be unique and nonmissing.")


def profile(df: pd.DataFrame) -> dict:
    _require_columns(df)
    summaries = {}
    missing_total = parse_total = invalid_total = category_format_total = 0
    for column in FEATURES:
        missing = _missing(df[column])
        item = {"missing": int(missing.sum()), "unique": int(df[column].nunique(dropna=False))}
        if column in MODEL_FEATURES:
            missing_total += item["missing"]
        if column in NUMERIC:
            strict, strict_errors = _parse_numeric(df[column], column, "strict")
            semantic, semantic_errors = _parse_numeric(df[column], column, "semantic")
            invalid = _invalid_numeric(semantic, column)
            item.update({"kind": "numeric", "strict_parse_errors": int(strict_errors.sum()),
                         "semantic_parse_errors": int(semantic_errors.sum()),
                         "invalid_range": int(invalid.sum()),
                         "examples_requiring_parse": df.loc[strict_errors, column].astype(str).head(3).tolist()})
            valid = semantic.loc[~invalid & semantic.notna()]
            item["range"] = [float(valid.min()), float(valid.max())] if len(valid) else None
            if column in MODEL_NUMERIC:
                parse_total += item["strict_parse_errors"]
                invalid_total += item["invalid_range"]
        else:
            text = df[column].map(_text)
            normalized = text.str.lower()
            changed = ~missing & (df[column].astype(str) != normalized)
            item.update({"kind": "categorical", "format_inconsistencies": int(changed.sum()),
                         "unknown_semantic_count": int((normalized == "unknown").sum()),
                         "top_values": normalized.value_counts().head(5).to_dict()})
            category_format_total += item["format_inconsistencies"]
        summaries[column] = item
    duplicate_count = int(df.duplicated(subset=MODEL_FEATURES + [TARGET]).sum())
    conflicts = df.groupby(MODEL_FEATURES, dropna=False)[TARGET].nunique()
    labels = {str(key): int(value) for key, value in df[TARGET].value_counts().items()}
    return {
        "rows": len(df), "feature_count": len(FEATURES), "prediction_feature_count": len(MODEL_FEATURES),
        "quality_count_scope": "prediction features only; duration is preserved metadata",
        "missing_cells": missing_total, "duplicate_rows": duplicate_count,
        "numeric_parse_errors": parse_total, "numeric_range_violations": invalid_total,
        "categorical_format_inconsistencies": category_format_total,
        "conflicting_label_groups": int((conflicts > 1).sum()),
        "label_counts": labels, "fields": summaries,
        "preserved_fields": {"duration": {"processing": "original values preserved unchanged",
                                           "excluded_from_quality_totals": True,
                                           "excluded_from_planning_and_prediction": True,
                                           "excluded_from_training_deduplication": True}},
        "duplicate_rule": "Equal prediction inputs and target; identifiers and duration are excluded. Cleaning deduplicates normalized values before outlier replacement and imputation.",
        "semantics": {"pdays_minus_one": "not previously contacted; valid sentinel",
                      "negative_balance": "valid balance, not automatically an anomaly",
                      "unknown": "a source category, not a missing marker",
                      "duration": "preserved for delivery, excluded from prediction"},
    }


class Cleaner:
    """Fit imputers and optional clipping on training data; never edit labels."""

    def __init__(self, plan: dict):
        self.plan = validate_plan(plan)
        self.missing_indicator_columns_ = (
            list(MISSING_INDICATOR_COLUMNS) if self.plan["add_missing_indicators"] else [])
        self.numeric_fill_: dict[str, float] = {}
        self.categorical_fill_: dict[str, str] = {}
        self.iqr_bounds_: dict[str, list[float]] = {}
        self.last_changes_: dict = {}

    def __setstate__(self, state: dict) -> None:
        """Restore fitted cleaners saved before optional plan fields existed."""
        self.__dict__.update(state)
        self.plan = validate_plan(self.plan)
        if "missing_indicator_columns_" not in state:
            self.missing_indicator_columns_ = (
                list(MISSING_INDICATOR_COLUMNS) if self.plan["add_missing_indicators"] else [])

    def _convert(self, df: pd.DataFrame, *, inference: bool = False) -> pd.DataFrame:
        _require_columns(df, require_target=not inference, require_duration=not inference)
        converted = df.copy()
        errors = {}
        for column in MODEL_NUMERIC:
            values, failed = _parse_numeric(df[column], column, self.plan["numeric_parser"])
            if failed.any():
                errors[column] = {"count": int(failed.sum()),
                                  "examples": df.loc[failed, column].astype(str).head(2).tolist()}
            converted[column] = values
        if errors:
            raise ValueError(f"numeric_parser={self.plan['numeric_parser']} could not parse values: {errors}. Use semantic for recognized currency/thousands/unit formats.")
        for column in CATEGORICAL:
            values = df[column].map(lambda value: _text(value).lower())
            converted[column] = values.mask(_missing(df[column]), np.nan)
        return converted

    def _training_rows(self, normalized: pd.DataFrame) -> pd.DataFrame:
        """Use one pre-imputation duplicate rule for fitting and train export."""
        if not self.plan["deduplicate"]:
            return normalized
        if ROW_ID in normalized:
            normalized = normalized.sort_values(ROW_ID, kind="stable")
        return normalized.drop_duplicates(subset=MODEL_FEATURES + [TARGET], keep="first")

    def _outliers(self, converted: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        result = converted.copy()
        changes = {"invalid_to_missing": 0, "clipped_legal_cells": 0,
                   "legal_numeric_cells": 0, "clipped_by_field": {}}
        for column in MODEL_NUMERIC:
            values = result[column]
            invalid = _invalid_numeric(values, column)
            legal = values.notna() & ~invalid
            changes["legal_numeric_cells"] += int(legal.sum())
            if self.plan["outlier_policy"] in {"invalid_to_missing", "iqr_clip"}:
                changes["invalid_to_missing"] += int(invalid.sum())
                result.loc[invalid, column] = np.nan
            if self.plan["outlier_policy"] == "iqr_clip" and column in self.iqr_bounds_:
                low, high = self.iqr_bounds_[column]
                eligible = legal & (values != -1) if column == "pdays" else legal
                clipped = eligible & ((values < low) | (values > high))
                count = int(clipped.sum())
                changes["clipped_legal_cells"] += count
                changes["clipped_by_field"][column] = count
                result.loc[eligible, column] = values.loc[eligible].clip(low, high)
        changes["clipped_legal_fraction"] = changes["clipped_legal_cells"] / max(1, changes["legal_numeric_cells"])
        return result, changes

    def _fit_training(self, train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        converted = self._training_rows(self._convert(train_df))
        if self.plan["outlier_policy"] == "iqr_clip":
            for column in MODEL_NUMERIC:
                values = converted.loc[~_invalid_numeric(converted[column], column), column].dropna()
                if column == "pdays":
                    values = values[values != -1]
                if not len(values):
                    continue
                q1, q3 = values.quantile([0.25, 0.75])
                width = float(q3 - q1)
                if width > 0:
                    self.iqr_bounds_[column] = [float(q1 - 1.5 * width), float(q3 + 1.5 * width)]
        converted, changes = self._outliers(converted)
        for column in MODEL_NUMERIC:
            values = converted[column]
            statistic = values.median() if self.plan["numeric_imputation"] == "median" else values.mean()
            self.numeric_fill_[column] = float(statistic) if pd.notna(statistic) else NUMERIC_FALLBACKS[column]
        for column in CATEGORICAL:
            mode = converted[column].dropna().mode()
            self.categorical_fill_[column] = (
                str(mode.iloc[0]) if self.plan["categorical_imputation"] == "most_frequent" and len(mode)
                else "unknown")
        return converted, changes

    def fit(self, train_df: pd.DataFrame) -> "Cleaner":
        self._fit_training(train_df)
        return self

    def transform(self, df: pd.DataFrame, training: bool = False) -> pd.DataFrame:
        if not self.numeric_fill_:
            raise ValueError("Cleaner must be fitted on training data before transform.")
        normalized = self._convert(df, inference=not training)
        if training:
            normalized = self._training_rows(normalized)
        converted, changes = self._outliers(normalized)
        return self._finish_transform(converted, changes, len(df))

    def _finish_transform(self, converted: pd.DataFrame, changes: dict, rows_before: int) -> pd.DataFrame:
        if self.plan["add_missing_indicators"]:
            for column, indicator in zip(MODEL_FEATURES, self.missing_indicator_columns_):
                converted[indicator] = converted[column].isna().astype("int8")
        changes["numeric_missing_filled"] = int(converted[MODEL_NUMERIC].isna().sum().sum())
        changes["categorical_missing_filled"] = int(converted[CATEGORICAL].isna().sum().sum())
        for column, value in self.numeric_fill_.items():
            converted[column] = converted[column].fillna(value)
        for column, value in self.categorical_fill_.items():
            converted[column] = converted[column].fillna(value)
        changes["duplicates_removed"] = rows_before - len(converted)
        changes["rows_before"] = rows_before
        changes["rows_after"] = len(converted)
        self.last_changes_ = changes
        return converted.reset_index(drop=True)

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        converted, changes = self._fit_training(train_df)
        return self._finish_transform(converted, changes, len(train_df))

    def description(self) -> dict:
        return {"plan": self.plan, "numeric_fill": self.numeric_fill_,
                "categorical_fill": self.categorical_fill_, "iqr_bounds": self.iqr_bounds_,
                "fit_scope": "training only",
                "prediction_features": MODEL_FEATURES + self.missing_indicator_columns_,
                "missing_indicator_columns": self.missing_indicator_columns_,
                "missing_indicator_definition": "Fixed prediction-field indicators: 1 after outlier replacement and before filling, otherwise 0. Real unknown categories and valid pdays=-1 are not missing. Columns are available even when training had no missing values; validation/test never fit statistics.",
                "deduplication": "Training only: normalized prediction inputs and target, before outlier replacement and imputation; no claim of customer-entity deduplication.",
                "preserved_fields": {"duration": "Unchanged; excluded from fitting, parsing, clipping, imputation, duplicate decisions and quality guardrails."},
                "max_legal_clip_fraction": MAX_LEGAL_CLIP_FRACTION,
                "missing_markers": sorted(MISSING_MARKERS),
                "retained_source_semantics": ["unknown category", "pdays=-1", "negative balance"]}


def evaluate_repairs(reference: pd.DataFrame, dirty: pd.DataFrame,
                     processed: pd.DataFrame, manifest: dict) -> dict:
    """Offline evaluator only. Never give its inputs or results to the planner."""
    clean_index = processed.set_index(ROW_ID, drop=False)
    reference_index = reference.set_index(ROW_ID, drop=False)
    counts: dict[str, dict] = {}
    numeric_errors: dict[str, dict[str, list[float]]] = {}
    numeric_units = {"age": "years", "balance": "EUR", "day": "day_of_month",
                     "campaign": "contacts", "pdays": "days", "previous": "contacts"}
    categorical_trials: dict[str, list[bool]] = {}
    corrupt_cells = {(entry["row_id"], entry["column"]) for entry in manifest.get("entries", [])}
    for entry in manifest.get("entries", []):
        kind, row_id, column = entry["kind"], entry["row_id"], entry["column"]
        item = counts.setdefault(kind, {"injected": 0, "retained_evaluable": 0,
                                        "exactly_restored": 0, "nonmissing_after": 0,
                                        "schema_valid_after": 0, "numeric_retained_evaluable": 0})
        item["injected"] += 1
        if row_id not in clean_index.index or row_id not in reference_index.index:
            continue
        item["retained_evaluable"] += 1
        actual, expected = clean_index.at[row_id, column], reference_index.at[row_id, column]
        if not (_text(actual).lower() in MISSING_MARKERS):
            item["nonmissing_after"] += 1
        if column in NUMERIC:
            item["numeric_retained_evaluable"] += 1
            try:
                actual_number, expected_number = float(actual), float(expected)
                if not np.isfinite(actual_number) or not np.isfinite(expected_number):
                    raise ValueError("Nonfinite reconstructed numeric value.")
                error = abs(actual_number - expected_number)
                exact = bool(np.isclose(actual_number, expected_number, rtol=0, atol=1e-9))
                numeric_errors.setdefault(kind, {}).setdefault(column, []).append(error)
                item["schema_valid_after"] += int(not _invalid_numeric(pd.Series([float(actual)]), column).iloc[0])
            except (TypeError, ValueError):
                exact = False
        else:
            exact = _text(actual).lower() == _text(expected).lower()
            categorical_trials.setdefault(kind, []).append(exact)
        item["exactly_restored"] += int(exact)
    for kind, item in counts.items():
        denominator = item["retained_evaluable"]
        item["exact_restoration_rate"] = item["exactly_restored"] / denominator if denominator else None
        item["filled_rate"] = item["nonmissing_after"] / denominator if denominator else None
        numeric_denominator = item["numeric_retained_evaluable"]
        item["schema_valid_rate_after"] = item["schema_valid_after"] / numeric_denominator if numeric_denominator else None
        item["numeric_reconstruction_mae_by_field"] = {
            column: {"mae": float(np.mean(errors)), "n": len(errors),
                     "unit": numeric_units.get(column, "source_field_units")}
            for column, errors in numeric_errors.get(kind, {}).items() if errors
        }
        trials = categorical_trials.get(kind, [])
        item["categorical_reconstruction_accuracy"] = float(np.mean(trials)) if trials else None
    retained = set(processed[ROW_ID])
    injected = manifest.get("duplicates", [])
    removed = sum(entry["row_id"] not in retained for entry in injected)
    original_present = sum(entry["original_row_id"] in retained for entry in injected)
    untouched_evaluated = untouched_changed = 0
    shared = reference_index.index.intersection(clean_index.index)
    for column in MODEL_FEATURES:
        expected = reference_index.loc[shared, column]
        actual = clean_index.loc[shared, column]
        eligible = np.array([(row_id, column) not in corrupt_cells for row_id in shared])
        if column in NUMERIC:
            expected_values, expected_errors = _parse_numeric(expected, column, "semantic")
            actual_values, actual_errors = _parse_numeric(actual, column, "semantic")
            # Already missing/invalid source values are outside the known-clean denominator.
            eligible &= ~(expected_values.isna() | expected_errors | _invalid_numeric(expected_values, column)).to_numpy()
            equal = np.isclose(expected_values.to_numpy(), actual_values.to_numpy(), rtol=0, atol=1e-9, equal_nan=False)
            equal &= ~actual_errors.to_numpy()
        else:
            eligible &= ~_missing(expected).to_numpy()
            equal = (expected.map(lambda value: _text(value).lower()).to_numpy()
                     == actual.map(lambda value: _text(value).lower()).to_numpy())
        untouched_evaluated += int(eligible.sum())
        untouched_changed += int((eligible & ~equal).sum())
    return {"cell_repairs": counts,
            "duplicates": {"injected": len(injected), "removed": removed,
                           "removal_recall": removed / len(injected) if injected else None,
                           "original_rows_retained": original_present},
            "reference_rows_retained": int(reference[ROW_ID].isin(retained).sum()),
            "reference_rows_removed": int((~reference[ROW_ID].isin(retained)).sum()),
            "reference_rows": len(reference), "dirty_rows": len(dirty),
            "untouched_cells": {"retained_evaluable": untouched_evaluated,
                                "changed": untouched_changed,
                                "unintended_change_rate": untouched_changed / untouched_evaluated if untouched_evaluated else None,
                                "scope": "Uncorrupted, retained source cells in the 15 prediction features only; duration, target and identifiers are excluded. Source-missing or schema-invalid numeric values are excluded from the denominator."},
            "note": "Exact restoration is meaningful for reversible formatting. Imputation is an estimate, so missing/outlier reconstruction is reported separately; source data is not absolute ground truth. Source rows removed by equal-input/equal-label merging may represent independent customers; report reference row retention as its cost."}
