"""Dataset contract and the deliberately small cleaning action space."""

FEATURES = [
    "age", "job", "marital", "education", "default", "balance", "housing",
    "loan", "contact", "day", "month", "duration", "campaign", "pdays",
    "previous", "poutcome",
]
TARGET = "y"
ROW_ID = "_row_id"
# Call duration is unavailable before making the marketing call.
MODEL_FEATURES = [name for name in FEATURES if name != "duration"]
MISSING_INDICATOR_COLUMNS = [f"{column}__missing" for column in MODEL_FEATURES]
NUMERIC = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAL = [name for name in FEATURES if name not in NUMERIC]
MODEL_NUMERIC = [name for name in NUMERIC if name in MODEL_FEATURES]

PLAN_OPTIONS = {
    "numeric_parser": {"strict", "semantic"},
    "numeric_imputation": {"median", "mean"},
    "categorical_imputation": {"unknown", "most_frequent"},
    "outlier_policy": {"keep", "invalid_to_missing", "iqr_clip"},
    "class_weight": {"none", "balanced"},
    "threshold_policy": {"fixed", "validation_macro_f1"},
}

OPTIONAL_PLAN_DEFAULTS = {
    "add_missing_indicators": False,
    "threshold_policy": "fixed",
}

DEFAULT_PLAN = {
    "name": "semantic_safe",
    "numeric_parser": "semantic",
    "numeric_imputation": "median",
    "categorical_imputation": "unknown",
    "outlier_policy": "invalid_to_missing",
    "deduplicate": True,
    "class_weight": "balanced",
    "reason": "Normalize recognized formats and repair schema violations using training statistics.",
    **OPTIONAL_PLAN_DEFAULTS,
}

NUMERIC_FALLBACKS = {
    "age": 40.0, "balance": 0.0, "day": 15.0, "duration": 0.0,
    "campaign": 1.0, "pdays": -1.0, "previous": 0.0,
}

DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
