"""Shared training-profile guidance for bounded candidate evaluation."""

from .schema import DEFAULT_PLAN


def training_guidance(profile: dict, *, validation_false_positive_rate: float | None = None) -> dict:
    parse_errors = profile.get("numeric_parse_errors", 0) or 0
    range_violations = profile.get("numeric_range_violations", 0) or 0
    recall_mode = validation_false_positive_rate is not None
    return {
        "strict_parse_errors": parse_errors,
        "numeric_range_violations": range_violations,
        "threshold_policy": "validation_macro_f1",
        "recommended_plan": {
            **DEFAULT_PLAN,
            "name": "profile_baseline",
            "reason": "按训练画像修复已知格式和范围问题，建立可比较的分类基线。",
            "numeric_parser": "semantic" if parse_errors else "strict",
            "outlier_policy": "invalid_to_missing" if range_violations else "keep",
            "deduplicate": bool(profile.get("duplicate_rows")),
            "categorical_imputation": "most_frequent" if recall_mode else "unknown",
            "class_weight": "balanced" if recall_mode else "none",
            "add_missing_indicators": bool(profile.get("missing_cells") or range_violations),
            "classifier": "hist_gradient_boosting" if recall_mode else DEFAULT_PLAN["classifier"],
            "threshold_policy": "validation_macro_f1",
        },
    }


def guide_plan(plan: dict, guidance: dict | None) -> tuple[dict, dict]:
    """Resolve known schema conflicts and use one threshold policy per guided run."""
    if guidance is None:
        return plan, {}
    settings = {"threshold_policy": guidance["threshold_policy"]}
    if guidance["strict_parse_errors"] and plan["numeric_parser"] == "strict":
        settings["numeric_parser"] = "semantic"
    if guidance["numeric_range_violations"] and plan["outlier_policy"] == "keep":
        settings["outlier_policy"] = "invalid_to_missing"
    adjustments = {key: {"requested": plan[key], "effective": value}
                   for key, value in settings.items() if plan[key] != value}
    return {**plan, **settings}, adjustments
