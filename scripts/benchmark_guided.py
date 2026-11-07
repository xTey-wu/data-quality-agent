"""Reproduce the guided search and classifier comparisons on a prepared split."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dqagent.data import load_table
from dqagent.engine import run_agent
from dqagent.experiment import _PresetSearchProvider, write_json
from dqagent.modeling import evaluate_plan, finalize_plan
from dqagent.quality import validate_plan


CANDIDATE_BUDGET = 4
FPR_LIMIT = 0.07
HISTORICAL_PLAN = {
    "name": "historical_fixed",
    "reason": "Replay the plan selected in the earlier live tool-calling experiment.",
    "numeric_parser": "semantic",
    "numeric_imputation": "median",
    "categorical_imputation": "most_frequent",
    "outlier_policy": "invalid_to_missing",
    "deduplicate": True,
    "class_weight": "balanced",
    "add_missing_indicators": True,
    "threshold_policy": "fixed",
    "classifier": "logistic_regression",
}


def load_recorded_table(path: Path) -> tuple:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    table = load_table(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"{path.name} changed while being read.")
    return table, digest


def validation_row(key, plan, threshold, selection, metrics, seconds):
    return {
        "id": key, "plan": validate_plan(plan), "classifier": plan.get("classifier", "logistic_regression"),
        "decision_threshold": threshold, "threshold_selection": selection,
        "validation": metrics, "validation_seconds": round(seconds, 3),
    }


def run_benchmark(data_dir: Path, output_dir: Path, seed: int | None = None) -> dict:
    started = perf_counter()
    metadata_bytes = (data_dir / "metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    prepared_seed = metadata.get("seed")
    if type(prepared_seed) is not int or (seed is not None and seed != prepared_seed):
        raise ValueError("Seed must match the prepared split's metadata.json.")
    seed = prepared_seed
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Choose a new or empty output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)

    train, train_hash = load_recorded_table(data_dir / "train_dirty.csv")
    validation, validation_hash = load_recorded_table(data_dir / "validation.csv")
    input_hashes = {
        "metadata.json": hashlib.sha256(metadata_bytes).hexdigest(),
        "train_dirty.csv": train_hash, "validation.csv": validation_hash,
    }
    report = {
        "schema_version": 1, "seed": seed, "dataset": metadata, "input_sha256": input_hashes,
        "environment": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "pandas", "scikit-learn")}},
        "comparison_protocol": {
            "scope": "Same-split regression comparison for each prepared split.",
            "test_used_for_selection": False, "historical_live_plan_replay": True,
            "candidate_budget_per_search": CANDIDATE_BUDGET,
            "validation_false_positive_rate_limit": FPR_LIMIT,
        },
        "searches": {}, "validation_results": [],
    }
    rows = report["validation_results"]

    # The two model families use the same frozen cleaning configuration.
    tuned_lr = {**HISTORICAL_PLAN, "threshold_policy": "validation_macro_f1"}
    tuned_hgb = {**tuned_lr, "classifier": "hist_gradient_boosting"}
    for key, plan, fpr_limit in (
        ("historical_fixed", HISTORICAL_PLAN, None),
        ("historical_tuned_lr", tuned_lr, None),
        ("historical_tuned_hgb", tuned_hgb, None),
        ("matched_lr", tuned_lr, FPR_LIMIT),
        ("matched_hgb", tuned_hgb, FPR_LIMIT),
        ("matched_hgb_calendar", {**tuned_hgb, "add_calendar_feature": True}, FPR_LIMIT),
    ):
        began = perf_counter()
        plan = {**plan, "name": key}
        options = {} if fpr_limit is None else {"validation_false_positive_rate": fpr_limit}
        result = evaluate_plan(train, validation, plan, seed=seed, **options)
        if not result["ok"]:
            write_json(output_dir / "validation.json", {**report, "failure": {"id": key, "error": result["error"]}})
            raise ValueError(f"Validation failed for {key}.")
        rows.append(validation_row(key, result["plan"], result["decision_threshold"],
                                   result["threshold_selection"], result["validation"], perf_counter() - began))

    for key, options in (
        ("preset_search", {"provider": _PresetSearchProvider()}),
        ("guided_search", {"guided": True}),
    ):
        began = perf_counter()
        summary = run_agent(train, validation, output_dir / key, seed=seed,
                            max_candidates=CANDIDATE_BUDGET, **options)
        report["searches"][key] = summary
        if summary["status"] != "completed":
            write_json(output_dir / "validation.json", {**report, "failure": {"id": key, "error": summary["status"]}})
            raise ValueError(f"Validation search failed for {key}.")
        rows.append(validation_row(key, summary["selected_plan"], summary["best_decision_threshold"],
                                   summary["best_threshold_selection"], summary["best_validation"],
                                   perf_counter() - began))

    plans = {row["id"]: row["plan"] for row in rows}
    thresholds = {row["id"]: row["decision_threshold"] for row in rows}
    report["validation_stage_seconds"] = round(perf_counter() - started, 3)
    write_json(output_dir / "validation.json", report)
    write_json(output_dir / "frozen_plans.json", plans)
    write_json(output_dir / "frozen_thresholds.json", thresholds)

    # The test file is first opened after all validation decisions are saved.
    test, input_hashes["test.csv"] = load_recorded_table(data_dir / "test.csv")
    test_rows = []
    for row in rows:
        key = row["id"]
        began = perf_counter()
        final = finalize_plan(train, validation, test, plans[key], output_dir / key / "artifacts",
                              seed=seed, decision_threshold=thresholds[key])
        test_rows.append({
            "id": key, "classifier": row["classifier"], "decision_threshold": thresholds[key],
            "test": final["test"], "artifacts": final["artifacts"],
            "final_fit_and_test_seconds": round(perf_counter() - began, 3),
        })
    write_json(output_dir / "test.json", {"seed": seed, "input_sha256": input_hashes, "test_results": test_rows})
    report.update(status="completed", test_results=test_rows, elapsed_seconds=round(perf_counter() - started, 3))
    write_json(output_dir / "metrics.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Prepared dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="New or empty results directory")
    parser.add_argument("--seed", type=int, help="Optional check against the prepared dataset seed")
    args = parser.parse_args()
    try:
        result = run_benchmark(args.data, args.output, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps({"status": result["status"], "seed": result["seed"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
