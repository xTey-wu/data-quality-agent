"""Persistent service and real stdio MCP protocol tests without an LLM API."""

import json
import multiprocessing
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from time import sleep
from unittest.mock import patch

import anyio
import pandas as pd
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import ValidationError

from dqagent import modeling
from dqagent.objective import selection_objective
from dqagent.schema import DEFAULT_PLAN, MODEL_FEATURES, plan_signature
from dqagent.service import CleaningPlan, CleaningService


def fixture_table(rows=80):
    frame = pd.DataFrame({
        "_row_id": [f"row-{index:09d}" for index in range(rows)],
        "age": [str(20 + index % 50) for index in range(rows)],
        "job": ["admin." if index % 2 else "services" for index in range(rows)],
        "marital": ["single"] * rows, "education": ["secondary"] * rows,
        "default": ["no"] * rows, "balance": [str(index * 15 - 100) for index in range(rows)],
        "housing": ["yes"] * rows, "loan": ["no"] * rows, "contact": ["unknown"] * rows,
        "day": [str(1 + index % 28) for index in range(rows)], "month": ["may"] * rows,
        "duration": ["PRIVATE-DURATION-SENTINEL"] * rows, "campaign": ["1"] * rows,
        "pdays": ["-1"] * rows, "previous": ["0"] * rows, "poutcome": ["unknown"] * rows,
        "y": ["yes" if index % 4 == 0 else "no" for index in range(rows)],
    })
    return frame


@pytest.fixture
def prepared(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    train = fixture_table()
    train.loc[0, "age"] = "20 years"
    for name, frame in (("train_dirty.csv", train), ("validation.csv", fixture_table(20)),
                        ("test.csv", fixture_table(24))):
        frame.to_csv(data / name, index=False)
    # These files must never be read by any host-facing tool.
    (data / "train_reference.csv").write_text("REFERENCE-ANSWER-SENTINEL")
    (data / "corruption_manifest.json").write_text('"MANIFEST-ANSWER-SENTINEL"')
    return data, tmp_path / "runs"


def test_schema_rejects_arbitrary_actions_and_nonboolean_values():
    with pytest.raises(ValidationError):
        CleaningPlan.model_validate({**DEFAULT_PLAN, "path": "/tmp/arbitrary"})
    with pytest.raises(ValidationError):
        CleaningPlan.model_validate({**DEFAULT_PLAN, "deduplicate": "false"})
    with pytest.raises(ValidationError):
        CleaningPlan.model_validate({**DEFAULT_PLAN, "name": " "})


def test_extended_plan_fields_are_optional_and_strict():
    legacy = {key: value for key, value in DEFAULT_PLAN.items()
              if key not in {"add_missing_indicators", "add_calendar_feature", "threshold_policy", "classifier"}}
    parsed = CleaningPlan.model_validate(legacy)
    assert parsed.add_missing_indicators is False
    assert parsed.add_calendar_feature is False
    assert parsed.threshold_policy == "fixed"
    assert parsed.classifier == "logistic_regression"
    required = CleaningPlan.model_json_schema()["required"]
    assert "add_missing_indicators" not in required and "threshold_policy" not in required
    assert "classifier" not in required
    assert "add_calendar_feature" not in required
    with pytest.raises(ValidationError):
        CleaningPlan.model_validate({**legacy, "add_missing_indicators": 1})


def test_guidance_persists_and_deduplicates_effective_plans(prepared):
    data, runs = prepared
    from dqagent.data import load_table
    path = data / "train_dirty.csv"
    train = load_table(path)
    train.loc[1, "day"] = "99"
    train.to_csv(path, index=False)
    service = CleaningService(data, runs)
    started = service.start_cleaning_run(guided=True)
    run_id = started["run_id"]
    assert started["guidance"]["recommended_plan"]["outlier_policy"] == "invalid_to_missing"

    restarted = CleaningService(data, runs)
    raw = {**DEFAULT_PLAN, "numeric_parser": "strict", "outlier_policy": "keep"}
    candidate = restarted.evaluate_cleaning_plan(run_id, raw)["candidate"]
    assert candidate["decision"] == "selected"
    assert set(candidate["plan_adjustments"]) == {"numeric_parser", "outlier_policy", "threshold_policy"}
    duplicate = restarted.evaluate_cleaning_plan(run_id, {**candidate["plan"], "name": "duplicate"})
    assert duplicate["error"]["code"] == "duplicate_plan"
    assert duplicate["remaining_candidates"] == 3
    final = restarted.finalize_cleaning_run(run_id)
    assert final["status"] == "finalized"
    assert final["frozen_decision_threshold"] == candidate["decision_threshold"]


def test_fpr_objective_persists_and_ranks_recall_then_fewer_false_positives(prepared):
    data, runs = prepared
    service = CleaningService(data, runs)
    started = service.start_cleaning_run(validation_false_positive_rate=0.07)
    run_id = started["run_id"]
    assert started["selection_objective"] == selection_objective(0.07)
    assert started["guidance"]["recommended_plan"]["threshold_policy"] == "validation_macro_f1"
    assert started["max_candidates"] == 4

    def result(tn, fp, fn, tp, threshold):
        macro_f1 = (tp / (2 * tp + fp + fn)) + (tn / (2 * tn + fp + fn))
        return {"ok": True, "guardrails": {"passed": True},
                "validation": {"macro_f1": macro_f1, "minority_recall": tp / (tp + fn),
                               "confusion_matrix": [[tn, fp], [fn, tp]]},
                "decision_threshold": threshold}

    outcomes = [result(100, 0, 7, 3, 0.8), result(93, 7, 6, 4, 0.7),
                result(96, 4, 6, 4, 0.6), result(92, 8, 5, 5, 0.5)]
    assert outcomes[0]["validation"]["macro_f1"] > outcomes[1]["validation"]["macro_f1"]
    plans = [
        {**DEFAULT_PLAN, "name": "higher-f1"},
        {**DEFAULT_PLAN, "name": "higher-recall", "numeric_imputation": "mean"},
        {**DEFAULT_PLAN, "name": "fewer-false-positives", "categorical_imputation": "most_frequent"},
        {**DEFAULT_PLAN, "name": "over-limit", "class_weight": "none"},
    ]
    restarted = CleaningService(data, runs)
    assert restarted.get_run_status(run_id)["validation_false_positive_rate"] == 0.07
    with patch("dqagent.service.modeling.evaluate_plan", side_effect=outcomes) as evaluate:
        candidates = [restarted.evaluate_cleaning_plan(run_id, plan)["candidate"] for plan in plans]
    assert [item["decision"] for item in candidates] == ["selected", "selected", "selected", "rejected"]
    assert candidates[-1]["ok"] is False
    assert "误报率上限" in candidates[-1]["error"]
    assert all(call.kwargs["validation_false_positive_rate"] == 0.07 for call in evaluate.call_args_list)
    assert all(call.args[2]["threshold_policy"] == "validation_macro_f1" for call in evaluate.call_args_list)
    retained = restarted.get_run_status(run_id)
    assert retained["best_plan"]["name"] == "fewer-false-positives"
    assert retained["best_decision_threshold"] == 0.6
    assert retained["remaining_candidates"] == 0


@pytest.mark.parametrize("limit", [True, -0.01, 1.01, float("nan"), "0.07"])
def test_invalid_fpr_limit_does_not_create_a_run(prepared, limit):
    data, runs = prepared
    reply = CleaningService(data, runs).start_cleaning_run(validation_false_positive_rate=limit)
    assert reply["error"]["code"] == "invalid_fpr_limit"
    assert list(runs.glob("run_*")) == []


def test_inspection_and_candidates_never_open_held_out_or_answer_files(prepared):
    data, runs = prepared
    service = CleaningService(data, runs)
    from dqagent.service import load_table
    loaded = []
    def tracked_load(path):
        loaded.append(Path(path).name)
        return load_table(path)
    with patch("dqagent.service.load_table", side_effect=tracked_load):
        inspection = service.inspect_training_data()
        started = service.start_cleaning_run()
        strict = service.evaluate_cleaning_plan(started["run_id"], {**DEFAULT_PLAN, "numeric_parser": "strict"})
        semantic = service.evaluate_cleaning_plan(started["run_id"], DEFAULT_PLAN)
    assert inspection["ok"] and semantic["candidate"]["decision"] == "selected"
    assert strict["candidate"]["decision"] == "rejected"
    assert set(loaded) == {"train_dirty.csv", "validation.csv"}
    encoded = json.dumps([inspection, strict, semantic])
    assert "PRIVATE-DURATION-SENTINEL" not in encoded
    assert "MANIFEST-ANSWER-SENTINEL" not in encoded
    assert "REFERENCE-ANSWER-SENTINEL" not in encoded
    assert inspection["prediction_schema"]["features"] == MODEL_FEATURES
    assert inspection["execution_mode"] == "host_tool_execution"


def test_budget_and_state_survive_restart(prepared):
    data, runs = prepared
    service = CleaningService(data, runs)
    run_id = service.start_cleaning_run()["run_id"]
    for numeric in ("mean", "median"):
        for weight in ("none", "balanced"):
            plan = {**DEFAULT_PLAN, "numeric_imputation": numeric, "class_weight": weight}
            assert service.evaluate_cleaning_plan(run_id, plan)["ok"]
    restarted = CleaningService(data, runs)
    assert restarted.get_run_status(run_id)["candidate_attempts"] == 4
    assert restarted.evaluate_cleaning_plan(run_id, DEFAULT_PLAN)["error"]["code"] == "candidate_budget"


def test_duplicate_plan_is_skipped_after_service_restart(prepared):
    data, runs = prepared
    service = CleaningService(data, runs)
    run_id = service.start_cleaning_run()["run_id"]
    legacy = {key: value for key, value in DEFAULT_PLAN.items()
              if key not in {"add_missing_indicators", "threshold_policy"}}
    assert service.evaluate_cleaning_plan(run_id, legacy)["candidate"]["decision"] == "selected"

    restarted = CleaningService(data, runs)
    with patch("dqagent.service.modeling.evaluate_plan", wraps=modeling.evaluate_plan) as evaluate:
        repeat = {**DEFAULT_PLAN, "name": "renamed", "reason": "Equivalent actions."}
        repeated = restarted.evaluate_cleaning_plan(run_id, repeat)
        assert repeated["error"]["code"] == "duplicate_plan"
        assert repeated["duplicate_of_attempt"] == 1
        assert repeated["remaining_candidates"] == 3
        evaluate.assert_not_called()
        updated = restarted.evaluate_cleaning_plan(run_id, {**repeat, "threshold_policy": "validation_macro_f1"})
        assert updated["candidate"]["attempt"] == 2
        evaluate.assert_called_once()


def test_rule_search_compares_multiple_real_legal_plans_without_an_api(prepared):
    from dqagent.data import load_table
    from dqagent.engine import run_agent
    data, runs = prepared
    result = run_agent(load_table(data / "train_dirty.csv"), load_table(data / "validation.csv"), runs / "rule_search")
    assert result["mode"] == "rule_simulation" and result["model_calls"] == 0
    assert len(result["candidates"]) == 4
    assert all(candidate["ok"] for candidate in result["candidates"])
    plans = [candidate["plan"] for candidate in result["candidates"]]
    assert all(plan["numeric_parser"] == "semantic" for plan in plans)
    assert {plan["class_weight"] for plan in plans} == {"none", "balanced"}
    assert {plan["categorical_imputation"] for plan in plans} == {"unknown", "most_frequent"}
    assert {plan["numeric_imputation"] for plan in plans} == {"median", "mean"}
    assert len({plan_signature(plan) for plan in plans}) == 4
    assert result["best_validation"]["macro_f1"] == max(candidate["validation"]["macro_f1"] for candidate in result["candidates"])
    assert result["best_decision_threshold"] == result["best_validation"]["decision_threshold"]


@pytest.mark.parametrize("run_id", ["../secret", "/tmp/secret", "run_" + "f" * 23, "run_" + "F" * 24])
def test_run_id_cannot_be_used_as_a_path(prepared, run_id):
    data, runs = prepared
    assert CleaningService(data, runs).get_run_status(run_id)["error"]["code"] == "invalid_run_id"


def test_dataset_changes_block_cross_version_comparisons(prepared):
    data, runs = prepared
    service = CleaningService(data, runs)
    run_id = service.start_cleaning_run()["run_id"]
    with (data / "validation.csv").open("a") as handle:
        handle.write("\n")
    assert service.evaluate_cleaning_plan(run_id, DEFAULT_PLAN)["error"]["code"] == "dataset_changed"


def test_finalization_freezes_before_test_and_blocks_repeats_after_restart(prepared):
    data, runs = prepared
    service = CleaningService(data, runs)
    run_id = service.start_cleaning_run()["run_id"]
    assert service.evaluate_cleaning_plan(run_id, DEFAULT_PLAN)["ok"]
    from dqagent.service import load_table
    def checked_load(path):
        if Path(path).name == "test.csv":
            state = json.loads((runs / run_id / "state.json").read_text())
            assert state["status"] == "finalizing"
            assert state["frozen_plan"] == state["best_plan"]
        return load_table(path)
    with patch("dqagent.service.load_table", side_effect=checked_load):
        final = service.finalize_cleaning_run(run_id)
    assert final["status"] == "finalized"
    assert final["final_metrics"]["test_evaluations"] == 1
    assert "artifacts/model.joblib" in final["artifacts"]
    assert all(not Path(name).is_absolute() for name in final["artifacts"])
    restarted = CleaningService(data, runs)
    assert restarted.finalize_cleaning_run(run_id)["error"]["code"] == "run_frozen"
    assert restarted.evaluate_cleaning_plan(run_id, DEFAULT_PLAN)["error"]["code"] == "run_frozen"


def test_failed_finalization_remains_frozen(prepared):
    data, runs = prepared
    service = CleaningService(data, runs)
    run_id = service.start_cleaning_run()["run_id"]
    service.evaluate_cleaning_plan(run_id, DEFAULT_PLAN)
    with patch("dqagent.service.modeling.finalize_plan", side_effect=ValueError("failure")):
        assert not service.finalize_cleaning_run(run_id)["ok"]
    restarted = CleaningService(data, runs)
    assert restarted.get_run_status(run_id)["status"] == "finalization_failed"
    assert restarted.evaluate_cleaning_plan(run_id, DEFAULT_PLAN)["error"]["code"] == "run_frozen"


def test_tuned_threshold_is_persisted_and_frozen_before_test_after_restart(prepared):
    data, runs = prepared
    service = CleaningService(data, runs)
    run_id = service.start_cleaning_run(validation_false_positive_rate=0.07)["run_id"]
    tuned = {**DEFAULT_PLAN, "add_missing_indicators": True, "threshold_policy": "validation_macro_f1"}
    candidate = service.evaluate_cleaning_plan(run_id, tuned)
    assert candidate["candidate"]["decision"] == "selected"
    threshold = candidate["candidate"]["decision_threshold"]
    assert candidate["best_decision_threshold"] == threshold
    restarted = CleaningService(data, runs)
    assert restarted.get_run_status(run_id)["best_decision_threshold"] == threshold
    from dqagent.service import load_table
    def checked_load(path):
        if Path(path).name == "test.csv":
            state = json.loads((runs / run_id / "state.json").read_text())
            assert state["status"] == "finalizing"
            assert state["frozen_decision_threshold"] == threshold
            assert state["selection_objective"] == selection_objective(0.07)
            assert state["validation_false_positive_rate"] == 0.07
        return load_table(path)
    with patch("dqagent.service.load_table", side_effect=checked_load), patch(
        "dqagent.service.modeling.finalize_plan", wraps=modeling.finalize_plan
    ) as finalizer, patch(
        "dqagent.service.modeling._select_threshold", side_effect=AssertionError("Reselected a frozen threshold.")
    ):
        final = restarted.finalize_cleaning_run(run_id)
    assert final["status"] == "finalized"
    assert finalizer.call_args.kwargs["decision_threshold"] == threshold
    exported = json.loads((runs / run_id / "artifacts" / "preprocessing.json").read_text())
    assert exported["decision_threshold"] == threshold
    assert final["final_metrics"]["test"]["decision_threshold"] == threshold


def test_legacy_fixed_state_can_finalize_but_missing_tuned_threshold_cannot(prepared):
    data, runs = prepared
    service = CleaningService(data, runs)
    for policy in ("fixed", "validation_macro_f1"):
        run_id = service.start_cleaning_run()["run_id"]
        service.evaluate_cleaning_plan(run_id, {**DEFAULT_PLAN, "threshold_policy": policy})
        path = runs / run_id / "state.json"
        state = json.loads(path.read_text())
        state.pop("best_decision_threshold")
        state.pop("validation_false_positive_rate")
        state.pop("selection_objective")
        if policy == "fixed":
            state["best_plan"].pop("threshold_policy")
            state["best_plan"].pop("add_missing_indicators")
        path.write_text(json.dumps(state))
        result = CleaningService(data, runs).finalize_cleaning_run(run_id)
        if policy == "fixed":
            assert result["status"] == "finalized"
            assert result["frozen_decision_threshold"] == 0.5
            assert result["selection_objective"] == selection_objective(None)
        else:
            assert result["error"]["code"] == "missing_locked_threshold"
            assert not (runs / run_id / "finalization.claim").exists()


def test_concurrent_finalize_only_executes_local_test_once(prepared):
    data, runs = prepared
    service = CleaningService(data, runs)
    run_id = service.start_cleaning_run()["run_id"]
    service.evaluate_cleaning_plan(run_id, DEFAULT_PLAN)
    with patch("dqagent.service.modeling.finalize_plan", wraps=modeling.finalize_plan) as finalize:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(service.finalize_cleaning_run, [run_id, run_id]))
        assert finalize.call_count == 1
    assert sum(result["ok"] for result in results) == 1


def test_independent_services_cannot_race_the_last_slot_or_finalization(prepared):
    data, runs = prepared
    first, second = CleaningService(data, runs), CleaningService(data, runs)
    run_id = first.start_cleaning_run()["run_id"]
    for options in ({}, {"numeric_imputation": "mean"}, {"class_weight": "none"}):
        assert first.evaluate_cleaning_plan(run_id, {**DEFAULT_PLAN, **options})["ok"]

    started, release = Event(), Event()

    original_evaluate = modeling.evaluate_plan

    def slow_evaluate(*args, **kwargs):
        started.set()
        assert release.wait(10)
        return original_evaluate(*args, **kwargs)

    fourth = {**DEFAULT_PLAN, "numeric_imputation": "mean", "class_weight": "none"}
    fifth = {**DEFAULT_PLAN, "deduplicate": False}
    with patch("dqagent.service.modeling.evaluate_plan", side_effect=slow_evaluate) as evaluator:
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(first.evaluate_cleaning_plan, run_id, fourth)
            try:
                assert started.wait(10)
                assert second.evaluate_cleaning_plan(run_id, fifth)["error"]["code"] == "run_busy"
                assert second.finalize_cleaning_run(run_id)["error"]["code"] == "run_busy"
            finally:
                release.set()
            assert pending.result(timeout=10)["candidate"]["attempt"] == 4
        evaluator.assert_called_once()
    assert second.evaluate_cleaning_plan(run_id, fifth)["error"]["code"] == "candidate_budget"
    final = second.finalize_cleaning_run(run_id)
    assert final["status"] == "finalized"
    assert final["candidate_attempts"] == len(final["candidates"]) == 4


def _pause_before_reading_test(data, runs, run_id, test_opening):
    from dqagent.service import load_table

    def pause(path):
        if Path(path).name == "test.csv":
            test_opening.set()
            sleep(60)
        return load_table(path)

    with patch("dqagent.service.load_table", side_effect=pause):
        CleaningService(data, runs).finalize_cleaning_run(run_id)


def test_killed_finalizer_releases_process_lock_but_cannot_repeat_test(prepared):
    data, runs = prepared
    service = CleaningService(data, runs)
    run_id = service.start_cleaning_run()["run_id"]
    assert service.evaluate_cleaning_plan(run_id, DEFAULT_PLAN)["ok"]
    context = multiprocessing.get_context("spawn")
    test_opening = context.Event()
    child = context.Process(target=_pause_before_reading_test, args=(data, runs, run_id, test_opening))
    child.start()
    try:
        assert test_opening.wait(15)
        assert service.get_run_status(run_id)["error"]["code"] == "run_busy"
    finally:
        child.terminate()
        child.join(10)
    assert not child.is_alive()
    assert service.get_run_status(run_id)["status"] == "finalization_interrupted"
    assert service.finalize_cleaning_run(run_id)["error"]["code"] == "run_frozen"


def test_stdio_protocol_initialize_tools_prompt_and_actual_workflow(prepared):
    data, runs = prepared
    root = Path(__file__).resolve().parents[1]
    async def exercise():
        params = StdioServerParameters(command=sys.executable,
            args=["-m", "dqagent.mcp_server", "--data-dir", str(data), "--runs-dir", str(runs)], cwd=root)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write, read_timeout_seconds=30) as client:
                initialized = await client.initialize()
                assert initialized.server_info.name == "data-quality-agent"
                listed = await client.list_tools()
                assert {tool.name for tool in listed.tools} == {
                    "inspect_training_data", "start_cleaning_run", "evaluate_cleaning_plan",
                    "get_run_status", "finalize_cleaning_run",
                }
                plan_tool = next(tool for tool in listed.tools if tool.name == "evaluate_cleaning_plan")
                schema = json.dumps(plan_tool.input_schema)
                assert "numeric_parser" in schema and "additionalProperties" in schema
                typed_plan = next(value for value in plan_tool.input_schema["$defs"].values()
                                  if "numeric_parser" in value.get("properties", {}))
                assert "add_missing_indicators" not in typed_plan["required"]
                assert "threshold_policy" not in typed_plan["required"]
                assert "classifier" not in typed_plan["required"]
                start_tool = next(tool for tool in listed.tools if tool.name == "start_cleaning_run")
                assert "validation_false_positive_rate" in start_tool.input_schema["properties"]
                assert "validation_false_positive_rate" not in start_tool.input_schema.get("required", [])
                prompt = await client.get_prompt("cleaning_workflow")
                assert "finalize_cleaning_run" in prompt.messages[0].content.text
                async def call(name, arguments=None):
                    response = await client.call_tool(name, arguments or {})
                    assert not response.is_error, response
                    assert isinstance(response.structured_content, dict)
                    return response.structured_content
                inspection = await call("inspect_training_data")
                assert inspection["training_profile"]["feature_count"] == 15
                old_run = await call("start_cleaning_run")
                old_plan = {key: value for key, value in DEFAULT_PLAN.items()
                            if key not in {"add_missing_indicators", "threshold_policy", "classifier"}}
                old_candidate = await call("evaluate_cleaning_plan", {"run_id": old_run["run_id"], "plan": old_plan})
                assert old_candidate["candidate"]["plan"]["threshold_policy"] == "fixed"
                assert old_candidate["candidate"]["plan"]["add_missing_indicators"] is False
                started = await call("start_cleaning_run", {"validation_false_positive_rate": 0.07})
                assert started["guidance"]["recommended_plan"]["numeric_parser"] == "semantic"
                assert started["selection_objective"] == selection_objective(0.07)
                run_id = started["run_id"]
                tuned = {**DEFAULT_PLAN, "classifier": "hist_gradient_boosting", "add_missing_indicators": True}
                candidate = await call("evaluate_cleaning_plan", {"run_id": run_id, "plan": tuned})
                assert candidate["candidate"]["decision"] == "selected"
                assert candidate["candidate"]["plan"]["classifier"] == "hist_gradient_boosting"
                assert candidate["candidate"]["plan_adjustments"]["threshold_policy"]["effective"] == "validation_macro_f1"
                assert candidate["candidate"]["threshold_selection"]["validation_false_positive_rate"] <= 0.07
                threshold = candidate["candidate"]["decision_threshold"]
                repeat = await call("evaluate_cleaning_plan", {"run_id": run_id, "plan": {**tuned, "name": "retry"}})
                assert repeat["error"]["code"] == "duplicate_plan"
                assert repeat["remaining_candidates"] == 3
                final = await call("finalize_cleaning_run", {"run_id": run_id})
                assert final["status"] == "finalized"
                assert final["frozen_decision_threshold"] == threshold
                assert final["validation_false_positive_rate"] == 0.07
                exported = json.loads((runs / run_id / "artifacts" / "preprocessing.json").read_text())
                assert exported["decision_threshold"] == threshold
                assert CleaningService(data, runs).get_run_status(run_id)["frozen_decision_threshold"] == threshold
                status = await call("get_run_status", {"run_id": run_id})
                assert status["internal_model_calls"] == 0
                assert status["candidate_attempts"] == 1
                rejected = await call("evaluate_cleaning_plan", {"run_id": run_id, "plan": DEFAULT_PLAN})
                assert rejected["error"]["code"] == "run_frozen"
    anyio.run(exercise)
