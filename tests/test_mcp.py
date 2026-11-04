"""Persistent service and real stdio MCP protocol tests without an LLM API."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import anyio
import pandas as pd
import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import ValidationError

from dqagent import modeling
from dqagent.schema import DEFAULT_PLAN, MODEL_FEATURES
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
    for index in range(4):
        result = service.evaluate_cleaning_plan(run_id, {**DEFAULT_PLAN, "name": f"candidate-{index}"})
        assert result["ok"]
    restarted = CleaningService(data, runs)
    assert restarted.get_run_status(run_id)["candidate_attempts"] == 4
    assert restarted.evaluate_cleaning_plan(run_id, DEFAULT_PLAN)["error"]["code"] == "candidate_budget"


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
                prompt = await client.get_prompt("cleaning_workflow")
                assert "finalize_cleaning_run" in prompt.messages[0].content.text
                async def call(name, arguments=None):
                    response = await client.call_tool(name, arguments or {})
                    assert not response.is_error, response
                    assert isinstance(response.structured_content, dict)
                    return response.structured_content
                inspection = await call("inspect_training_data")
                assert inspection["training_profile"]["feature_count"] == 15
                started = await call("start_cleaning_run")
                run_id = started["run_id"]
                candidate = await call("evaluate_cleaning_plan", {"run_id": run_id, "plan": DEFAULT_PLAN})
                assert candidate["candidate"]["decision"] == "selected"
                final = await call("finalize_cleaning_run", {"run_id": run_id})
                assert final["status"] == "finalized"
                status = await call("get_run_status", {"run_id": run_id})
                assert status["internal_model_calls"] == 0
                assert status["candidate_attempts"] == 1
                rejected = await call("evaluate_cleaning_plan", {"run_id": run_id, "plan": DEFAULT_PLAN})
                assert rejected["error"]["code"] == "run_frozen"
    anyio.run(exercise)
