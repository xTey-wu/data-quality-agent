"""Orchestration and provider security checks; these tests never call an API."""

import copy
import io
import json
import tempfile
import types
import unittest
import urllib.error
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import dqagent
from dqagent.engine import _usage_summary, run_agent
from dqagent.providers import ChatCompletionsProvider, MockProvider, ProviderError, ProviderResponse


def plan(name="candidate", parser="semantic"):
    return {
        "name": name, "numeric_parser": parser, "numeric_imputation": "median",
        "categorical_imputation": "unknown", "outlier_policy": "invalid_to_missing",
        "deduplicate": True, "class_weight": "none", "reason": "检查受控方案。",
    }


def tool_call(call_id, candidate=None, name="evaluate_plan", arguments=None):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": arguments if arguments is not None else json.dumps({"plan": candidate or plan()}),
    }}


def evaluation(train, val, candidate, seed=42):
    del train, val, seed
    if candidate["numeric_parser"] == "strict":
        return {"ok": False, "plan": candidate, "error": "numeric_parse_failed"}
    passed = candidate["outlier_policy"] != "iqr_clip"
    return {
        "ok": True, "plan": candidate, "validation": {"macro_f1": 0.71},
        "quality_before": {"rows": 10}, "quality_after": {"rows": 10},
        "guardrails": {"passed": passed, "reasons": [] if passed else ["excessive_clipping"]},
        "changes": {}, "error": None,
    }


@contextmanager
def fake_tools(evaluator=evaluation):
    quality = types.SimpleNamespace(profile=lambda frame: {"rows": 10}, validate_plan=lambda candidate: candidate)
    modeling = types.SimpleNamespace(evaluate_plan=Mock(side_effect=evaluator))
    with patch.object(dqagent, "quality", quality, create=True), patch.object(dqagent, "modeling", modeling, create=True):
        yield modeling.evaluate_plan


class ScriptedProvider(MockProvider):
    def __init__(self, responses):
        super().__init__()
        self.responses = iter(responses)
        self.received = []

    def complete(self, messages, tools):
        self.received.append(copy.deepcopy(messages))
        return ProviderResponse(next(self.responses))


class AgentTests(unittest.TestCase):
    def test_offline_mock_repairs_strict_error_and_rolls_back_bad_candidate(self):
        with tempfile.TemporaryDirectory() as directory, fake_tools():
            summary = run_agent(object(), object(), directory)
            events = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text().splitlines()]
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["mode"], "rule_simulation")
            self.assertFalse(summary["is_real_model"])
            self.assertEqual(summary["real_model_validation"], "not_run")
            self.assertEqual(summary["selected_plan"]["name"], "semantic_repair")
            self.assertEqual(summary["candidates"][0]["decision"], "rejected")
            self.assertTrue(any(entry["event"] == "rollback" for entry in events))
            self.assertIsNone(summary["usage"]["estimated_cost"])
            self.assertEqual(summary["model_calls"], 0)

    def test_every_tool_call_gets_a_response_and_assistant_fields_survive(self):
        first = {"role": "assistant", "content": None, "reasoning_content": "provider-specific state", "tool_calls": [
            tool_call("unknown", name="read_file"),
            tool_call("invalid", arguments=json.dumps({"plan": {**plan(), "path": "/tmp/anything"}})),
            tool_call("valid", plan("valid")),
        ]}
        provider = ScriptedProvider([first, {"role": "assistant", "content": "done"}])
        with tempfile.TemporaryDirectory() as directory, fake_tools() as evaluate:
            summary = run_agent(object(), object(), directory, provider=provider)
            evaluate.assert_called_once()
            second_messages = provider.received[1]
            responses = [message for message in second_messages if message["role"] == "tool"]
            self.assertEqual([message["tool_call_id"] for message in responses], ["unknown", "invalid", "valid"])
            self.assertEqual(json.loads(responses[0]["content"])["error"]["code"], "unsupported_tool")
            self.assertEqual(json.loads(responses[1]["content"])["error"]["code"], "invalid_arguments")
            self.assertEqual(second_messages[2]["reasoning_content"], "provider-specific state")
            self.assertEqual(summary["status"], "completed")

    def test_candidate_budget_still_responds_to_all_calls(self):
        provider = ScriptedProvider([{"role": "assistant", "content": None, "tool_calls": [
            tool_call(f"call_{index}", plan(f"candidate_{index}")) for index in range(6)
        ]}])
        with tempfile.TemporaryDirectory() as directory, fake_tools() as evaluate:
            summary = run_agent(object(), object(), directory, provider=provider, max_candidates=2)
            events = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text().splitlines()]
            responses = [entry for entry in events if entry["event"] == "tool_result"]
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(len(responses), 6)
            self.assertIn(responses[-1]["result"]["error"]["code"], {"candidate_budget", "tool_budget"})
            self.assertEqual(summary["termination"], "candidate_budget")

    def test_no_valid_plan_is_failure_and_clears_stale_selected_plan(self):
        provider = ScriptedProvider([{"role": "assistant", "content": "I cannot select a plan."}])
        with tempfile.TemporaryDirectory() as directory, fake_tools():
            selected = Path(directory) / "selected_plan.json"
            selected.write_text('{"name":"old"}')
            summary = run_agent(object(), object(), directory, provider=provider)
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(json.loads(selected.read_text()))

    def test_invalid_candidate_attempts_consume_the_candidate_budget(self):
        provider = ScriptedProvider([{"role": "assistant", "content": None, "tool_calls": [
            tool_call(f"invalid_{index}", arguments='{"plan": {}}') for index in range(8)
        ]}])
        with tempfile.TemporaryDirectory() as directory, fake_tools() as evaluate:
            summary = run_agent(object(), object(), directory, provider=provider)
            evaluate.assert_not_called()
            self.assertEqual(summary["status"], "failed")
            self.assertEqual(summary["termination"], "candidate_budget")
            self.assertEqual(summary["budgets"]["candidate_attempts"], 4)
            self.assertEqual(summary["budgets"]["invalid_tool_calls"], 4)

    def test_unknown_tools_are_bounded_but_all_receive_responses(self):
        provider = ScriptedProvider([{"role": "assistant", "content": None, "tool_calls": [
            tool_call(f"unknown_{index}", name="read_file") for index in range(20)
        ]}])
        with tempfile.TemporaryDirectory() as directory, fake_tools() as evaluate:
            summary = run_agent(object(), object(), directory, provider=provider)
            events = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text().splitlines()]
            responses = [entry for entry in events if entry["event"] == "tool_result"]
            evaluate.assert_not_called()
            self.assertEqual(summary["termination"], "tool_budget")
            self.assertEqual(summary["budgets"]["tool_actions_processed"], 8)
            self.assertEqual(len(responses), 20)
            self.assertEqual(responses[-1]["result"]["error"]["code"], "tool_budget")

    def test_provider_failure_after_a_valid_candidate_is_partial(self):
        class FailAfterCandidate(ScriptedProvider):
            def describe(self):
                return {**super().describe(), "provider": "qwen", "model": "test-model",
                        "mode": "llm_api", "is_real_model": True}

            def complete(self, messages, tools):
                if self.received:
                    raise ProviderError("http_error", "模型接口返回 HTTP 503；响应正文未记录。")
                return super().complete(messages, tools)

        provider = FailAfterCandidate([{"role": "assistant", "content": None, "tool_calls": [tool_call("valid")]}])
        with tempfile.TemporaryDirectory() as directory, fake_tools():
            summary = run_agent(object(), object(), directory, provider=provider)
            self.assertEqual(summary["status"], "partial")
            self.assertEqual(summary["real_model_validation"], "failed_or_partial")
            self.assertIsNotNone(summary["selected_plan"])

    def test_planner_never_receives_unavailable_feature_statistics(self):
        hidden = "UNAVAILABLE-CALL-DURATION-SENTINEL"
        profile = {"rows": 10, "fields": {
            "age": {"kind": "numeric", "missing": 1},
            "duration": {"range": [hidden, hidden], "missing": 999},
        }, "preserved_fields": {"duration": hidden}, "semantics": {"duration": hidden}}
        provider = ScriptedProvider([
            {"role": "assistant", "content": None, "tool_calls": [tool_call("valid")]},
            {"role": "assistant", "content": "done"},
        ])
        def result_with_hidden_statistics(*args, **kwargs):
            result = evaluation(*args, **kwargs)
            result["quality_before"] = result["quality_after"] = profile
            result["changes"] = {"clipped_by_field": {"duration": hidden, "age": 1}}
            return result
        with tempfile.TemporaryDirectory() as directory, fake_tools(result_with_hidden_statistics):
            dqagent.quality.profile = lambda frame: profile
            run_agent(object(), object(), directory, provider=provider)
            requests = json.dumps(provider.received)
            self.assertNotIn(hidden, requests)
            self.assertNotIn('"duration"', requests)
            payload = json.loads(provider.received[0][1]["content"])
            self.assertEqual(len(payload["prediction_schema"]["features"]), 15)
            self.assertEqual(payload["training_profile"]["missing_cells"], 1)


class ProviderTests(unittest.TestCase):
    def config(self):
        return {"provider": "qwen", "model": "qwen3.7-plus", "api_key_env": "DQAGENT_TEST_KEY", "max_retries": 0}

    def test_missing_key_does_not_send_http(self):
        with patch.dict("os.environ", {"DQAGENT_TEST_KEY": ""}):
            provider = ChatCompletionsProvider(self.config(), allow_network=True)
            provider._opener.open = Mock()
            with self.assertRaises(ProviderError) as raised:
                provider.complete([], [])
            self.assertEqual(raised.exception.code, "missing_api_key")
            provider._opener.open.assert_not_called()

    def test_generation_parameters_are_bounded_and_nonthinking(self):
        invalid = [
            {"max_tokens": 0}, {"max_tokens": 8193}, {"max_tokens": True}, {"max_tokens": 1.5},
            {"temperature": True}, {"temperature": float("nan")}, {"temperature": 2.1},
            {"temperature": -0.1}, {"top_p": float("inf")}, {"top_p": True}, {"top_p": 1.1},
        ]
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ProviderError):
                ChatCompletionsProvider({**self.config(), "parameters": parameters})
        provider = ChatCompletionsProvider(self.config())
        self.assertEqual(provider.parameters["max_tokens"], 4096)
        self.assertIs(provider.extra_body["enable_thinking"], False)
        for extra in [{"enable_thinking": True}, {"enable_thinking": 0}]:
            with self.assertRaises(ProviderError):
                ChatCompletionsProvider({**self.config(), "extra_body": extra})
        deepseek = {**self.config(), "provider": "deepseek", "model": "deepseek-flash"}
        self.assertEqual(ChatCompletionsProvider(deepseek).extra_body, {"thinking": {"type": "disabled"}})
        for extra in [{"thinking": {"type": "enabled"}}, {"reasoning_effort": "high"}]:
            with self.assertRaises(ProviderError):
                ChatCompletionsProvider({**deepseek, "extra_body": extra})

    def test_network_requires_explicit_opt_in_even_with_key(self):
        with patch.dict("os.environ", {"DQAGENT_TEST_KEY": "not-a-real-key"}):
            provider = ChatCompletionsProvider(self.config())
            provider._opener.open = Mock()
            with self.assertRaises(ProviderError) as raised:
                provider.complete([], [])
            self.assertEqual(raised.exception.code, "live_calls_disabled")
            provider._opener.open.assert_not_called()

    def test_disabled_live_call_is_not_counted_as_a_model_request(self):
        provider = ChatCompletionsProvider(self.config())
        with tempfile.TemporaryDirectory() as directory, fake_tools():
            summary = run_agent(object(), object(), directory, provider=provider)
            self.assertEqual(summary["model_calls"], 0)
            self.assertEqual(summary["provider_calls"], 1)
            self.assertEqual(summary["real_model_validation"], "not_run")

    def test_rejects_foreign_or_credential_bearing_endpoints(self):
        for endpoint in ["https://evil.example/v1", "http://dashscope.aliyuncs.com/v1",
                         "https://dashscope.aliyuncs.com.evil.example/v1",
                         "https://user:password@dashscope.aliyuncs.com/v1"]:
            with self.subTest(endpoint=endpoint), self.assertRaises(ProviderError):
                ChatCompletionsProvider({**self.config(), "base_url": endpoint})

    def test_accepts_only_single_label_beijing_workspace_endpoints(self):
        accepted = "https://workspace-123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        provider = ChatCompletionsProvider({**self.config(), "base_url": accepted})
        self.assertEqual(provider.base_url, accepted)
        bad_hosts = [
            "workspace.cn-beijing.maas.aliyuncs.com.evil.example",
            "nested.workspace.cn-beijing.maas.aliyuncs.com",
            "cn-beijing.maas.aliyuncs.com",
            "-workspace.cn-beijing.maas.aliyuncs.com",
            "workspace-.cn-beijing.maas.aliyuncs.com",
            "work_space.cn-beijing.maas.aliyuncs.com",
            "workspace.cn-shanghai.maas.aliyuncs.com",
            "a" * 64 + ".cn-beijing.maas.aliyuncs.com",
            "workspace.cn-beijing.maas.aliyuncs.com:8443",
        ]
        for host in bad_hosts:
            with self.subTest(host=host), self.assertRaises(ProviderError):
                ChatCompletionsProvider({**self.config(), "base_url": "https://" + host + "/compatible-mode/v1"})

    def test_http_error_body_and_secret_never_reach_run_files(self):
        secret = "TEST-ONLY-SECRET-MUST-NOT-BE-LOGGED"
        with patch.dict("os.environ", {"DQAGENT_TEST_KEY": secret}):
            provider = ChatCompletionsProvider(self.config(), allow_network=True)
            provider._opener.open = Mock(side_effect=urllib.error.HTTPError(
                "https://dashscope.aliyuncs.com", 401, secret, {}, io.BytesIO(secret.encode()),
            ))
            with tempfile.TemporaryDirectory() as directory, fake_tools():
                summary = run_agent(object(), object(), directory, provider=provider)
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["provider_error"]["code"], "http_error")
                for path in Path(directory).iterdir():
                    self.assertNotIn(secret, path.read_text())

    def test_padded_key_is_stripped_for_sending_and_both_forms_redacted(self):
        normalized = "FICTIONAL-ONLY-STRIPPED-KEY"
        padded = "  " + normalized + " \n"
        with patch.dict("os.environ", {"DQAGENT_TEST_KEY": padded}):
            provider = ChatCompletionsProvider(self.config(), allow_network=True)
            document = {"choices": [{"message": {"role": "assistant", "content": normalized}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
            provider._opener.open = Mock(return_value=io.BytesIO(json.dumps(document).encode()))
            with tempfile.TemporaryDirectory() as directory, fake_tools():
                run_agent(object(), object(), directory, provider=provider)
                request = provider._opener.open.call_args.args[0]
                self.assertEqual(request.get_header("Authorization"), "Bearer " + normalized)
                self.assertNotIn(normalized, json.dumps(provider.redact({"raw": padded, "normalized": normalized})))
                for path in Path(directory).iterdir():
                    self.assertNotIn(normalized, path.read_text())

    def test_retries_only_report_known_response_cost(self):
        records = [{"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500}]
        info = {"is_real_model": True, "pricing": {"input_per_million": 2, "output_per_million": 8, "currency": "CNY"}}
        retried = _usage_summary(records, info, retries=1)
        self.assertAlmostEqual(retried["known_response_cost"], 0.006)
        self.assertIsNone(retried["estimated_cost"])
        self.assertIn("失败请求", retried["cost_note"])
        completed = _usage_summary(records, info, retries=0)
        self.assertAlmostEqual(completed["estimated_cost"], 0.006)

    def test_complete_assistant_preserved_and_unknown_cost_remains_null(self):
        provider = ChatCompletionsProvider(self.config())
        response = provider._validate_response({
            "model": "qwen3.7-plus-2026-05-26",
            "choices": [{"message": {"role": "assistant", "content": None,
                "reasoning_content": "state", "tool_calls": [tool_call("a")]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }, 0)
        self.assertEqual(response.message["reasoning_content"], "state")
        self.assertEqual(response.usage["returned_model"], "qwen3.7-plus-2026-05-26")
        self.assertIsNone(provider.describe()["pricing"]["input_per_million"])

    def test_retry_is_bounded_and_does_not_log_body(self):
        config = {**self.config(), "max_retries": 1}
        with patch.dict("os.environ", {"DQAGENT_TEST_KEY": "test-only"}):
            provider = ChatCompletionsProvider(config, allow_network=True)
            provider._opener.open = Mock(side_effect=lambda *args, **kwargs: (_ for _ in ()).throw(
                urllib.error.HTTPError("https://dashscope.aliyuncs.com", 429, "private", {}, io.BytesIO(b"private"))
            ))
            with patch("dqagent.providers.time.sleep"), self.assertRaises(ProviderError) as raised:
                provider.complete([], [])
            self.assertEqual(raised.exception.retries, 1)
            self.assertEqual(provider._opener.open.call_count, 2)
            self.assertNotIn("private", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
