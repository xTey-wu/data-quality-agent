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
from dqagent.engine import PLAN_SCHEMA, _usage_summary, _validate_arguments, run_agent
from dqagent.providers import ChatCompletionsProvider, MockProvider, ProviderError, ProviderResponse
from dqagent.schema import plan_signature


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
        "decision_threshold": 0.42 if candidate.get("threshold_policy") == "validation_macro_f1" else 0.5,
        "quality_before": {"rows": 10}, "quality_after": {"rows": 10},
        "guardrails": {"passed": passed, "reasons": [] if passed else ["excessive_clipping"]},
        "changes": {}, "error": None,
    }


@contextmanager
def fake_tools(evaluator=evaluation, training_profile=None):
    quality = types.SimpleNamespace(profile=lambda frame: training_profile or {"rows": 10},
                                    validate_plan=lambda candidate: candidate)
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
    def test_offline_mock_repairs_strict_error_and_compares_valid_candidates(self):
        with tempfile.TemporaryDirectory() as directory, fake_tools():
            summary = run_agent(object(), object(), directory)
            events = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text().splitlines()]
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["mode"], "rule_simulation")
            self.assertFalse(summary["is_real_model"])
            self.assertEqual(summary["real_model_validation"], "not_run")
            self.assertEqual(summary["selected_plan"]["numeric_parser"], "semantic")
            self.assertGreaterEqual(sum(candidate["ok"] for candidate in summary["candidates"]), 2)
            self.assertEqual(summary["best_decision_threshold"], 0.42)
            self.assertEqual(summary["candidates"][0]["decision"], "rejected")
            self.assertTrue(any(entry["event"] == "rollback" for entry in events))
            self.assertIsNone(summary["usage"]["estimated_cost"])
            self.assertEqual(summary["model_calls"], 0)

    def test_legacy_and_extended_plan_arguments_are_compatible(self):
        legacy = _validate_arguments(json.dumps({"plan": plan()}))
        self.assertFalse(legacy["add_missing_indicators"])
        self.assertFalse(legacy["add_calendar_feature"])
        self.assertEqual(legacy["threshold_policy"], "fixed")
        self.assertEqual(legacy["classifier"], "logistic_regression")
        self.assertNotIn("add_missing_indicators", PLAN_SCHEMA["required"])
        self.assertNotIn("threshold_policy", PLAN_SCHEMA["required"])
        self.assertNotIn("classifier", PLAN_SCHEMA["required"])
        self.assertNotIn("add_calendar_feature", PLAN_SCHEMA["required"])
        extended = _validate_arguments(json.dumps({"plan": {**plan(), "add_missing_indicators": True,
                                                           "threshold_policy": "validation_macro_f1"}}))
        self.assertTrue(extended["add_missing_indicators"])
        with self.assertRaises(ValueError):
            _validate_arguments(json.dumps({"plan": {**plan(), "add_missing_indicators": 1}}))

    def test_guided_search_recovers_tool_call_and_compares_effective_candidates(self):
        conflicted = {**plan("conflicted", parser="strict"), "outlier_policy": "keep"}
        alternate = {**plan("tree"), "classifier": "hist_gradient_boosting"}
        provider = ScriptedProvider([
            {"role": "assistant", "content": "先评估候选。"},
            {"role": "assistant", "tool_calls": [tool_call("first", conflicted)]},
            {"role": "assistant", "tool_calls": [tool_call("duplicate", plan("duplicate")), tool_call("tree", alternate)]},
        ])
        training = {"rows": 10, "fields": {"age": {"strict_parse_errors": 2, "invalid_range": 1}}}
        with tempfile.TemporaryDirectory() as directory, fake_tools(training_profile=training) as evaluate:
            summary = run_agent(object(), object(), directory, provider=provider, guided=True, max_candidates=2)
            self.assertTrue(summary["tool_call_reminder_sent"])
            self.assertIn("evaluate_plan", provider.received[1][-1]["content"])
            self.assertEqual(summary["budgets"]["tool_calls_received"], 3)
            self.assertEqual(evaluate.call_count, 2)
            first, second = summary["candidates"]
            self.assertEqual(set(first["plan_adjustments"]), {"numeric_parser", "outlier_policy", "threshold_policy"})
            self.assertEqual(first["plan"]["outlier_policy"], "invalid_to_missing")
            self.assertEqual(second["plan"]["classifier"], "hist_gradient_boosting")
            self.assertEqual(summary["best_decision_threshold"], 0.42)

    def test_capped_search_prefers_recall_and_rejects_ineligible_validation_results(self):
        trials = [plan("a"), {**plan("b"), "class_weight": "balanced"},
                  {**plan("c"), "add_calendar_feature": True}, {**plan("d"), "numeric_imputation": "mean"}]
        provider = ScriptedProvider([{"role": "assistant", "tool_calls": [
            tool_call(candidate["name"], candidate) for candidate in trials]}])
        values = iter([(0.9, 3, 0), (0.7, 4, 7), (0.7, 4, 4), (0.9, 9, 8)])

        def capped(train, val, candidate, seed=42, *, validation_false_positive_rate=None):
            self.assertEqual(validation_false_positive_rate, 0.07)
            self.assertEqual(candidate["threshold_policy"], "validation_macro_f1")
            result = evaluation(train, val, candidate, seed)
            macro, tp, fp = next(values)
            result["validation"] = {"macro_f1": macro, "confusion_matrix": [[100 - fp, fp], [10 - tp, tp]]}
            return result

        with tempfile.TemporaryDirectory() as directory, fake_tools(capped) as evaluate:
            summary = run_agent(object(), object(), directory, provider=provider, validation_false_positive_rate=0.07)
            self.assertEqual(evaluate.call_count, 4)
            self.assertEqual(summary["selection_objective"]["metric"], "minority_recall")
            self.assertEqual(summary["selected_plan"]["name"], "c")
            self.assertEqual([c["decision"] for c in summary["candidates"]],
                             ["selected", "selected", "selected", "rejected"])

    def test_tuned_plan_without_validation_threshold_is_rejected(self):
        tuned = {**plan(), "threshold_policy": "validation_macro_f1"}
        provider = ScriptedProvider([
            {"role": "assistant", "content": None, "tool_calls": [tool_call("tuned", tuned)]},
            {"role": "assistant", "content": "done"},
        ])
        def missing_threshold(*args, **kwargs):
            result = evaluation(*args, **kwargs)
            result.pop("decision_threshold")
            return result
        with tempfile.TemporaryDirectory() as directory, fake_tools(missing_threshold):
            summary = run_agent(object(), object(), directory, provider=provider)
            self.assertEqual(summary["status"], "failed")
            self.assertIsNone(summary["best_decision_threshold"])

    def test_unsafe_iqr_candidate_rolls_back_without_discarding_best(self):
        provider = ScriptedProvider([
            {"role": "assistant", "content": None, "tool_calls": [tool_call("safe", plan("safe"))]},
            {"role": "assistant", "content": None, "tool_calls": [tool_call("unsafe", {**plan("unsafe"), "outlier_policy": "iqr_clip"})]},
            {"role": "assistant", "content": "done"},
        ])
        with tempfile.TemporaryDirectory() as directory, fake_tools():
            summary = run_agent(object(), object(), directory, provider=provider)
            self.assertEqual(summary["selected_plan"]["name"], "safe")
            self.assertEqual(summary["candidates"][1]["decision"], "rejected")
            self.assertIn('"event": "rollback"', (Path(directory) / "trace.jsonl").read_text())

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
            tool_call(f"call_{index}", {**plan(f"candidate_{index}"), "deduplicate": index % 2 == 0})
            for index in range(6)
        ]}])
        with tempfile.TemporaryDirectory() as directory, fake_tools() as evaluate:
            summary = run_agent(object(), object(), directory, provider=provider, max_candidates=2)
            events = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text().splitlines()]
            responses = [entry for entry in events if entry["event"] == "tool_result"]
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(len(responses), 6)
            self.assertIn(responses[-1]["result"]["error"]["code"], {"candidate_budget", "tool_budget"})
            self.assertEqual(summary["termination"], "candidate_budget")

    def test_duplicate_plan_consumes_only_a_tool_action(self):
        first = plan("first")
        repeat = {**first, "name": "renamed", "reason": "相同参数，另一种说明。",
                  "add_missing_indicators": False, "threshold_policy": "fixed"}
        changed = {**repeat, "threshold_policy": "validation_macro_f1"}
        provider = ScriptedProvider([{"role": "assistant", "content": None, "tool_calls": [
            tool_call("first", first), tool_call("repeat", repeat), tool_call("changed", changed),
        ]}])
        with tempfile.TemporaryDirectory() as directory, fake_tools() as evaluate:
            summary = run_agent(object(), object(), directory, provider=provider, max_candidates=2)
            events = [json.loads(line) for line in (Path(directory) / "trace.jsonl").read_text().splitlines()]
            repeated = next(event["result"] for event in events if event.get("tool_call_id") == "repeat")
            self.assertEqual(repeated["error"]["code"], "duplicate_plan")
            self.assertEqual(repeated["duplicate_of_attempt"], 1)
            self.assertEqual(repeated["remaining_candidates"], 1)
            self.assertEqual(evaluate.call_count, 2)
            self.assertEqual(summary["budgets"]["candidate_attempts"], 2)
            self.assertEqual(summary["budgets"]["tool_actions_processed"], 3)

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

    def mock_first(self, profile=None):
        provider = MockProvider()
        messages = [] if profile is None else [{"role": "user", "content": json.dumps({"training_profile": profile})}]
        response = provider.complete(messages, [])
        candidate = json.loads(response.message["tool_calls"][0]["function"]["arguments"])["plan"]
        return provider, messages, response, candidate

    def mock_feedback(self, provider, messages, response, candidate, score=0.7, retained=None, error=None):
        result = {"ok": error is None, "plan": candidate, "validation": {"macro_f1": score} if error is None else None,
                  "guardrails": {"passed": error is None}, "error": error,
                  "retained_plan": retained or candidate["name"], "agent_decision": "selected" if error is None else "rejected"}
        messages.extend([response.message, {"role": "tool", "tool_call_id": response.message["tool_calls"][0]["id"],
                                           "content": json.dumps(result)}])
        new_response = provider.complete(messages, [])
        candidate = json.loads(new_response.message["tool_calls"][0]["function"]["arguments"])["plan"]
        return new_response, candidate

    def test_mock_uses_known_format_errors_without_wasting_strict_attempt(self):
        _, _, _, formatted = self.mock_first({"rows": 100, "numeric_parse_errors": 3, "numeric_range_violations": 2})
        _, _, _, clean = self.mock_first({"rows": 100, "numeric_parse_errors": 0, "numeric_range_violations": 0})
        self.assertEqual(formatted["numeric_parser"], "semantic")
        self.assertEqual(clean["numeric_parser"], "strict")
        self.assertEqual(formatted["outlier_policy"], "invalid_to_missing")
        self.assertEqual(clean["outlier_policy"], "keep")
        self.assertTrue(formatted["add_missing_indicators"])
        self.assertEqual(formatted["threshold_policy"], "validation_macro_f1")

    def test_mock_error_feedback_changes_parser_not_only_reason(self):
        provider, messages, response, first = self.mock_first()
        self.assertEqual(first["numeric_parser"], "strict")
        _, corrected = self.mock_feedback(provider, messages, response, first, error="numeric_parse_failed")
        self.assertEqual(corrected["numeric_parser"], "semantic")

    def test_mock_balanced_candidate_is_separate_from_outlier_clipping(self):
        provider, messages, response, first = self.mock_first({"numeric_parse_errors": 1, "numeric_range_violations": 1})
        _, second = self.mock_feedback(provider, messages, response, first)
        self.assertEqual(second["class_weight"], "balanced")
        self.assertEqual(second["outlier_policy"], first["outlier_policy"])
        self.assertNotEqual(second["outlier_policy"], "iqr_clip")

    def test_mock_retained_feedback_controls_later_categorical_and_numeric_candidates(self):
        branches = []
        for balanced_wins in (False, True):
            provider, messages, response, first = self.mock_first({"numeric_parse_errors": 1})
            second_response, second = self.mock_feedback(provider, messages, response, first, score=0.7)
            third_response, third = self.mock_feedback(provider, messages, second_response, second,
                score=0.8 if balanced_wins else 0.6, retained=second["name"] if balanced_wins else first["name"])
            self.assertEqual(third["class_weight"], "balanced" if balanced_wins else "none")
            self.assertEqual(third["categorical_imputation"], "most_frequent")
            fourth_response, fourth = self.mock_feedback(provider, messages, third_response, third,
                score=0.9, retained=third["name"])
            self.assertEqual(fourth["numeric_imputation"], "mean")
            self.assertEqual(fourth["categorical_imputation"], third["categorical_imputation"])
            signatures = {plan_signature(candidate) for candidate in (first, second, third, fourth)}
            self.assertEqual(len(signatures), 4)
            branches.append(third)
        self.assertNotEqual(branches[0]["class_weight"], branches[1]["class_weight"])

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
