"""Offline simulation and opt-in official Chat Completions providers.

Constructing a provider never performs a network request. Real calls require
``allow_network=True`` as well as an API key in the configured environment variable.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .objective import validate_fpr_limit, validation_rank
from .schema import OPTIONAL_PLAN_DEFAULTS, plan_signature


class ProviderError(RuntimeError):
    """An intentionally sanitized error: never include HTTP response bodies."""

    def __init__(self, code: str, message: str, retries: int = 0):
        super().__init__(message)
        self.code = code
        self.retries = retries


@dataclass
class ProviderResponse:
    message: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)
    retries: int = 0


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


class MockProvider:
    """Deterministic rule search driven by the visible profile and tool feedback."""

    def __init__(self):
        self._issued: list[dict[str, Any]] = []

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "mock", "model": "rule-simulation-v2", "mode": "rule_simulation",
            "is_real_model": False, "parameters": {"deterministic": True},
            "pricing": {"input_per_million": None, "output_per_million": None, "currency": None},
        }

    def complete(self, messages: list[dict], tools: list[dict]) -> ProviderResponse:
        del tools
        profile, recommended, budget, call_plans, feedback = None, None, 4, {}, []
        limit = None
        for message in messages:
            if message.get("role") == "user":
                try:
                    payload = json.loads(message.get("content", ""))
                    if isinstance(payload.get("training_profile"), dict):
                        profile = payload["training_profile"]
                    if isinstance(payload.get("guidance"), dict):
                        recommended = payload["guidance"].get("recommended_plan")
                    objective = payload.get("selection_objective") or {}
                    limit = validate_fpr_limit(objective.get("validation_false_positive_rate_limit"))
                    requested = payload.get("max_candidates", 4)
                    if type(requested) is int:
                        budget = max(1, min(requested, 4))
                except (ValueError, TypeError, AttributeError):
                    pass
            elif message.get("role") == "assistant":
                for call in message.get("tool_calls") or []:
                    try:
                        if call["function"]["name"] == "evaluate_plan":
                            previous = json.loads(call["function"]["arguments"])["plan"]
                            call_plans[call["id"]] = previous
                    except (KeyError, ValueError, TypeError):
                        pass
            elif message.get("role") == "tool":
                try:
                    result = json.loads(message.get("content", ""))
                    previous = result.get("plan") or call_plans.get(message.get("tool_call_id"))
                    if isinstance(previous, dict):
                        feedback.append((previous, result))
                except (ValueError, TypeError, AttributeError):
                    pass
        seen = {plan_signature(plan) for plan in self._issued}
        for previous in call_plans.values():
            signature = plan_signature(previous)
            if signature not in seen:
                self._issued.append(copy.deepcopy(previous))
                seen.add(signature)
        if len(self._issued) >= budget:
            return ProviderResponse({"role": "assistant", "content": "离线规则模拟预算已用完；保留通过检查的最佳方案。"})

        fields = (profile or {}).get("fields", {})
        parse_errors = (profile or {}).get("numeric_parse_errors", 0) or sum(
            field.get("strict_parse_errors", 0) for field in fields.values())
        invalid_values = (profile or {}).get("numeric_range_violations", 0)
        missing_categories = [field for field in fields.values()
                              if field.get("kind") == "categorical" and field.get("missing", 0)]
        rows = (profile or {}).get("rows") or 1
        mode_is_dominant = bool(missing_categories) and all(
            not field.get("unknown_semantic_count", 0)
            and max(field.get("top_values", {}).values(), default=0) >= 0.8 * max(1, rows - field.get("missing", 0))
            for field in missing_categories)
        base = {
            "numeric_parser": "semantic" if parse_errors else "strict",
            "numeric_imputation": "median",
            "categorical_imputation": "most_frequent" if mode_is_dominant else "unknown",
            "outlier_policy": "invalid_to_missing" if profile is None or invalid_values else "keep",
            "deduplicate": profile is None or bool(profile.get("duplicate_rows", 0)),
            "class_weight": "none", "add_missing_indicators": True,
            "threshold_policy": "validation_macro_f1",
        }
        if recommended:
            base = {key: value for key, value in recommended.items() if key not in {"name", "reason"}}
        valid_feedback = [(plan, result) for plan, result in feedback
                          if result.get("ok") and result.get("guardrails", {}).get("passed")
                          and validation_rank(result.get("validation"), limit) is not None]
        retained = next((result.get("retained_plan") for _, result in reversed(feedback) if result.get("retained_plan")), None)
        anchors = [(plan, result) for plan, result in valid_feedback if plan.get("name") == retained]
        anchor = copy.deepcopy((anchors[-1] if anchors else max(
            valid_feedback, key=lambda pair: validation_rank(pair[1]["validation"], limit), default=(base, {})))[0])
        anchor.pop("name", None)
        anchor.pop("reason", None)
        options = []
        for previous, result in reversed(feedback):
            error = str(result.get("error", "")).lower()
            if not result.get("ok") and previous.get("numeric_parser") == "strict" and ("parse" in error or "解析" in error):
                repaired = {key: value for key, value in previous.items() if key not in {"name", "reason"}}
                repaired["numeric_parser"] = "semantic"
                options.append((repaired, "严格解析失败，按实际工具反馈切换为语义解析。"))
                break
            if not result.get("guardrails", {}).get("passed", True) and previous.get("outlier_policy") == "iqr_clip":
                repaired = {key: value for key, value in previous.items() if key not in {"name", "reason"}}
                repaired["outlier_policy"] = "invalid_to_missing"
                options.append((repaired, "裁剪未通过约束，保留其他设置并改为修复非法值。"))
                break
        if not self._issued:
            options.append((base, "根据训练画像选择解析、缺失与去重策略。"))
        elif feedback:
            # Once an axis has been compared, spend the next slot on another axis.
            axes = [("class_weight", ("none", "balanced")),
                    ("categorical_imputation", ("unknown", "most_frequent")),
                    ("numeric_imputation", ("median", "mean"))]
            if limit is not None:
                axes.insert(0, ("add_calendar_feature", (False, True)))
            elif recommended:
                axes.insert(0, ("classifier", ("logistic_regression", "hist_gradient_boosting")))
            for key, choices in axes:
                default = OPTIONAL_PLAN_DEFAULTS.get(key)
                tested = {plan.get(key, default) for plan in self._issued}
                if set(choices).issubset(tested):
                    continue
                alternative = copy.deepcopy(anchor)
                alternative[key] = choices[1] if anchor.get(key, default) == choices[0] else choices[0]
                options.append((alternative, f"以反馈保留的方案为基础，单独比较 {key}。"))
        normalized_options = []
        for candidate, reason in options:
            candidate = {"add_missing_indicators": True, "threshold_policy": "validation_macro_f1", **candidate}
            normalized_options.append((candidate, reason))
        selected = next(((candidate, reason) for candidate, reason in normalized_options
                         if plan_signature(candidate) not in seen), None)
        if selected is None:
            return ProviderResponse({"role": "assistant", "content": "离线规则模拟没有未重复的受控候选；保留已有结果。"})
        plan, reason = selected
        plan = {"name": f"rule_candidate_{len(self._issued) + 1}", **plan,
                "reason": "离线规则模拟：" + reason}
        plan.setdefault("add_missing_indicators", True)
        plan.setdefault("threshold_policy", "validation_macro_f1")
        self._issued.append(copy.deepcopy(plan))
        return ProviderResponse({
            "role": "assistant", "content": None,
            "tool_calls": [{"id": f"mock_call_{len(self._issued)}", "type": "function", "function": {
                "name": "evaluate_plan", "arguments": json.dumps({"plan": plan}, ensure_ascii=False),
            }}],
        })

    def redact(self, value: Any) -> Any:
        return value


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Authorization headers must never follow a redirect to another host.
        return None


class ChatCompletionsProvider:
    """Small stdlib client; a configured official endpoint is the only destination."""

    _DEFAULT_URLS = {
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "deepseek": "https://api.deepseek.com/v1",
    }

    def __init__(self, config: dict[str, Any], *, allow_network: bool = False):
        self.config = copy.deepcopy(config)
        self.provider = str(config.get("provider", ""))
        self.model = str(config.get("model", ""))
        if self.provider not in self._DEFAULT_URLS or not self.model.strip():
            raise ProviderError("invalid_config", "供应商或模型名称配置不正确。")
        self.api_key_env = str(config.get("api_key_env", ""))
        if not self.api_key_env or not self.api_key_env.replace("_", "a").isalnum():
            raise ProviderError("invalid_config", "必须配置 API Key 的环境变量名称。")
        if any(k.lower() in {"api_key", "authorization", "token", "secret"} for k in config):
            raise ProviderError("inline_secret_forbidden", "配置中只能保存密钥环境变量名，不能保存密钥。")
        base_url_env = str(config.get("base_url_env", ""))
        base_url = os.environ.get(base_url_env, "") if base_url_env else ""
        base_url = base_url or config.get("base_url") or self._DEFAULT_URLS[self.provider]
        self.base_url = self._validate_url(str(base_url))
        self.allow_network = bool(allow_network)
        self.timeout = config.get("timeout_seconds", 60)
        self.max_retries = config.get("max_retries", 2)
        if (not _finite_nonnegative(self.timeout) or not 1 <= self.timeout <= 300
                or type(self.max_retries) is not int or not 0 <= self.max_retries <= 3):
            raise ProviderError("invalid_config", "超时应为 1—300 秒，重试次数应为 0—3。")
        self.parameters = copy.deepcopy(config.get("parameters", {}))
        if not isinstance(self.parameters, dict) or set(self.parameters) - {"temperature", "max_tokens", "top_p"}:
            raise ProviderError("invalid_config", "仅允许 temperature、max_tokens、top_p 模型参数。")
        self.parameters.setdefault("max_tokens", 4096)
        self.parameters.setdefault("temperature", 0)
        if type(self.parameters["max_tokens"]) is not int or not 1 <= self.parameters["max_tokens"] <= 8192:
            raise ProviderError("invalid_config", "max_tokens 必须是 1—8192 的整数，默认 4096。")
        for parameter, ceiling in (("temperature", 2), ("top_p", 1)):
            if parameter in self.parameters:
                value = self.parameters[parameter]
                if not _finite_nonnegative(value) or value > ceiling:
                    raise ProviderError("invalid_config", f"{parameter} 必须是 0—{ceiling} 范围内的有限实数。")
        self.extra_body = copy.deepcopy(config.get("extra_body", {}))
        allowed_extra = {"enable_thinking"} if self.provider == "qwen" else {"thinking"}
        if not isinstance(self.extra_body, dict) or set(self.extra_body) - allowed_extra:
            raise ProviderError("invalid_config", "包含未允许的供应商参数。")
        if self.provider == "qwen":
            self.extra_body.setdefault("enable_thinking", False)
            nonthinking = self.extra_body["enable_thinking"] is False
        else:
            self.extra_body.setdefault("thinking", {"type": "disabled"})
            nonthinking = self.extra_body["thinking"] == {"type": "disabled"}
        if not nonthinking:
            raise ProviderError("invalid_config", "此版本固定使用非思考模式，不能启用额外推理。")
        self.pricing = {
            "input_per_million": config.get("input_cost_per_million"),
            "output_per_million": config.get("output_cost_per_million"),
            "currency": config.get("currency"),
        }
        for price in (self.pricing["input_per_million"], self.pricing["output_per_million"]):
            if price is not None and not _finite_nonnegative(price):
                raise ProviderError("invalid_config", "价格必须为非负数或 null。")
        self._opener = urllib.request.build_opener(_NoRedirect())

    @classmethod
    def from_config(cls, path: str | Path, profile: str = "qwen", *, allow_network: bool = False):
        try:
            with Path(path).open(encoding="utf-8") as handle:
                document = json.load(handle)
            profiles = document.get("profiles", document)
            config = profiles[profile]
            if not isinstance(config, dict):
                raise ValueError
        except (OSError, ValueError, TypeError, KeyError):
            raise ProviderError("invalid_config", "无法读取指定模型配置。") from None
        return cls(config, allow_network=allow_network)

    def _validate_url(self, base_url: str) -> str:
        try:
            parsed = urllib.parse.urlsplit(base_url)
            host = (parsed.hostname or "").lower()
            port = parsed.port
        except ValueError:
            raise ProviderError("unsafe_endpoint", "模型地址不是有效的官方 HTTPS 地址。") from None
        if self.provider == "qwen":
            allowed = host in {"dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com"}
            # The current Beijing workspace endpoint has exactly one DNS label.
            workspace_host = re.fullmatch(
                r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.cn-beijing\.maas\.aliyuncs\.com", host,
            )
            allowed = allowed or workspace_host is not None
        else:
            allowed = host == "api.deepseek.com"
        if (parsed.scheme != "https" or not allowed or parsed.username or parsed.password
                or parsed.query or parsed.fragment or port not in (None, 443)):
            raise ProviderError("unsafe_endpoint", "仅允许当前供应商官方域名下的 HTTPS 模型接口。")
        return base_url.rstrip("/")

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.provider, "model": self.model, "mode": "llm_api", "is_real_model": True,
            "parameters": {**self.parameters, **self.extra_body}, "pricing": self.pricing,
            "network_enabled": self.allow_network,
            "request_policy": {"timeout_seconds": self.timeout, "max_retries": self.max_retries},
            "version_note": "正式实验应固定可调用快照，并保留服务端实际返回的模型名称。",
        }

    def redact(self, value: Any) -> Any:
        raw_secret = os.environ.get(self.api_key_env, "")
        secrets = sorted({secret for secret in (raw_secret, raw_secret.strip()) if secret}, key=len, reverse=True)
        def text_without_secrets(text):
            for secret in secrets:
                text = text.replace(secret, "[REDACTED]")
            return text
        if isinstance(value, dict):
            return {text_without_secrets(str(k)): "[REDACTED]" if str(k).strip().lower() in {"api_key", "authorization"} else self.redact(v)
                    for k, v in value.items()}
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if isinstance(value, str):
            return text_without_secrets(value)
        return value

    def _validate_response(self, document: Any, retries: int) -> ProviderResponse:
        try:
            message = document["choices"][0]["message"]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError
            calls = message.get("tool_calls") or []
            if not isinstance(calls, list) or len(calls) > 32:
                raise ValueError
            ids = set()
            for call in calls:
                function = call.get("function", {})
                if (not isinstance(call.get("id"), str) or not call["id"] or call["id"] in ids
                        or call.get("type") != "function" or not isinstance(function.get("name"), str)
                        or not isinstance(function.get("arguments"), str)):
                    raise ValueError
                ids.add(call["id"])
            if not calls and not isinstance(message.get("content"), str):
                raise ValueError
            usage = document.get("usage") or {}
            if not isinstance(usage, dict):
                raise ValueError
            usage = {k: v for k, v in usage.items()
                     if k in {"prompt_tokens", "completion_tokens", "total_tokens", "prompt_tokens_details", "completion_tokens_details"}}
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                if name in usage and (not isinstance(usage[name], int) or isinstance(usage[name], bool) or usage[name] < 0):
                    raise ValueError
            if isinstance(document.get("model"), str):
                usage["returned_model"] = document["model"]
            return ProviderResponse(copy.deepcopy(message), usage, retries)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise ProviderError("invalid_response", "模型响应结构不完整，无法安全继续工具循环。", retries) from None

    def complete(self, messages: list[dict], tools: list[dict]) -> ProviderResponse:
        if not self.allow_network:
            raise ProviderError("live_calls_disabled", "真实 API 调用尚未显式启用。")
        secret = os.environ.get(self.api_key_env, "").strip()
        if not secret:
            raise ProviderError("missing_api_key", "未设置所需的 API Key 环境变量；没有发送网络请求。")
        body = {"model": self.model, "messages": messages, "tools": tools, "tool_choice": "auto",
                "stream": False, **self.parameters, **self.extra_body}
        try:
            payload = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (TypeError, ValueError):
            raise ProviderError("invalid_request", "请求包含不可序列化的数据。") from None
        endpoint = self.base_url + "/chat/completions"
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(endpoint, data=payload, method="POST", headers={
                "Content-Type": "application/json", "Authorization": "Bearer " + secret,
            })
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    # Bound response size, including reasoning content.
                    raw = response.read(8 * 1024 * 1024 + 1)
                if len(raw) > 8 * 1024 * 1024:
                    raise ProviderError("response_too_large", "模型响应超出允许大小。", attempt)
                try:
                    document = json.loads(raw)
                except (ValueError, UnicodeError):
                    raise ProviderError("invalid_response", "模型接口未返回有效 JSON。", attempt) from None
                return self._validate_response(document, attempt)
            except urllib.error.HTTPError as error:
                status = error.code
                error.close()
                retriable = status == 429 or 500 <= status <= 599
                if not retriable or attempt == self.max_retries:
                    raise ProviderError("http_error", f"模型接口返回 HTTP {status}；响应正文未记录。", attempt) from None
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError):
                if attempt == self.max_retries:
                    raise ProviderError("network_error", "模型请求超时或连接失败；底层错误正文未记录。", attempt) from None
            time.sleep(min(2 ** attempt, 4))
        raise ProviderError("network_error", "模型请求未完成。", self.max_retries)
