"""Fair baseline comparison; held-out test evaluation follows plan selection."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from .data import load_table
from .engine import run_agent
from .modeling import evaluate_plan, finalize_plan
from .providers import ChatCompletionsProvider, MockProvider, ProviderResponse
from .quality import evaluate_repairs, profile
from .schema import DEFAULT_PLAN


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class _PresetSearchProvider(MockProvider):
    """Fixed four-plan control with the same tool and candidate budget."""

    def __init__(self):
        super().__init__()
        self._step = 0
        self.plans = [{
            **DEFAULT_PLAN, "name": f"grid_{category}_{weight}",
            "categorical_imputation": category, "class_weight": weight,
            "add_missing_indicators": True, "threshold_policy": "validation_macro_f1",
            "reason": "Evaluate a predeclared categorical-imputation/class-weight combination.",
        } for category in ("unknown", "most_frequent") for weight in ("none", "balanced")]

    def describe(self):
        return {**super().describe(), "model": "preset-search-v1", "mode": "preset_search"}

    def complete(self, messages, tools):
        del messages, tools
        if self._step >= len(self.plans):
            return ProviderResponse({"role": "assistant", "content": "预设候选搜索完成。"})
        plan = self.plans[self._step]
        self._step += 1
        return ProviderResponse({"role": "assistant", "content": None, "tool_calls": [{
            "id": f"grid_{self._step}", "type": "function", "function": {
                "name": "evaluate_plan", "arguments": json.dumps({"plan": plan}, ensure_ascii=False),
            },
        }]})


def run_experiment(data_dir, output_dir, *, mode="mock", config=None, provider_profile="qwen", seed=42,
                   compare=False) -> dict:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if mode not in {"mock", "api"}:
        raise ValueError("mode must be mock or api")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("结果目录已有文件，请换一个新目录，避免覆盖之前的实验。")
    required = ["train_dirty.csv", "validation.csv", "test.csv", "metadata.json"]
    for name in required:
        if not (data_dir / name).is_file():
            raise ValueError(f"缺少 {name}，请先运行 prepare。")
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("seed") != seed:
        raise ValueError("运行种子与数据构造种子不一致，请使用 metadata.json 中的种子。")
    provider = MockProvider() if mode == "mock" else ChatCompletionsProvider.from_config(
        config, profile=provider_profile, allow_network=True)
    train = load_table(data_dir / "train_dirty.csv")
    validation = load_table(data_dir / "validation.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    info = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "execution": {"mode": mode, "is_real_model": mode == "api",
                      "real_model_status": "pending" if mode == "mock" else "running"},
        "dataset": metadata,
        "environment": {"python": platform.python_version(), **{
            name: importlib.metadata.version(name) for name in ["pandas", "scikit-learn", "numpy"]}},
        "input_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
                         for name in required if name != "test.csv"},
        "quality_before": profile(train),
        "results": [],
    }
    # This call cannot access the test, reference data, or synthetic answer manifest.
    planning_started = time.perf_counter()
    agent = run_agent(train, validation, output_dir / "agent", provider=provider, seed=seed)
    info["agent"] = agent
    info["agent_planning_seconds"] = round(time.perf_counter() - planning_started, 3)
    info["execution"]["real_model_status"] = agent["real_model_validation"]
    baseline_plans = [
        ("minimal", "最低必要预处理", {
            **DEFAULT_PLAN, "name": "minimal", "outlier_policy": "invalid_to_missing",
            "numeric_imputation": "mean", "categorical_imputation": "most_frequent",
            "deduplicate": False, "class_weight": "none",
            "reason": "Basic format/schema repair with mean/mode filling, without deduplication or class weighting.",
        }),
        ("fixed", "固定清洗流程", {
            **DEFAULT_PLAN, "name": "fixed", "reason": "Fixed semantic parsing, schema repair, median filling, deduplication and balanced weights.",
        }),
    ]
    reference_plan = {
        **DEFAULT_PLAN, "name": "previous", "class_weight": "none",
        "add_missing_indicators": False, "threshold_policy": "fixed",
        "reason": "Frozen previous strategy for a same-split regression comparison.",
    }
    if compare:
        baseline_plans.extend([
            ("previous", "原方案", reference_plan),
            ("indicators_only", "仅加入缺失标记", {
                **reference_plan, "name": "indicators_only", "add_missing_indicators": True,
                "reason": "Change missing indicators alone; keep the 0.5 threshold and original imputation.",
            }),
            ("threshold_only", "仅优化分类阈值", {
                **reference_plan, "name": "threshold_only", "threshold_policy": "validation_macro_f1",
                "reason": "Change validation threshold selection alone; keep the original prediction features.",
            }),
        ])
        search_started = time.perf_counter()
        control = run_agent(train, validation, output_dir / "preset_search", provider=_PresetSearchProvider(), seed=seed)
        info["preset_search"] = control
        info["preset_search_seconds"] = round(time.perf_counter() - search_started, 3)
        if control["selected_plan"] is not None:
            baseline_plans.append(("preset_search", "等预算预设搜索", control["selected_plan"]))
    if agent["selected_plan"] is not None:
        baseline_plans.append(("agent", "智能体（真实模型）" if mode == "api" else "智能体（规则模拟）", agent["selected_plan"]))
    # Validate and freeze every plan and probability threshold before reading test.
    evaluated = {}
    validation_seconds = {}
    for key, _, plan in baseline_plans:
        validation_started = time.perf_counter()
        evaluated[key] = evaluate_plan(train, validation, plan, seed=seed)
        validation_seconds[key] = time.perf_counter() - validation_started
        if key in {"agent", "preset_search"} and evaluated[key].get("ok"):
            source = agent if key == "agent" else info["preset_search"]
            if evaluated[key]["decision_threshold"] != source["best_decision_threshold"]:
                raise ValueError("方案复核的阈值与已选择阈值不一致，未打开测试集。")
    write_json(output_dir / "frozen_plans.json", {key: plan for key, _, plan in baseline_plans})
    frozen_thresholds = {key: candidate["decision_threshold"] for key, candidate in evaluated.items()
                         if candidate.get("ok")}
    write_json(output_dir / "frozen_thresholds.json", frozen_thresholds)
    test = load_table(data_dir / "test.csv")
    info["input_sha256"]["test.csv"] = hashlib.sha256((data_dir / "test.csv").read_bytes()).hexdigest()
    reference = load_table(data_dir / "train_reference.csv") if (data_dir / "train_reference.csv").exists() else None
    manifest = json.loads((data_dir / "corruption_manifest.json").read_text(encoding="utf-8")) if (data_dir / "corruption_manifest.json").exists() else None
    for key, label, plan in baseline_plans:
        row_start = time.perf_counter()
        candidate = evaluated[key]
        if not candidate.get("ok"):
            info["results"].append({"id": key, "label": label, "ok": False, "error": candidate.get("error")})
            continue
        result_dir = output_dir / key / "artifacts"
        final = finalize_plan(train, validation, test, plan, result_dir, seed=seed,
                              decision_threshold=frozen_thresholds[key])
        result = {
            "id": key, "label": label, "ok": True, "plan": plan,
            "validation": candidate["validation"], "test": final["test"],
            "decision_threshold": frozen_thresholds[key],
            "threshold_selection": candidate["threshold_selection"],
            "prediction_features": final["prediction_features"],
            "quality_after": candidate["quality_after"], "guardrails": candidate["guardrails"],
            "changes": candidate.get("changes"), "artifacts": final.get("artifacts", {}),
            "final_fit_eval_seconds": round(validation_seconds[key] + time.perf_counter() - row_start, 3),
        }
        if reference is not None and manifest is not None:
            cleaned = load_table(result_dir / "train_processed.csv")
            result["repair_evaluation"] = evaluate_repairs(reference, train, cleaned, manifest)
        info["results"].append(result)
    info["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    search_completed = not compare or info["preset_search"]["status"] == "completed"
    info["status"] = "completed" if agent["status"] == "completed" and search_completed and all(x["ok"] for x in info["results"]) else "incomplete"
    info["note"] = "离线规则模拟用于工程验证；所有分类指标来自本地真实训练，真实大模型决策效果待接入。" if mode == "mock" else "模型通过API决策；分类指标来自固定数据划分和本地训练。"
    info["comparison_protocol"] = {
        "same_split": True, "test_used_for_selection": False,
        "threshold_fit_scope": "validation only; thresholds frozen before opening test",
        "candidate_budget_per_search": 4,
        "comparison_note": "原方案及单项对照在同一已报告的数据划分上进行回归比较；不据此声称独立外部泛化。",
    }
    write_json(output_dir / "metrics.json", info)
    _write_summary(output_dir, info)
    return info


def _write_summary(directory: Path, info: dict) -> None:
    lines = ["# 实验结果", "", info["note"], "",
             "| 方案 | 阈值 | 验证 Macro-F1 | 测试 Macro-F1 | 测试少数类召回 | 测试精确率 | 测试 AP |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in info["results"]:
        if row["ok"]:
            lines.append(f'| {row["label"]} | {row["decision_threshold"]:.2f} | {row["validation"]["macro_f1"]:.4f} | {row["test"]["macro_f1"]:.4f} | {row["test"]["minority_recall"]:.4f} | {row["test"]["precision"]:.4f} | {row["test"]["average_precision"]:.4f} |')
        else:
            lines.append(f'| {row["label"]} | — | 执行失败 | — | — | — | — |')
    lines += ["", "AP 为平均精确率，用作 PR 曲线表现的摘要。", "",
              "测试集在方案锁定后打开，所有方案使用同一组样本和标签。", "",
              f'真实大模型调用次数：{info["agent"]["model_calls"]}；决策模式：{info["agent"]["mode"]}。']
    (directory / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
