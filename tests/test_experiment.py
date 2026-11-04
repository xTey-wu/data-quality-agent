import json
from pathlib import Path

import pandas as pd
import pytest

from dqagent.experiment import run_experiment
from dqagent.schema import DEFAULT_PLAN


def fixture_table(count=120):
    return pd.DataFrame([{
        "_row_id": f"row-{i:06d}", "age": str(22 + i % 55), "job": "admin.",
        "marital": "married" if i % 2 else "single", "education": "secondary",
        "default": "no", "balance": str(i * 29 - 500), "housing": "yes" if i % 3 else "no",
        "loan": "no", "contact": "cellular", "day": str(1 + i % 28), "month": "may",
        "duration": str(i * 10), "campaign": str(1 + i % 4), "pdays": "-1", "previous": "0",
        "poutcome": "unknown", "y": "yes" if i % 4 == 0 else "no",
    } for i in range(count)])


def test_experiment_freezes_plans_before_opening_test_and_exports(tmp_path, monkeypatch):
    from dqagent import experiment
    source, output = tmp_path / "input", tmp_path / "run"
    source.mkdir()
    table = fixture_table()
    table.iloc[:80].to_csv(source / "train_dirty.csv", index=False)
    table.iloc[80:100].to_csv(source / "validation.csv", index=False)
    table.iloc[100:].to_csv(source / "test.csv", index=False)
    (source / "metadata.json").write_text(json.dumps({"seed": 42, "source_rows": 120, "source_features": 16}), encoding="utf-8")
    original_load = experiment.load_table

    def audited_load(path):
        if Path(path).name == "test.csv":
            assert (output / "frozen_plans.json").is_file()
            frozen = json.loads((output / "frozen_plans.json").read_text())
            assert set(frozen) == {"minimal", "fixed", "agent"}
        return original_load(path)

    monkeypatch.setattr(experiment, "load_table", audited_load)
    result = run_experiment(source, output)
    assert result["status"] == "completed"
    assert result["agent"]["model_calls"] == 0
    assert result["execution"]["real_model_status"] == "not_run"
    assert len(result["results"]) == 3
    assert all(row["ok"] for row in result["results"])
    for row in result["results"]:
        assert row["test"]["rows"] == 20
        assert (output / row["id"] / "artifacts/model.joblib").exists()
        assert row["final_fit_eval_seconds"] >= 0
    with pytest.raises(ValueError, match="已有文件"):
        run_experiment(source, output)


def test_seed_mismatch_rejected_before_running(tmp_path):
    for name in ["train_dirty.csv", "validation.csv", "test.csv"]:
        (tmp_path / name).write_text("unused\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text('{"seed": 7}', encoding="utf-8")
    with pytest.raises(ValueError, match="种子"):
        run_experiment(tmp_path, tmp_path / "output", seed=42)
