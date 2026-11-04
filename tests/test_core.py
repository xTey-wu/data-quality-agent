import hashlib
from io import BytesIO
import json
import ssl
from unittest.mock import Mock
import urllib.error
import urllib.request
import zipfile

import pandas as pd
import pytest

from dqagent import data
from dqagent.modeling import evaluate_plan, finalize_plan
from dqagent.data import download_bank, prepare_dataset, load_table
from dqagent.quality import Cleaner, evaluate_repairs, profile, validate_plan
from dqagent.schema import DEFAULT_PLAN, FEATURES, MODEL_FEATURES, MODEL_NUMERIC, ROW_ID, TARGET


def table(rows=80):
    frame = pd.DataFrame({
        "age": [str(20 + index % 50) for index in range(rows)],
        "job": ["admin." if index % 2 else "services" for index in range(rows)],
        "marital": ["single"] * rows, "education": ["secondary"] * rows,
        "default": ["no"] * rows,
        "balance": [str(-100 + index * 15) for index in range(rows)],
        "housing": ["yes"] * rows, "loan": ["no"] * rows,
        "contact": ["unknown"] * rows, "day": [str(1 + index % 28) for index in range(rows)],
        "month": ["may"] * rows, "duration": ["100"] * rows,
        "campaign": ["1"] * rows, "pdays": ["-1"] * rows,
        "previous": ["0"] * rows, "poutcome": ["unknown"] * rows,
        "y": ["yes" if index % 4 == 0 else "no" for index in range(rows)],
    })
    frame.insert(0, ROW_ID, [f"row-{index:09d}" for index in range(rows)])
    return frame


@pytest.fixture
def synthetic_source_archive():
    """Tiny in-memory source for downloader CI; no real data or network required."""
    csv_bytes = table(12).drop(columns=ROW_ID).to_csv(index=False, sep=";").encode("utf-8")
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bank-full.csv", csv_bytes)
        archive.writestr("../ignored-source-file.txt", "never extract other files")
    return csv_bytes, buffer.getvalue()


def test_download_retries_another_official_url_with_tls_verification_and_tracks_source(
        tmp_path, monkeypatch, synthetic_source_archive):
    csv_bytes, archive_bytes = synthetic_source_archive
    expected = hashlib.sha256(csv_bytes).hexdigest()
    (tmp_path / "source.json").write_text(json.dumps({"csv_sha256": expected}), encoding="utf-8")
    opener = Mock()
    opener.open.side_effect = [
        urllib.error.URLError(ssl.SSLCertVerificationError("synthetic TLS failure")),
        BytesIO(archive_bytes),
    ]
    build = Mock(return_value=opener)
    monkeypatch.setattr(data.urllib.request, "build_opener", build)
    target = download_bank(tmp_path)
    assert target.read_bytes() == csv_bytes
    assert [call.args[0].full_url for call in opener.open.call_args_list] == list(data.UCI_ARCHIVE_URLS)
    https = next(handler for handler in build.call_args.args if isinstance(handler, urllib.request.HTTPSHandler))
    assert https._context.verify_mode == ssl.CERT_REQUIRED
    assert https._context.check_hostname
    record = json.loads((tmp_path / "source.json").read_text())
    assert record["source_url"] == data.UCI_ARCHIVE_URLS[1]
    assert record["archive_sha256"] == hashlib.sha256(archive_bytes).hexdigest()
    assert record["csv_sha256"] == expected
    assert record["repository_checksum_verified"]
    assert record["archive_urls_attempted"] == list(data.UCI_ARCHIVE_URLS)
    assert str(tmp_path) not in json.dumps(record)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bank-full.csv", "source.json"]


def test_download_supports_official_nested_zip_without_extracting_other_members(
        tmp_path, monkeypatch, synthetic_source_archive):
    csv_bytes, nested = synthetic_source_archive
    outer_buffer = BytesIO()
    with zipfile.ZipFile(outer_buffer, "w", compression=zipfile.ZIP_DEFLATED) as outer:
        outer.writestr("bank.zip", nested)
        outer.writestr("ignored.txt", "not part of the requested source")
    opener = Mock()
    opener.open.return_value = BytesIO(outer_buffer.getvalue())
    monkeypatch.setattr(data.urllib.request, "build_opener", Mock(return_value=opener))
    assert download_bank(tmp_path).read_bytes() == csv_bytes
    assert sorted(path.name for path in tmp_path.iterdir()) == ["bank-full.csv", "source.json"]


def test_download_rejects_repository_checksum_mismatch_without_installing(
        tmp_path, monkeypatch, synthetic_source_archive):
    _, archive_bytes = synthetic_source_archive
    provenance = {"csv_sha256": "0" * 64}
    (tmp_path / "source.json").write_text(json.dumps(provenance), encoding="utf-8")
    opener = Mock()
    opener.open.return_value = BytesIO(archive_bytes)
    monkeypatch.setattr(data.urllib.request, "build_opener", Mock(return_value=opener))
    with pytest.raises(ValueError, match="checksum"):
        download_bank(tmp_path)
    assert not (tmp_path / "bank-full.csv").exists()
    assert json.loads((tmp_path / "source.json").read_text()) == provenance


def test_download_keeps_cached_source_and_detects_local_modification(
        tmp_path, monkeypatch, synthetic_source_archive):
    csv_bytes, _ = synthetic_source_archive
    (tmp_path / "bank-full.csv").write_bytes(csv_bytes)
    (tmp_path / "source.json").write_text(
        json.dumps({"csv_sha256": hashlib.sha256(csv_bytes).hexdigest()}), encoding="utf-8")
    build = Mock()
    monkeypatch.setattr(data.urllib.request, "build_opener", build)
    assert download_bank(tmp_path).read_bytes() == csv_bytes
    build.assert_not_called()
    (tmp_path / "bank-full.csv").write_bytes(csv_bytes + b"\n")
    with pytest.raises(ValueError, match="not overwritten"):
        download_bank(tmp_path)
    assert (tmp_path / "bank-full.csv").read_bytes() == csv_bytes + b"\n"


def test_download_failure_never_disables_tls_or_creates_partial_source(tmp_path, monkeypatch):
    opener = Mock()
    opener.open.side_effect = urllib.error.URLError(ssl.SSLCertVerificationError("synthetic TLS failure"))
    monkeypatch.setattr(data.urllib.request, "build_opener", Mock(return_value=opener))
    with pytest.raises(RuntimeError, match="certificate verification was retained"):
        download_bank(tmp_path)
    assert opener.open.call_count == len(data.UCI_ARCHIVE_URLS)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("url", [
    "http://archive.ics.uci.edu/bank.zip", "https://example.com/bank.zip",
    "https://archive.ics.uci.edu.evil.example/bank.zip",
    "https://user:password@archive.ics.uci.edu/bank.zip",
    "https://archive.ics.uci.edu:8443/bank.zip",
])
def test_download_rejects_foreign_or_insecure_redirects(url):
    handler = data._OfficialUciRedirect()
    request = urllib.request.Request(data.UCI_ARCHIVE_URLS[0])
    with pytest.raises(urllib.error.URLError, match="official HTTPS"):
        handler.redirect_request(request, None, 302, "redirect", {}, url)


def test_download_limits_response_and_decompressed_member_sizes(tmp_path, monkeypatch):
    opener = Mock()
    monkeypatch.setattr(data.urllib.request, "build_opener", Mock(return_value=opener))
    monkeypatch.setattr(data, "MAX_ARCHIVE_BYTES", 1024)
    opener.open.return_value = BytesIO(b"x" * 1025)
    with pytest.raises(ValueError, match="100 MB"):
        download_bank(tmp_path)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bank-full.csv", b"a" * 4096)
    assert len(buffer.getvalue()) < 1024
    opener.open.return_value = BytesIO(buffer.getvalue())
    with pytest.raises(ValueError, match="Source CSV exceeds"):
        download_bank(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_semantic_parser_preserves_domain_values_and_never_uses_duration():
    dirty = table()
    dirty.loc[0, "age"] = " 20 years "
    dirty.loc[0, "balance"] = "€-1,200"
    dirty.loc[0, "job"] = "  ADMIN. "
    cleaned = Cleaner(DEFAULT_PLAN).fit_transform(dirty)
    assert cleaned.loc[0, "age"] == 20
    assert cleaned.loc[0, "balance"] == -1200
    assert cleaned.loc[0, "pdays"] == -1
    assert cleaned.loc[0, "contact"] == "unknown"
    assert cleaned.loc[0, "job"] == "admin."
    assert "duration" not in MODEL_FEATURES
    assert ROW_ID not in MODEL_FEATURES


def test_strict_parse_failure_is_reviewable_and_semantic_retry_succeeds():
    dirty = table()
    dirty.loc[0, "age"] = "20 years"
    strict = {**DEFAULT_PLAN, "numeric_parser": "strict"}
    failed = evaluate_plan(dirty, table(20), strict)
    assert not failed["ok"]
    assert "semantic" in failed["error"]
    assert failed["validation"] is None
    recovered = evaluate_plan(dirty, table(20), DEFAULT_PLAN)
    assert recovered["ok"], recovered
    assert 0 <= recovered["validation"]["macro_f1"] <= 1


@pytest.mark.parametrize("currency", ["$100", "£100", "¥100", "100 USD", "100 CNY", "100 元"])
def test_cross_currency_values_are_not_silently_converted(currency):
    dirty = table()
    dirty.loc[0, "balance"] = currency
    with pytest.raises(ValueError, match="could not parse"):
        Cleaner(DEFAULT_PLAN).fit(dirty)


def test_imputation_uses_training_only_and_validation_rows_are_preserved():
    training = table(20)
    training.loc[0, "balance"] = ""
    cleaner = Cleaner(DEFAULT_PLAN).fit(training)
    learned = cleaner.numeric_fill_["balance"]
    validation = table(4)
    validation.loc[0, "balance"] = ""
    validation.loc[1:, "balance"] = "999999999"
    validation.loc[3, FEATURES] = validation.loc[2, FEATURES]
    result = cleaner.transform(validation)
    assert result.loc[0, "balance"] == learned
    assert cleaner.numeric_fill_["balance"] == learned
    assert len(result) == len(validation)
    assert result["y"].tolist() == validation["y"].tolist()


def test_duplicate_ids_are_not_features_and_conflicting_labels_are_retained():
    training = table(20)
    duplicate = training.iloc[[0]].copy()
    duplicate[ROW_ID] = "row-000999999"
    conflicting = training.iloc[[0]].copy()
    conflicting[ROW_ID] = "row-000999998"
    conflicting["y"] = "no"
    dirty = pd.concat([training, duplicate, conflicting], ignore_index=True)
    assert profile(dirty)["duplicate_rows"] == 1
    cleaned = Cleaner(DEFAULT_PLAN).fit_transform(dirty)
    assert len(cleaned) == len(training) + 1
    assert "row-000999998" in set(cleaned[ROW_ID])
    assert "row-000999999" not in set(cleaned[ROW_ID])


def test_duration_is_preserved_and_cannot_affect_quality_guardrails_dedup_or_model():
    training = table()
    duplicate = training.iloc[[0]].copy()
    duplicate[ROW_ID] = "row-000999999"
    duplicate["duration"] = "999999"
    training = pd.concat([training, duplicate], ignore_index=True)
    altered = training.copy()
    altered["duration"] = ["bad duration", "", "-100"] * 27
    validation = table(20)
    altered_validation = validation.copy()
    altered_validation["duration"] = "this field is unavailable before a call"
    first_cleaner = Cleaner(DEFAULT_PLAN)
    first = first_cleaner.fit_transform(training)
    second_cleaner = Cleaner(DEFAULT_PLAN)
    second = second_cleaner.fit_transform(altered)
    assert len(first) == len(second) == 80
    assert first[ROW_ID].tolist() == second[ROW_ID].tolist()
    assert first_cleaner.numeric_fill_ == second_cleaner.numeric_fill_
    assert "duration" not in first_cleaner.numeric_fill_
    assert first_cleaner.last_changes_ == second_cleaner.last_changes_
    retained_duration = altered.set_index(ROW_ID).loc[second[ROW_ID], "duration"].tolist()
    assert second["duration"].tolist() == retained_duration
    assert profile(altered)["numeric_parse_errors"] == 0
    assert profile(altered)["numeric_range_violations"] == 0
    assert profile(altered)["missing_cells"] == 0
    assert profile(altered)["duplicate_rows"] == profile(training)["duplicate_rows"] == 1
    a = evaluate_plan(training, validation, DEFAULT_PLAN)
    b = evaluate_plan(altered, altered_validation, DEFAULT_PLAN)
    assert a["ok"] and b["ok"]
    assert a["validation"] == b["validation"]
    assert a["guardrails"] == b["guardrails"]
    assert a["changes"] == b["changes"]
    assert second_cleaner.last_changes_["legal_numeric_cells"] == 80 * len(MODEL_NUMERIC)
    iqr_plan = {**DEFAULT_PLAN, "outlier_policy": "iqr_clip"}
    first_iqr, second_iqr = Cleaner(iqr_plan), Cleaner(iqr_plan)
    first_iqr.fit_transform(training)
    second_iqr.fit_transform(altered)
    assert "duration" not in first_iqr.iqr_bounds_
    assert first_iqr.iqr_bounds_ == second_iqr.iqr_bounds_
    assert first_iqr.last_changes_ == second_iqr.last_changes_


def test_imputation_collisions_do_not_merge_distinct_training_records():
    training = table(4)
    training.loc[1, FEATURES + [TARGET]] = training.loc[0, FEATURES + [TARGET]]
    training.loc[0, "age"] = ""
    training.loc[1:, "age"] = "40"
    cleaner = Cleaner(DEFAULT_PLAN)
    cleaned = cleaner.fit_transform(training)
    assert len(cleaned) == 4
    assert cleaner.last_changes_["duplicates_removed"] == 0
    assert cleaned.loc[0, "age"] == cleaned.loc[1, "age"] == 40
    assert cleaned.duplicated(subset=MODEL_FEATURES + [TARGET]).sum() == 1


def test_training_statistics_and_export_use_same_pre_imputation_duplicate_rule():
    training = table(4)
    training.loc[1, FEATURES + [TARGET]] = training.loc[0, FEATURES + [TARGET]]
    training["balance"] = ["0", "0", "10", "20"]
    cleaner = Cleaner({**DEFAULT_PLAN, "numeric_imputation": "mean"})
    cleaned = cleaner.fit_transform(training)
    assert len(cleaned) == 3
    assert cleaner.numeric_fill_["balance"] == 10


def test_invalid_range_policy_and_no_label_mutation():
    training = table()
    training.loc[0, "age"] = "-12"
    keep = {**DEFAULT_PLAN, "outlier_policy": "keep"}
    assert not evaluate_plan(training, table(20), keep)["guardrails"]["passed"]
    clean = Cleaner(DEFAULT_PLAN).fit_transform(training)
    assert (clean["age"] >= 0).all()
    assert clean["y"].tolist() == training["y"].tolist()


def test_iqr_clipping_of_many_valid_cells_is_rejected():
    training = table(100)
    # Six prediction numeric columns: >12 legal clips cross the 2% guardrail.
    training.loc[:79, "balance"] = [str(index) for index in range(80)]
    training.loc[80:, "balance"] = "10000"
    plan = {**DEFAULT_PLAN, "outlier_policy": "iqr_clip"}
    result = evaluate_plan(training, table(20), plan)
    assert not result["guardrails"]["passed"]
    assert "IQR" in result["error"]


def test_plan_rejects_extra_actions_and_nonboolean_deduplicate():
    with pytest.raises(ValueError):
        validate_plan({**DEFAULT_PLAN, "python_code": "anything"})
    with pytest.raises(ValueError):
        validate_plan({**DEFAULT_PLAN, "deduplicate": "false"})


def test_finalization_writes_frozen_model_and_blocks_repeat_test(tmp_path):
    result = finalize_plan(table(), table(20), table(24), DEFAULT_PLAN, tmp_path)
    assert result["test_evaluations"] == 1
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "preprocessing.json").exists()
    assert result["artifacts"]["model"] == str(tmp_path / "model.joblib")
    assert len(pd.read_csv(tmp_path / "test_processed.csv")) == 24
    with pytest.raises(ValueError, match="already contains"):
        finalize_plan(table(), table(20), table(24), DEFAULT_PLAN, tmp_path)


def test_repair_evaluation_separates_format_and_imputation():
    reference = table(20)
    dirty = reference.copy()
    dirty.loc[0, "age"] = "20 years"
    dirty.loc[1, "age"] = ""
    dirty.loc[2, "balance"] = ""
    processed = Cleaner(DEFAULT_PLAN).fit_transform(dirty)
    manifest = {"entries": [
        {"kind": "numeric_format", "row_id": reference.loc[0, ROW_ID], "column": "age"},
        {"kind": "missing", "row_id": reference.loc[1, ROW_ID], "column": "age"},
        {"kind": "missing", "row_id": reference.loc[2, ROW_ID], "column": "balance"}], "duplicates": []}
    result = evaluate_repairs(reference, dirty, processed, manifest)
    assert result["cell_repairs"]["numeric_format"]["exact_restoration_rate"] == 1
    fields = result["cell_repairs"]["missing"]["numeric_reconstruction_mae_by_field"]
    assert fields["age"]["mae"] > 0
    assert fields["age"]["unit"] == "years"
    assert fields["balance"]["mae"] > 0
    assert fields["balance"]["unit"] == "EUR"
    assert fields["age"]["n"] == fields["balance"]["n"] == 1
    assert "numeric_reconstruction_mae" not in result["cell_repairs"]["missing"]
    assert result["untouched_cells"]["unintended_change_rate"] == 0
    assert result["untouched_cells"]["retained_evaluable"] == len(reference) * len(MODEL_FEATURES) - 3
    assert "duration" in result["untouched_cells"]["scope"]


def test_prepare_groups_identical_model_inputs_before_training_only_pollution(tmp_path):
    source = table(9000).drop(columns=ROW_ID)
    # Different duration and label still belong to one prediction-input group.
    copy = source.iloc[[0]].copy()
    copy["duration"] = "9999"
    copy["y"] = "no"
    source = pd.concat([source, copy], ignore_index=True)
    raw = tmp_path / "source.csv"
    source.to_csv(raw, sep=";", index=False)
    prepared = prepare_dataset(raw, tmp_path / "prepared", seed=42)
    paths = prepared["paths"]
    train = load_table(paths["train_reference"])
    validation = load_table(paths["validation"])
    test = load_table(paths["test"])
    assert len(train) >= 5000
    groups = [set(pd.util.hash_pandas_object(split[MODEL_FEATURES], index=False))
              for split in [train, validation, test]]
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert profile(validation)["numeric_parse_errors"] == 0
    assert profile(test)["missing_cells"] == 0
    dirty = load_table(paths["train_dirty"])
    before = profile(dirty)
    assert before["missing_cells"] > 0
    assert before["numeric_parse_errors"] > 0
    assert before["numeric_range_violations"] > 0
    assert before["duplicate_rows"] > 0
