"""Acquire the public source and construct a reproducible, isolated benchmark."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import ssl
import urllib.error
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .schema import DATASET_URL, FEATURES, MODEL_FEATURES, MODEL_NUMERIC, ROW_ID, TARGET

UCI_ARCHIVE_URLS = (
    DATASET_URL,
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
)
MAX_ARCHIVE_BYTES = 100_000_000
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/222/bank+marketing"


class _OfficialUciRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        try:
            target = urllib.parse.urlsplit(newurl)
            allowed = (target.scheme == "https" and target.hostname == "archive.ics.uci.edu"
                       and target.port in (None, 443) and not target.username and not target.password)
        except ValueError:
            allowed = False
        if not allowed:
            raise urllib.error.URLError("UCI download redirect must remain on the official HTTPS host.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _read_source_record(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        raise ValueError("source.json is not a valid provenance record.") from None
    if not isinstance(record, dict):
        raise ValueError("source.json must contain a provenance object.")
    expected = record.get("csv_sha256")
    if expected is not None and (not isinstance(expected, str) or len(expected) != 64
                                 or any(character not in "0123456789abcdefABCDEF" for character in expected)):
        raise ValueError("source.json contains an invalid CSV SHA-256 checksum.")
    return record


def _extract_bank_csv(payload: bytes, depth: int = 0) -> bytes | None:
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        for member in archive.infolist():
            if Path(member.filename).name == "bank-full.csv":
                if member.file_size > MAX_ARCHIVE_BYTES:
                    raise ValueError("Source CSV exceeds the size limit.")
                with archive.open(member) as handle:
                    content = handle.read(MAX_ARCHIVE_BYTES + 1)
                if len(content) > MAX_ARCHIVE_BYTES:
                    raise ValueError("Source CSV exceeds the size limit.")
                return content
        if depth == 0:
            for member in archive.infolist():
                if Path(member.filename).name == "bank.zip":
                    if member.file_size > MAX_ARCHIVE_BYTES:
                        raise ValueError("Nested source archive exceeds the size limit.")
                    with archive.open(member) as handle:
                        nested = handle.read(MAX_ARCHIVE_BYTES + 1)
                    if len(nested) > MAX_ARCHIVE_BYTES:
                        raise ValueError("Nested source archive exceeds the size limit.")
                    return _extract_bank_csv(nested, depth=1)
    return None


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        first_line = handle.readline()
    separator = ";" if first_line.count(";") > first_line.count(",") else ","
    return pd.read_csv(path, sep=separator, dtype=object, keep_default_na=False,
                       encoding="utf-8-sig")


def download_bank(destination: Path) -> Path:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "bank-full.csv"
    source_path = destination / "source.json"
    previous_source = _read_source_record(source_path)
    expected_checksum = previous_source.get("csv_sha256")
    if target.exists():
        checksum = hashlib.sha256(target.read_bytes()).hexdigest()
        if expected_checksum and checksum != expected_checksum.lower():
            raise ValueError("Existing bank-full.csv does not match source.json; the file was not overwritten.")
        if not previous_source:
            source_path.write_text(json.dumps({
                "csv_sha256": checksum, "source_url": None, "archive_sha256": None,
                "acquisition": "pre-existing local CSV; original download provenance is unverified",
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return target
    # Explicit default TLS verification is retained on both official URLs.
    opener = urllib.request.build_opener(
        _OfficialUciRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    failures, attempted = [], []
    archive_bytes = None
    downloaded_url = None
    for source_url in UCI_ARCHIVE_URLS:
        attempted.append(source_url)
        request = urllib.request.Request(source_url, headers={"User-Agent": "data-quality-course-project/1.0"})
        try:
            with opener.open(request, timeout=90) as response:
                archive_bytes = response.read(MAX_ARCHIVE_BYTES + 1)
            downloaded_url = source_url
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, ssl.SSLError) as error:
            if isinstance(error, urllib.error.HTTPError):
                error.close()
            failures.append(f"{source_url}: {type(error).__name__}")
    if archive_bytes is None:
        raise RuntimeError("Official UCI HTTPS download failed; certificate verification was retained. "
                           + "; ".join(failures)) from None
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ValueError("The source archive exceeds the 100 MB download limit.")
    try:
        csv_bytes = _extract_bank_csv(archive_bytes)
    except (zipfile.BadZipFile, RuntimeError):
        raise ValueError("The official response is not a valid readable Bank Marketing ZIP archive.") from None
    if csv_bytes is None:
        raise ValueError("The official archive does not contain bank-full.csv.")
    checksum = hashlib.sha256(csv_bytes).hexdigest()
    if expected_checksum and checksum != expected_checksum.lower():
        raise ValueError("Downloaded CSV does not match the repository's source.json checksum; nothing was installed.")
    try:
        header = pd.read_csv(BytesIO(csv_bytes), sep=";", nrows=0).columns
    except (ValueError, UnicodeError, pd.errors.ParserError):
        raise ValueError("Downloaded bank-full.csv has an unreadable schema.") from None
    if set(header) != set(FEATURES + [TARGET]):
        raise ValueError("Downloaded bank-full.csv does not have the expected source columns.")
    temporary = target.with_suffix(".csv.part")
    temporary.write_bytes(csv_bytes)
    temporary.replace(target)
    record = {
        "source_url": downloaded_url, "archive_urls_attempted": attempted,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(), "csv_sha256": checksum,
        "repository_checksum_verified": expected_checksum is not None,
        "license": "CC BY 4.0 (UCI dataset page)", "dataset_page": DATASET_PAGE,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "citation": "Moro, S., Cortez, P., & Rita, P. Bank Marketing. UCI Machine Learning Repository. DOI:10.24432/C5K306",
    }
    source_temporary = source_path.with_suffix(".json.part")
    source_temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_temporary.replace(source_path)
    return target


def prepare_dataset(raw_csv: str | Path, destination: str | Path, seed: int = 42) -> dict:
    raw_csv, destination = Path(raw_csv), Path(destination)
    source = load_table(raw_csv)
    missing = sorted(set(FEATURES + [TARGET]) - set(source.columns))
    if missing:
        raise ValueError(f"Missing required source columns: {missing}")
    source = source[FEATURES + [TARGET]].copy()
    if len(source) < 8500:
        raise ValueError("At least 8,500 source rows are needed to retain 5,000 training rows.")
    if not set(source[TARGET]).issubset({"yes", "no"}):
        raise ValueError("Source labels must be 'yes' or 'no'; labels are never repaired automatically.")
    source.insert(0, ROW_ID, [f"row-{index:09d}" for index in range(len(source))])

    # Group by the actual prediction inputs, excluding both labels and duration.
    hashes = pd.util.hash_pandas_object(source[MODEL_FEATURES], index=False).astype(str)
    group_labels = pd.DataFrame({"group": hashes, "label": source[TARGET]}).groupby("group")["label"].agg(
        lambda values: values.mode().iloc[0])
    train_groups, holdout_groups = train_test_split(
        group_labels.index.to_numpy(), test_size=0.4, random_state=seed,
        stratify=group_labels.to_numpy())
    validation_groups, test_groups = train_test_split(
        holdout_groups, test_size=0.5, random_state=seed + 1,
        stratify=group_labels.loc[holdout_groups].to_numpy())
    train = source.loc[hashes.isin(train_groups)].reset_index(drop=True)
    validation = source.loc[hashes.isin(validation_groups)].reset_index(drop=True)
    test = source.loc[hashes.isin(test_groups)].reset_index(drop=True)
    if len(train) < 5000:
        raise ValueError("The group split produced fewer than 5,000 training rows.")
    dirty = train.copy()
    rng = np.random.default_rng(seed)
    available = [(row, column) for row in range(len(train)) for column in MODEL_FEATURES]
    rng.shuffle(available)
    occupied: set[tuple[int, str]] = set()
    entries: list[dict] = []

    def record(row: int, column: str, value: str, kind: str) -> None:
        original = str(dirty.at[row, column])
        dirty.at[row, column] = value
        occupied.add((row, column))
        entries.append({"kind": kind, "row_id": str(dirty.at[row, ROW_ID]),
                        "column": column, "original": original, "corrupted": value})

    for row, column in available[:round(len(available) * 0.025)]:
        record(row, column, "", "missing")
    numeric_cells = [(row, column) for row in range(len(train)) for column in MODEL_NUMERIC
                     if (row, column) not in occupied]
    rng.shuffle(numeric_cells)
    invalid_values = {"age": "-12", "day": "44", "campaign": "0", "pdays": "-9", "previous": "-5"}
    candidates = [(row, column) for row, column in numeric_cells if column in invalid_values]
    for row, column in candidates[:round(len(train) * 0.04)]:
        record(row, column, invalid_values[column], "outlier")
    remaining_numeric = [(row, column) for row, column in numeric_cells if (row, column) not in occupied]
    for row, column in remaining_numeric[:round(len(train) * 0.06)]:
        original = str(train.at[row, column])
        if column == "balance":
            formatted = f"€{int(float(original)):,}"
        elif column == "age":
            formatted = f"{original} years"
        elif column in {"day", "pdays"}:
            formatted = f"{original} days"
        else:
            formatted = f"{original} 次"
        record(row, column, formatted, "numeric_format")
    categorical_cells = [(row, column) for row, column in available
                         if column not in MODEL_NUMERIC and (row, column) not in occupied]
    for row, column in categorical_cells[:round(len(train) * 0.06)]:
        record(row, column, f"  {str(train.at[row, column]).upper()}  ", "categorical_format")
    duplicates = []
    duplicate_rows = rng.choice(len(train), size=max(1, round(len(train) * 0.01)), replace=False)
    copies = dirty.iloc[duplicate_rows].copy()
    for offset, (index, copied) in enumerate(copies.iterrows()):
        new_id = f"row-{len(source) + offset:09d}"
        duplicates.append({"kind": "duplicate", "row_id": new_id,
                           "original_row_id": str(copied[ROW_ID])})
        copies.at[index, ROW_ID] = new_id
    dirty = pd.concat([dirty, copies], ignore_index=True)
    manifest = {"version": 1, "seed": seed, "entries": entries, "duplicates": duplicates,
                "note": "Synthetic corruption ground truth only; the source is not assumed error-free."}
    metadata = {
        "dataset": "UCI Bank Marketing, bank-full.csv", "source_url": DATASET_URL,
        "source_sha256": hashlib.sha256(raw_csv.read_bytes()).hexdigest(), "seed": seed,
        "source_rows": len(source), "source_features": len(FEATURES),
        "split": {"train_reference": len(train), "train_dirty": len(dirty),
                  "validation": len(validation), "test": len(test)},
        "group_columns": MODEL_FEATURES,
        "split_protocol": "60/20/20 group-stratified IID benchmark; identical prediction inputs stay together. No reliable customer identifier exists, so this is not customer-entity isolation.",
        "excluded_prediction_columns": ["duration", ROW_ID, TARGET],
        "preserved_fields": {"duration": "Original values retained; excluded from parsing, fitted statistics, clipping, imputation, planning, quality guardrails and training duplicate comparison."},
        "training_deduplication_rule": "If enabled: merge normalized identical modeling inputs with identical labels before outlier replacement and imputation. This is not customer-entity deduplication; independent source records can be merged. Report reference_rows_retained/reference_rows_removed.",
        "pollution_scope": "training only; validation and test are unmodified source holdouts",
        "source_is_verified_ground_truth": False,
        "corruptions": {kind: sum(entry["kind"] == kind for entry in entries)
                        for kind in ["missing", "outlier", "numeric_format", "categorical_format"]},
        "injected_duplicates": len(duplicates),
    }
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in {"train_dirty": dirty, "validation": validation, "test": test,
                        "train_reference": train}.items():
        path = destination / f"{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = str(path)
    for name, content in {"corruption_manifest": manifest, "metadata": metadata}.items():
        path = destination / f"{name}.json"
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
        paths[name] = str(path)
    return {"metadata": metadata, "paths": paths}
