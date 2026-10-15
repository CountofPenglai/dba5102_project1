"""Verified flat dataset artifacts, safe publication, and model-facing loading."""

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import platform
import tempfile
import numpy as np
import pandas as pd
from .config import CONFIG_PATH, DEFAULT_DATA, DEFAULT_OUTPUT, PROJECT, PROTOCOL_PATH, load_config, load_protocol
from .features import BASE_FEATURE, FEATURE_COLUMNS, RISK_FEATURES
from .pipeline import prepare, pool_statistics

TABLE_KINDS = ("features", "targets", "metadata")
ARTIFACT_PATHS = {f"datasets/{kind}.csv" for kind in TABLE_KINDS} | {
    "reports/dataset_statistics.csv", "reports/exclusion_statistics.csv",
}
TARGET_COLUMNS = ["payout_usdc", "payout_per_btc_usdc", "positive_payout"]
METADATA_COLUMNS = [
    "source_row_number", "created_at_utc", "closed_at_utc", "expiration_at_utc",
    "last_candle_close_utc", "benchmark_eligible", "base_missing",
    "nominal_duration_deviation_seconds", "recorded_premium_usdc",
    "recorded_price_per_btc_usdc", "recorded_markup_per_btc_usdc",
]
DATE_COLUMNS = ("created_at_utc", "closed_at_utc", "expiration_at_utc", "last_candle_close_utc")


@dataclass
class ModelData:
    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_checksums():
    return {str(path.relative_to(PROJECT)): checksum(path)
            for package in ("pp_audit", "pp_data")
            for path in sorted((PROJECT / package).glob("*.py"))}


def _check_nonoverlap(source, output):
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output and source directories must not overlap")


def _artifact_bytes(root, name):
    path = root / name
    parts = Path(name).parts
    if root.is_symlink() or any(root.joinpath(*parts[:i]).is_symlink() for i in range(1, len(parts) + 1)):
        raise ValueError(f"Dataset artifact must not use symbolic links: {name}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"Dataset checksum unavailable: {name}") from error


def _parse_artifact(payload, record, name, require_dtypes=False):
    if not isinstance(record, dict) or hashlib.sha256(payload).hexdigest() != record.get("sha256"):
        raise ValueError(f"Dataset checksum mismatch: {name}")
    dtypes = record.get("dtypes")
    if require_dtypes and (not isinstance(dtypes, dict) or set(dtypes) != set(record.get("columns", []))):
        raise ValueError(f"Dataset dtype schema mismatch: {name}")
    if dtypes is not None and any(value not in ("object", "bool", "int64", "float64") for value in dtypes.values()):
        raise ValueError(f"Unsupported dataset dtype: {name}")
    try:
        table = pd.read_csv(BytesIO(payload), float_precision="round_trip", dtype=dtypes)
    except (ValueError, TypeError, pd.errors.ParserError) as error:
        raise ValueError(f"Dataset schema mismatch: {name}") from error
    if len(table) != record.get("rows") or table.columns.tolist() != record.get("columns"):
        raise ValueError(f"Dataset schema mismatch: {name}")
    return table


def _validate_tables(tables, config, research_end):
    expected = {"features": FEATURE_COLUMNS, "targets": TARGET_COLUMNS, "metadata": METADATA_COLUMNS}
    for kind in TABLE_KINDS:
        table = tables[kind]
        if table.columns.tolist() != ["order_id"] + expected[kind]:
            raise ValueError(f"Dataset {kind} whitelist/schema mismatch")
        if table.order_id.isna().any() or table.order_id.duplicated().any():
            raise ValueError(f"Invalid order IDs: {kind}")
    features, targets, metadata = (tables[kind].set_index("order_id") for kind in TABLE_KINDS)
    if not features.index.equals(targets.index) or not features.index.equals(metadata.index):
        raise ValueError("Features, targets and metadata IDs are not aligned")
    metadata = metadata.copy()
    for name in DATE_COLUMNS:
        try:
            metadata[name] = pd.to_datetime(metadata[name], utc=True, errors="raise", format="mixed")
        except (ValueError, TypeError) as error:
            raise ValueError("Dataset violates temporal schema") from error
    if not pd.api.types.is_integer_dtype(metadata.source_row_number) or (metadata.source_row_number <= 0).any():
        raise ValueError("Invalid original source row numbers")
    if metadata.source_row_number.duplicated().any() or metadata.index.tolist() != [
            f"training_{number:06d}" for number in metadata.source_row_number]:
        raise ValueError("Order IDs do not match original source row numbers")
    ordered = metadata.sort_values(["created_at_utc", "source_row_number"], kind="stable")
    if not metadata.index.equals(ordered.index):
        raise ValueError("Dataset is not in chronological order")
    age = (metadata.created_at_utc - metadata.last_candle_close_utc).dt.total_seconds()
    valid_time = (
        metadata[list(DATE_COLUMNS)].notna().all(axis=1)
        & metadata.created_at_utc.ge(pd.Timestamp(config["research_start"]))
        & metadata.created_at_utc.lt(pd.Timestamp(research_end))
        & metadata.closed_at_utc.ge(metadata.created_at_utc)
        & metadata.closed_at_utc.lt(pd.Timestamp(research_end))
        & metadata.expiration_at_utc.gt(metadata.created_at_utc)
        & age.ge(0) & age.le(config["max_candle_age_seconds"])
    )
    if not valid_time.all():
        raise ValueError("Dataset violates temporal boundaries or completed-candle requirements")
    if not all(pd.api.types.is_bool_dtype(metadata[name]) for name in ("benchmark_eligible", "base_missing")):
        raise ValueError("Dataset cohort flags must be boolean")
    try:
        valid_features = np.isfinite(features[RISK_FEATURES]).all().all()
        valid_targets = np.isfinite(targets).all().all() and targets.ge(0).all().all()
    except TypeError as error:
        raise ValueError("Dataset feature/target schema must be numeric") from error
    if not valid_features:
        raise ValueError("Dataset features contain nonfinite values")
    if not valid_targets or not targets.positive_payout.isin([0, 1]).all():
        raise ValueError("Dataset targets contain invalid values")
    if not targets.positive_payout.eq(targets.payout_usdc.gt(0)).all():
        raise ValueError("Dataset targets disagree about positive payout")
    common_base = features.loc[metadata.benchmark_eligible, BASE_FEATURE]
    if not np.isfinite(common_base).all() or not common_base.gt(0).all():
        raise ValueError("Benchmark features contain invalid API Base values")
    return features, targets, metadata


def _read_dataset(root):
    root = Path(root)
    manifest_bytes = _artifact_bytes(root, "manifest.json")
    try:
        manifest = json.loads(manifest_bytes)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("Invalid dataset manifest") from error
    if not isinstance(manifest, dict) or manifest.get("producer") != "pp_data" or manifest.get("version") != "preprocessing_v2":
        raise ValueError("Invalid flat dataset manifest")
    config = load_config()
    if manifest.get("config") != config:
        raise ValueError("Dataset config differs from the flat preprocessing contract")
    if manifest.get("research_end") != load_protocol()["validation_end"]:
        raise ValueError("Dataset research cutoff differs from the sole validation protocol")
    if manifest.get("feature_sets") != {"risk": RISK_FEATURES, "benchmark": FEATURE_COLUMNS} or manifest.get("target") != "payout_per_btc_usdc":
        raise ValueError("Dataset feature/target schema mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != ARTIFACT_PATHS:
        raise ValueError("Dataset artifact list is incomplete or unexpected")
    # Snapshot all bytes first. A concurrent publication cannot mix parsed tables.
    snapshots = {name: _artifact_bytes(root, name) for name in sorted(ARTIFACT_PATHS)}
    loaded = {name: _parse_artifact(payload, artifacts[name], name, require_dtypes=True)
              for name, payload in snapshots.items()}
    report_columns = {
        "reports/dataset_statistics.csv": ["cohort", "rows", "first_created_utc", "last_created_utc",
                                           "last_label_available_utc", "base_missing_rows"],
        "reports/exclusion_statistics.csv": ["reason", "rows", "definition"],
    }
    for name, columns in report_columns.items():
        if list(loaded[name]) != columns:
            raise ValueError(f"Dataset report schema mismatch: {name}")
    tables = {kind: loaded[f"datasets/{kind}.csv"] for kind in TABLE_KINDS}
    validated = _validate_tables(tables, config, manifest["research_end"])
    identity = {
        "version": manifest["version"], "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "artifacts": {name: artifacts[name]["sha256"] for name in sorted(artifacts)},
    }
    return manifest, tables, validated, identity


def canonical_identity(root=DEFAULT_OUTPUT):
    """Return the full manifest and artifact identity after verifying the flat pool."""
    return _read_dataset(root)[3]


def load_dataset(output_dir=None, cohort="benchmark", use_base=None) -> ModelData:
    """Load one pool; model evaluation roles are assigned only by its protocol.

    Benchmark uses common-price rows and includes API Base by default. Risk uses
    every retained order, excludes API Base, and never imputes it.
    """
    if cohort not in ("benchmark", "risk"):
        raise ValueError("Unknown cohort")
    if use_base is None:
        use_base = cohort == "benchmark"
    if not isinstance(use_base, bool) or (cohort == "risk" and use_base):
        raise ValueError("Risk cohort cannot request API Base")
    _, _, (features, targets, metadata), _ = _read_dataset(output_dir or DEFAULT_OUTPUT)
    mask = metadata.benchmark_eligible if cohort == "benchmark" else pd.Series(True, index=features.index)
    columns = RISK_FEATURES + ([BASE_FEATURE] if use_base else [])
    X = features.loc[mask, columns].copy()
    if X.empty:
        raise ValueError(f"No eligible rows for cohort={cohort}; inspect preprocessing reports")
    return ModelData(X=X, y=targets.loc[mask, "payout_per_btc_usdc"].copy(), metadata=metadata.loc[mask].copy())


def _guard_output(source, output):
    _check_nonoverlap(source, output)
    if output.is_symlink():
        raise ValueError("Output must not be a symbolic link")
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError("Output must be a directory")
    entries = list(output.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("Output must not contain symbolic links")
    files = {str(path.relative_to(output)) for path in entries if path.is_file()}
    directories = {str(path.relative_to(output)) for path in entries if path.is_dir()}
    if files or directories:
        if files != ARTIFACT_PATHS | {"manifest.json"} or directories != {"datasets", "reports"}:
            raise ValueError("Output contains foreign files; choose a dedicated output directory")
        canonical_identity(output)


def _new_manifest(cohort_origin):
    config = load_config()
    return {
        "producer": "pp_data", "version": config["version"], "config": config,
        "config_sha256": checksum(CONFIG_PATH), "code_sha256": code_checksums(),
        "protocol_sha256": checksum(PROTOCOL_PATH), "research_end": load_protocol()["validation_end"],
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "feature_sets": {"risk": RISK_FEATURES, "benchmark": FEATURE_COLUMNS},
        "target": "payout_per_btc_usdc", "cohort_origin": cohort_origin,
        "source_id_definition": "training_ + one-based original data row number, zero-padded to 6 digits",
        "limitations": [
            "API Base quote timestamps unavailable; historical nearest matching not independently verified",
            "Source negative markup filtering and purchased-orders-only selection remain",
            "No imputation, scaling, demand fitting, volatility thresholds, or model fitting performed",
            "A raw rebuild can expand the historical report pool; an explicit cohort reference is required to retain its source IDs",
        ],
        "artifacts": {},
    }


def _publish_dataset(source, output, result, manifest):
    """Publish a complete new directory atomically; identical existing stores are no-ops.

    A different or damaged existing store is never overwritten. Build to a new
    output directory to keep provenance and avoid partial replacement.
    """
    source, output = Path(source).resolve(), Path(output).absolute()
    if output.is_symlink():
        raise ValueError("Output must not be a symbolic link")
    output = output.resolve()
    _guard_output(source, output)
    tables = {f"datasets/{kind}.csv": getattr(result, kind) for kind in TABLE_KINDS}
    tables.update({"reports/dataset_statistics.csv": result.statistics,
                   "reports/exclusion_statistics.csv": result.exclusions})
    _validate_tables({kind: getattr(result, kind) for kind in TABLE_KINDS}, manifest["config"], manifest["research_end"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pp_data_", dir=output.parent) as temporary:
        stage = Path(temporary) / "dataset"
        stage.mkdir()
        for name, table in sorted(tables.items()):
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g")
            dtypes = {column: (str(dtype) if dtype.kind in "bif" else "object")
                      for column, dtype in table.dtypes.items()}
            manifest["artifacts"][name] = {
                "sha256": checksum(path), "rows": len(table), "columns": list(table), "dtypes": dtypes,
            }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        canonical_identity(stage)
        _guard_output(source, output)
        if output.exists() and any(output.iterdir()):
            same = all((output / name).read_bytes() == (stage / name).read_bytes()
                       for name in ARTIFACT_PATHS | {"manifest.json"})
            if same:
                return
            raise FileExistsError("Output already contains a different dataset; choose a new output directory")
        # rename is an atomic publication; it only replaces an empty directory.
        os.replace(stage, output)


def build_dataset(data_dir=DEFAULT_DATA, output_dir=DEFAULT_OUTPUT, cohort_reference=None):
    if Path(output_dir).is_symlink():
        raise ValueError("Output must not be a symbolic link")
    source, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    _guard_output(source, output)
    input_names = ["pp_transactions_training.csv", "btc_market_history.csv"]
    metadata_path = source / "pp_research_datasets_metadata.json"
    if metadata_path.exists():
        input_names.append(metadata_path.name)
    inputs = {name: checksum(source / name) for name in input_names}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if not isinstance(metadata, dict):
        raise ValueError("Source metadata must be a JSON object")
    official_start = metadata.get("validation_first_createdAt_utc")
    if official_start is not None:
        parsed = pd.to_datetime(official_start, utc=True, errors="raise")
        if pd.isna(parsed) or parsed != pd.Timestamp(load_protocol()["validation_end"]):
            raise ValueError("Metadata official boundary differs from the sole validation protocol")
    result = prepare(pd.read_csv(source / input_names[0], float_precision="round_trip"),
                     pd.read_csv(source / input_names[1], float_precision="round_trip"))
    manifest = _new_manifest("raw_quality_filtered_pool")
    if cohort_reference is not None:
        reference_path = Path(cohort_reference).resolve()
        _check_nonoverlap(reference_path, output)
        _, reference, reference_values, identity = _read_dataset(reference_path)
        ids = reference["metadata"].order_id
        available = result.metadata.order_id
        missing = set(ids) - set(available)
        if missing:
            raise ValueError(f"Cohort reference has {len(missing)} source IDs missing from rebuilt pool")
        mask = available.isin(ids)
        removed = int((~mask).sum())
        for kind in TABLE_KINDS:
            setattr(result, kind, getattr(result, kind).loc[mask].reset_index(drop=True))
        rebuilt_values = _validate_tables(
            {kind: getattr(result, kind) for kind in TABLE_KINDS},
            manifest["config"], manifest["research_end"])
        # Row-number IDs alone do not identify orders if the raw file was reordered.
        # Allow only negligible CSV/arithmetic rounding, never changed contracts,
        # labels, timestamps or pricing features within the retained cohort.
        try:
            for rebuilt, previous in zip(rebuilt_values, reference_values):
                pd.testing.assert_frame_equal(rebuilt, previous, check_dtype=False,
                                              check_exact=False, rtol=1e-12, atol=1e-9)
        except AssertionError as error:
            raise ValueError("Rebuilt rows differ from reference data; do not reuse changed source IDs") from error
        result.statistics = pool_statistics(result.metadata)
        result.exclusions.loc[result.exclusions.reason.eq("retained"), "rows"] = len(result.metadata)
        result.exclusions = pd.concat([result.exclusions, pd.DataFrame([{
            "reason": "outside_reference_cohort", "rows": removed,
            "definition": "Eligible source IDs not present in the explicitly supplied reference pool",
        }])], ignore_index=True)
        manifest["cohort_origin"] = "reference_source_ids"
        manifest["cohort_reference"] = identity
    if inputs != {name: checksum(source / name) for name in input_names}:
        raise ValueError("Source files changed during preprocessing; retry on a stable snapshot")
    manifest["input_sha256"] = inputs
    _publish_dataset(source, output, result, manifest)
    return result
