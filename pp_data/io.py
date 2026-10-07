"""Reproducible dataset artifacts and guarded model-facing loading."""

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
from .config import CONFIG_PATH, DEFAULT_DATA, DEFAULT_OUTPUT, PROJECT, load_config
from .features import BASE_FEATURE, FEATURE_COLUMNS, RISK_FEATURES
from .pipeline import prepare

SPLITS = ("train", "validation", "test")
ARTIFACT_PATHS = {
    f"datasets/{split}/{table}.csv" for split in SPLITS
    for table in ("features", "targets", "metadata")
} | {"reports/split_statistics.csv", "reports/exclusion_statistics.csv"}


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_checksums():
    return {str(p.relative_to(PROJECT)): checksum(p)
            for package in ("pp_audit", "pp_data")
            for p in sorted((PROJECT / package).glob("*.py"))}


def _guard_output(source, output):
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output and source directories must not overlap")
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError("Output must be a directory")
    entries = list(output.rglob("*"))
    if any(p.is_symlink() for p in entries):
        raise ValueError("Output must not contain symbolic links")
    files = {str(p.relative_to(output)) for p in entries if p.is_file()}
    if files:
        if files - (ARTIFACT_PATHS | {"manifest.json"}) or "manifest.json" not in files:
            raise ValueError("Output contains foreign files; choose a dedicated output directory")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("producer") != "pp_data":
            raise ValueError("Output is not a generated pp_data dataset")


def build_dataset(data_dir=DEFAULT_DATA, output_dir=DEFAULT_OUTPUT):
    if Path(output_dir).is_symlink():
        raise ValueError("Output directory must not be a symbolic link")
    source, output = Path(data_dir).resolve(), Path(output_dir).resolve()
    _guard_output(source, output)
    config = load_config()
    input_names = ["pp_transactions_training.csv", "btc_market_history.csv"]
    metadata_path = source / "pp_research_datasets_metadata.json"
    if metadata_path.exists():
        input_names.append(metadata_path.name)
    inputs = {name: checksum(source / name) for name in input_names}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object")
    official_start = metadata.get("validation_first_createdAt_utc")
    if official_start is not None:
        parsed = pd.to_datetime(official_start, utc=True, errors="raise")
        if pd.isna(parsed) or parsed != pd.Timestamp(config["splits"]["test"]["end"]):
            raise ValueError("Metadata official holdout boundary differs from frozen config")
    result = prepare(pd.read_csv(source / input_names[0]), pd.read_csv(source / input_names[1]))
    tables = {
        f"datasets/{split}/{kind}.csv": getattr(bundle, kind)
        for split, bundle in result.splits.items()
        for kind in ("features", "targets", "metadata")
    }
    tables.update({"reports/split_statistics.csv": result.statistics,
                   "reports/exclusion_statistics.csv": result.exclusions})
    if inputs != {name: checksum(source / name) for name in input_names}:
        raise ValueError("Source files changed during preprocessing; retry on a stable snapshot")
    manifest = {
        "producer": "pp_data", "version": config["version"], "config": config,
        "config_sha256": checksum(CONFIG_PATH), "code_sha256": code_checksums(),
        "input_sha256": inputs,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "feature_sets": {"risk": RISK_FEATURES, "benchmark": FEATURE_COLUMNS},
        "target": "payout_per_btc_usdc",
        "source_id_definition": "training_ + one-based original data row number, zero-padded to 6 digits",
        "official_validation_file_present": (source / "pp_transactions_validation.csv").is_file(),
        "limitations": [
            "API Base quote timestamps unavailable; historical nearest matching not independently verified",
            "July internal test descriptive statistics were previously inspected; not an untouched blind test",
            "Source negative markup filtering and purchased-orders-only selection remain",
            "No imputation, scaling, demand fitting, volatility thresholds, or model fitting performed",
        ],
        "artifacts": {},
    }
    # Stage every artifact first. Publish the manifest last; checksum verification rejects mixed runs.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pp_data_", dir=output.parent) as temporary:
        stage = Path(temporary)
        for name, table in sorted(tables.items()):
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(destination, index=False, encoding="utf-8-sig", float_format="%.17g")
            manifest["artifacts"][name] = {
                "sha256": checksum(destination), "rows": len(table), "columns": table.columns.tolist(),
            }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        _guard_output(source, output)
        for name in sorted(tables) + ["manifest.json"]:
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, destination)
    return result


@dataclass
class ModelSplit:
    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame


def load_split(split, cohort="benchmark", use_base=None, output_dir=None) -> ModelSplit:
    """Load one fixed split. Do not merge metadata into X.

    benchmark: common-price subset, API Base included by default.
    risk: all retained orders, excludes API Base and never imputes it.
    use_base=False on benchmark preserves identical rows for feature ablations.
    """
    if split not in SPLITS or cohort not in ("benchmark", "risk"):
        raise ValueError("Unknown split or cohort")
    if use_base is None:
        use_base = cohort == "benchmark"
    if not isinstance(use_base, bool) or (cohort == "risk" and use_base):
        raise ValueError("Risk cohort cannot request API Base; choose benchmark or use_base=False")
    root = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("producer") != "pp_data":
        raise ValueError("Invalid dataset manifest")
    if manifest.get("config") != load_config() or manifest.get("config_sha256") != checksum(CONFIG_PATH):
        raise ValueError("Dataset config differs from the canonical config; regenerate")
    if manifest.get("code_sha256") != code_checksums():
        raise ValueError("Preprocessing code changed; regenerate the canonical dataset")
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != ARTIFACT_PATHS:
        raise ValueError("Dataset artifact list is incomplete or unexpected")
    snapshots = {}
    for name in sorted(ARTIFACT_PATHS):
        path = root / name
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError(f"Dataset checksum unavailable: {name}; regenerate") from error
        if hashlib.sha256(payload).hexdigest() != artifacts[name]["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {name}; regenerate")
        if name.startswith(f"datasets/{split}/"):
            snapshots[name] = payload
    loaded = {}
    for kind in ("features", "targets", "metadata"):
        name = f"datasets/{split}/{kind}.csv"
        # Parse exactly the bytes verified above, even if a rebuild replaces the files now.
        table = pd.read_csv(BytesIO(snapshots[name]), float_precision="round_trip")
        if len(table) != artifacts[name]["rows"] or table.columns.tolist() != artifacts[name]["columns"]:
            raise ValueError(f"Dataset schema mismatch: {name}")
        if table.order_id.isna().any() or table.order_id.duplicated().any():
            raise ValueError(f"Invalid order IDs: {name}")
        loaded[kind] = table.set_index("order_id")
    features, targets, metadata = (loaded[k] for k in ("features", "targets", "metadata"))
    if features.columns.tolist() != FEATURE_COLUMNS:
        raise ValueError("Feature whitelist mismatch")
    if not features.index.equals(targets.index) or not features.index.equals(metadata.index):
        raise ValueError("Features, targets and metadata IDs are not aligned")
    for name in ("created_at_utc", "closed_at_utc", "expiration_at_utc", "last_candle_close_utc"):
        metadata[name] = pd.to_datetime(metadata[name], utc=True, errors="raise")
    bounds = manifest["config"]["splits"][split]
    valid_time = (
        metadata.created_at_utc.ge(pd.Timestamp(bounds["start"]))
        & metadata.created_at_utc.lt(pd.Timestamp(bounds["end"]))
        & metadata.closed_at_utc.lt(pd.Timestamp(bounds["end"]))
        & metadata.last_candle_close_utc.le(metadata.created_at_utc)
    )
    if not valid_time.all():
        raise ValueError("Dataset violates fixed temporal boundaries")
    mask = metadata.benchmark_eligible.eq(True) if cohort == "benchmark" else pd.Series(True, index=features.index)
    columns = RISK_FEATURES + ([BASE_FEATURE] if use_base else [])
    X = features.loc[mask, columns].copy()
    y = targets.loc[mask, "payout_per_btc_usdc"].copy()
    if X.empty:
        raise ValueError(f"No eligible rows for split={split}, cohort={cohort}; inspect preprocessing reports")
    if not np.isfinite(X).all().all() or not np.isfinite(y).all() or y.lt(0).any():
        raise ValueError("Selected dataset contains invalid model values")
    return ModelSplit(X=X, y=y, metadata=metadata.loc[mask].copy())
