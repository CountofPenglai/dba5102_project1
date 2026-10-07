"""Read only canonical train/validation bytes; never open test artifacts."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pp_data.config import CONFIG_PATH, DEFAULT_OUTPUT, load_config
from pp_data.features import BASE_FEATURE, FEATURE_COLUMNS, RISK_FEATURES
from pp_data.io import ModelSplit, checksum, code_checksums

DEVELOPMENT_FILES = tuple(f"datasets/{split}/{kind}.csv" for split in ("train", "validation")
                          for kind in ("features", "targets", "metadata"))


def development_snapshot(dataset_dir=DEFAULT_OUTPUT):
    root = Path(dataset_dir or DEFAULT_OUTPUT)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("producer") != "pp_data" or manifest.get("config") != load_config()
            or manifest.get("config_sha256") != checksum(CONFIG_PATH)
            or manifest.get("code_sha256") != code_checksums()):
        raise ValueError("Canonical preprocessing identity changed")
    captured, artifacts = {}, {}
    for name in DEVELOPMENT_FILES:
        info = manifest.get("artifacts", {}).get(name)
        if not isinstance(info, dict):
            raise ValueError(f"Missing development artifact: {name}")
        payload = (root / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != info["sha256"]:
            raise ValueError(f"Development checksum mismatch: {name}")
        captured[name], artifacts[name] = payload, info
    identity = {"scope": ["train", "validation"], "config_sha256": manifest["config_sha256"],
                "preprocessing_code": manifest["code_sha256"], "artifacts": artifacts}
    return identity, captured


def load_validation_pool(resolved, dataset_dir=DEFAULT_OUTPUT):
    identity, captured = development_snapshot(dataset_dir)
    cohort, use_base = resolved["cohort"], resolved["use_base"]
    if cohort not in ("risk", "benchmark") or not isinstance(use_base, bool) or (use_base and cohort != "benchmark"):
        raise ValueError("Invalid development cohort/base selection")
    parts = []
    for split in ("train", "validation"):
        tables = {}
        for kind in ("features", "targets", "metadata"):
            name = f"datasets/{split}/{kind}.csv"
            table = pd.read_csv(BytesIO(captured[name]), float_precision="round_trip")
            info = identity["artifacts"][name]
            if (len(table) != info["rows"] or table.columns.tolist() != info["columns"]
                    or table.order_id.isna().any() or table.order_id.duplicated().any()):
                raise ValueError(f"Development schema mismatch: {name}")
            tables[kind] = table.set_index("order_id")
        X, targets, meta = (tables[k] for k in ("features", "targets", "metadata"))
        if X.columns.tolist() != FEATURE_COLUMNS or not X.index.equals(targets.index) or not X.index.equals(meta.index):
            raise ValueError("Development feature whitelist or IDs mismatch")
        for column in ("created_at_utc", "closed_at_utc", "expiration_at_utc", "last_candle_close_utc"):
            meta[column] = pd.to_datetime(meta[column], utc=True, errors="raise")
        bounds = load_config()["splits"][split]
        valid = (meta.created_at_utc.ge(pd.Timestamp(bounds["start"]))
                 & meta.created_at_utc.lt(pd.Timestamp(bounds["end"]))
                 & meta.closed_at_utc.lt(pd.Timestamp(bounds["end"]))
                 & meta.closed_at_utc.ge(meta.created_at_utc)
                 & meta.last_candle_close_utc.le(meta.created_at_utc))
        if not valid.all():
            raise ValueError("Development temporal bounds violated")
        mask = meta.benchmark_eligible.eq(True) if cohort == "benchmark" else pd.Series(True, index=X.index)
        X = X.loc[mask, RISK_FEATURES + ([BASE_FEATURE] if use_base else [])].copy()
        y = targets.loc[mask, "payout_per_btc_usdc"].copy()
        if X.empty or not np.isfinite(X).all().all() or not np.isfinite(y).all() or y.lt(0).any():
            raise ValueError(f"Invalid or empty development split: {split}")
        parts.append(ModelSplit(X, y, meta.loc[mask].copy()))
    pool = ModelSplit(pd.concat([p.X for p in parts]), pd.concat([p.y for p in parts]),
                      pd.concat([p.metadata for p in parts]))
    if not pool.X.index.is_unique:
        raise ValueError("Duplicate development IDs")
    order = pool.metadata.sort_values("created_at_utc", kind="stable").index
    return ModelSplit(pool.X.loc[order], pool.y.loc[order], pool.metadata.loc[order])
