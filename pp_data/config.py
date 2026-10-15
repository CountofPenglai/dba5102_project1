"""Flat preprocessing settings and the sole model evaluation protocol."""

import json
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "config" / "preprocessing_v2.json"
PROTOCOL_PATH = PROJECT / "config" / "validation_protocol_v3.json"
DEFAULT_DATA = PROJECT.parent / "docs" / "Jasper Vault Options Pricing Optimisation(Included Base Premium)"
DEFAULT_OUTPUT = PROJECT / "outputs" / "preprocessing"
WINDOWS = (60, 240, 1440)


def load_protocol():
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["version"] != "rolling_validation_v3" or protocol["timezone"] != "UTC":
        raise ValueError("Unsupported validation protocol or timezone")
    dates = [pd.Timestamp(protocol[key])
             for key in ("training_start", "validation_start", "validation_end")]
    if any(pd.isna(value) or value.tzinfo is None for value in dates) or not dates[0] < dates[1] < dates[2]:
        raise ValueError("Validation dates must be timezone-aware and increasing")
    if protocol.get("official_validation") != "forbidden":
        raise ValueError("Official validation must remain forbidden")
    return protocol


def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["version"] != "preprocessing_v2" or config["timezone"] != "UTC" or "splits" in config:
        raise ValueError("Unsupported flat preprocessing config")
    start = pd.Timestamp(config["research_start"])
    if pd.isna(start) or start.tzinfo is None or not start < pd.Timestamp(load_protocol()["training_start"]):
        raise ValueError("Research start must precede model training")
    if config["research_end_source"] != "validation_protocol_v3.json:validation_end":
        raise ValueError("Research end must use the sole validation protocol")
    if tuple(config["market_windows_minutes"]) != WINDOWS or config["max_candle_age_seconds"] != 60:
        raise ValueError("Feature window changes require a new preprocessing version")
    return config
