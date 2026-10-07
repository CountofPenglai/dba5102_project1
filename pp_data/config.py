"""The single versioned split definition; no per-experiment split overrides."""

import json
from pathlib import Path
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "config" / "preprocessing_v1.json"
DEFAULT_DATA = PROJECT.parent / "docs" / "Jasper Vault Options Pricing Optimisation(Included Base Premium)"
DEFAULT_OUTPUT = PROJECT / "outputs" / "preprocessing"
WINDOWS = (60, 240, 1440)


def load_config():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["version"] != "preprocessing_v1" or config["timezone"] != "UTC":
        raise ValueError("Unsupported preprocessing version or timezone")
    if list(config["splits"]) != ["train", "validation", "test"]:
        raise ValueError("Expected train, validation, test in chronological order")
    previous_end = None
    for bounds in config["splits"].values():
        start, end = (pd.Timestamp(bounds[key]) for key in ("start", "end"))
        if start.tzinfo is None or end.tzinfo is None or not start < end:
            raise ValueError("Split boundaries must be timezone-aware and increasing")
        if previous_end is not None and start != previous_end:
            raise ValueError("Split windows must be contiguous and non-overlapping")
        previous_end = end
    if tuple(config["market_windows_minutes"]) != WINDOWS or config["max_candle_age_seconds"] != 60:
        raise ValueError("Feature window changes require a new preprocessing version")
    return config
