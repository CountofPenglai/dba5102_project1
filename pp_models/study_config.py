"""Bounded, sequential search specifications restricted to validation."""
import copy
import json
from pathlib import Path
import math
import pandas as pd
from pp_data.config import PROJECT, load_config
from .config import MODEL_NAMES, PARAMETERS, resolve_config, validate_config
from .runs import fingerprint

STUDY_CONFIG = PROJECT / "config/validation_study_v1.json"


def load_study_config(path=None):
    return json.loads(Path(path or STUDY_CONFIG).read_text(encoding="utf-8"))


def validate_study(base, study, models=None):
    keys = {"version", "cohort", "use_base", "threads", "rmse_tolerance", "baseline_smoothing",
            "periods", "stages", "seconds_per_fit_assumption"}
    if not isinstance(study, dict) or set(study) != keys or study["version"] != "validation_study_v1":
        raise ValueError("Unknown study configuration")
    validate_config(base)
    if base["window_days"] != 30 or study["threads"] != 1 or study["cohort"] != "risk" or study["use_base"] is not False:
        raise ValueError("First study fixes 30 days, one thread, risk cohort and no Base")
    for name in ("rmse_tolerance", "baseline_smoothing"):
        value = study[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {name}")
    if study["rmse_tolerance"] > .05 or study["baseline_smoothing"] == 0:
        raise ValueError("Invalid selection tolerance or baseline smoothing")
    bounds = load_config()["splits"]["validation"]
    expected = {"tuning": [pd.Timestamp(bounds["start"]), pd.Timestamp("2026-06-21T00:00Z")],
                "confirmation": [pd.Timestamp("2026-06-21T00:00Z"), pd.Timestamp(bounds["end"])]}
    if (not isinstance(study["periods"], dict) or set(study["periods"]) != set(expected)
            or any([pd.Timestamp(t) for t in study["periods"][k]] != expected[k] for k in expected)):
        raise ValueError("Study must use the frozen June tuning/confirmation periods")
    if set(study["stages"]) != set(MODEL_NAMES) or set(study["seconds_per_fit_assumption"]) != set(MODEL_NAMES):
        raise ValueError("Expected three model search specifications")
    for model in MODEL_NAMES:
        stages, seen = study["stages"][model], set()
        if not isinstance(stages, list) or not stages:
            raise ValueError("Empty search stages")
        count = 1
        for stage in stages:
            if not isinstance(stage, dict) or set(stage) != {"parameter", "values"}:
                raise ValueError("Invalid search stage")
            parameter, values = stage["parameter"], stage["values"]
            if parameter not in PARAMETERS[model] or parameter in seen or not isinstance(values, list) or not values:
                raise ValueError("Invalid/repeated search parameter")
            seen.add(parameter)
            if len({fingerprint(v) for v in values}) != len(values) or base["models"][model][parameter] not in values:
                raise ValueError("Each stage requires unique values including the initial setting")
            for value in values:
                candidate = copy.deepcopy(base)
                candidate["models"][model][parameter] = value
                validate_config(candidate)
            count += len(values) - 1
        if count > 6:
            raise ValueError("At most six unique candidates per model in this study")
        timing = study["seconds_per_fit_assumption"][model]
        if (not isinstance(timing, list) or len(timing) != 2 or
                not all(type(v) in (int, float) and math.isfinite(v) for v in timing) or
                not 0 < timing[0] <= timing[1]):
            raise ValueError("Invalid unmeasured timing interval")
    return resolve_config(base, study["cohort"], study["use_base"], models)


def candidate_config(base, model, parameter, value, study):
    raw = copy.deepcopy(base)
    raw["models"][model][parameter] = copy.deepcopy(value)
    return raw, resolve_config(raw, study["cohort"], study["use_base"], [model])
