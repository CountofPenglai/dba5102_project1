"""Validated, explicit experiment configuration; temporal splits belong to pp_data."""
import copy
import json
import math
from pathlib import Path
from pp_data.config import PROJECT
from pp_data.features import BASE_FEATURE, RISK_FEATURES

CONFIG_PATH = PROJECT / "config" / "models_v1.json"
MODEL_NAMES = ("gam", "ebm", "shallow_gbdt")
PARAMETERS = {
    "gam": {"n_splines", "lam", "max_iter", "tol"},
    "ebm": {"max_bins", "max_interaction_bins", "learning_rate", "max_rounds",
            "min_samples_leaf", "reg_lambda", "smoothing_rounds",
            "interaction_smoothing_rounds", "interactions"},
    "shallow_gbdt": {"max_depth", "max_leaf_nodes", "max_iter", "learning_rate",
                     "min_samples_leaf", "l2_regularization"},
}
INTEGER_PARAMETERS = {"n_splines", "max_iter", "max_bins", "max_interaction_bins",
                      "max_rounds", "min_samples_leaf", "smoothing_rounds",
                      "interaction_smoothing_rounds", "max_depth", "max_leaf_nodes"}


def _number(value, name, minimum=0, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < minimum
            or (integer and not isinstance(value, int))):
        raise ValueError(f"Invalid {name}: {value!r}")


def load_model_config(path=None):
    cfg = json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    keys = {"version", "window_days", "min_train_rows", "seed", "features", "models"}
    if not isinstance(cfg, dict) or set(cfg) != keys or cfg["version"] != "models_v1":
        raise ValueError("Unknown model configuration keys/version")
    for key in ("window_days", "min_train_rows"):
        _number(cfg[key], key, minimum=1, integer=True)
    _number(cfg["seed"], "seed", integer=True)
    if cfg["seed"] > 2**31 - 1:
        raise ValueError("seed exceeds supported range")
    features = cfg["features"]
    if (not isinstance(features, list) or not features
            or not all(isinstance(f, str) for f in features)
            or len(features) != len(set(features)) or not set(features) <= set(RISK_FEATURES)):
        raise ValueError("Features must be unique members of the pricing-time risk whitelist")
    if not isinstance(cfg["models"], dict) or set(cfg["models"]) != set(MODEL_NAMES):
        raise ValueError("Expected exactly gam, ebm and shallow_gbdt configurations")
    for name, params in cfg["models"].items():
        if not isinstance(params, dict) or set(params) != PARAMETERS[name]:
            raise ValueError(f"Unknown/missing parameters for {name}")
        for key, value in params.items():
            if key != "interactions":
                minimum = 0 if key in {"lam", "reg_lambda", "l2_regularization",
                                      "smoothing_rounds", "interaction_smoothing_rounds"} else 1e-12
                _number(value, key, minimum, key in INTEGER_PARAMETERS)
        if name == "gam" and params["n_splines"] < 4:
            raise ValueError("Cubic GAM requires at least four splines")
        if name == "shallow_gbdt" and (params["max_depth"] > 3 or params["max_leaf_nodes"] < 2):
            raise ValueError("shallow_gbdt requires depth <= 3 and at least two leaves")
        if name == "ebm":
            if params["max_bins"] < 3 or params["max_interaction_bins"] < 3:
                raise ValueError("EBM requires at least three bins")
            pairs = params["interactions"]
            if not isinstance(pairs, list) or len(pairs) > 2:
                raise ValueError("At most two explicit interactions are allowed")
            seen = set()
            for pair in pairs:
                if (not isinstance(pair, list) or len(pair) != 2
                        or not all(isinstance(f, str) and f in features for f in pair)
                        or pair[0] == pair[1] or frozenset(pair) in seen):
                    raise ValueError("Interaction must name two distinct selected features")
                seen.add(frozenset(pair))


def resolve_config(cfg, cohort="risk", use_base=False, models=None):
    validate_config(cfg)
    if cohort not in ("risk", "benchmark") or not isinstance(use_base, bool):
        raise ValueError("Invalid cohort or use_base")
    if use_base and cohort != "benchmark":
        raise ValueError("API Base requires the benchmark common cohort")
    names = list(MODEL_NAMES if models is None else models)
    if not names or len(names) != len(set(names)) or not set(names) <= set(MODEL_NAMES):
        raise ValueError("Invalid model selection")
    result = copy.deepcopy(cfg)
    result.update(cohort=cohort, use_base=use_base, selected_models=names,
                  target="payout_per_btc_usdc", protocol="daily_prequential_v1")
    if use_base:
        result["features"].append(BASE_FEATURE)
    return result
