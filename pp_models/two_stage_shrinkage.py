"""Fixed prediction blending and direction-level admission for branch 3."""
import numpy as np
import pandas as pd

WEIGHTS = (1.0, 0.5, 0.25, 0.0)
REQUIRED_METRICS = ("all_mae", "body_mae", "tail_rmse", "gross_underprediction", "absolute_total_bias")


def _series(value, name):
    if not isinstance(value, pd.Series) or value.empty or not value.index.is_unique:
        raise ValueError(f"Invalid {name} predictions")
    data = value.to_numpy(float)
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError(f"Invalid {name} predictions")
    return data


def blend_predictions(base, raw, weight):
    if weight not in WEIGHTS:
        raise ValueError("Unsupported shrinkage weight")
    b, r = _series(base, "base"), _series(raw, "raw")
    if not base.index.equals(raw.index):
        raise ValueError("Misaligned shrinkage predictions")
    return pd.Series(b + weight * (r - b), index=base.index, name=raw.name)


def combine_direction_weights(base, raw, option_type, weights):
    if (not isinstance(option_type, pd.Series) or not base.index.equals(option_type.index)
            or set(weights) != {0, 1} or not option_type.isin([0, 1]).all()):
        raise ValueError("Invalid direction weights or alignment")
    result = base.copy().astype(float)
    for side in (0, 1):
        mask = option_type.eq(side)
        result.loc[mask] = blend_predictions(base.loc[mask], raw.loc[mask], weights[side])
    return result


def select_direction_weight(metrics_by_weight, base_metrics, tail_orders, tail_dates):
    if tail_orders < 30 or tail_dates < 5:
        return {"weight": 0.0, "passed": False, "reason": "insufficient_tail_support"}
    if set(metrics_by_weight) != set(WEIGHTS) or any(k not in base_metrics for k in REQUIRED_METRICS):
        raise ValueError("Incomplete shrinkage metrics")
    for weight in WEIGHTS:
        current = metrics_by_weight[weight]
        if any(k not in current for k in REQUIRED_METRICS):
            raise ValueError("Incomplete direction metric row")
        values = np.array([current[k] for k in REQUIRED_METRICS] + [base_metrics[k] for k in REQUIRED_METRICS], float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid direction metric values")
        checks = {
            "all_mae_pass": current["all_mae"] <= base_metrics["all_mae"],
            "body_mae_pass": current["body_mae"] <= 1.02 * base_metrics["body_mae"],
            "tail_rmse_pass": current["tail_rmse"] <= 1.02 * base_metrics["tail_rmse"],
            "gross_underprediction_pass": current["gross_underprediction"]
                                               <= 1.02 * base_metrics["gross_underprediction"],
            "absolute_total_bias_pass": current["absolute_total_bias"] <= base_metrics["absolute_total_bias"],
        }
        if all(checks.values()):
            return {"weight": weight, "passed": True, "reason": "first_passing_weight", **checks}
    return {"weight": 0.0, "passed": False, "reason": "no_weight_passed"}
