"""Fixed rolling mean comparators; daily history is supplied by the evaluator."""
import numpy as np
import pandas as pd

BASELINES = ("rolling_mean_30d", "rolling_group_mean_30d")


class RollingMean:
    def __init__(self, grouped=False, smoothing=30):
        self.grouped, self.smoothing = grouped, smoothing

    def fit(self, X, y):
        if not X.index.equals(y.index) or y.empty or not np.isfinite(y).all() or y.lt(0).any():
            raise ValueError("Invalid mean baseline training data")
        self.mean = float(y.mean())
        self.groups = {}
        if self.grouped:
            frame = X[["option_type", "product_duration_seconds"]].assign(target=y)
            for key, group in frame.groupby(["option_type", "product_duration_seconds"]):
                self.groups[key] = (group.target.sum() + self.smoothing * self.mean) / (len(group) + self.smoothing)
        return self

    def predict_raw(self, X):
        if not hasattr(self, "mean"):
            raise ValueError("Mean baseline is not fitted")
        if not self.grouped:
            return np.full(len(X), self.mean)
        return np.array([self.groups.get(tuple(row), self.mean) for row in
                         X[["option_type", "product_duration_seconds"]].to_numpy()])


def build_baseline(name, resolved):
    if name not in BASELINES:
        raise ValueError("Unknown baseline")
    return RollingMean(name == BASELINES[1], resolved.get("baseline_smoothing", 30))
