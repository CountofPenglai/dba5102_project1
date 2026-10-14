"""Historical payout-probability controls for branch 4."""
import numpy as np
import pandas as pd

LEVELS = ("direction_frequency", "direction_tenor_frequency")


class HistoricalPayoutProbability:
    def __init__(self, level, smoothing=30):
        if level not in LEVELS or isinstance(smoothing, bool) or smoothing != 30:
            raise ValueError("Unsupported historical probability definition")
        self.level, self.smoothing = level, smoothing

    def fit(self, X, y):
        required = ["option_type", "product_duration_seconds"]
        if (not isinstance(X, pd.DataFrame) or X.empty or list(X) != required or not X.index.is_unique
                or not isinstance(y, pd.Series) or not X.index.equals(y.index)
                or not np.isfinite(X.to_numpy(float)).all() or not np.isfinite(y.to_numpy(float)).all()
                or not X.option_type.isin([0, 1]).all() or X.product_duration_seconds.le(0).any() or y.lt(0).any()):
            raise ValueError("Invalid historical probability training data")
        positive = y.gt(0)
        self.direction, self.cells, self.detail = {}, {}, {}
        for side in (0, 1):
            mask = X.option_type.eq(side)
            self.direction[side] = float(positive.loc[mask].mean()) if mask.any() else 1.0
            self.detail[f"direction_{side}_orders"] = int(mask.sum())
            self.detail[f"direction_{side}_positive_orders"] = int(positive.loc[mask].sum())
            for tenor, part in X.loc[mask].groupby("product_duration_seconds"):
                ids = part.index
                self.cells[(side, tenor)] = (int(positive.loc[ids].sum()), len(ids))
        return self

    def predict_probability(self, X):
        if not hasattr(self, "direction"):
            raise ValueError("Historical probability not fitted")
        required = ["option_type", "product_duration_seconds"]
        if (not isinstance(X, pd.DataFrame) or X.empty or list(X) != required or not X.index.is_unique
                or not np.isfinite(X.to_numpy(float)).all() or not X.option_type.isin([0, 1]).all()
                or X.product_duration_seconds.le(0).any()):
            raise ValueError("Invalid historical probability scoring data")
        values, missing = [], 0
        for side, tenor in X.to_numpy():
            p = self.direction[int(side)]
            if self.level == "direction_tenor_frequency":
                key = (int(side), tenor)
                if key in self.cells:
                    positive, total = self.cells[key]
                    p = (positive + self.smoothing * p) / (total + self.smoothing)
                else:
                    missing += 1
            values.append(p)
        self.detail["scoring_missing_tenor_fallbacks"] = missing
        result = np.array(values, float)
        if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
            raise ValueError("Invalid historical probabilities")
        return result

    def diagnostics(self):
        if not hasattr(self, "detail"):
            raise ValueError("Historical probability not fitted")
        return dict(self.detail)


def combine_components(probability, severity, index):
    p, s = np.asarray(probability, float), np.asarray(severity, float)
    if (p.shape != s.shape or p.shape != (len(index),) or not np.isfinite(p).all()
            or not np.isfinite(s).all() or ((p < 0) | (p > 1)).any() or (s < 0).any()):
        raise ValueError("Invalid probability/severity components")
    return pd.Series(p * s, index=index)
