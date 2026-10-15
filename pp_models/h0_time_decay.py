"""Original H0 with exponential creation-age weights in both fitted stages."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pp_data.features import BASE_FEATURE
from .h0 import H0, hurdle_design


def time_weights(created, origin, half_life_days):
    origin = pd.Timestamp(origin)
    if (not isinstance(created, pd.Series) or created.isna().any() or created.dt.tz is None
            or origin.tzinfo is None or not created.lt(origin).all()
            or (half_life_days is not None and (not np.isfinite(half_life_days) or half_life_days <= 0))):
        raise ValueError('Time weights need strictly historical timezone-aware creation times')
    if half_life_days is None:
        return np.ones(len(created))
    age = (origin-created).dt.total_seconds().to_numpy()/86400
    weights = np.exp2(-age/half_life_days)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Time weights underflowed; use a finite history window')
    return weights


def effective_size(weights):
    return float(weights.sum()**2/(weights@weights)) if len(weights) and (weights@weights) > 0 else 0.


class TimeDecayH0(H0):
    def __init__(self, half_life_days):
        super().__init__()
        self.half_life_days = half_life_days

    def fit(self, X, y, created, origin):
        self._check(X)
        if (not isinstance(y, pd.Series) or not X.index.equals(y.index) or not np.isfinite(y).all()
                or y.lt(0).any() or not isinstance(created, pd.Series) or not X.index.equals(created.index)):
            raise ValueError('Invalid aligned payout targets or creation times')
        raw = time_weights(created, origin, self.half_life_days)
        self.fitted = False; self.parameters = {}; self.detail = {}; self.weight_diagnostics = {}
        # Invoke the original implementation for the unweighted control, including its fallbacks.
        if self.half_life_days is None:
            super().fit(X, y, created)
        for side in (0, 1):
            mask = X.option_type.eq(side); n = int(mask.sum())
            b = X.loc[mask, BASE_FEATURE].to_numpy(float); target = y.loc[mask].to_numpy(float)
            positive = target > 0; w = raw[mask.to_numpy()]
            weights = w/w.mean() if n else w
            if self.half_life_days is not None:
                par = dict(c=1., probability=1., severity=1.)
                if n:
                    par['probability'] = 1. if positive.all() else float(weights@positive/weights.sum())
                    par['severity'] = float(np.sum(w[positive]*b[positive]*target[positive]) /
                        np.sum(w[positive]*b[positive]**2)) if positive.any() else 0.
                    if 0 < positive.sum() < n:
                        design = hurdle_design(X.loc[mask]); scaler = StandardScaler().fit(design)
                        model = LogisticRegression(C=.1, max_iter=1000, tol=1e-9, solver='lbfgs')
                        model.fit(np.clip(scaler.transform(design), -3, 3), positive.astype(int), sample_weight=weights)
                        par.update(scaler=scaler, model=model)
                self.parameters[side] = par
            pos_w = w[positive]; leverage = pos_w*b[positive]**2
            months = created.loc[mask].dt.strftime('%Y-%m').to_numpy()
            numerator = w*b*target; denominator = w*b*b*positive
            self.weight_diagnostics[side] = dict(orders=n, positive_orders=int(positive.sum()),
                creation_days=int(created.loc[mask].dt.floor('D').nunique()),
                raw_weight_sum=float(w.sum()), normalized_weight_sum=float(weights.sum()),
                effective_orders=effective_size(w), effective_positive_orders=effective_size(pos_w),
                severity_leverage_effective_orders=effective_size(leverage),
                weighted_mean_age_days=float(np.average((pd.Timestamp(origin)-created.loc[mask]).dt.total_seconds()/86400, weights=w)) if n else None,
                numerator=float(numerator.sum()), denominator=float(denominator.sum()),
                historical_months={month: dict(orders=int((months == month).sum()),
                    weight_sum=float(w[months == month].sum()),
                    severity_numerator=float(numerator[months == month].sum()),
                    severity_denominator=float(denominator[months == month].sum())) for month in sorted(set(months))})
        self.fitted = True
        return self
