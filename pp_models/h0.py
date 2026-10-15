"""Direction-specific two-stage H0, with pricing-time inputs only."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pp_data.features import BASE_FEATURE

MODEL_INPUTS = ['option_type', BASE_FEATURE, 'remaining_seconds',
                'product_duration_seconds', 'realized_vol_60m', 'log_moneyness']


def probability_frame(X):
    return X[MODEL_INPUTS].copy()


def hurdle_design(X):
    uncertainty = np.maximum(X.realized_vol_60m.to_numpy()*np.sqrt(X.remaining_seconds.to_numpy()/3600), 1e-6)
    aligned = (1-2*X.option_type.to_numpy())*X.log_moneyness.to_numpy()/uncertainty
    return np.column_stack([np.log(X[BASE_FEATURE]), np.log(X.remaining_seconds), np.clip(aligned, -5, 5)])


class H0:
    def __init__(self):
        self.fitted = False

    def _check(self, X):
        if (not isinstance(X, pd.DataFrame) or X.empty or not X.index.is_unique or list(X) != MODEL_INPUTS
                or not np.isfinite(X.to_numpy(float)).all() or not X.option_type.isin([0, 1]).all()
                or X[[BASE_FEATURE, 'remaining_seconds', 'product_duration_seconds']].le(0).any().any()
                or X.realized_vol_60m.lt(0).any()):
            raise ValueError('Invalid pricing-time H0 inputs')

    def fit(self, X, y, created=None):
        self._check(X)
        if not isinstance(y, pd.Series) or not X.index.equals(y.index) or not np.isfinite(y).all() or y.lt(0).any():
            raise ValueError('Invalid payout target alignment')
        self.parameters = {}; self.detail = {}; self.fitted = False
        for side in (0, 1):
            mask = X.option_type.eq(side)
            b = X.loc[mask, BASE_FEATURE].to_numpy(float); target = y.loc[mask].to_numpy(float)
            positive = target > 0; n = len(target)
            parameter = dict(probability=1., severity=1.)
            if n:
                parameter['probability'] = float(positive.mean())
                parameter['severity'] = float(b[positive]@target[positive]/(b[positive]@b[positive])) if positive.any() else 0.
                if 0 < positive.sum() < n:
                    design = hurdle_design(X.loc[mask]); scaler = StandardScaler().fit(design)
                    model = LogisticRegression(C=.1, max_iter=1000, tol=1e-9, solver='lbfgs')
                    model.fit(np.clip(scaler.transform(design), -3, 3), positive.astype(int))
                    parameter.update(scaler=scaler, model=model)
            self.parameters[side] = parameter
        self.fitted = True
        return self

    def predict_components(self, X):
        self._check(X)
        if not self.fitted: raise ValueError('H0 is not fitted')
        probabilities = np.empty(len(X)); severity = np.empty(len(X))
        for side in (0, 1):
            mask = X.option_type.eq(side).to_numpy(); parameter = self.parameters[side]
            if not mask.any(): continue
            probabilities[mask] = parameter['probability']
            if 'model' in parameter:
                z = np.clip(parameter['scaler'].transform(hurdle_design(X.loc[mask])), -3, 3)
                probabilities[mask] = parameter['model'].predict_proba(z)[:, 1]
            severity[mask] = X.loc[mask, BASE_FEATURE]*parameter['severity']
        return probabilities, severity

    def predict_raw(self, X):
        p, s = self.predict_components(X)
        result = p*s
        if not np.isfinite(result).all() or (result < 0).any(): raise ValueError('Invalid H0 prediction')
        return result


def h0_state(model):
    result = {}
    for side, p in model.parameters.items():
        v = {k: float(p[k]) for k in ('probability', 'severity')}
        if 'model' in p:
            v.update(mean=p['scaler'].mean_.tolist(), scale=p['scaler'].scale_.tolist(),
                     coef=p['model'].coef_[0].tolist(), intercept=float(p['model'].intercept_[0]))
        result[str(side)] = v
    return result
