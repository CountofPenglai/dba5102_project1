"""Direction-specific two-stage H0, with pricing-time inputs only."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pp_data.features import BASE_FEATURE
from .pricing_inputs import MODEL_INPUTS, check_pricing_inputs, hurdle_design


def probability_frame(X):
    """Select the pricing whitelist; settlement metadata must stay outside the model."""
    return X[MODEL_INPUTS].copy()


class H0:
    """Expected payout per BTC = P(positive payout) * conditional payout amount.

    Call (0) and Put (1) are fitted separately. The caller is responsible for
    selecting orders settled strictly before the weekly fitting timestamp.
    """
    def __init__(self):
        self.fitted = False

    def _check(self, X):
        check_pricing_inputs(X)

    def fit(self, X, y, created=None):
        # y is realized payout in USDC per BTC, not the premium or total order payout.
        # `created` is accepted for estimator compatibility; H0 itself is unweighted.
        self._check(X)
        if not isinstance(y, pd.Series) or not X.index.equals(y.index) or not np.isfinite(y).all() or y.lt(0).any():
            raise ValueError('Invalid payout target alignment')
        self.parameters = {}; self.detail = {}; self.fitted = False
        for side in (0, 1):
            mask = X.option_type.eq(side)
            b = X.loc[mask, BASE_FEATURE].to_numpy(float); target = y.loc[mask].to_numpy(float)
            # Zero payouts are valid negative labels for occurrence, but are excluded
            # from the conditional-amount fit below.
            positive = target > 0; n = len(target)
            # With no history for a direction, p=1 and c=1 reproduce Base.
            parameter = dict(probability=1., severity=1.)
            if n:
                parameter['probability'] = float(positive.mean())
                # Zero-intercept least squares on positive claims only: c=sum(b*y)/sum(b*b).
                parameter['severity'] = float(b[positive]@target[positive]/(b[positive]@b[positive])) if positive.any() else 0.
                # A single-class history uses its observed probability (0 or 1).
                # Logistic regression is needed only when both outcomes are present.
                if 0 < positive.sum() < n:
                    design = hurdle_design(X.loc[mask]); scaler = StandardScaler().fit(design)
                    # Fit scaling on this direction's history; use the same clipped
                    # transform at prediction time. C=0.1 applies L2 regularization.
                    model = LogisticRegression(C=.1, max_iter=1000, tol=1e-9, solver='lbfgs')
                    model.fit(np.clip(scaler.transform(design), -3, 3), positive.astype(int))
                    parameter.update(scaler=scaler, model=model)
            self.parameters[side] = parameter
        self.fitted = True
        return self

    def predict_components(self, X):
        # Return occurrence probability and conditional amount separately so research
        # branches can adjust one stage while holding the other fixed.
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
            # c is one fitted scalar per direction/update; Base supplies order-level
            # variation in the conditional payout amount.
            severity[mask] = X.loc[mask, BASE_FEATURE]*parameter['severity']
        return probabilities, severity

    def predict_raw(self, X):
        p, s = self.predict_components(X)
        result = p*s
        if not np.isfinite(result).all() or (result < 0).any(): raise ValueError('Invalid H0 prediction')
        return result


def h0_state(model):
    """Export plain numeric parameters for reproducible, pickle-free inference."""
    result = {}
    for side, p in model.parameters.items():
        v = {k: float(p[k]) for k in ('probability', 'severity')}
        # Missing logistic fields deliberately encode the constant-probability case.
        if 'model' in p:
            v.update(mean=p['scaler'].mean_.tolist(), scale=p['scaler'].scale_.tolist(),
                     coef=p['model'].coef_[0].tolist(), intercept=float(p['model'].intercept_[0]))
        result[str(side)] = v
    return result
