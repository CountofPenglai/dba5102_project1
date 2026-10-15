"""Independent, small refinements of either H0 severity or occurrence probability."""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler

from pp_data.features import BASE_FEATURE
from .calibration_models import hurdle_design

INPUTS = ["option_type", BASE_FEATURE, "remaining_seconds", "realized_vol_60m", "log_moneyness",
          "up_vol_60m", "up_vol_240m", "down_vol_60m", "down_vol_240m"]
AMOUNT_BOUNDS = (.5, 1.5)


def check_X(X):
    if (not isinstance(X, pd.DataFrame) or X.empty or not X.index.is_unique or list(X) != INPUTS
            or not np.isfinite(X.to_numpy(float)).all() or not X.option_type.isin([0, 1]).all()
            or X[[BASE_FEATURE, "remaining_seconds"]].le(0).any().any()
            or X[["realized_vol_60m"] + INPUTS[-4:]].lt(0).any().any()):
        raise ValueError("Invalid pricing-time component features")


def check_labels(X, y, created):
    check_X(X)
    if (not isinstance(y, pd.Series) or not X.index.equals(y.index) or not np.isfinite(y).all()
            or y.lt(0).any() or not isinstance(created, pd.Series) or not X.index.equals(created.index)
            or not isinstance(created.dtype, pd.DatetimeTZDtype) or created.isna().any()):
        raise ValueError("Invalid component labels/dates")


def check_anchor(X, anchor, probability=False):
    if (not isinstance(anchor, pd.Series) or not X.index.equals(anchor.index)
            or not np.isfinite(anchor).all() or anchor.lt(0).any()
            or (probability and anchor.gt(1).any())):
        raise ValueError("Invalid frozen H0 component anchor")


class FeatureMap:
    """Train-only standardization, smooth probability main effects, one interaction."""
    def __init__(self, component, feature_set):
        self.component, self.feature_set = component, feature_set

    def raw(self, X):
        result = hurdle_design(X)
        if self.component == "amount":
            result = result[:, 1:]
        if self.feature_set == "market":
            v60 = np.where(X.option_type.eq(0), X.up_vol_60m, X.down_vol_60m)
            v240 = np.where(X.option_type.eq(0), X.up_vol_240m, X.down_vol_240m)
            result = np.column_stack([result, np.log(v60 + 1e-12), np.log((v60+1e-12)/(v240+1e-12))])
        return result

    def fit(self, X):
        raw = self.raw(X)
        self.scaler = StandardScaler().fit(raw)
        z = np.clip(self.scaler.transform(raw), -3, 3)
        if self.feature_set == "market":
            product = z[:, -2] * z[:, -1]
            self.product_mean = float(product.mean())
            self.product_scale = float(product.std()) or 1.
        if self.component == "probability":
            self.spline = SplineTransformer(degree=2, knots=np.tile([-3., 0., 3.], (z.shape[1], 1)).T,
                include_bias=False, extrapolation="constant").fit(z)
            self.basis_scaler = StandardScaler().fit(self._basis(z))
        return self

    def _basis(self, z):
        result = self.spline.transform(z) if self.component == "probability" else z
        if self.feature_set == "market":
            interaction = np.clip((z[:, -2]*z[:, -1]-self.product_mean)/self.product_scale, -3, 3)
            result = np.column_stack([result, interaction])
        return result

    def transform(self, X):
        z = np.clip(self.scaler.transform(self.raw(X)), -3, 3)
        basis = self._basis(z)
        if self.component == "probability":
            return np.clip(self.basis_scaler.transform(basis), -3, 3)
        return np.column_stack([np.ones(len(X)), basis])


class AmountRefinement:
    def __init__(self, feature_set, strength):
        if feature_set not in ("contract", "market") or strength not in (30, 100):
            raise ValueError("Unknown fixed amount refinement")
        self.feature_set, self.strength, self.fitted = feature_set, strength, False

    def fit(self, X, y, created, anchor):
        check_labels(X, y, created)
        check_anchor(X, anchor)
        self.fitted, self.parameters, self.details = False, {}, []
        for side in (0, 1):
            mask = X.option_type.eq(side) & y.gt(0)
            n, days = int(mask.sum()), int(created.loc[mask].dt.floor("D").nunique())
            parameter, reason = {}, "none"
            if n < 40 or days < 5:
                reason = "insufficient_positive_support"
            elif anchor.loc[mask].eq(0).all():
                reason = "zero_anchor"
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        fmap = FeatureMap("amount", self.feature_set).fit(X.loc[mask])
                        phi = fmap.transform(X.loc[mask])
                        s0, target = anchor.loc[mask].to_numpy(), y.loc[mask].to_numpy()
                        # Divide by a common scale only for numerical conditioning:
                        # exactly equivalent to raw-currency SSE + k*mean(s0**2)*||beta||².
                        scale = float(np.sqrt(np.mean(s0**2)))
                        design = phi * (s0/scale)[:, None]
                        beta = np.linalg.solve(design.T@design + self.strength*np.eye(phi.shape[1]),
                                               design.T@((target-s0)/scale))
                        if not np.isfinite(beta).all():
                            raise FloatingPointError("Invalid amount coefficients")
                        parameter = {"features": fmap, "beta": beta}
                except (ArithmeticError, ValueError, np.linalg.LinAlgError, Warning):
                    reason = "numerical_failure"
            self.parameters[side] = parameter
            self.details.append({"side": side, "training_orders": int(X.option_type.eq(side).sum()),
                "positive_orders": n, "creation_days": days, "fallback": reason,
                "coefficient_norm": float(np.linalg.norm(parameter["beta"])) if parameter else 0.})
        self.fitted = True
        return self

    def predict_severity(self, X, anchor):
        check_X(X); check_anchor(X, anchor)
        if not self.fitted:
            raise ValueError("Amount refinement not fitted")
        output = anchor.to_numpy(float).copy()
        for side, parameter in self.parameters.items():
            mask = X.option_type.eq(side).to_numpy()
            if parameter and mask.any():
                ratio = 1 + parameter["features"].transform(X.loc[mask]) @ parameter["beta"]
                if not np.isfinite(ratio).all():
                    raise ValueError("Nonfinite amount prediction")
                output[mask] *= np.clip(ratio, *AMOUNT_BOUNDS)
        return output

    def diagnostics(self):
        return [row.copy() for row in self.details]


class ProbabilityRefinement:
    def __init__(self, feature_set, C):
        if feature_set not in ("contract", "market") or C not in (.03, .1):
            raise ValueError("Unknown fixed probability refinement")
        self.feature_set, self.C, self.fitted = feature_set, C, False

    def fit(self, X, y, created):
        check_labels(X, y, created)
        self.fitted, self.parameters, self.details = False, {}, []
        for side in (0, 1):
            mask = X.option_type.eq(side)
            target = y.loc[mask].gt(0).astype(int)
            n, positive = len(target), int(target.sum())
            days = int(created.loc[mask].dt.floor("D").nunique())
            parameter, reason = {}, "none"
            if n < 80 or min(positive, n-positive) < 20 or days < 5:
                reason = "insufficient_class_support"
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        fmap = FeatureMap("probability", self.feature_set).fit(X.loc[mask])
                        design = fmap.transform(X.loc[mask])
                        model = LogisticRegression(C=self.C, max_iter=1000, tol=1e-9, solver="lbfgs").fit(design, target)
                        if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
                            raise FloatingPointError("Invalid logistic coefficients")
                        parameter = {"features": fmap, "model": model}
                except (ArithmeticError, ValueError, Warning):
                    reason = "numerical_failure"
            self.parameters[side] = parameter
            self.details.append({"side": side, "training_orders": n, "positive_orders": positive,
                "creation_days": days, "fallback": reason,
                "coefficient_norm": float(np.linalg.norm(parameter["model"].coef_)) if parameter else 0.})
        self.fitted = True
        return self

    def predict_probability(self, X, anchor):
        check_X(X); check_anchor(X, anchor, probability=True)
        if not self.fitted:
            raise ValueError("Probability refinement not fitted")
        output = anchor.to_numpy(float).copy()
        for side, parameter in self.parameters.items():
            mask = X.option_type.eq(side).to_numpy()
            if parameter and mask.any():
                output[mask] = parameter["model"].predict_proba(parameter["features"].transform(X.loc[mask]))[:, 1]
        if not np.isfinite(output).all() or ((output < 0) | (output > 1)).any():
            raise ValueError("Invalid probability prediction")
        return output

    def diagnostics(self):
        return [row.copy() for row in self.details]
