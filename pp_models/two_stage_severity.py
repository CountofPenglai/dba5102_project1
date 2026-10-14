"""Positive-payout severity candidates for the fixed two-stage probability model."""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor
from pp_data.features import BASE_FEATURE

SEVERITY_INPUTS = ["option_type", BASE_FEATURE, "remaining_seconds", "realized_vol_60m", "log_moneyness"]
DIRECTIONAL_INPUTS = SEVERITY_INPUTS + ["up_vol_60m", "up_vol_240m", "down_vol_60m", "down_vol_240m"]
SEVERITY_NAMES = ("h0", "g0", "g1a", "g1b")
DIRECTIONAL_NAMES = ("g2a", "g2b")


def directional_acceleration(X):
    required = ["option_type", "up_vol_60m", "up_vol_240m", "down_vol_60m", "down_vol_240m"]
    if (not isinstance(X, pd.DataFrame) or X.empty or not X.index.is_unique
            or any(column not in X for column in required)
            or not np.isfinite(X[required].to_numpy(float)).all()
            or not X.option_type.isin([0, 1]).all() or X[required[1:]].lt(0).any().any()):
        raise ValueError("Invalid directional acceleration inputs")
    call = np.log((X.up_vol_60m.to_numpy(float) + 1e-12) / (X.up_vol_240m.to_numpy(float) + 1e-12))
    put = np.log((X.down_vol_60m.to_numpy(float) + 1e-12) / (X.down_vol_240m.to_numpy(float) + 1e-12))
    return pd.Series(np.where(X.option_type.eq(0), call, put), index=X.index, name="directional_acceleration")


def severity_design(X, include_directional=False):
    """Two contract variables, calculated only from pricing-time fields."""
    uncertainty = np.maximum(
        X.realized_vol_60m.to_numpy(float) * np.sqrt(X.remaining_seconds.to_numpy(float) / 3600), 1e-6)
    aligned = ((1 - 2 * X.option_type.to_numpy(float)) * X.log_moneyness.to_numpy(float) / uncertainty)
    columns = [np.log(X.remaining_seconds.to_numpy(float)), np.clip(aligned, -5, 5)]
    if include_directional:
        columns.append(directional_acceleration(X).to_numpy(float))
    return np.column_stack(columns)


class TwoStageSeverity:
    def __init__(self, name, alpha=None, minimum_positive_orders=40, minimum_creation_days=5):
        if name not in SEVERITY_NAMES + DIRECTIONAL_NAMES:
            raise ValueError("Unknown severity candidate")
        expected = {"h0": None, "g0": None, "g1a": 1.0, "g1b": 10.0,
                    "g2a": 1.0, "g2b": 10.0}[name]
        if alpha is None:
            alpha = expected
        if alpha != expected:
            raise ValueError("Severity alpha differs from the fixed candidate")
        if (isinstance(minimum_positive_orders, bool) or minimum_positive_orders < 1
                or isinstance(minimum_creation_days, bool) or minimum_creation_days < 1):
            raise ValueError("Invalid severity support threshold")
        self.name, self.alpha = name, alpha
        self.minimum_positive_orders = int(minimum_positive_orders)
        self.minimum_creation_days = int(minimum_creation_days)
        self.fitted = False

    def _check_X(self, X):
        columns = DIRECTIONAL_INPUTS if self.name in DIRECTIONAL_NAMES else SEVERITY_INPUTS
        if (not isinstance(X, pd.DataFrame) or X.empty or not X.index.is_unique
                or list(X) != columns or not np.isfinite(X.to_numpy(float)).all()
                or not X.option_type.isin([0, 1]).all()
                or X[[BASE_FEATURE, "remaining_seconds"]].le(0).any().any()
                or X.realized_vol_60m.lt(0).any()):
            raise ValueError("Invalid pricing-time severity inputs")

    def fit(self, X, y, created):
        self._check_X(X)
        if (not isinstance(y, pd.Series) or not X.index.equals(y.index)
                or not np.isfinite(y.to_numpy(float)).all() or y.lt(0).any()):
            raise ValueError("Invalid severity target alignment")
        if (not isinstance(created, pd.Series) or not X.index.equals(created.index)
                or created.isna().any() or not isinstance(created.dtype, pd.DatetimeTZDtype)):
            raise ValueError("Invalid severity creation-time alignment")
        self.parameters, self.detail = {}, {}
        for side, label in ((0, "call"), (1, "put")):
            side_mask = X.option_type.eq(side)
            positive_mask = side_mask & y.gt(0)
            xp, yp, cp = X.loc[positive_mask], y.loc[positive_mask], created.loc[positive_mask]
            n, days = len(xp), int(cp.dt.floor("D").nunique())
            parameter = {"mode": "base", "coefficient": 1.0, "fallback_reason": "no_history"}
            if side_mask.any():
                parameter.update(coefficient=0.0, fallback_reason="no_positive_payout")
            if n:
                base = xp[BASE_FEATURE].to_numpy(float)
                target = yp.to_numpy(float)
                coefficient = float(np.dot(base, target) / np.dot(base, base))
                parameter.update(coefficient=coefficient, fallback_reason=None)
                ratios = target / base
                self.detail.update({f"{label}_ratio_mean": float(ratios.mean()),
                                    f"{label}_ratio_median": float(np.median(ratios)),
                                    f"{label}_ratio_max": float(ratios.max())})
                supported = n >= self.minimum_positive_orders and days >= self.minimum_creation_days
                if self.name != "h0" and not supported:
                    parameter["fallback_reason"] = "insufficient_positive_support"
                elif self.name == "g0":
                    parameter.update(mode="ratio", coefficient=float(ratios.mean()))
                elif self.name in ("g1a", "g1b") + DIRECTIONAL_NAMES:
                    design = severity_design(xp, self.name in DIRECTIONAL_NAMES)
                    mean, scale = design.mean(axis=0), design.std(axis=0)
                    scale = np.where(scale > 0, scale, 1.0)
                    z = np.clip((design - mean) / scale, -3, 3)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("error")
                            model = GammaRegressor(alpha=self.alpha, max_iter=1000, tol=1e-9).fit(z, ratios)
                        fitted = model.predict(z)
                        if not np.isfinite(fitted).all() or (fitted <= 0).any():
                            raise FloatingPointError("non-finite Gamma fit")
                        parameter.update(mode="gamma", mean=mean, scale=scale, model=model)
                    except (ArithmeticError, FloatingPointError, ValueError, Warning) as error:
                        parameter["fallback_reason"] = "numerical_failure"
                        parameter["failure_type"] = type(error).__name__
            self.parameters[side] = parameter
            self.detail.update({f"{label}_training_orders": int(side_mask.sum()),
                                f"{label}_positive_orders": n,
                                f"{label}_positive_creation_days": days,
                                f"{label}_mode": parameter["mode"],
                                f"{label}_fallback_reason": parameter["fallback_reason"],
                                f"{label}_h0_coefficient": parameter["coefficient"]})
            if parameter["mode"] == "gamma":
                self.detail.update({f"{label}_intercept": float(parameter["model"].intercept_),
                                    f"{label}_log_remaining_coefficient": float(parameter["model"].coef_[0]),
                                    f"{label}_aligned_position_coefficient": float(parameter["model"].coef_[1])})
                if self.name in DIRECTIONAL_NAMES:
                    self.detail[f"{label}_directional_acceleration_coefficient"] = float(parameter["model"].coef_[2])
        self.fitted = True
        return self

    def predict_severity(self, X):
        self._check_X(X)
        if not self.fitted:
            raise ValueError("Severity model not fitted")
        result = np.empty(len(X), dtype=float)
        for side in (0, 1):
            mask = X.option_type.eq(side).to_numpy()
            if not mask.any():
                continue
            part, parameter = X.loc[mask], self.parameters[side]
            if parameter["mode"] == "gamma":
                z = np.clip((severity_design(part, self.name in DIRECTIONAL_NAMES)
                             - parameter["mean"]) / parameter["scale"], -3, 3)
                ratio = parameter["model"].predict(z)
            else:
                ratio = np.full(mask.sum(), parameter["coefficient"], dtype=float)
            result[mask] = part[BASE_FEATURE].to_numpy(float) * ratio
        if not np.isfinite(result).all() or (result < 0).any():
            raise ValueError("Invalid severity prediction")
        return result

    def diagnostics(self):
        if not self.fitted:
            raise ValueError("Severity model not fitted")
        return dict(self.detail)
