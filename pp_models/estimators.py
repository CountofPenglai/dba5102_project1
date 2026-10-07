"""Three real estimator adapters, with explicit feature and target boundaries."""
import numpy as np
import pandas as pd


class PayoutRegressor:
    def __init__(self, estimator, features):
        self.estimator = estimator
        self.features = list(features)
        self.fitted = False
        self.constant = None

    def _matrix(self, X):
        if not isinstance(X, pd.DataFrame) or X.columns.tolist() != self.features:
            raise ValueError("Model feature names/order differ from the frozen configuration")
        if X.empty or not X.index.is_unique or not all(pd.api.types.is_numeric_dtype(t) for t in X.dtypes):
            raise ValueError("Expected nonempty numeric features with unique row IDs")
        values = X.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite model features")
        if "option_type" in X and not X.option_type.isin([0, 1]).all():
            raise ValueError("option_type must be 0 or 1")
        return values

    def fit(self, X, y):
        values = self._matrix(X)
        if not isinstance(y, pd.Series) or not X.index.equals(y.index):
            raise ValueError("Target IDs/order must exactly match features")
        target = y.to_numpy(dtype=float)
        if not np.isfinite(target).all() or (target < 0).any():
            raise ValueError("Payout targets must be finite and nonnegative")
        self.fitted, self.constant = False, None
        # A constant target has an exact constant least-squares solution; in particular
        # this avoids zero-scale GAM optimization on all-zero daily histories.
        if np.all(target == target[0]):
            self.constant = float(target[0])
        else:
            self.estimator.fit(values, target)
            # pyGAM can return after max_iter without raising on non-convergence.
            # Treat that as a failed candidate, never as a successful search result.
            if hasattr(self.estimator, "logs_") and hasattr(self.estimator, "tol"):
                diffs = self.estimator.logs_.get("diffs", [])
                if not len(diffs) or not np.isfinite(diffs[-1]) or diffs[-1] >= self.estimator.tol:
                    raise RuntimeError("GAM solver did not converge; candidate must be investigated")
        self.fitted = True
        return self

    def predict_raw(self, X):
        values = self._matrix(X)
        if not self.fitted:
            raise ValueError("Model is not fitted")
        result = (np.full(len(X), self.constant) if self.constant is not None
                  else np.asarray(self.estimator.predict(values), dtype=float))
        if result.shape != (len(X),) or not np.isfinite(result).all():
            raise ValueError("Model produced invalid predictions")
        return result

    def predict(self, X):
        return np.maximum(0., self.predict_raw(X))


def build_model(name, resolved):
    if name not in resolved["selected_models"]:
        raise ValueError(f"Model is not selected: {name}")
    features, params = resolved["features"], resolved["models"][name].copy()
    seed = resolved["seed"]
    try:
        if name == "gam":
            from pygam import LinearGAM, s, l
            # Binary linear term is equivalent to a two-level effect with intercept,
            # and remains defined when only one option direction occurs in history.
            terms = [l(i, lam=params["lam"]) if f == "option_type" else
                     s(i, n_splines=params["n_splines"], spline_order=3,
                       lam=params["lam"], penalties="derivative") for i, f in enumerate(features)]
            combined = terms[0]
            for term in terms[1:]:
                combined += term
            estimator = LinearGAM(combined, max_iter=params["max_iter"], tol=params["tol"])
        elif name == "ebm":
            from interpret.glassbox import ExplainableBoostingRegressor
            pairs = [(features.index(a), features.index(b)) for a, b in params.pop("interactions")]
            estimator = ExplainableBoostingRegressor(
                feature_names=features,
                feature_types=["nominal" if f == "option_type" else "continuous" for f in features],
                interactions=pairs, objective="rmse", validation_size=0, outer_bags=1,
                early_stopping_rounds=0, max_leaves=2, n_jobs=1, random_state=seed, **params)
        elif name == "shallow_gbdt":
            from sklearn.ensemble import HistGradientBoostingRegressor
            estimator = HistGradientBoostingRegressor(
                loss="squared_error", early_stopping=False, categorical_features=None,
                random_state=seed, **params)
        else:
            raise ValueError(f"Unknown model: {name}")
    except ImportError as error:
        raise RuntimeError("Install model dependencies: python -m pip install -r requirements-models.txt") from error
    return PayoutRegressor(estimator, features)
