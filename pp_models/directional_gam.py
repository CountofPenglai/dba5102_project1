"""Four finite GAM ablations with strict pricing-time feature whitelists."""
from pygam import LinearGAM, s, l
from pygam.terms import SplineTerm
import numpy as np
from scipy import sparse
from .config import resolve_config
from .estimators import PayoutRegressor, build_model

VARIANTS = ("original", "multiwindow", "directional", "semivol")
ORIGINAL_FEATURES = ["option_type", "remaining_seconds", "log_moneyness", "market_price_usdc_per_btc",
                     "realized_vol_60m", "realized_vol_1440m", "log_return_60m"]


def features_for(variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown GAM variant")
    if variant == "original":
        return ORIGINAL_FEATURES.copy()
    features = ORIGINAL_FEATURES[:4] + [f"log_return_{w}m" for w in (60, 240, 1440)]
    labels = ("up", "down") if variant == "semivol" else ("total",)
    return features + [f"{label}_vol_{w}m" for w in (60, 240, 1440) for label in labels]


def variant_config(base, variant, lam):
    if lam not in (10, 100, 1000) or isinstance(lam, bool):
        raise ValueError("Unsupported directional study lambda")
    resolved = resolve_config(base, "risk", False, ["gam"])
    resolved["features"] = features_for(variant)
    resolved["models"]["gam"] = {"lam": lam, "n_splines": 4, "max_iter": 100, "tol": 1e-4}
    resolved["gam_variant"] = variant
    resolved["daily_direction_metrics"] = True
    return resolved


class DiagnosticGAM(PayoutRegressor):
    def diagnostics(self):
        if self.constant is not None:
            return {"gam_edof": 1., "gam_iterations": 0, "gam_final_diff": 0.}
        return {"gam_edof": float(self.estimator.statistics_["edof"]),
                "gam_iterations": len(self.estimator.logs_["diffs"]),
                "gam_final_diff": float(self.estimator.logs_["diffs"][-1])}


class CenteredDirectionSpline(SplineTerm):
    """Center the by-basis on training Put rows; do not add a free Put intercept."""
    def compile(self, X, verbose=False):
        super().compile(X, verbose=verbose)
        indicator = X[:, self.by]
        count = indicator.sum()
        columns = super().build_columns(X, verbose=verbose)
        self._training_center = (np.asarray(columns.sum(axis=0)).ravel() / count
                                 if count else np.zeros(self.n_splines))
        return self

    def build_columns(self, X, verbose=False):
        columns = super().build_columns(X, verbose=verbose)
        # Fitted centering is reused unchanged for prediction, including unseen values.
        return columns - sparse.csc_matrix(X[:, self.by, None] * self._training_center)


def build_directional_model(name, resolved):
    variant = resolved["gam_variant"]
    features, p = resolved["features"], resolved["models"]["gam"]
    if name != "gam" or resolved["selected_models"] != ["gam"] or features != features_for(variant):
        raise ValueError("Invalid directional GAM model/features")
    if (set(p) != {"lam", "n_splines", "max_iter", "tol"} or p["lam"] not in (10, 100, 1000)
            or p["n_splines"] != 4 or p["max_iter"] != 100 or p["tol"] != 1e-4):
        raise ValueError("Invalid directional GAM parameters")
    if variant == "original":
        estimator = build_model(name, resolved).estimator
    else:
        terms = l(0, lam=p["lam"])
        for i, feature in enumerate(features[1:], 1):
            terms += s(i, n_splines=4, spline_order=3, lam=p["lam"], penalties="derivative")
            if variant in ("directional", "semivol") and feature != "market_price_usdc_per_btc":
                # Shared curve + Put-specific deviation: no split of the training pool.
                terms += CenteredDirectionSpline(i, by=0, n_splines=4, spline_order=3,
                                                 lam=p["lam"], penalties="derivative")
        estimator = LinearGAM(terms, max_iter=p["max_iter"], tol=p["tol"])
    return DiagnosticGAM(estimator, features)
