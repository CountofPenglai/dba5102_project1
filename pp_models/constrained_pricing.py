"""Direction-specific Base calibration with small, strongly regularized corrections."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from pp_data.features import BASE_FEATURE

RISK = ['contract_momentum', 'contract_vol_change']
FLOW = ['flow_market', 'flow_call', 'flow_put']
INPUTS = ['option_type', BASE_FEATURE, 'constraint_supported', 'contract_dynamic',
          'contract_calibrated'] + RISK + FLOW
MODES = ('raw', 'scaled', 'contract', 'tails', 'flow')


def stationary_within_bounds(beta, gradient, bounds):
    """Check first-order stationarity when line search hits floating-point limits."""
    lower, upper = np.asarray(bounds, float).T
    if (not np.isfinite(beta).all() or not np.isfinite(gradient).all()
            or (beta < lower).any() or (beta > upper).any()):
        return False
    return bool(np.max(np.abs(beta - np.clip(beta-gradient, lower, upper))) <= 1e-6)


def profile_loss(beta, Z, base, target, penalty):
    """Optimize the unpenalized nonnegative scalar analytically at each beta."""
    w = base * np.exp(Z @ beta)
    scale = float(w @ target / (w @ w))
    prediction = scale * w
    norm = float(target @ target)
    residual = prediction - target
    loss = float(residual @ residual / norm + penalty @ (beta * beta))
    gradient = 2 * (Z.T @ (residual * prediction)) / norm + 2 * penalty * beta
    return loss, gradient, scale


class ConstrainedModel:
    def __init__(self, mode, lam=None):
        if (mode not in MODES or (mode in ('raw', 'scaled') and lam is not None)
                or (mode not in ('raw', 'scaled') and (isinstance(lam, bool) or lam not in (.1, 1., 10.)))):
            raise ValueError('Invalid constrained pricing mode/penalty')
        self.mode, self.lam = mode, lam
        self.features = ([] if mode in ('raw', 'scaled') else
                         ['contract_dynamic' if mode == 'contract' else 'contract_calibrated'] + RISK)
        if mode == 'flow': self.features += FLOW
        self.fits = None

    def _validate(self, X):
        if (not isinstance(X, pd.DataFrame) or X.empty or not X.index.is_unique or list(X) != INPUTS
                or not X.option_type.isin([0, 1]).all() or not X.constraint_supported.isin([0, 1]).all()
                or X[BASE_FEATURE].le(0).any() or np.isinf(X.to_numpy(float)).any()
                or not np.isfinite(X[[c for c in INPUTS if c not in FLOW]]).all().all()):
            raise ValueError('Invalid constrained pricing inputs')

    def _design(self, X, fitted):
        values = X[self.features].to_numpy(float)
        values = np.where(np.isfinite(values), values, fitted['center'])
        Z = np.clip((values - fitted['center']) / fitted['spread'], -3., 3.)
        Z[X.constraint_supported.eq(0).to_numpy()] = 0.
        return Z

    def fit(self, X, y):
        self._validate(X)
        if (not isinstance(y, pd.Series) or not X.index.equals(y.index)
                or not np.isfinite(y).all() or y.lt(0).any()):
            raise ValueError('Invalid constrained pricing targets')
        self.fits = {}
        for side in (0, 1):
            subset = X.loc[X.option_type.eq(side)]
            target = y.loc[subset.index].to_numpy(float)
            base = subset[BASE_FEATURE].to_numpy(float)
            size = len(self.features)
            fitted = dict(scale=1., beta=np.zeros(size), center=np.zeros(size), spread=np.ones(size),
                          rows=len(subset), iterations=0, loss=0., stationary_fallback=0)
            if len(subset) and self.mode != 'raw':
                fitted['scale'] = float(base @ target / (base @ base))
                for j, feature in enumerate(self.features):
                    valid = subset.loc[subset.constraint_supported.eq(1), feature].dropna()
                    if len(valid):
                        fitted['center'][j] = float(valid.median())
                        fitted['spread'][j] = max(float(valid.std(ddof=0)), 1e-6)
                if size and target @ target > 0:
                    Z = self._design(subset, fitted)
                    penalty = self.lam * np.array([10. if f in FLOW else 1. for f in self.features])
                    def objective(beta):
                        loss, gradient, _ = profile_loss(beta, Z, base, target, penalty)
                        return loss, gradient
                    # Positive proxy effect; trend and frequency may have either sign.
                    bounds = [(0., 1.)] + [(-1., 1.)] * (size - 1)
                    result = minimize(objective, np.zeros(size), jac=True, method='L-BFGS-B',
                                      bounds=bounds, options={'maxiter':200, 'ftol':1e-12, 'gtol':1e-8})
                    if (not result.success and (not np.isfinite(result.fun)
                            or not stationary_within_bounds(result.x, objective(result.x)[1], bounds)
                            or result.fun > objective(np.zeros(size))[0] + 1e-12)):
                        raise ValueError(f'Constrained fit failed: {result.message}')
                    fitted.update(beta=result.x, iterations=int(result.nit), loss=float(result.fun),
                                  stationary_fallback=int(not result.success),
                                  scale=profile_loss(result.x, Z, base, target, penalty)[2])
            self.fits[side] = fitted
        self.prediction_stats = {}
        return self

    def predict_raw(self, X):
        self._validate(X)
        if self.fits is None: raise ValueError('Model is not fitted')
        prediction = X[BASE_FEATURE].to_numpy(float).copy()
        for side, fitted in self.fits.items():
            mask = X.option_type.eq(side).to_numpy()
            if not mask.any(): continue
            Z = self._design(X.loc[mask], fitted)
            multiplier = fitted['scale'] * np.exp(Z @ fitted['beta'])
            prediction[mask] *= multiplier
            self.prediction_stats.update({f'side{side}_multiplier_mean':float(multiplier.mean()),
                                          f'side{side}_multiplier_max':float(multiplier.max())})
        if not np.isfinite(prediction).all() or (prediction < 0).any():
            raise ValueError('Invalid constrained predictions')
        return prediction

    def diagnostics(self):
        result = dict(self.prediction_stats)
        for side, fitted in self.fits.items():
            for name in ('scale', 'rows', 'iterations', 'loss', 'stationary_fallback'):
                result[f'side{side}_{name}'] = fitted[name]
            for feature, beta in zip(self.features, fitted['beta']):
                result[f'side{side}_beta_{feature}'] = float(beta)
        return result


def build_constrained_model(name, resolved):
    mode = resolved['constraint_mode']
    if (name != f'constrained_{mode}' or resolved['selected_models'] != [name]
            or resolved['features'] != INPUTS or resolved['cohort'] != 'benchmark'
            or resolved['use_base'] is not True):
        raise ValueError('Invalid constrained factory configuration')
    return ConstrainedModel(mode, resolved['constraint_lambda'])
