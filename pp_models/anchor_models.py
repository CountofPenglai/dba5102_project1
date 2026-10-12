"""Two fixed anchors with an optional scalar or four-variable log multiplier."""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from pp_data.features import BASE_FEATURE
from .baselines import RollingMean

INPUTS = ['option_type', 'product_duration_seconds', BASE_FEATURE, 'realized_vol_60m',
          'realized_vol_1440m', 'log_return_60m', 'log_moneyness']
MARKET_TERMS = ['log_vol_60m', 'log_vol_ratio_60m_1440m', 'aligned_return_60m', 'aligned_moneyness']


def group_leave_one_out(X, y, smoothing=30):
    if len(y) < 2 or not X.index.equals(y.index) or smoothing <= 0:
        raise ValueError('Leave-one-out requires aligned history with at least two rows and positive smoothing')
    frame = X[['option_type', 'product_duration_seconds']].assign(target=y)
    grouped = frame.groupby(['option_type', 'product_duration_seconds']).target
    own_sum = grouped.transform('sum').to_numpy() - y.to_numpy()
    own_count = grouped.transform('size').to_numpy() - 1
    global_without = (float(y.sum()) - y.to_numpy()) / (len(y)-1)
    return (own_sum + smoothing*global_without) / (own_count+smoothing)


def market_matrix(X):
    short = np.maximum(X.realized_vol_60m.to_numpy()/np.sqrt(60), 1e-12)
    long = np.maximum(X.realized_vol_1440m.to_numpy()/np.sqrt(1440), 1e-12)
    sign = 1 - 2*X.option_type.to_numpy()
    return np.column_stack([np.log(short), np.log(short/long), sign*X.log_return_60m, sign*X.log_moneyness])


class AnchorModel:
    def __init__(self, anchor, mode, lam=None, smoothing=30):
        if anchor not in ('official', 'grouped') or mode not in ('raw', 'scaled', 'dynamic'):
            raise ValueError('Unknown anchor or multiplier mode')
        if mode == 'dynamic' and (isinstance(lam, bool) or lam not in (.1, 1., 10.)):
            raise ValueError('Unsupported correction penalty')
        if smoothing != 30:
            raise ValueError('Group smoothing is fixed at 30')
        self.anchor, self.mode, self.lam, self.smoothing = anchor, mode, lam, smoothing
        self.fitted = False

    def _validate_X(self, X):
        if (not isinstance(X, pd.DataFrame) or X.empty or not X.index.is_unique or X.columns.tolist() != INPUTS
                or not np.isfinite(X.to_numpy(dtype=float)).all() or not X.option_type.isin([0, 1]).all()
                or X[BASE_FEATURE].le(0).any() or X.product_duration_seconds.le(0).any()
                or X[['realized_vol_60m', 'realized_vol_1440m']].lt(0).any().any()):
            raise ValueError('Invalid anchor inputs; require positive Base and legal pricing features')

    def fit(self, X, y):
        self._validate_X(X)
        if not isinstance(y, pd.Series) or not y.index.equals(X.index) or not np.isfinite(y).all() or y.lt(0).any():
            raise ValueError('Invalid aligned nonnegative payout targets')
        self.fitted = False
        self.beta, self.alpha, self.multiplier = np.zeros(4), 0., 1.
        self.nfev, self.optimality, self.zero_scale = 0, 0., False
        if self.anchor == 'grouped':
            self.group_model = RollingMean(grouped=True, smoothing=self.smoothing).fit(X, y)
            anchor = (self.group_model.predict_raw(X) if self.mode == 'raw'
                      else group_leave_one_out(X, y, self.smoothing))
        else:
            anchor = X[BASE_FEATURE].to_numpy()
        if self.mode != 'raw':
            denominator = float(np.dot(anchor, anchor))
            self.multiplier = float(np.dot(anchor, y)/denominator) if denominator > 0 else 0.
            self.zero_scale = self.multiplier == 0.
            if not self.zero_scale:
                self.alpha = float(np.log(self.multiplier))
            if self.mode == 'dynamic' and not self.zero_scale:
                values = market_matrix(X)
                self.center, self.scale = values.mean(axis=0), values.std(axis=0)
                self.scale = np.where(self.scale > 1e-12, self.scale, 1.)
                design = np.column_stack([np.ones(len(X)), (values-self.center)/self.scale])
                target = y.to_numpy()
                normalizer = float(np.sqrt(np.dot(target, target)))
                penalty = np.sqrt(self.lam)
                def parts(theta):
                    eta = design@theta
                    # Iteration guard only. Accepted fits/predictions may not reach it.
                    prediction = anchor*np.exp(np.clip(eta, -100., 100.))
                    return eta, prediction
                def residual(theta):
                    _, prediction = parts(theta)
                    return np.r_[(prediction-target)/normalizer, penalty*theta[1:]]
                def jacobian(theta):
                    eta, prediction = parts(theta)
                    derivative = prediction * (np.abs(eta) < 100.) / normalizer
                    data_jac = derivative[:, None]*design
                    return np.vstack([data_jac, np.column_stack([np.zeros(4), penalty*np.eye(4)])])
                result = least_squares(residual, np.r_[self.alpha, self.beta], jac=jacobian,
                    max_nfev=200, ftol=1e-8, xtol=1e-8, gtol=1e-8)
                if (not result.success or not np.isfinite(result.x).all()
                        or (np.abs(design@result.x) >= 100.).any()):
                    raise RuntimeError('Anchor correction solver failed or reached numerical exponent guard')
                self.alpha, self.beta = float(result.x[0]), result.x[1:]
                self.multiplier = float(np.exp(self.alpha))
                self.nfev, self.optimality = result.nfev, float(result.optimality)
        self.fitted = True
        return self

    def predict_raw(self, X):
        self._validate_X(X)
        if not self.fitted:
            raise ValueError('Anchor model is not fitted')
        base = (X[BASE_FEATURE].to_numpy() if self.anchor == 'official' else self.group_model.predict_raw(X))
        if self.zero_scale:
            return np.zeros(len(X))
        if self.mode == 'dynamic':
            eta = self.alpha + ((market_matrix(X)-self.center)/self.scale)@self.beta
            if not np.isfinite(eta).all() or (np.abs(eta) >= 100.).any():
                raise RuntimeError('Prediction reached numerical exponent guard')
            result = base*np.exp(eta)
        else:
            result = base*self.multiplier
        if not np.isfinite(result).all() or (result < 0).any():
            raise RuntimeError('Invalid anchor prediction')
        return result

    def diagnostics(self):
        return {'calibration_multiplier': self.multiplier, 'correction_beta_norm': float(np.linalg.norm(self.beta)),
            'solver_evaluations': self.nfev, 'solver_optimality': self.optimality,
            'zero_scale': self.zero_scale, **{'beta_'+name: float(v) for name, v in zip(MARKET_TERMS, self.beta)}}


def build_anchor_model(name, resolved):
    if (name != resolved['selected_models'][0] or resolved['selected_models'] != [name]
            or resolved['features'] != INPUTS or name != resolved['anchor']+'_'+resolved['correction']
            or resolved['cohort'] != 'benchmark' or resolved['use_base'] is not True):
        raise ValueError('Invalid anchor experiment configuration')
    return AnchorModel(resolved['anchor'], resolved['correction'], resolved.get('correction_lambda'))
