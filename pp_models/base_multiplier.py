"""Small, bounded Put multiplier. Prediction never accepts Base or outcomes."""
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

INPUTS = ['option_type', 'product_duration_seconds', 'remaining_seconds', 'log_moneyness',
          'total_vol_240m', 'down_vol_240m', 'log_return_60m']
FAMILIES = ('constant', 'contract', 'market')


def validate_input(X):
    if (not isinstance(X, pd.DataFrame) or set(X) != set(INPUTS)
            or not X.index.is_unique or X.empty):
        raise ValueError('Multiplier input whitelist or unique IDs violated')
    if not X.option_type.isin([0, 1]).all():
        raise ValueError('Unknown option direction')


def transform(X, family):
    """Put direction; all fields are available at quoting time."""
    result = pd.DataFrame(index=X.index)
    if family != 'constant':
        valid_remaining = X.remaining_seconds.where(X.remaining_seconds > 0)
        sigma = X.total_vol_240m.where(X.total_vol_240m >= 0)
        result['log_remaining'] = np.log(valid_remaining)
        result['position'] = -X.log_moneyness / np.maximum(sigma*np.sqrt(valid_remaining/60), 1e-6)
    if family == 'market':
        result['log_down_vol'] = np.log(X.down_vol_240m.where(X.down_vol_240m >= 0).clip(lower=1e-12))
        result['adverse_trend'] = -X.log_return_60m / np.maximum(sigma*np.sqrt(60), 1e-6)
    return result


class BaseMultiplier:
    def __init__(self, family, strength, delta):
        if family not in FAMILIES or strength not in (.1, 1., 10.) or delta not in (.1, .2):
            raise ValueError('Unregistered multiplier specification')
        self.family, self.strength, self.delta = family, float(strength), float(delta)

    def fit(self, X, base, y, created):
        validate_input(X)
        if (any(not X.index.equals(v.index) for v in (base,y,created))
                or not np.isfinite(X.to_numpy(float)).all()
                or not np.isfinite(base).all() or (base <= 0).any()
                or not np.isfinite(y).all() or (y < 0).any()
                or created.isna().any() or created.dt.tz is None):
            raise ValueError('Invalid or misaligned multiplier training data')
        put = X.option_type.eq(1)
        raw = transform(X.loc[put], self.family)
        if not np.isfinite(raw).all().all():
            raise ValueError('Invalid training feature transform')
        self.feature_names_ = ['intercept'] + list(raw)
        self.mean_, self.std_ = raw.mean(), raw.std(ddof=0).replace(0, 1).fillna(1.)
        self.lower_, self.upper_ = raw.min(), raw.max()
        days = created.dt.floor('D')
        self.supported_ = int(put.sum()) >= 100 and days[put].nunique() >= 10
        self.tenors_ = [float(tenor) for tenor, ids in X.loc[put].groupby('product_duration_seconds').groups.items()
                        if len(ids) >= 30 and days.loc[ids].nunique() >= 5]
        self.put_training_orders_ = int(put.sum())
        self.coef_ = np.zeros(len(self.feature_names_))
        self.scale_ = max(float(np.sqrt(np.mean(y[put]**2))) if put.any() else 0., 1e-6)
        self.optimization_success_, self.solver_ = True, 'unsupported_identity'
        self.max_training_raw_adjustment_ = 0.
        self.objective_ = self.null_objective_ = float(np.mean(((base[put]-y[put])/self.scale_)**2)) if put.any() else 0.
        if not self.supported_ or not self.tenors_:
            return self
        phi, _, _ = self._design(X.loc[put])
        a = phi*base[put].to_numpy()[:,None]/self.scale_
        target = (y[put]-base[put]).to_numpy()/self.scale_
        n, k = a.shape
        h = a.T@a/n + self.strength*np.eye(k)
        v = a.T@target/n
        coef = np.linalg.solve(h,v)
        self.solver_ = 'closed_form'
        if np.max(np.abs(phi@coef)) > self.delta:
            self.solver_ = 'SLSQP_linear_constraints'
            objective = lambda beta: float(np.mean((a@beta-target)**2)+self.strength*np.sum(beta**2))
            gradient = lambda beta: 2*(h@beta-v)
            opt = minimize(objective,np.zeros(k),jac=gradient,method='SLSQP',
                           constraints=[LinearConstraint(phi,-self.delta,self.delta)],
                           options={'maxiter':500,'ftol':1e-12})
            if not opt.success or np.max(np.abs(phi@opt.x)) > self.delta+1e-8:
                raise RuntimeError(f'Multiplier optimization failed: {opt.message}')
            coef = opt.x
        self.coef_ = coef
        self.max_training_raw_adjustment_ = float(np.max(np.abs(phi@coef)))
        residual = (base[put].to_numpy()*self.predict_multiplier(X.loc[put])-y[put].to_numpy())/self.scale_
        self.objective_ = float(np.mean(residual**2)+self.strength*np.sum(coef**2))
        if self.objective_ > self.null_objective_+1e-8:
            raise RuntimeError('Multiplier loss exceeds feasible identity')
        return self

    def _design(self, X):
        raw = transform(X,self.family)
        valid_state = (np.isfinite(raw).all(axis=1) & raw.ge(self.lower_).all(axis=1)
                       & raw.le(self.upper_).all(axis=1))
        put = X.option_type.eq(1)
        tenor_ok = X.product_duration_seconds.isin(self.tenors_)
        supported = put & self.supported_ & tenor_ok & valid_state
        reason = np.where(~put,'call_identity',np.where(not self.supported_,'global_support',
                         np.where(~tenor_ok,'tenor_support',np.where(~valid_state,'outside_training_state','supported'))))
        z = ((raw-self.mean_)/self.std_).clip(-3,3).fillna(0).to_numpy()
        phi = np.column_stack([np.ones(len(X)),z])*supported.to_numpy()[:,None]
        return phi,supported,reason

    def predict_diagnostics(self, X):
        validate_input(X)
        if not hasattr(self,'coef_'):
            raise ValueError('Multiplier is not fitted')
        phi,supported,reason=self._design(X)
        multiplier=1+np.clip(phi@self.coef_,-self.delta,self.delta)
        return pd.DataFrame({'multiplier':multiplier,'supported':supported,'fallback_reason':reason},index=X.index)

    def predict_multiplier(self, X):
        return self.predict_diagnostics(X).multiplier.to_numpy()

    def diagnostics(self):
        return {'family':self.family,'strength':self.strength,'delta':self.delta,
                'coefficients':dict(zip(self.feature_names_,self.coef_.tolist())),
                'mean':self.mean_.to_dict(),'std':self.std_.to_dict(),
                'lower':self.lower_.to_dict(),'upper':self.upper_.to_dict(),
                'supported_tenors':self.tenors_,'global_support':self.supported_,
                'put_training_orders':self.put_training_orders_,'loss_scale':self.scale_,
                'objective':self.objective_,'null_objective':self.null_objective_,
                'solver':self.solver_,'max_training_raw_adjustment':self.max_training_raw_adjustment_}
