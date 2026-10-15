"""Shared calendar correction with strongly regularized direction deviations."""
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize
from .macro_multiplier_features import BASIS, FEATURE_COLUMNS


def validate_features(X):
    if (not isinstance(X, pd.DataFrame) or set(X) != set(FEATURE_COLUMNS) or not X.index.is_unique
            or not X.option_type.isin([0, 1]).all()
            or not np.isfinite(X[BASIS]).all().all()
            or not X[BASIS].ge(0).all().all() or not X[BASIS].le(1+1e-12).all().all()):
        raise ValueError('Multiplier feature whitelist or bounded exposure violated')
    active = X[BASIS].gt(0).any(axis=1)
    if (active & (~X.calendar_status.eq('known_event') | X.event_time.isna() | X.event_id.eq(''))).any():
        raise ValueError('Positive exposure without dated event evidence')


def support(X, minimum):
    rows = []
    for basis in BASIS:
        g = X.loc[X[basis].gt(0)]
        n, events, months = len(g), g.event_id.nunique(), g.event_month.nunique()
        rows.append({'basis': basis, 'orders': n, 'events': events, 'months': months,
                     'enabled': bool(n >= minimum[0] and events >= minimum[1] and months >= minimum[2])})
    return rows


def constraint_matrix():
    corners = np.array(list(itertools.product([0., 1.], repeat=3)))
    rows, caps = [], []
    for corner in corners:
        rows.append(np.r_[corner, np.zeros(6)]); caps.append(.05)
        for side in (0, 1):
            deviation = np.zeros(9); deviation[3+3*side:6+3*side] = corner
            rows.extend([deviation, deviation+np.r_[corner, np.zeros(6)]])
            caps.extend([.02, .05])
    return np.array(rows), np.array(caps)


class MacroMultiplier:
    def __init__(self, strength):
        if strength not in (1., 10., 100.): raise ValueError('Unregistered shrinkage strength')
        self.strength = float(strength)

    def fit(self, X, base, y, origin):
        validate_features(X); origin = pd.Timestamp(origin)
        if (origin.tzinfo is None or any(not X.index.equals(v.index) for v in (base, y))
                or not np.isfinite(base).all() or base.le(0).any()
                or not np.isfinite(y).all() or y.lt(0).any()):
            raise ValueError('Invalid historical training data')
        mask = X.calendar_status.eq('known_event') & X.event_time.lt(origin) & X[BASIS].gt(0).any(axis=1)
        f = X.loc[mask]; b, target_y = base.loc[f.index].to_numpy(), y.loc[f.index].to_numpy()
        self.origin_ = origin.isoformat()
        shared = support(f, (30, 5, 2))
        directions = [support(f.loc[f.option_type.eq(side)], (60, 8, 3)) for side in (0, 1)]
        self.shared_ = np.array([r['enabled'] for r in shared])
        self.direction_ = np.array([[r['enabled'] for r in rows] for rows in directions]) & self.shared_
        self.support_ = {'shared': shared, 'call': directions[0], 'put': directions[1]}
        self.coef_ = np.zeros(9); self.solver_ = 'unsupported_identity'
        self.training_orders_ = len(f); self.training_events_ = f.event_id.nunique()
        w = (1/f.event_id.map(f.event_id.value_counts()).to_numpy(float)) if len(f) else np.array([])
        if len(w): w /= w.sum()
        self.scale_ = max(float(np.sqrt(np.sum(w*target_y**2))), 1e-6)
        target = (target_y-b)/self.scale_
        self.null_objective_ = float(np.sum(w*target**2))
        self.objective_ = self.null_objective_
        if not self.shared_.any() or not len(f): return self
        phi = self._design(f)
        a = phi*b[:, None]/self.scale_
        penalty = self.strength*np.r_[np.ones(3), np.full(6, 10.)]
        h = a.T@(w[:, None]*a)+np.diag(penalty); v = a.T@(w*target)
        coef = np.linalg.solve(h, v)
        matrix, caps = constraint_matrix()
        enabled = np.r_[self.shared_, self.direction_.reshape(-1)]
        self.solver_ = 'closed_form'
        objective = lambda beta: float(np.sum(w*(a@beta-target)**2)+np.sum(penalty*beta**2))
        if np.any(np.abs(matrix@coef) > caps):
            self.solver_ = 'SLSQP_cube_constraints'
            fit = minimize(objective, np.zeros(9), jac=lambda beta: 2*(h@beta-v), method='SLSQP',
                           bounds=[(None, None) if e else (0., 0.) for e in enabled],
                           constraints=[LinearConstraint(matrix, -caps, caps)],
                           options={'ftol': 1e-12, 'maxiter': 500})
            if not fit.success: raise RuntimeError('Announcement multiplier optimizer failed: '+fit.message)
            coef = fit.x
        if np.any(np.abs(matrix@coef) > caps+1e-8) or np.any(np.abs(coef[~enabled]) > 1e-10):
            raise RuntimeError('Announcement multiplier violated whole-domain bounds')
        self.coef_ = coef; self.objective_ = objective(coef)
        if self.objective_ > self.null_objective_+1e-8:
            raise RuntimeError('Announcement objective exceeds feasible identity')
        return self

    def _design(self, X):
        values = X[BASIS].to_numpy(float)
        shared = values*self.shared_
        direction = [values*self.direction_[s]*X.option_type.eq(s).to_numpy()[:, None] for s in (0, 1)]
        return np.column_stack([shared, *direction])

    def predict_diagnostics(self, X):
        validate_features(X)
        if not hasattr(self, 'coef_'): raise ValueError('Multiplier not fitted')
        phi = self._design(X)
        shared = phi[:, :3]@self.coef_[:3]; deviation = phi[:, 3:]@self.coef_[3:]
        supported = np.any(phi != 0, axis=1)
        reasons = np.where(~X.calendar_status.eq('known_event'), X.calendar_status,
                           np.where(supported, 'supported', 'insufficient_event_support'))
        return pd.DataFrame({'multiplier': 1+np.clip(shared+deviation, -.05, .05),
                             'shared_adjustment': shared, 'direction_adjustment': deviation,
                             'supported': supported, 'fallback_reason': reasons}, index=X.index)

    def diagnostics(self):
        return {'strength': self.strength, 'origin': self.origin_, 'shared_enabled': self.shared_.tolist(),
                'direction_enabled': self.direction_.tolist(), 'support': self.support_,
                'coefficients': self.coef_.tolist(), 'training_orders': self.training_orders_,
                'training_events': int(self.training_events_), 'loss_scale': self.scale_,
                'objective': self.objective_, 'null_objective': self.null_objective_, 'solver': self.solver_}
