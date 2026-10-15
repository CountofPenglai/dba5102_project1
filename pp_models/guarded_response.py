"""Squared response with calm-market penalty and optional daily upward limit."""
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from .fast_response_model import INPUTS as BASE_INPUTS, FastResponseModel

INPUTS = BASE_INPUTS+['realized_vol_60m', 'log_return_60m']


def downside_score(X):
    return (-X.log_return_60m).clip(lower=0)/X.realized_vol_60m.clip(lower=1e-8)


def fit_risk_context(X, y):
    """All thresholds use the current day's legal Put history, never today's labels."""
    if (not X.index.equals(y.index) or not np.isfinite(y).all() or y.lt(0).any()
            or not np.isfinite(X[['realized_vol_60m', 'log_return_60m']]).all().all()
            or X.realized_vol_60m.lt(0).any()):
        raise ValueError('Invalid historical risk context')
    put = X.option_type.eq(1)
    if not put.any():
        return {'context_put_rows': 0, 'calm_vol_q80': 0., 'calm_downside_q80': 0., 'payout_q90': 0.}
    return {'context_put_rows': int(put.sum()),
        'calm_vol_q80': float(X.loc[put, 'realized_vol_60m'].quantile(.8)),
        'calm_downside_q80': float(downside_score(X.loc[put]).quantile(.8)),
        'payout_q90': float(y.loc[put].quantile(.9))}


def calm_mask(X, context):
    if not context['context_put_rows']:
        return pd.Series(False, index=X.index)
    return (X.option_type.eq(1) & X.realized_vol_60m.le(context['calm_vol_q80'])
            & downside_score(X).le(context['calm_downside_q80']))


class GuardedResponseModel(FastResponseModel):
    def __init__(self, calm_penalty, upward_step=None):
        if (isinstance(calm_penalty, bool) or calm_penalty not in (.01, .1, 1.)
                or (upward_step is not None and (isinstance(upward_step, bool) or upward_step not in (.05, .1, .2)))):
            raise ValueError('Invalid guarded response parameters')
        self.calm_penalty, self.upward_step = float(calm_penalty), upward_step
        self.fitted, self.updates = False, 0

    def _validate_X(self, X):
        if not isinstance(X, pd.DataFrame) or list(X) != INPUTS:
            raise ValueError('Invalid guarded response feature schema')
        super()._validate_X(X[BASE_INPUTS])
        if (not np.isfinite(X[['realized_vol_60m', 'log_return_60m']]).all().all()
                or X.realized_vol_60m.lt(0).any()):
            raise ValueError('Invalid guarded market inputs')

    def fit(self, X, y):
        self._validate_X(X)
        context = fit_risk_context(X, y)
        previous = self.beta if self.fitted else None
        self.fitted = False
        put = X.option_type.eq(1)
        target = y.loc[put].to_numpy(dtype=float)
        base = X.loc[put, BASE_FEATURE].to_numpy(dtype=float)
        a = base*X.loc[put, 'fast_response_state'].to_numpy(dtype=float)
        calm = calm_mask(X, context)
        self.calm_state_second_moment = float(X.loc[calm, 'fast_response_state'].pow(2).mean()) if calm.any() else 0.
        denominator = float(a@a + self.calm_penalty*(target@target)*self.calm_state_second_moment)
        optimum = max(0., float(a@(target-base))/denominator) if denominator > 0 else 0.
        self.unconstrained_beta = optimum
        optimum = min(optimum, 3.)
        self.previous_beta = previous
        self.upward_limited = previous is not None and self.upward_step is not None and optimum > previous+self.upward_step
        self.beta = min(optimum, previous+self.upward_step) if self.upward_limited else optimum
        if not np.isfinite(self.beta):
            raise ValueError('Nonfinite guarded response amplitude')
        self.put_training_rows = int(put.sum())
        self.calm_training_rows = int(calm.sum())
        self.updates += 1
        self.fitted = True
        return self

    def diagnostics(self):
        return super().diagnostics() | {'response_beta_before_constraints': self.unconstrained_beta,
            'response_previous_beta': float('nan') if self.previous_beta is None else self.previous_beta,
            'response_upward_limited': self.upward_limited, 'response_fit_updates': self.updates,
            'response_amplitude_capped': self.unconstrained_beta > 3.,
            'calm_training_rows': self.calm_training_rows, 'calm_state_second_moment': self.calm_state_second_moment}
