"""Bounded linear and squared five-minute pressure responses on official Base."""
import pandas as pd
from pp_data.io import ModelSplit
from .fast_response_features import augment_pool as augment_legacy, response_features
from .fast_response_model import FastResponseModel


class AcceleratedResponseModel(FastResponseModel):
    def __init__(self, lam):
        if isinstance(lam, bool) or lam not in (0., .01, .1):
            raise ValueError('Unsupported accelerated response penalty')
        self.lam, self.fitted = float(lam), False

    def fit(self, X, y):
        super().fit(X, y)
        self.unconstrained_beta = self.beta
        # This is the exact constrained optimum of the one-dimensional quadratic loss.
        self.beta = min(self.beta, 3.)
        return self

    def diagnostics(self):
        return super().diagnostics() | {'response_amplitude_capped': self.unconstrained_beta > 3.,
                                       'response_beta_before_cap': self.unconstrained_beta}


def augment_pool(pool, market):
    extended, legacy_info = augment_legacy(pool, market)
    X, meta = extended.X.copy(), extended.metadata.copy()
    for name, power in [('linear5', 1), ('power5', 2)]:
        extra = response_features(pool.metadata.created_at_utc, market, lookback=288, interval_minutes=5, power=power)
        X['response_'+name] = extra.fast_response_state
        X['phase_'+name] = extra.fast_response_phase
        meta['response_available_'+name+'_utc'] = extra.response_available_at_utc
    max_age = float((meta.created_at_utc-meta.response_available_power5_utc).dt.total_seconds().max())
    return ModelSplit(X, pool.y.copy(), meta), {
        'version': 'accelerated_response_v1', 'orders': len(X), 'interval_minutes': 5,
        'scale_lookback_bars': 288, 'scale_floor': .0001, 'noise_threshold_sigma': .1,
        'pressure_cap': 3., 'powers': [1, 2], 'recovery_factor_per_bar': .5,
        'maximum_multiplier': 4., 'latest_state_age_seconds_max': max_age,
        'legacy_15m': legacy_info, 'labels_used_for_state': False}
