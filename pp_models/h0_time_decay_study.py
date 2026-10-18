"""Report recipe kernels; execution is owned by report_unified_study."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT
from pp_data.features import BASE_FEATURE
from .h0 import probability_frame, h0_state
from .validation import PROTOCOL, prediction_context
from .training import training_session

CONFIG = PROJECT / 'config/h0_time_decay_v1.json'
MODELS = ('official_raw', 'h0', 'h0_decay7', 'h0_decay14')
HALF_LIVES = {'h0': None, 'h0_decay7': 7, 'h0_decay14': 14}
SETTINGS = dict(version='h0_time_decay_v1', validation_protocol='config/validation_protocol_v3.json', models=list(MODELS), half_life_days=HALF_LIVES, primary='h0_decay7', age_basis='creation_time', probability_weight_normalization='sum_equals_direction_order_count', scaler='unweighted_original_h0', logistic_C=0.1, residual_correction=False, trend_overlay=False, parameter_search=False, bootstrap_blocks=[3, 7, 14], bootstrap_repeats=2000)

@threadpool_limits.wrap(limits=1)
def predict_period(pool, protocol=None, *, training=None):
    protocol = PROTOCOL if protocol is None else protocol
    session = training_session(pool, protocol, training)
    frames = []; states = {}; diagnostics = []
    for window in session.splitter.windows():
        origin = window.origin
        q = window.query_ids
        x = pool.X.loc[q]
        common = prediction_context(pool, window, session.splitter)
        params = {}; components = {}
        for name, half in HALF_LIVES.items():
            model = session.fit(window, half)
            probability, severity = model.predict_components(probability_frame(x))
            components[name] = (probability, severity)
            params[name] = dict(parameters=h0_state(model), weights=model.weight_diagnostics)
            for side, diag in model.weight_diagnostics.items():
                for month, part in diag['historical_months'].items():
                    diagnostics.append(dict(origin=origin.isoformat(), model=name, side=side, historical_month=month,
                        **{k: v for k, v in diag.items() if k != 'historical_months'},
                        month_orders=part['orders'], month_weight=part['weight_sum'],
                        month_severity_numerator=part['severity_numerator'], month_severity_denominator=part['severity_denominator'],
                        severity_coefficient=model.parameters[side]['severity']))
        p0, s0 = components['h0']
        for name in MODELS:
            p, s = components.get(name, (np.full(len(q), np.nan), np.full(len(q), np.nan)))
            pred = x[BASE_FEATURE].to_numpy() if name == 'official_raw' else p*s
            if not np.isfinite(pred).all() or (pred < 0).any(): raise ValueError('Invalid predictions')
            frames.append(common.assign(model=name, prediction=pred, probability=p, severity=s,
                h0_probability=p0, h0_severity=s0, multiplier=pred/x[BASE_FEATURE]))
        states[origin.isoformat()] = dict(**session.splitter.describe(window), models=params)
    if not frames: raise ValueError('Empty validation interval')
    return pd.concat(frames), states, pd.DataFrame(diagnostics)
