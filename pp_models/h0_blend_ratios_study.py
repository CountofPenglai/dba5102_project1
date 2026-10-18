"""Report recipe kernels; execution is owned by report_unified_study."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT
from pp_data.features import BASE_FEATURE
from .h0 import probability_frame, h0_state
from .validation import PROTOCOL, prediction_context
from .training import training_session, MIN_TRAINING_ORDERS
from .forecast_history import ForwardPredictionHistory
from .h0_blend_ratios import GRID, FIXED, DYNAMIC, MODELS, grid_weight

CONFIG = PROJECT / 'config/h0_blend_ratios_v1.json'
SETTINGS = dict(version='h0_blend_ratios_v1', protocol='config/validation_protocol_v3.json', models=list(MODELS), fixed_grid=list(GRID), fixed_ratios={k: list(v) for k, v in FIXED.items()}, dynamic_windows=DYNAMIC, primary_dynamic='dynamic14', warmup_start='2026-04-03T00:00:00Z', history_days=30, update_days=7, minimum_oos_orders=100, minimum_oos_days=10, weight_shrinkage_orders=100, weight_prior=0.5, insufficient_history_alpha=0.0, ties='smaller_h0_weight', weekly_label_feedback=True, time_decay=False, contract_residual=False, hourly_amount=False, state_gate=False, fixed_grid_exploration=True, monthly_oracle_selection=False, adaptive_search=False, bootstrap_blocks=[3, 7, 14], bootstrap_repeats=2000, simultaneous_comparisons=51)

@threadpool_limits.wrap(limits=1)
def predict_period(pool, protocol=None, *, training=None):
    protocol = PROTOCOL if protocol is None else protocol
    session = training_session(pool, protocol, training)
    cache = ForwardPredictionHistory(pool)
    frames = []; states = {}
    for window in session.splitter.windows(warmup_days=28):
        origin = window.origin
        h, q = window.history_ids, window.query_ids
        state = dict(**session.splitter.describe(window), is_validation=window.is_validation,
                     h0_supported=len(h) >= MIN_TRAINING_ORDERS)
        if not state['h0_supported']:
            if window.is_validation: raise ValueError('Insufficient validation H0 training history')
            states[origin.isoformat()] = state
            continue
        x = pool.X.loc[q]
        model = session.fit(window)
        hp = model.predict_raw(probability_frame(x)); bp = x[BASE_FEATURE].to_numpy()
        state['h0'] = h0_state(model)
        if window.is_validation:
            past = cache.settled(origin, history_days=protocol['history_days'])
            weights = {name: {str(s): grid_weight(past.loc[past.side.eq(s)], origin, days) for s in (0, 1)}
                       for name, days in DYNAMIC.items()}
            alphas = {name: np.where(x.option_type.eq(0), pair[0], pair[1]) for name, pair in FIXED.items()}
            for name, pars in weights.items():
                alphas[name] = x.option_type.map({s: pars[str(s)]['alpha'] for s in (0, 1)}).to_numpy()
            common = prediction_context(pool, window, session.splitter).assign(h0_prediction=hp, base_prediction=bp)
            for name, a in alphas.items():
                prediction = hp if name == 'h0' else bp if name == 'official_raw' else a*hp+(1-a)*bp
                if not np.isfinite(prediction).all() or (prediction < 0).any(): raise ValueError('Invalid convex prediction')
                frames.append(common.assign(model=name, prediction=prediction, h0_weight=a))
            state['weights'] = weights
        states[origin.isoformat()] = state
        cache.append(window, hp, bp)
    if not frames: raise ValueError('Empty forward validation')
    return pd.concat(frames), states
