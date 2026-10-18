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
from .h0_state_fallback import StateGate, history_weight, match_states

CANDIDATES = ('half_blend', 'state_fallback', 'history_blend', 'state_history_blend')
MODELS = ('official_raw', 'h0') + CANDIDATES
CONFIG = PROJECT / 'config/h0_state_fallback_v1.json'
SETTINGS = dict(version='h0_state_fallback_v1', protocol='config/validation_protocol_v3.json', models=list(MODELS), primary='state_history_blend', warmup_start='2026-04-03T00:00:00Z', history_days=30, update_days=7, state_smoothing_hours=6, state_minimum_history_hours=168, state_distance_quantiles=[0.9, 0.99], state_scale='median_iqr_div_1.349', minimum_oos_orders=100, minimum_oos_days=10, weight_shrinkage_orders=100, weight_prior=0.5, weight_bounds=[0.0, 1.0], insufficient_history_alpha=0.0, probability_model='original_h0', time_decay=False, contract_residual=False, hourly_amount=False, parameter_search=False, market_prefix_sha256='cbf5d491d167ac377beb844dc9794e4f5e9986f72859035426f7789e67924b50', bootstrap_blocks=[3, 7, 14], bootstrap_repeats=2000)

@threadpool_limits.wrap(limits=1)
def predict_period(pool, market, protocol=None, *, training=None):
    protocol = PROTOCOL if protocol is None else protocol
    session = training_session(pool, protocol, training)
    at_quote = match_states(pool.metadata.created_at_utc, market)
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
            weights = {str(s): history_weight(past.loc[past.side.eq(s)], origin) for s in (0, 1)}
            alpha = x.option_type.map({s: weights[str(s)]['alpha'] for s in (0, 1)}).to_numpy()
            gate_model = StateGate().fit(market, origin, protocol['training_start'])
            g, distance = gate_model.predict(at_quote.loc[q])
            alphas = dict(official_raw=np.zeros(len(q)), h0=np.ones(len(q)), half_blend=np.full(len(q), .5),
                state_fallback=1-g, history_blend=alpha, state_history_blend=alpha*(1-g))
            common = prediction_context(pool, window, session.splitter).assign(h0_prediction=hp, base_prediction=bp,
                market_distance=distance, state_gate=g, state_available=np.isfinite(at_quote.loc[q]).all(axis=1))
            for name, a in alphas.items():
                prediction = hp if name == 'h0' else bp if name == 'official_raw' else a*hp+(1-a)*bp
                if not np.isfinite(prediction).all() or (prediction < 0).any(): raise ValueError('Invalid convex prediction')
                frames.append(common.assign(model=name, prediction=prediction, h0_weight=a))
            state.update(weights=weights, market=gate_model.state)
        states[origin.isoformat()] = state
        cache.append(window, hp, bp)
    if not frames: raise ValueError('Empty forward validation')
    return pd.concat(frames), states
