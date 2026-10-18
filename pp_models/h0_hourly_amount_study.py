"""Report recipe kernels; execution is owned by report_unified_study."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT
from pp_data.features import BASE_FEATURE
from .h0 import probability_frame, h0_state
from .validation import PROTOCOL, prediction_context
from .training import training_session
from .h0_hourly_amount import HourlyAmount, amount_features, CANDIDATES

CONFIG = PROJECT / 'config/h0_hourly_amount_v1.json'
MODELS = ('official_raw', 'h0') + CANDIDATES
SETTINGS = dict(version='h0_hourly_amount_v1', protocol='config/validation_protocol_v3.json', models=list(MODELS), primary='amount_joint', probability='unchanged_unweighted_h0', time_decay=False, contract_residual=False, windows_minutes=[60, 240], ridge_strength=100, continuous_bounds=[0.9, 1.1], gate_bounds=[1.0, 1.1], minimum_positive_orders=40, minimum_positive_days=5, gate_level_quantiles=[0.75, 0.95], gate_speed_ratios=[1.25, 2.0], gate_minimum_positive_orders=15, gate_minimum_positive_days=3, standardized_clip=3, parameter_search=False, market_prefix_sha256='cbf5d491d167ac377beb844dc9794e4f5e9986f72859035426f7789e67924b50', bootstrap_blocks=[3, 7, 14], bootstrap_repeats=2000)

@threadpool_limits.wrap(limits=1)
def predict_period(pool, market, protocol=None, *, training=None):
    protocol = PROTOCOL if protocol is None else protocol
    session = training_session(pool, protocol, training)
    if not pool.X.index.equals(market.index):
        raise ValueError('Hourly market features must align with the common pool')
    created = pool.metadata.created_at_utc
    frames = []; states = {}
    for window in session.splitter.windows():
        origin = window.origin
        h, q = window.history_ids, window.query_ids
        hx, x = pool.X.loc[h], pool.X.loc[q]
        h0 = session.fit(window)
        p0, s0 = h0.predict_components(probability_frame(x))
        _, sh = h0.predict_components(probability_frame(hx))
        ah = pd.Series(sh, index=h); aq = pd.Series(s0, index=q)
        fh = amount_features(hx, market.loc[h], ah); fq = amount_features(x, market.loc[q], aq)
        common = prediction_context(pool, window, session.splitter).assign(
            h0_probability=p0, h0_severity=s0, hourly_available=fq.available, level=fq.level, speed=fq.speed)
        severity = {'h0': s0}; supports = {'h0': np.ones(len(q), bool)}; params = {}
        for name in CANDIDATES:
            m = HourlyAmount(name).fit(hx.option_type, fh, pool.y.loc[h], created.loc[h], ah)
            severity[name], supports[name] = m.predict(x.option_type, fq, aq)
            params[name] = m.state()
        for name in MODELS:
            s = severity.get(name, np.full(len(q), np.nan))
            prediction = x[BASE_FEATURE].to_numpy() if name == 'official_raw' else p0*s
            if not np.isfinite(prediction).all() or (prediction < 0).any(): raise ValueError('Invalid payout prediction')
            multiplier = np.divide(s, s0, out=np.ones(len(q)), where=s0 > 0) if name != 'official_raw' else np.ones(len(q))
            frames.append(common.assign(model=name, prediction=prediction, probability=p0 if name != 'official_raw' else np.nan,
                severity=s, amount_multiplier=multiplier, supported=supports.get(name, np.ones(len(q), bool))))
        details = session.splitter.describe(window)
        details.pop('origin')  # Preserve the established hourly archive schema.
        states[origin.isoformat()] = dict(**details, h0=h0_state(h0), models=params)
    if not frames: raise ValueError('Empty validation interval')
    return pd.concat(frames), states
