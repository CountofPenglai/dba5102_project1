"""Report recipe kernels; execution is owned by report_unified_study."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT
from pp_data.features import BASE_FEATURE
from .h0 import H0, hurdle_design, probability_frame, h0_state
from .validation import PROTOCOL, PROTOCOL_PATH, history_ids, tail_context
from .h0_hourly_amount import HourlyAmount, market_features, amount_features, CANDIDATES

CONFIG = PROJECT / 'config/h0_hourly_amount_v1.json'
MODELS = ('official_raw', 'h0') + CANDIDATES
SETTINGS = dict(version='h0_hourly_amount_v1', protocol='config/validation_protocol_v3.json', models=list(MODELS), primary='amount_joint', probability='unchanged_unweighted_h0', time_decay=False, contract_residual=False, windows_minutes=[60, 240], ridge_strength=100, continuous_bounds=[0.9, 1.1], gate_bounds=[1.0, 1.1], minimum_positive_orders=40, minimum_positive_days=5, gate_level_quantiles=[0.75, 0.95], gate_speed_ratios=[1.25, 2.0], gate_minimum_positive_orders=15, gate_minimum_positive_days=3, standardized_clip=3, parameter_search=False, market_prefix_sha256='cbf5d491d167ac377beb844dc9794e4f5e9986f72859035426f7789e67924b50', bootstrap_blocks=[3, 7, 14], bootstrap_repeats=2000)

@threadpool_limits.wrap(limits=1)
def predict_period(pool, market, protocol=None):
    protocol = PROTOCOL if protocol is None else protocol
    start, end = pd.Timestamp(protocol['validation_start']), pd.Timestamp(protocol['validation_end'])
    if (start.tzinfo is None or end.tzinfo is None or not pd.Timestamp(protocol['training_start']) < start < end
            or protocol['update_days'] != 7 or protocol['history_days'] != 30 or start != start.floor('D')
            or not pool.X.index.is_unique or not pool.X.index.equals(pool.y.index)
            or not pool.X.index.equals(pool.metadata.index) or not pool.X.index.equals(market.index)):
        raise ValueError('Invalid hourly study pool or weekly dates')
    created, closed = pool.metadata.created_at_utc, pool.metadata.closed_at_utc
    frames = []; states = {}
    for origin in pd.date_range(start, end, freq='7D', inclusive='left'):
        stop = min(origin+pd.Timedelta(days=7), end)
        q = pool.X.index[created.ge(origin) & created.lt(stop)]
        if not len(q): continue
        h = history_ids(pool, origin, protocol)
        if len(h) < 100: raise ValueError('Insufficient H0 historical support')
        hx, x = pool.X.loc[h], pool.X.loc[q]
        h0 = H0().fit(probability_frame(hx), pool.y.loc[h], created.loc[h])
        p0, s0 = h0.predict_components(probability_frame(x)); _, sh = h0.predict_components(probability_frame(hx))
        ah = pd.Series(sh, index=h); aq = pd.Series(s0, index=q)
        fh = amount_features(hx, market.loc[h], ah); fq = amount_features(x, market.loc[q], aq)
        threshold = pd.Series(index=q, dtype=float); source = pd.Series(index=q, dtype=object)
        for month in sorted(created.loc[q].dt.strftime('%Y-%m').unique()):
            mo = pd.Timestamp(month+'-01', tz='UTC'); mq = q[created.loc[q].dt.strftime('%Y-%m').eq(month)]
            tv, sv = tail_context(pool, history_ids(pool, mo, protocol, window=False), mq)
            threshold.loc[mq] = tv; source.loc[mq] = sv
        common = pd.DataFrame(dict(day=created.loc[q].dt.floor('D'), month=created.loc[q].dt.strftime('%Y-%m'),
            side=x.option_type.astype(int), tenor=x.product_duration_seconds.astype(int), quantity=x.quantity_btc,
            tail_threshold=threshold, tail_source=source, origin=origin.isoformat(), h0_probability=p0, h0_severity=s0,
            hourly_available=fq.available, level=fq.level, speed=fq.speed), index=q)
        severity = {'h0': s0}; supports = {'h0': np.ones(len(q), bool)}; params = {}
        for name in CANDIDATES:
            m = HourlyAmount(name).fit(hx.option_type, fh, pool.y.loc[h], created.loc[h], ah)
            severity[name], supports[name] = m.predict(x.option_type, fq, aq); params[name] = m.state()
        for name in MODELS:
            s = severity.get(name, np.full(len(q), np.nan))
            prediction = x[BASE_FEATURE].to_numpy() if name == 'official_raw' else p0*s
            if not np.isfinite(prediction).all() or (prediction < 0).any(): raise ValueError('Invalid payout prediction')
            multiplier = np.divide(s, s0, out=np.ones(len(q)), where=s0 > 0) if name != 'official_raw' else np.ones(len(q))
            frames.append(common.assign(model=name, prediction=prediction, probability=p0 if name != 'official_raw' else np.nan,
                severity=s, amount_multiplier=multiplier, supported=supports.get(name, np.ones(len(q), bool))))
        states[origin.isoformat()] = dict(stop=stop.isoformat(), history_orders=len(h), query_orders=len(q),
            earliest_training_create=created.loc[h].min().isoformat(), latest_training_close=closed.loc[h].max().isoformat(),
            h0=h0_state(h0), models=params)
    if not frames: raise ValueError('Empty validation interval')
    return pd.concat(frames), states
