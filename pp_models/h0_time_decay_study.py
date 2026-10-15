"""Report recipe kernels; execution is owned by report_unified_study."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT
from pp_data.features import BASE_FEATURE
from .h0 import H0, hurdle_design, probability_frame, h0_state
from .validation import PROTOCOL, PROTOCOL_PATH, history_ids, tail_context
from .h0_time_decay import TimeDecayH0

CONFIG = PROJECT / 'config/h0_time_decay_v1.json'
MODELS = ('official_raw', 'h0', 'h0_decay7', 'h0_decay14')
HALF_LIVES = {'h0': None, 'h0_decay7': 7, 'h0_decay14': 14}
SETTINGS = dict(version='h0_time_decay_v1', validation_protocol='config/validation_protocol_v3.json', models=list(MODELS), half_life_days=HALF_LIVES, primary='h0_decay7', age_basis='creation_time', probability_weight_normalization='sum_equals_direction_order_count', scaler='unweighted_original_h0', logistic_C=0.1, residual_correction=False, trend_overlay=False, parameter_search=False, bootstrap_blocks=[3, 7, 14], bootstrap_repeats=2000)

@threadpool_limits.wrap(limits=1)
def predict_period(pool, protocol=None):
    protocol = PROTOCOL if protocol is None else protocol
    start = pd.Timestamp(protocol['validation_start']); end = pd.Timestamp(protocol['validation_end'])
    train_start = pd.Timestamp(protocol['training_start'])
    if (start.tzinfo is None or end.tzinfo is None or train_start.tzinfo is None or not train_start < start < end
            or start != start.floor('D') or protocol['update_days'] != 7 or protocol['history_days'] != 30
            or not pool.X.index.is_unique or not pool.X.index.equals(pool.y.index)
            or not pool.X.index.equals(pool.metadata.index)):
        raise ValueError('Invalid rolling validation pool or dates')
    created = pool.metadata.created_at_utc; closed = pool.metadata.closed_at_utc
    frames = []; states = {}; diagnostics = []
    for origin in pd.date_range(start, end, freq='7D', inclusive='left'):
        stop = min(origin+pd.Timedelta(days=7), end)
        q = pool.X.index[created.ge(origin) & created.lt(stop)]
        if not len(q): continue
        train = history_ids(pool, origin, protocol)
        if len(train) < 100: raise ValueError('Insufficient historical H0 training orders')
        x = pool.X.loc[q]; threshold = pd.Series(index=q, dtype=float); source = pd.Series(index=q, dtype=object)
        for month in sorted(created.loc[q].dt.strftime('%Y-%m').unique()):
            mo = pd.Timestamp(month+'-01', tz='UTC'); mq = q[created.loc[q].dt.strftime('%Y-%m').eq(month)]
            tv, sv = tail_context(pool, history_ids(pool, mo, protocol, window=False), mq)
            threshold.loc[mq] = tv; source.loc[mq] = sv
        common = pd.DataFrame(dict(day=created.loc[q].dt.floor('D'), month=created.loc[q].dt.strftime('%Y-%m'),
            side=x.option_type.astype(int), tenor=x.product_duration_seconds.astype(int), quantity=x.quantity_btc,
            tail_threshold=threshold, tail_source=source, origin=origin.isoformat()), index=q)
        params = {}; components = {}
        for name, half in HALF_LIVES.items():
            model = TimeDecayH0(half).fit(probability_frame(pool.X.loc[train]), pool.y.loc[train], created.loc[train], origin)
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
        states[origin.isoformat()] = dict(origin=origin.isoformat(), stop=stop.isoformat(), history_orders=len(train),
            query_orders=len(q), earliest_training_create=created.loc[train].min().isoformat(),
            latest_training_close=closed.loc[train].max().isoformat(), models=params)
    if not frames: raise ValueError('Empty validation interval')
    return pd.concat(frames), states, pd.DataFrame(diagnostics)
