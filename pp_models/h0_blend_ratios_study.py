"""Report recipe kernels; execution is owned by report_unified_study."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT
from pp_data.features import BASE_FEATURE
from .h0 import H0, hurdle_design, probability_frame, h0_state
from .validation import PROTOCOL, PROTOCOL_PATH, history_ids, tail_context
from .h0_blend_ratios import GRID, FIXED, DYNAMIC, MODELS, CANDIDATES, grid_weight

CONFIG = PROJECT / 'config/h0_blend_ratios_v1.json'
SETTINGS = dict(version='h0_blend_ratios_v1', protocol='config/validation_protocol_v3.json', models=list(MODELS), fixed_grid=list(GRID), fixed_ratios={k: list(v) for k, v in FIXED.items()}, dynamic_windows=DYNAMIC, primary_dynamic='dynamic14', warmup_start='2026-04-03T00:00:00Z', history_days=30, update_days=7, minimum_oos_orders=100, minimum_oos_days=10, weight_shrinkage_orders=100, weight_prior=0.5, insufficient_history_alpha=0.0, ties='smaller_h0_weight', weekly_label_feedback=True, time_decay=False, contract_residual=False, hourly_amount=False, state_gate=False, fixed_grid_exploration=True, monthly_oracle_selection=False, adaptive_search=False, bootstrap_blocks=[3, 7, 14], bootstrap_repeats=2000, simultaneous_comparisons=51)

@threadpool_limits.wrap(limits=1)
def predict_period(pool, protocol=None):
    protocol = PROTOCOL if protocol is None else protocol
    start, end = pd.Timestamp(protocol['validation_start']), pd.Timestamp(protocol['validation_end'])
    warmup = start-pd.Timedelta(days=28)
    if (start.tzinfo is None or end.tzinfo is None or not pd.Timestamp(protocol['training_start']) <= warmup < start < end
            or start != start.floor('D') or protocol['history_days'] != 30 or protocol['update_days'] != 7
            or not pool.X.index.is_unique or not pool.X.index.equals(pool.y.index) or not pool.X.index.equals(pool.metadata.index)):
        raise ValueError('Invalid forward validation pool/grid')
    created, closed = pool.metadata.created_at_utc, pool.metadata.closed_at_utc
    cache = []; frames = []; states = {}
    for origin in pd.date_range(warmup, end, freq='7D', inclusive='left'):
        stop = min(origin+pd.Timedelta(days=7), end)
        q = pool.X.index[created.ge(origin) & created.lt(stop)]
        if not len(q): continue
        h = history_ids(pool, origin, protocol)
        state = dict(origin=origin.isoformat(), stop=stop.isoformat(), history_orders=len(h), query_orders=len(q),
            is_validation=bool(origin >= start), h0_supported=len(h) >= 100,
            earliest_training_create=created.loc[h].min().isoformat() if len(h) else None,
            latest_training_close=closed.loc[h].max().isoformat() if len(h) else None)
        if len(h) < 100:
            if origin >= start: raise ValueError('Insufficient validation H0 training history')
            states[origin.isoformat()] = state; continue
        hx, x = pool.X.loc[h], pool.X.loc[q]
        model = H0().fit(probability_frame(hx), pool.y.loc[h], created.loc[h])
        hp = model.predict_raw(probability_frame(x)); bp = x[BASE_FEATURE].to_numpy()
        state['h0'] = h0_state(model)
        if origin >= start:
            previous = pd.concat(cache) if cache else pd.DataFrame({
                **{k: pd.Series(dtype=float) for k in ('h0', 'base', 'side')},
                **{k: pd.Series(dtype='datetime64[ns, UTC]') for k in ('created', 'forecast_origin')}})
            use = previous.created.ge(origin-pd.Timedelta(days=30)) & closed.reindex(previous.index).lt(origin)
            past = previous.loc[use].copy(); past['closed'] = closed.reindex(past.index); past['actual'] = pool.y.reindex(past.index)
            weights = {name: {str(s): grid_weight(past.loc[past.side.eq(s)], origin, window) for s in (0, 1)}
                       for name, window in DYNAMIC.items()}
            alphas = {name: np.where(x.option_type.eq(0), pair[0], pair[1]) for name, pair in FIXED.items()}
            for name, pars in weights.items():
                alphas[name] = x.option_type.map({s: pars[str(s)]['alpha'] for s in (0, 1)}).to_numpy()
            threshold = pd.Series(index=q, dtype=float); source = pd.Series(index=q, dtype=object)
            for month in sorted(created.loc[q].dt.strftime('%Y-%m').unique()):
                mo = pd.Timestamp(month+'-01', tz='UTC'); mq = q[created.loc[q].dt.strftime('%Y-%m').eq(month)]
                tv, sv = tail_context(pool, history_ids(pool, mo, protocol, window=False), mq)
                threshold.loc[mq] = tv; source.loc[mq] = sv
            common = pd.DataFrame(dict(day=created.loc[q].dt.floor('D'), month=created.loc[q].dt.strftime('%Y-%m'),
                side=x.option_type.astype(int), tenor=x.product_duration_seconds.astype(int), quantity=x.quantity_btc,
                tail_threshold=threshold, tail_source=source, origin=origin.isoformat(), h0_prediction=hp, base_prediction=bp), index=q)
            for name, a in alphas.items():
                prediction = hp if name == 'h0' else bp if name == 'official_raw' else a*hp+(1-a)*bp
                if not np.isfinite(prediction).all() or (prediction < 0).any(): raise ValueError('Invalid convex prediction')
                frames.append(common.assign(model=name, prediction=prediction, h0_weight=a))
            state['weights'] = weights
        states[origin.isoformat()] = state
        cache.append(pd.DataFrame(dict(h0=hp, base=bp, side=x.option_type, created=created.loc[q], forecast_origin=origin), index=q))
    if not frames: raise ValueError('Empty forward validation')
    return pd.concat(frames), states
