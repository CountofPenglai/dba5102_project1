"""Report recipe kernels; execution is owned by report_unified_study."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT
from pp_data.features import BASE_FEATURE
from .h0 import H0, hurdle_design, probability_frame, h0_state
from .validation import PROTOCOL, PROTOCOL_PATH, history_ids, tail_context
from scipy.special import expit

CALL_GRID = tuple((i / 100 for i in range(25, 66, 5)))
PUT_GRID = tuple((i / 100 for i in range(10, 41, 5)))
FINE = {f'blend_c{round(c * 100):03d}_p{round(p * 100):03d}': (c, p) for c in CALL_GRID for p in PUT_GRID}
FIXED = {'official_raw': (0.0, 0.0), 'h0': (1.0, 1.0), 'half_blend': (0.5, 0.5), 'blend_c000_p025': (0.0, 0.25), **FINE}
MODELS = tuple(FIXED)
PREVIOUS_BEST = 'blend_c050_p025'
PAIRS = [(c, r) for c in FINE for r in ('official_raw', 'h0', PREVIOUS_BEST) if c != r]
CONFIG = PROJECT / 'config/h0_blend_fine_v1.json'
SETTINGS = dict(version='h0_blend_fine_v1', protocol='config/validation_protocol_v3.json', models=list(MODELS), call_grid=list(CALL_GRID), put_grid=list(PUT_GRID), fine_models=list(FINE), fixed_ratios={k: list(v) for k, v in FIXED.items()}, pairs=[list(p) for p in PAIRS], previous_best=PREVIOUS_BEST, h0_mode='replay_current_run_weekly_parameters', h0_history_days=30, update_days=7, ratio_mode='fixed_all_validation_months', new_h0_fits=0, new_dynamic_weights=False, primary_metric='month_equal_rmse', pooled_rmse_required=True, sequential_validation_tuning=True, monthly_oracle_selection=False, adaptive_grid_expansion=False, bootstrap_blocks=[3, 7, 14], bootstrap_repeats=2000, simultaneous_comparisons=len(PAIRS))

def replay_h0(x, state):
    design = hurdle_design(x); out = np.empty(len(x))
    for side in (0, 1):
        mask = x.option_type.eq(side).to_numpy(); par = state[str(side)]
        if 'coef' in par:
            z = np.clip((design[mask]-np.asarray(par['mean']))/np.asarray(par['scale']), -3, 3)
            probability = expit(z@np.asarray(par['coef'])+par['intercept'])
        else: probability = par['probability']
        out[mask] = probability*x.loc[mask, BASE_FEATURE].to_numpy()*par['severity']
    if not np.isfinite(out).all() or (out < 0).any(): raise ValueError('Invalid replayed H0')
    return out


@threadpool_limits.wrap(limits=1)
def predict_period(pool, archived_states, protocol=None):
    protocol = PROTOCOL if protocol is None else protocol
    start, end = pd.Timestamp(protocol['validation_start']), pd.Timestamp(protocol['validation_end'])
    if (start.tzinfo is None or end.tzinfo is None or not pd.Timestamp(protocol['training_start']) < start < end
            or start != start.floor('D') or protocol['history_days'] != 30 or protocol['update_days'] != 7
            or not pool.X.index.is_unique or not pool.X.index.equals(pool.y.index) or not pool.X.index.equals(pool.metadata.index)
            or not pool.X.option_type.isin([0, 1]).all()):
        raise ValueError('Invalid archived weekly replay inputs')
    created, closed = pool.metadata.created_at_utc, pool.metadata.closed_at_utc
    frames = []
    for origin in pd.date_range(start, end, freq='7D', inclusive='left'):
        stop = min(origin+pd.Timedelta(days=7), end); text = origin.isoformat()
        q = pool.X.index[created.ge(origin) & created.lt(stop)]
        if not len(q): continue
        state = archived_states.get(text)
        if state is None or not state['h0_supported'] or not state['is_validation']:
            raise ValueError('Missing or unsupported archived validation week: '+text)
        h = history_ids(pool, origin, protocol)
        if (state['origin'] != text or pd.Timestamp(state['stop']) != stop or state['history_orders'] != len(h)
                or state['query_orders'] != len(q) or len(h) < 100
                or state['earliest_training_create'] != created.loc[h].min().isoformat()
                or state['latest_training_close'] != closed.loc[h].max().isoformat()):
            raise ValueError('Archived training history or weekly grid mismatch')
        x = pool.X.loc[q]; hp = replay_h0(x, state['h0']); bp = x[BASE_FEATURE].to_numpy()
        threshold = pd.Series(index=q, dtype=float); source = pd.Series(index=q, dtype=object)
        for month in sorted(created.loc[q].dt.strftime('%Y-%m').unique()):
            mo = pd.Timestamp(month+'-01', tz='UTC'); mq = q[created.loc[q].dt.strftime('%Y-%m').eq(month)]
            tv, sv = tail_context(pool, history_ids(pool, mo, protocol, window=False), mq)
            threshold.loc[mq] = tv; source.loc[mq] = sv
        common = pd.DataFrame(dict(day=created.loc[q].dt.floor('D'), month=created.loc[q].dt.strftime('%Y-%m'),
            side=x.option_type.astype(int), tenor=x.product_duration_seconds.astype(int), quantity=x.quantity_btc,
            tail_threshold=threshold, tail_source=source, origin=text, h0_prediction=hp, base_prediction=bp), index=q)
        for name, pair in FIXED.items():
            a = np.where(x.option_type.eq(0), pair[0], pair[1]); pred = a*hp+(1-a)*bp
            if not np.isfinite(pred).all() or (pred < 0).any(): raise ValueError('Invalid fixed convex prediction')
            frames.append(common.assign(model=name, prediction=pred, h0_weight=a))
    if not frames: raise ValueError('Empty validation replay')
    return pd.concat(frames)
