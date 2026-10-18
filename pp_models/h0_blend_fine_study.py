"""Report recipe kernels; execution is owned by report_unified_study."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT
from pp_data.features import BASE_FEATURE
from .pricing_inputs import hurdle_design
from .validation import PROTOCOL, prediction_context
from pp_data.splitting import RollingSplitter
from .training import MIN_TRAINING_ORDERS
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
    splitter = RollingSplitter(pool, protocol)
    if not pool.X.option_type.isin([0, 1]).all():
        raise ValueError('Invalid replay option direction')
    frames = []
    for window in splitter.windows():
        origin, stop = window.origin, window.stop
        q, h = window.query_ids, window.history_ids
        text = origin.isoformat()
        state = archived_states.get(text)
        if state is None or not state['h0_supported'] or not state['is_validation']:
            raise ValueError('Missing or unsupported archived validation week: '+text)
        expected = splitter.describe(window)
        if len(h) < MIN_TRAINING_ORDERS or any(state[k] != value for k, value in expected.items()):
            raise ValueError('Archived training history or weekly grid mismatch')
        x = pool.X.loc[q]; hp = replay_h0(x, state['h0']); bp = x[BASE_FEATURE].to_numpy()
        common = prediction_context(pool, window, splitter).assign(h0_prediction=hp, base_prediction=bp)
        for name, pair in FIXED.items():
            a = np.where(x.option_type.eq(0), pair[0], pair[1]); pred = a*hp+(1-a)*bp
            if not np.isfinite(pred).all() or (pred < 0).any(): raise ValueError('Invalid fixed convex prediction')
            frames.append(common.assign(model=name, prediction=pred, h0_weight=a))
    if not frames: raise ValueError('Empty validation replay')
    return pd.concat(frames)
