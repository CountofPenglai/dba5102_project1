"""Five-percentage-point fixed mixtures of the previously verified weekly H0."""
import argparse
import json
import resource
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT
from pp_data.features import BASE_FEATURE
from .base_multiplier_study import AccessGuard
from .calibration_models import hurdle_design
from .data import load_pool
from .frozen_candidate_study import canonical_identity, write_tables
from .scoped_data import development_snapshot
from .runs import run_destination, snapshot
from .study_storage import atomic_json, file_digest
from .two_stage_study import tail_context
from .h0_time_decay_study import PROTOCOL, PROTOCOL_PATH, history_ids

CALL_GRID = tuple(i/100 for i in range(25, 66, 5))
PUT_GRID = tuple(i/100 for i in range(10, 41, 5))
FINE = {f'blend_c{round(c*100):03d}_p{round(p*100):03d}': (c, p) for c in CALL_GRID for p in PUT_GRID}
FIXED = {'official_raw': (0., 0.), 'h0': (1., 1.), 'half_blend': (.5, .5), 'blend_c000_p025': (0., .25), **FINE}
MODELS = tuple(FIXED)
PREVIOUS_BEST = 'blend_c050_p025'
PAIRS = [(c, r) for c in FINE for r in ('official_raw', 'h0', PREVIOUS_BEST) if c != r]
PAIRS += [('h0', 'official_raw'), ('half_blend', 'official_raw'), ('blend_c000_p025', 'official_raw')]
CONFIG = PROJECT/'config/h0_blend_fine_v1.json'
PLAN = PROJECT/'docs/plans/2026-10-15-h0-blend-fine.md'
SETTINGS = dict(version='h0_blend_fine_v1', protocol='config/validation_protocol_v3.json', models=list(MODELS),
    call_grid=list(CALL_GRID), put_grid=list(PUT_GRID), fine_models=list(FINE),
    fixed_ratios={k: list(v) for k, v in FIXED.items()}, pairs=[list(p) for p in PAIRS],
    previous_best=PREVIOUS_BEST, reference_run='outputs/model_studies/h0_blend_ratios_20261015',
    h0_mode='replay_verified_weekly_parameters', h0_history_days=30, update_days=7,
    ratio_mode='fixed_all_validation_months', new_h0_fits=0, new_dynamic_weights=False,
    primary_metric='month_equal_rmse', pooled_rmse_required=True, sequential_validation_tuning=True,
    monthly_oracle_selection=False, adaptive_grid_expansion=False,
    bootstrap_blocks=[3, 7, 14], bootstrap_repeats=2000, simultaneous_comparisons=len(PAIRS))


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


def score(pool, predicted, end):
    ref = predicted.loc[predicted.model.eq('official_raw')]
    if not ref.index.is_unique: raise ValueError('Duplicate query orders')
    known = pool.metadata.closed_at_utc.reindex(ref.index).lt(pd.Timestamp(end)); ids = ref.index[known]
    f = predicted.loc[predicted.index.isin(ids)].copy(); f['actual'] = pool.y.reindex(f.index).to_numpy()
    if not np.isfinite(f.actual).all() or f.actual.lt(0).any(): raise ValueError('Invalid scoring labels')
    f['tail_group'] = np.where(f.tail_threshold.isna(), 'unavailable', np.where(f.actual > f.tail_threshold, 'tail', 'body'))
    for name in MODELS:
        if not f.loc[f.model.eq(name)].index.equals(ids): raise ValueError('Unequal fixed-grid cohorts')
    return f, dict(query_orders=len(ref), scored_orders=len(ids), labels_unavailable=int((~known).sum()))


def run(run_id):
    from .fixed_blend_statistics import tables
    if json.loads(CONFIG.read_text()) != SETTINGS or json.loads(PROTOCOL_PATH.read_text()) != PROTOCOL:
        raise ValueError('Fixed configuration changed')
    guard = AccessGuard(); guard.allow_pressure = True; sys.addaudithook(guard.check)
    root = run_destination(PROJECT/'outputs/model_studies', run_id)
    root.mkdir(parents=True); (root/'provenance').mkdir(); began = perf_counter()
    canonical = canonical_identity(include_test=True); dev, _ = development_snapshot()
    reference = PROJECT/SETTINGS['reference_run']; rm = json.loads((reference/'provenance/manifest.json').read_text())
    if rm['status'] != 'complete' or json.loads((reference/'provenance/preflight.json').read_text())['canonical'] != canonical:
        raise ValueError('Reference/canonical mismatch')
    for name, digest in rm['artifacts'].items():
        if file_digest(reference/name) != digest: raise ValueError('Reference changed')
    protected = {str(p.relative_to(PROJECT)): file_digest(p) for p in reference.rglob('*') if p.is_file()}
    paths = {CONFIG, PROTOCOL_PATH, PLAN, PROJECT/'scripts/verify_h0_blend_fine.py',
        PROJECT/'scripts/verify_h0_hourly_amount.py', PROJECT/'scripts/verify_frozen_candidates.py',
        PROJECT/'tests/test_h0_blend_fine.py', PROJECT/'tests/test_h0_blend_ratios.py', PROJECT/'tests/test_h0_state_fallback.py'}
    for folder in ('pp_models', 'pp_data'): paths.update((PROJECT/folder).glob('*.py'))
    sources = {str(p.relative_to(PROJECT)): file_digest(p) for p in sorted(paths)}
    with ZipFile(root/'provenance/source.zip', 'w', ZIP_DEFLATED) as archive:
        for name in sources: archive.write(PROJECT/name, name)
    atomic_json(root/'provenance/config.json', SETTINGS); atomic_json(root/'provenance/protocol.json', PROTOCOL)
    atomic_json(root/'provenance/preflight.json', dict(sources=sources, protected=protected, canonical=canonical, development_data=dev))
    try:
        pool = load_pool('test', {'cohort': 'benchmark', 'use_base': True})
        states = json.loads((reference/'provenance/parameters.json').read_text())
        predicted = predict_period(pool, states); scored, coverage = score(pool, predicted, PROTOCOL['validation_end'])
        result = tables(scored, PROTOCOL['validation_start'], PROTOCOL['validation_end'], FIXED, PAIRS)
        prior = pd.read_csv(reference/'metrics/validation/summary.csv', dtype={'value': str})
        overlaps = sorted(set(prior.model)&set(FIXED))
        for name in overlaps:
            pd.testing.assert_frame_equal(result['summary'].loc[result['summary'].model.eq(name)].reset_index(drop=True),
                prior.loc[prior.model.eq(name)].reset_index(drop=True), check_exact=False, atol=1e-7, rtol=1e-10)
        write_tables(root, 'validation', result)
        saved = {k: {a: b for a, b in v.items() if a != 'weights'} for k, v in states.items() if v['is_validation']}
        atomic_json(root/'provenance/parameters.json', saved)
        if canonical_identity(include_test=True) != canonical or development_snapshot()[0] != dev: raise ValueError('Canonical changed')
        for name, digest in {**sources, **protected}.items():
            if file_digest(PROJECT/name) != digest: raise ValueError('Source/reference changed: '+name)
        artifacts = {str(p.relative_to(root)): file_digest(p) for p in root.rglob('*') if p.is_file()}
        atomic_json(root/'provenance/manifest.json', dict(status='complete', producer=SETTINGS['version'], artifacts=artifacts,
            coverage=coverage, weekly_h0_states_replayed=len(saved), new_model_fit_calls=0, fine_grid_points=len(FINE),
            models=len(FIXED), comparisons=len(PAIRS), overlapping_reference_models=overlaps, references_match=True,
            monthly_oracle_selection=False, official_validation_open_attempts=guard.official_attempts,
            premature_test_attempts=guard.premature_test_attempts, elapsed_seconds=perf_counter()-began,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, environment=snapshot()['environment']))
    except Exception as e:
        atomic_json(root/'provenance/failure.json', dict(error=repr(e))); raise
    return root


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    print(run(parser.parse_args().run_id))


if __name__ == '__main__': main()
