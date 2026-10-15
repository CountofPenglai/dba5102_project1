"""Fixed directional H0/Base ratios and strictly prequential weekly grid choice."""
import argparse
import json
import resource
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT
from pp_data.features import BASE_FEATURE
from .base_multiplier_study import AccessGuard
from .calibration_models import CalibrationModel
from .data import load_pool
from .frozen_candidate_study import canonical_identity, write_tables
from .scoped_data import development_snapshot
from .runs import run_destination, snapshot
from .study_storage import atomic_json, file_digest
from .two_stage_study import probability_frame, tail_context
from .weekly_residual_study import h0_state
from .h0_time_decay_study import PROTOCOL, PROTOCOL_PATH, history_ids
from .h0_blend_ratios import GRID, FIXED, DYNAMIC, MODELS, CANDIDATES, grid_weight

CONFIG = PROJECT/'config/h0_blend_ratios_v1.json'
PLAN = PROJECT/'docs/plans/2026-10-15-h0-blend-ratios.md'
SETTINGS = dict(version='h0_blend_ratios_v1', protocol='config/validation_protocol_v3.json', models=list(MODELS),
    fixed_grid=list(GRID), fixed_ratios={k: list(v) for k, v in FIXED.items()}, dynamic_windows=DYNAMIC,
    primary_dynamic='dynamic14', warmup_start='2026-04-03T00:00:00Z', history_days=30, update_days=7,
    minimum_oos_orders=100, minimum_oos_days=10, weight_shrinkage_orders=100, weight_prior=.5,
    insufficient_history_alpha=0., ties='smaller_h0_weight', weekly_label_feedback=True,
    time_decay=False, contract_residual=False, hourly_amount=False, state_gate=False,
    fixed_grid_exploration=True, monthly_oracle_selection=False, adaptive_search=False,
    reference_run='outputs/model_studies/h0_state_fallback_20261015',
    bootstrap_blocks=[3, 7, 14], bootstrap_repeats=2000, simultaneous_comparisons=51)


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
        model = CalibrationModel('hurdle_base').fit(probability_frame(hx), pool.y.loc[h], created.loc[h])
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


def score(pool, predicted, end):
    ref = predicted.loc[predicted.model.eq('official_raw')]
    if not ref.index.is_unique: raise ValueError('Duplicate query orders')
    known = pool.metadata.closed_at_utc.reindex(ref.index).lt(pd.Timestamp(end)); ids = ref.index[known]
    f = predicted.loc[predicted.index.isin(ids)].copy(); f['actual'] = pool.y.reindex(f.index).to_numpy()
    if not np.isfinite(f.actual).all() or f.actual.lt(0).any(): raise ValueError('Invalid scoring labels')
    f['tail_group'] = np.where(f.tail_threshold.isna(), 'unavailable', np.where(f.actual > f.tail_threshold, 'tail', 'body'))
    for name in MODELS:
        if not f.loc[f.model.eq(name)].index.equals(ids): raise ValueError('Unequal candidate cohorts')
    return f, dict(query_orders=len(ref), scored_orders=len(ids), labels_unavailable=int((~known).sum()))


def run(run_id):
    from .h0_blend_ratios_statistics import tables
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
    paths = {CONFIG, PROTOCOL_PATH, PLAN, PROJECT/'scripts/verify_h0_blend_ratios.py',
        PROJECT/'scripts/verify_h0_hourly_amount.py', PROJECT/'scripts/verify_frozen_candidates.py',
        PROJECT/'tests/test_h0_blend_ratios.py', PROJECT/'tests/test_h0_state_fallback.py', PROJECT/'tests/test_h0_time_decay.py'}
    for folder in ('pp_models', 'pp_data'): paths.update((PROJECT/folder).glob('*.py'))
    sources = {str(p.relative_to(PROJECT)): file_digest(p) for p in sorted(paths)}
    with ZipFile(root/'provenance/source.zip', 'w', ZIP_DEFLATED) as archive:
        for name in sources: archive.write(PROJECT/name, name)
    atomic_json(root/'provenance/config.json', SETTINGS); atomic_json(root/'provenance/protocol.json', PROTOCOL)
    atomic_json(root/'provenance/preflight.json', dict(sources=sources, protected=protected, canonical=canonical, development_data=dev))
    try:
        pool = load_pool('test', {'cohort': 'benchmark', 'use_base': True})
        predicted, states = predict_period(pool); scored, coverage = score(pool, predicted, PROTOCOL['validation_end'])
        result = tables(scored, PROTOCOL['validation_start'], PROTOCOL['validation_end'])
        prior = pd.read_csv(reference/'metrics/validation/summary.csv', dtype={'value': str})
        for name in ('official_raw', 'h0', 'half_blend'):
            pd.testing.assert_frame_equal(result['summary'].loc[result['summary'].model.eq(name)].reset_index(drop=True),
                prior.loc[prior.model.eq(name)].reset_index(drop=True), check_exact=False, atol=1e-7, rtol=1e-10)
        prior_states = json.loads((reference/'provenance/parameters.json').read_text())
        for origin, state in states.items():
            if state.get('h0') != prior_states[origin].get('h0'): raise ValueError('H0 parameters changed')
        write_tables(root, 'validation', result); atomic_json(root/'provenance/parameters.json', states)
        if canonical_identity(include_test=True) != canonical or development_snapshot()[0] != dev: raise ValueError('Canonical changed')
        for name, digest in {**sources, **protected}.items():
            if file_digest(PROJECT/name) != digest: raise ValueError('Source/reference changed: '+name)
        artifacts = {str(p.relative_to(root)): file_digest(p) for p in root.rglob('*') if p.is_file()}
        atomic_json(root/'provenance/manifest.json', dict(status='complete', producer=SETTINGS['version'], artifacts=artifacts,
            coverage=coverage, evaluation_updates=sum(s['is_validation'] for s in states.values()),
            h0_fit_objects=sum(s['h0_supported'] for s in states.values()),
            warmup_fits=sum(s['h0_supported'] and not s['is_validation'] for s in states.values()),
            supported_grid_direction_fits=sum(int(w['supported']) for s in states.values() for pp in s.get('weights', {}).values() for w in pp.values()),
            grid_direction_fits=sum(len(pp) for s in states.values() for pp in s.get('weights', {}).values()),
            references_match=True, within_week_label_updates=0, monthly_oracle_selection=False,
            official_validation_open_attempts=guard.official_attempts, premature_test_attempts=guard.premature_test_attempts,
            elapsed_seconds=perf_counter()-began, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            environment=snapshot()['environment']))
    except Exception as e:
        atomic_json(root/'provenance/failure.json', dict(error=repr(e))); raise
    return root


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    print(run(parser.parse_args().run_id))


if __name__ == '__main__': main()
