"""April training only, May--July weekly forward validation; no residual overlays."""
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
from .data import load_pool
from .frozen_candidate_study import canonical_identity, write_tables
from .scoped_data import development_snapshot
from .runs import run_destination, snapshot
from .study_storage import atomic_json, file_digest
from .two_stage_study import probability_frame, tail_context
from .weekly_residual_study import h0_state
from .h0_time_decay import TimeDecayH0

CONFIG = PROJECT/'config/h0_time_decay_v1.json'
PROTOCOL_PATH = PROJECT/'config/validation_protocol_v3.json'
PLAN = PROJECT/'docs/plans/2026-10-15-h0-time-decay.md'
MODELS = ('official_raw', 'h0', 'h0_decay7', 'h0_decay14')
HALF_LIVES = {'h0': None, 'h0_decay7': 7, 'h0_decay14': 14}
PROTOCOL = dict(version='rolling_validation_v3', timezone='UTC', training_start='2026-04-01T00:00:00Z',
    validation_start='2026-05-01T00:00:00Z', validation_end='2026-07-31T17:11:48Z',
    report_months=['2026-05', '2026-06', '2026-07'], history_days=30, update_days=7,
    label_rule='created in [max(training_start,D-30d),D), closed strictly before D', weekly_label_feedback=True,
    tail_history='monthly q90 from April onward, created and closed strictly before month start',
    old_july_status='rolling_validation; canonical test storage retained without repartitioning',
    official_validation='forbidden', selection='fixed exploratory candidates, no parameter search or automatic adoption')
SETTINGS = dict(version='h0_time_decay_v1', validation_protocol='config/validation_protocol_v3.json', models=list(MODELS),
    half_life_days=HALF_LIVES, primary='h0_decay7', age_basis='creation_time',
    probability_weight_normalization='sum_equals_direction_order_count', scaler='unweighted_original_h0',
    logistic_C=.1, residual_correction=False, trend_overlay=False, parameter_search=False,
    bootstrap_blocks=[3, 7, 14], bootstrap_repeats=2000)


def history_ids(pool, origin, protocol, window=True):
    created = pool.metadata.created_at_utc; closed = pool.metadata.closed_at_utc
    earliest = pd.Timestamp(protocol['training_start'])
    if window: earliest = max(earliest, origin-pd.Timedelta(days=protocol['history_days']))
    return pool.X.index[created.ge(earliest) & created.lt(origin) & closed.lt(origin)]


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


def score(pool, predicted, end):
    ref = predicted.loc[predicted.model.eq('official_raw')]
    if not ref.index.is_unique: raise ValueError('Duplicate prediction IDs')
    known = pool.metadata.closed_at_utc.reindex(ref.index).lt(pd.Timestamp(end)); ids = ref.index[known]
    frame = predicted.loc[predicted.index.isin(ids)].copy(); frame['actual'] = pool.y.reindex(frame.index).to_numpy()
    if not np.isfinite(frame.actual).all() or frame.actual.lt(0).any(): raise ValueError('Invalid scoring targets')
    frame['tail_group'] = np.where(frame.tail_threshold.isna(), 'unavailable', np.where(frame.actual > frame.tail_threshold, 'tail', 'body'))
    for name in MODELS:
        if not frame.loc[frame.model.eq(name)].index.equals(ids): raise ValueError('Unequal cohorts')
    return frame, dict(query_orders=len(ref), scored_orders=len(ids), labels_unavailable=int((~known).sum()))


def run(run_id):
    from .h0_time_decay_statistics import tables
    if json.loads(CONFIG.read_text()) != SETTINGS or json.loads(PROTOCOL_PATH.read_text()) != PROTOCOL:
        raise ValueError('Fixed study settings changed; version a new protocol')
    guard = AccessGuard()
    # User explicitly redesignates old July as validation. This flag only opens canonical test storage.
    guard.allow_pressure = True; sys.addaudithook(guard.check)
    root = run_destination(PROJECT/'outputs/model_studies', run_id)
    root.mkdir(parents=True); (root/'provenance').mkdir(); began = perf_counter()
    canonical = canonical_identity(include_test=True); dev, _ = development_snapshot()
    paths = {CONFIG, PROTOCOL_PATH, PLAN, PROJECT/'scripts/verify_h0_time_decay.py', PROJECT/'tests/test_h0_time_decay.py'}
    for folder in ('pp_models', 'pp_data'): paths.update((PROJECT/folder).glob('*.py'))
    sources = {str(p.relative_to(PROJECT)): file_digest(p) for p in sorted(paths)}
    with ZipFile(root/'provenance/source.zip', 'w', ZIP_DEFLATED) as z:
        for name in sources: z.write(PROJECT/name, name)
    atomic_json(root/'provenance/config.json', SETTINGS); atomic_json(root/'provenance/protocol.json', PROTOCOL)
    atomic_json(root/'provenance/preflight.json', dict(sources=sources, canonical=canonical, development_data=dev))
    try:
        pool = load_pool('test', {'cohort': 'benchmark', 'use_base': True})
        predicted, states, diag = predict_period(pool)
        scored, coverage = score(pool, predicted, PROTOCOL['validation_end'])
        result = tables(scored, PROTOCOL['validation_start'], PROTOCOL['validation_end'])
        result['weight_diagnostics'] = diag
        write_tables(root, 'validation', result); atomic_json(root/'provenance/parameters.json', states)
        if canonical_identity(include_test=True) != canonical or development_snapshot()[0] != dev:
            raise ValueError('Canonical data changed during study')
        for name, digest in sources.items():
            if file_digest(PROJECT/name) != digest: raise ValueError('Source changed: '+name)
        artifacts = {str(p.relative_to(root)): file_digest(p) for p in root.rglob('*') if p.is_file()}
        atomic_json(root/'provenance/manifest.json', dict(status='complete', producer=SETTINGS['version'], artifacts=artifacts,
            coverage=coverage, update_origins=len(states), model_fit_objects=len(states)*len(HALF_LIVES),
            within_week_label_updates=0, official_validation_open_attempts=guard.official_attempts,
            premature_test_attempts=guard.premature_test_attempts, old_july_role='rolling_validation_authorized',
            elapsed_seconds=perf_counter()-began, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            environment=snapshot()['environment']))
    except Exception as e:
        atomic_json(root/'provenance/failure.json', dict(error=repr(e))); raise
    return root


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    print(run(parser.parse_args().run_id))


if __name__ == '__main__': main()
