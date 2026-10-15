"""Fixed original probabilities, three hourly amount responses, weekly forward validation."""
import argparse
import json
import resource
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT, DEFAULT_DATA
from pp_data.features import BASE_FEATURE
from .base_multiplier_study import AccessGuard
from .calibration_models import CalibrationModel
from .data import load_pool
from .frozen_candidate_study import canonical_identity, write_tables
from .scoped_data import development_snapshot
from .put_final_io import read_test_market_prefix
from .runs import run_destination, snapshot
from .study_storage import atomic_json, file_digest
from .two_stage_study import probability_frame, tail_context
from .weekly_residual_study import h0_state
from .h0_time_decay_study import PROTOCOL, PROTOCOL_PATH, history_ids
from .h0_hourly_amount import HourlyAmount, market_features, amount_features, CANDIDATES

CONFIG = PROJECT/'config/h0_hourly_amount_v1.json'
PLAN = PROJECT/'docs/plans/2026-10-15-h0-hourly-amount.md'
MODELS = ('official_raw', 'h0')+CANDIDATES
SETTINGS = dict(version='h0_hourly_amount_v1', protocol='config/validation_protocol_v3.json', models=list(MODELS),
    primary='amount_joint', probability='unchanged_unweighted_h0', time_decay=False, contract_residual=False,
    windows_minutes=[60, 240], ridge_strength=100, continuous_bounds=[.9, 1.1], gate_bounds=[1., 1.1],
    minimum_positive_orders=40, minimum_positive_days=5, gate_level_quantiles=[.75, .95], gate_speed_ratios=[1.25, 2.],
    gate_minimum_positive_orders=15, gate_minimum_positive_days=3, standardized_clip=3, parameter_search=False,
    market_prefix_sha256='cbf5d491d167ac377beb844dc9794e4f5e9986f72859035426f7789e67924b50',
    reference_run='outputs/model_studies/h0_time_decay_20261015', bootstrap_blocks=[3, 7, 14], bootstrap_repeats=2000)


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
        h0 = CalibrationModel('hurdle_base').fit(probability_frame(hx), pool.y.loc[h], created.loc[h])
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


def score(pool, predicted, end):
    base = predicted.loc[predicted.model.eq('official_raw')]
    if not base.index.is_unique: raise ValueError('Duplicate queries')
    known = pool.metadata.closed_at_utc.reindex(base.index).lt(pd.Timestamp(end)); ids = base.index[known]
    frame = predicted.loc[predicted.index.isin(ids)].copy(); frame['actual'] = pool.y.reindex(frame.index).to_numpy()
    if not np.isfinite(frame.actual).all() or frame.actual.lt(0).any(): raise ValueError('Invalid scoring target')
    frame['tail_group'] = np.where(frame.tail_threshold.isna(), 'unavailable', np.where(frame.actual > frame.tail_threshold, 'tail', 'body'))
    h0 = frame.loc[frame.model.eq('h0')]
    for name in MODELS:
        p = frame.loc[frame.model.eq(name)]
        if not p.index.equals(ids): raise ValueError('Unequal cohorts')
        if name != 'official_raw':
            np.testing.assert_array_equal(p.probability, h0.probability)
            np.testing.assert_array_equal(p.prediction, p.probability*p.severity)
    return frame, dict(query_orders=len(base), scored_orders=len(ids), labels_unavailable=int((~known).sum()))


def run(run_id):
    from .h0_hourly_amount_statistics import tables
    if json.loads(CONFIG.read_text()) != SETTINGS or json.loads(PROTOCOL_PATH.read_text()) != PROTOCOL:
        raise ValueError('Fixed settings changed; version another study')
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
    paths = {CONFIG, PROTOCOL_PATH, PLAN, PROJECT/'scripts/verify_h0_hourly_amount.py',
        PROJECT/'scripts/verify_frozen_candidates.py', PROJECT/'tests/test_h0_hourly_amount.py',
        PROJECT/'tests/test_h0_time_decay.py'}
    for folder in ('pp_models', 'pp_data'): paths.update((PROJECT/folder).glob('*.py'))
    sources = {str(p.relative_to(PROJECT)): file_digest(p) for p in sorted(paths)}
    with ZipFile(root/'provenance/source.zip', 'w', ZIP_DEFLATED) as z:
        for name in sources: z.write(PROJECT/name, name)
    atomic_json(root/'provenance/config.json', SETTINGS); atomic_json(root/'provenance/protocol.json', PROTOCOL)
    atomic_json(root/'provenance/preflight.json', dict(sources=sources, protected=protected, canonical=canonical, development_data=dev))
    try:
        pool = load_pool('test', {'cohort': 'benchmark', 'use_base': True})
        raw, market_info = read_test_market_prefix(DEFAULT_DATA/'btc_market_history.csv')
        if market_info['prefix_sha256'] != SETTINGS['market_prefix_sha256']: raise ValueError('Market prefix changed')
        f = market_features(pool.metadata.created_at_utc, raw)
        if not f.loc[f.available, 'feature_last_close_utc'].equals(pool.metadata.loc[f.available, 'last_candle_close_utc'].rename('feature_last_close_utc')):
            raise ValueError('Hourly and canonical quote timestamps disagree')
        for w in (60, 240):
            reconstructed = np.sqrt(w*(f[f'up_vol_{w}m']**2+f[f'down_vol_{w}m']**2))
            np.testing.assert_allclose(reconstructed.loc[f.available], pool.X.loc[f.available, f'realized_vol_{w}m'], atol=1e-12, rtol=1e-9)
        predicted, states = predict_period(pool, f); scored, coverage = score(pool, predicted, PROTOCOL['validation_end'])
        result = tables(scored, PROTOCOL['validation_start'], PROTOCOL['validation_end'])
        old = pd.read_csv(reference/'metrics/validation/summary.csv', dtype={'value': str})
        for name in ('official_raw', 'h0'):
            a = result['summary'].loc[result['summary'].model.eq(name)].reset_index(drop=True)
            b = old.loc[old.model.eq(name)].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-7, rtol=1e-10)
        write_tables(root, 'validation', result); atomic_json(root/'provenance/parameters.json', states)
        if canonical_identity(include_test=True) != canonical or development_snapshot()[0] != dev: raise ValueError('Canonical changed')
        for name, digest in {**sources, **protected}.items():
            if file_digest(PROJECT/name) != digest: raise ValueError('Source/reference changed: '+name)
        artifacts = {str(p.relative_to(root)): file_digest(p) for p in root.rglob('*') if p.is_file()}
        atomic_json(root/'provenance/manifest.json', dict(status='complete', producer=SETTINGS['version'], artifacts=artifacts,
            coverage=coverage, market=market_info, update_origins=len(states), h0_fit_objects=len(states),
            amount_fit_objects=len(states)*len(CANDIDATES), supported_direction_amount_fits=sum(
                int(p['supported']) for s in states.values() for m in s['models'].values() for p in m.values()),
            probability_unchanged=True, references_match=True, within_week_parameter_updates=0,
            missing_hourly_signal_quotes=int((~scored.loc[scored.model.eq('h0'), 'hourly_available']).sum()),
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
