"""Rerun the report's existing recipes on one weekly v3 cohort, including announcements."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pp_data.config import PROJECT
from pp_data.features import BASE_FEATURE
from . import h0_time_decay_study as decay
from . import h0_state_fallback_study as state
from . import h0_hourly_amount_study as hourly
from . import h0_blend_ratios_study as coarse
from . import h0_blend_fine_study as fine
from .runtime import AccessGuard
from pp_data.io import load_dataset
from .fixed_blend_statistics import tables
from .runtime import canonical_identity, write_tables
from .h0_hourly_amount import market_features
from .h0_state_fallback import market_states
from .macro_multiplier import MacroMultiplier
from .macro_multiplier_features import pool_features
from .runtime import read_market_prefix
from .runtime import run_destination, environment_snapshot
from .runtime import atomic_json, file_digest

PROTOCOL = decay.PROTOCOL
COMMON = ['day', 'month', 'side', 'tenor', 'quantity', 'tail_threshold', 'tail_source', 'origin']
CORE = COMMON + ['model', 'prediction']
STRENGTHS = {'announcement_l1': 1., 'announcement_l10': 10., 'announcement_l100': 100.}
FIXED = {**coarse.FIXED, **fine.FINE}
EXTRAS = ('h0_decay7', 'h0_decay14', *hourly.CANDIDATES,
          'state_fallback', 'history_blend', 'state_history_blend', *coarse.DYNAMIC, *STRENGTHS)
CATALOG = {**FIXED, **{m: (np.nan, np.nan) for m in EXTRAS}}
PAIRS = [('h0', 'official_raw')] + [(c, r) for c in CATALOG if c not in ('official_raw', 'h0')
                                  for r in ('official_raw', 'h0')]
PAIRS += [(c, fine.PREVIOUS_BEST) for c in fine.FINE if c != fine.PREVIOUS_BEST]
SETTINGS = dict(version='report_unified_v2', protocol='config/validation_protocol_v3.json',
    models=list(CATALOG), announcement_strengths=STRENGTHS,
    announcement_history='same_30_day_created_and_settled_pool_as_H0',
    announcement_support='unchanged_shared_5_events_30_orders_2_months_direction_8_60_3',
    new_official_validation='forbidden', adaptive_grid_expansion=False, monthly_oracle_selection=False,
    fresh_h0_fits=True, bootstrap_blocks=[3, 7, 14], bootstrap_repeats=2000,
    pairs=[list(p) for p in PAIRS], simultaneous_comparisons=len(PAIRS))


def merge_predictions(frames):
    """Every recipe must forecast the same orders with the same as-of tail/context."""
    if not frames: raise ValueError('No predictions')
    ref = frames[0].loc[frames[0].model.eq('official_raw'), CORE]
    if ref.empty or not ref.index.is_unique: raise ValueError('Missing/duplicate Base reference')
    combined = {}
    for frame in frames:
        if not set(CORE).issubset(frame): raise ValueError('Missing prediction columns')
        for name, part in frame.groupby('model', sort=False):
            if not part.index.is_unique or set(part.index) != set(ref.index):
                raise ValueError('Different prediction cohort: '+name)
            part = part.loc[ref.index, CORE].copy()
            try:
                pd.testing.assert_frame_equal(part[COMMON], ref[COMMON], check_dtype=False,
                                              check_exact=False, rtol=1e-12, atol=1e-9)
                if name in combined:
                    np.testing.assert_allclose(part.prediction, combined[name].prediction, rtol=1e-12, atol=1e-9)
            except AssertionError as error:
                raise ValueError('Prediction context/reference disagreement: '+name) from error
            if not np.isfinite(part.prediction).all() or part.prediction.lt(0).any():
                raise ValueError('Invalid prediction: '+name)
            if name not in combined: combined[name] = part
    return pd.concat(combined.values())


def score_predictions(pool, predicted, end):
    # Revalidate cohort equality before adding any evaluation targets.
    predicted = merge_predictions([predicted])
    ref = predicted.loc[predicted.model.eq('official_raw')]
    known = pool.metadata.closed_at_utc.reindex(ref.index).lt(pd.Timestamp(end))
    ids = ref.index[known]
    f = predicted.loc[predicted.index.isin(ids)].copy()
    f['actual'] = pool.y.reindex(f.index).to_numpy()
    if not np.isfinite(f.actual).all() or f.actual.lt(0).any(): raise ValueError('Invalid evaluation targets')
    f['tail_group'] = np.where(f.tail_threshold.isna(), 'unavailable',
                             np.where(f.actual > f.tail_threshold, 'tail', 'body'))
    return f, dict(query_orders=len(ref), scored_orders=len(ids), labels_unavailable=int((~known).sum()))


def predict_announcements(pool, calendar, reference, protocol=None):
    """Keep the original announcement recipe, change only its lawful training clock/pool."""
    protocol = PROTOCOL if protocol is None else protocol
    start, end = pd.Timestamp(protocol['validation_start']), pd.Timestamp(protocol['validation_end'])
    created, closed = pool.metadata.created_at_utc, pool.metadata.closed_at_utc
    qids = pool.X.index[created.ge(start) & created.lt(end)]
    if (protocol['update_days'] != 7 or protocol['history_days'] != 30
            or start.tzinfo is None or end.tzinfo is None or not pd.Timestamp(protocol['training_start']) < start < end
            or not pool.X.index.is_unique or not pool.X.index.equals(pool.y.index)
            or not pool.X.index.equals(pool.metadata.index) or not reference.index.is_unique
            or not reference.index.equals(qids) or not reference.model.eq('official_raw').all()):
        raise ValueError('Invalid common announcement cohort/protocol')
    features = pool_features(pool, calendar)
    frames, fits, cohorts = [], [], []
    for origin in pd.date_range(start, end, freq='7D', inclusive='left'):
        stop = min(origin+pd.Timedelta(days=7), end)
        q = qids[created.loc[qids].ge(origin) & created.loc[qids].lt(stop)]
        if not len(q): continue
        if not reference.loc[q, 'origin'].eq(origin.isoformat()).all():
            raise ValueError('Announcement reference uses a different update clock')
        h = decay.history_ids(pool, origin, protocol)
        if len(h) < 100: raise ValueError('Insufficient common historical pool')
        base = pool.X.loc[q, BASE_FEATURE]
        for name, strength in STRENGTHS.items():
            model = MacroMultiplier(strength).fit(features.loc[h], pool.X.loc[h, BASE_FEATURE], pool.y.loc[h], origin)
            diagnostics = model.predict_diagnostics(features.loc[q])
            common = reference.loc[q, COMMON].copy()
            frames.append(common.assign(model=name, prediction=base*diagnostics.multiplier,
                **{c: diagnostics[c] for c in diagnostics}, event_id=features.loc[q, 'event_id'],
                calendar_status=features.loc[q, 'calendar_status']))
            fits.append(dict(model=name, **model.diagnostics()))
        cohorts.append(dict(origin=origin.isoformat(), stop=stop.isoformat(), history_orders=len(h), query_orders=len(q),
            earliest_training_create=created.loc[h].min().isoformat(), latest_training_close=closed.loc[h].max().isoformat(),
            history_ids_sha256=hashlib.sha256('\n'.join(h.astype(str)).encode()).hexdigest()))
    if not frames: raise ValueError('Empty announcement evaluation period')
    return pd.concat(frames), fits, pd.DataFrame(cohorts)


def predict_all(pool, market, calendar, protocol=None):
    protocol = PROTOCOL if protocol is None else protocol
    print('Refitting H0 and fixed time-decay recipes', flush=True)
    a, ds, weights = decay.predict_period(pool, protocol)
    print('Refitting state fallback and historical-error mixtures', flush=True)
    b, ss = state.predict_period(pool, market_states(market), protocol)
    print('Refitting hourly amount responses', flush=True)
    c, hs = hourly.predict_period(pool, market_features(pool.metadata.created_at_utc, market), protocol)
    print('Refitting coarse grid / causal dynamic weights; replaying fine grid from fresh parameters', flush=True)
    e, cs = coarse.predict_period(pool, protocol)
    f = fine.predict_period(pool, cs, protocol)
    print('Refitting announcements with the same weekly 30-day history', flush=True)
    g, af, cohorts = predict_announcements(pool, calendar, a.loc[a.model.eq('official_raw')], protocol)
    merged = merge_predictions([a, b, c, e, f, g])
    if set(merged.model) != set(CATALOG): raise ValueError('Incomplete report recipe catalog')
    support = []
    for fit in af:
        for scope, rows in fit['support'].items():
            support.extend(dict(model=fit['model'], origin=fit['origin'], scope=scope, **row) for row in rows)
    routing = []
    for (month, model, side, reason), group in g.groupby(['month', 'model', 'side', 'fallback_reason']):
        routing.append(dict(month=month, model=model, side=side, reason=reason, orders=len(group),
            supported_orders=int(group.supported.sum()), changed_orders=int(group.multiplier.ne(1.).sum()),
            event_count=group.loc[group.event_id.ne(''), 'event_id'].nunique(),
            minimum_multiplier=group.multiplier.min(), maximum_multiplier=group.multiplier.max()))
    states = dict(time_decay=ds, state_fallback=ss, hourly=hs, coarse=cs, announcement=af)
    diagnostics = dict(weight_diagnostics=weights, announcement_support=pd.DataFrame(support),
                       announcement_routing=pd.DataFrame(routing), training_cohorts=cohorts)
    return merged, states, diagnostics


def compare_reference(summary, comparison):
    """Verify the report's frozen evidence, without loading any archived model."""
    root = PROJECT/'docs/results/report_unified_v3_20261015_r2/validation'
    keys = ['month', 'model', 'group', 'value']
    columns = ['orders', 'rmse_per_btc', 'mae_per_btc', 'total_bias_usdc', 'gross_underprediction_usdc']
    old = pd.read_csv(root/'summary.csv', dtype={'value': str}).set_index(keys)
    current = summary.set_index(keys).reindex(old.index)
    np.testing.assert_allclose(current[columns].to_numpy(float), old[columns].to_numpy(float), rtol=1e-10, atol=1e-7)
    grid = pd.read_csv(root/'model_comparison.csv').set_index('model')
    cols = ['macro_rmse_per_btc', 'macro_mae_per_btc', 'pooled_rmse_per_btc', 'pooled_mae_per_btc']
    new = comparison.set_index('model').loc[grid.index, cols]
    np.testing.assert_allclose(new, grid[cols], rtol=1e-10, atol=1e-7)
    return pd.DataFrame([dict(reference='report_unified_v3_20261015_r2', summary_groups=len(old),
        models=len(grid), maximum_absolute_difference=float(np.abs(current[columns]-old[columns]).to_numpy().max()),
        reproduced=True)])


@threadpool_limits.wrap(limits=1)
def run(run_id, data_dir, market_path):
    if json.loads(decay.PROTOCOL_PATH.read_text()) != PROTOCOL: raise ValueError('Protocol changed')
    for module in (decay, state, hourly, coarse, fine):
        if json.loads(module.CONFIG.read_text()) != module.SETTINGS: raise ValueError('Existing recipe configuration changed')
    macro_settings = json.loads((PROJECT/'config/macro_multiplier_v1.json').read_text())
    calendar_path = PROJECT/macro_settings['calendar']
    if file_digest(calendar_path) != macro_settings['calendar_sha256']: raise ValueError('Calendar evidence changed')
    guard = AccessGuard(); sys.addaudithook(guard.check)
    root = run_destination(PROJECT/'outputs/model_studies', run_id, data_dir)
    root.mkdir(parents=True); (root/'provenance').mkdir(); began = perf_counter()
    canonical = canonical_identity(data_dir)
    environment = environment_snapshot()
    paths = {calendar_path, PROJECT/'docs/plans/2026-10-15-report-unified-rerun.md',
             PROJECT/'scripts/verify_report_unified.py'}
    for folder, pattern in [('pp_models', '*.py'), ('pp_data', '*.py'), ('pp_audit', '*.py'), ('pp_calendar', '*.py'), ('config', '*.json'), ('tests', '*.py'), ('scripts', '*.py')]:
        paths.update((PROJECT/folder).glob(pattern))
    sources = {str(p.relative_to(PROJECT)): file_digest(p) for p in sorted(paths)}
    protected = {str(p.relative_to(PROJECT)): file_digest(p) for p in (PROJECT/'docs/results').rglob('*') if p.is_file()}
    with ZipFile(root/'provenance/source.zip', 'w', ZIP_DEFLATED) as archive:
        for name in sources: archive.write(PROJECT/name, name)
    atomic_json(root/'provenance/config.json', SETTINGS)
    atomic_json(root/'provenance/protocol.json', PROTOCOL)
    atomic_json(root/'provenance/preflight.json', dict(canonical=canonical,
        sources=sources, previous_results=protected, data_dir=str(Path(data_dir).resolve()), market_path=str(Path(market_path).resolve())))
    try:
        pool = load_dataset(data_dir, cohort='benchmark', use_base=True)
        market, market_identity = read_market_prefix(market_path)
        if market_identity['prefix_sha256'] != state.SETTINGS['market_prefix_sha256']:
            raise ValueError('Historical market prefix changed')
        predicted, states, diagnostics = predict_all(pool, market, json.loads(calendar_path.read_text()))
        scored, coverage = score_predictions(pool, predicted, PROTOCOL['validation_end'])
        if coverage != dict(query_orders=4640, scored_orders=4640, labels_unavailable=0):
            raise ValueError('Report comparison cohort changed')
        print(f'Scoring {len(CATALOG)} recipes on {coverage["scored_orders"]} identical orders', flush=True)
        result = tables(scored, PROTOCOL['validation_start'], PROTOCOL['validation_end'], CATALOG, PAIRS)
        result['model_comparison'] = result['fixed_grid'].copy()
        result['fixed_grid'] = result['fixed_grid'].loc[result['fixed_grid'].model.isin(FIXED)].copy()
        result['reference_checks'] = compare_reference(result['summary'], result['model_comparison'])
        result.update(diagnostics)
        write_tables(root, 'validation', result)
        atomic_json(root/'provenance/parameters.json', states)
        atomic_json(root/'provenance/market.json', market_identity)
        if canonical_identity(data_dir) != canonical:
            raise ValueError('Canonical data changed')
        for name, digest in {**sources, **protected}.items():
            if file_digest(PROJECT/name) != digest: raise ValueError('Source/reference changed: '+name)
        fits = dict(h0_time_decay=len(states['time_decay'])*3,
                    h0_state=sum(s['h0_supported'] for s in states['state_fallback'].values()),
                    h0_hourly=len(states['hourly']), h0_coarse=sum(s['h0_supported'] for s in states['coarse'].values()),
                    hourly_amount=len(states['hourly'])*3, announcement=len(states['announcement']))
        manifest = dict(status='complete', producer=SETTINGS['version'], coverage=coverage, recipes=len(CATALOG),
            fits=fits, reference_results_reproduced=True, official_validation_open_attempts=guard.official_attempts,
            july_role='seen_rolling_validation',
            elapsed_seconds=perf_counter()-began, max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            environment=environment, artifacts={str(p.relative_to(root)): file_digest(p) for p in root.rglob('*') if p.is_file()})
        atomic_json(root/'provenance/manifest.json', manifest)
        print(json.dumps({k: v for k, v in manifest.items() if k not in ('artifacts', 'environment')}, ensure_ascii=False), flush=True)
    except Exception as error:
        atomic_json(root/'provenance/failure.json', dict(error=repr(error))); raise
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--data-dir', type=Path, required=True, help='Verified canonical preprocessing directory')
    parser.add_argument('--market-path', type=Path, required=True, help='Market CSV; only the old-data prefix is read')
    args = parser.parse_args()
    print(run(args.run_id, args.data_dir, args.market_path))


if __name__ == "__main__":
    main()
