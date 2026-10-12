"""Validation-only sequential study: python -m pp_models.market_activity_study plan|run."""
import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
from pp_data.config import DEFAULT_DATA, DEFAULT_OUTPUT, PROJECT
from .baselines import BASELINES, build_baseline
from .config import load_model_config
from .data import coverage, daily_batches
from .directional_features import read_market_prefix
from .directional_study import install_development_guard, validate_study as validate_directional_study
from .evaluation import evaluate
from .market_activity_features import read_order_prefix, augment_market_activity
from .market_activity_gam import VARIANTS, build_activity_model, variant_config
from .runs import fingerprint
from .scoped_data import load_validation_pool
from .study import choose_candidate, study_snapshot
from .study_storage import atomic_json, file_digest, study_workspace

CONFIG_PATH = PROJECT / 'config/gam_market_activity_v1.json'
REFERENCE_LAMBDAS = {'original': 100, 'semivol': 1000}


def load_study_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_study(config)
    return config


def validate_study(config):
    if config.get('version') != 'gam_market_activity_v1':
        raise ValueError('Unsupported activity study configuration')
    validate_directional_study(config | {'version': 'gam_directional_v1'})


def plan_study(pool, base, config):
    validate_study(config)
    resolved = variant_config(base, 'original', 100)
    periods = {p: coverage(pool, 'validation', resolved, bounds) for p, bounds in config['periods'].items()}
    if any(p['insufficient_history_days'] for p in periods.values()):
        raise ValueError('Insufficient history for activity study')
    for phase, bounds in config['periods'].items():
        ids = [str(i) for _, _, rows in daily_batches(pool, 'validation', resolved['window_days'], bounds) for i in rows]
        periods[phase]['prediction_ids_sha256'] = fingerprint(ids)
    return {'study': 'gam_price_dynamics_then_platform_activity_exploration', 'periods': periods,
        'variants': list(VARIANTS), 'reference_lambdas': REFERENCE_LAMBDAS, 'tuning_candidates': 8,
        'max_gam_fits': 8*periods['tuning']['prediction_days'] + 6*periods['followup']['prediction_days'],
        'baseline_updates': 2*sum(p['prediction_days'] for p in periods.values()),
        'selection': 'tuning RMSE within 1% of minimum, prefer larger lambda; freeze before followup',
        'fixed_regularization_control': 'D, E, F lambda=1000; no followup reselection',
        'fixed_parameters': {'n_splines': 4, 'max_iter': 100, 'tol': 1e-4},
        'test_used': False, 'margin_used': False, 'frequency_markup_used': False, 'threads': 1}


def _tables(records, selected, phase):
    summaries, days = [], []
    for record in records.values():
        if record['phase'] != phase:
            continue
        cfg = record['resolved']
        variant = cfg.get('gam_variant', cfg['selected_models'][0])
        baseline = variant in BASELINES
        chosen = baseline or cfg == selected.get(variant)
        lam = None if baseline else cfg['models']['gam']['lam']
        fixed = variant in ('semivol', 'price_dynamics', 'platform_activity') and lam == 1000
        role = ('baseline' if baseline else 'reference' if variant in REFERENCE_LAMBDAS else
                'selected' if chosen else 'fixed_lam1000' if fixed else 'candidate')
        tags = dict(variant=variant, phase=phase, role=role, selected=chosen,
            fixed_regularization_control=fixed, lam=lam, candidate_id=record['id'])
        summaries.append(record['summary'].assign(**tags))
        days.append(record['daily'].assign(**tags))
    summary, daily = pd.concat(summaries, ignore_index=True), pd.concat(days, ignore_index=True)
    keys = ['cohort', 'group', 'value']
    for name in BASELINES:
        ref = summary.loc[summary.variant.eq(name), keys+['rmse_per_btc']].rename(columns={'rmse_per_btc': 'baseline_rmse'})
        summary = summary.merge(ref, on=keys, how='left', validate='many_to_one')
        summary['rmse_change_vs_'+name] = summary.rmse_per_btc / summary.baseline_rmse.replace(0, float('nan')) - 1
        summary = summary.drop(columns='baseline_rmse')
    return summary, daily


def run_activity_study(root, run_id, base, config, pool, source, resume=False, evaluator=evaluate,
                       progress=None, source_check=None, dataset_dir=DEFAULT_OUTPUT, feature_statistics=None):
    plan = plan_study(pool, base, config)
    identity = {'base': base, 'study': config, 'source': source, 'plan': plan}
    with study_workspace(root, run_id, identity, resume, dataset_dir) as store:
        selected, records = {}, {}
        atomic_json(store.path/'provenance/plan.json', plan)
        def trial(cfg, phase, baseline=False):
            if source_check:
                source_check()
            result = store.trial(cfg, phase, config['periods'][phase], lambda: evaluator(pool, 'validation', cfg,
                model_factory=build_baseline if baseline else build_activity_model, period=config['periods'][phase]))
            overall = result['summary'].query("cohort == 'risk' and group == 'overall'")
            if int(overall.orders.iloc[0]) != plan['periods'][phase]['prediction_rows']:
                raise ValueError('Candidate scoring cohort changed')
            if source_check:
                source_check()
            records[result['id']] = result
            if progress:
                progress(f"{phase}: {cfg.get('gam_variant', cfg['selected_models'][0])} "
                    f"lambda={cfg['models']['gam']['lam']} RMSE={result['score']:.4f}")
            return result
        references = []
        for name in BASELINES:
            cfg = variant_config(base, 'original', 100)
            cfg.pop('gam_variant')
            cfg.update(selected_models=[name], features=['option_type', 'product_duration_seconds'], baseline_smoothing=30)
            references.append(cfg)
            trial(cfg, 'tuning', True)
        for variant in VARIANTS:
            lambdas = [REFERENCE_LAMBDAS[variant]] if variant in REFERENCE_LAMBDAS else config['lambdas']
            candidates = [trial(variant_config(base, variant, lam), 'tuning') for lam in lambdas]
            selected[variant] = choose_candidate(candidates, 'gam', config['rmse_tolerance'])['resolved']
        atomic_json(store.path/'provenance/selected_models.json', selected)
        for variant in VARIANTS:
            trial(selected[variant], 'followup')
            if variant not in REFERENCE_LAMBDAS and selected[variant]['models']['gam']['lam'] != 1000:
                trial(variant_config(base, variant, 1000), 'followup')
        for cfg in references:
            trial(cfg, 'followup', True)
        paths = [store.path/'provenance/plan.json', store.path/'provenance/selected_models.json']
        (store.path/'metrics').mkdir(exist_ok=True)
        for phase in config['periods']:
            summary, daily = _tables(records, selected, phase)
            for suffix, table in [('', summary), ('_daily', daily)]:
                path = store.path/f'metrics/{phase}{suffix}.csv'
                table.to_csv(path, index=False, float_format='%.17g')
                paths.append(path)
        if feature_statistics is not None:
            path = store.path/'metrics/feature_statistics.csv'
            feature_statistics.to_csv(path, index=False, float_format='%.17g')
            paths.append(path)
        store.status('complete', completed_candidates=len(records),
            files={str(p.relative_to(store.path)): file_digest(p) for p in paths},
            limitations=['June is repeatedly observed exploratory validation, not independent confirmation',
                         'Platform event counts reflect provided data coverage; no causal demand or profit claim',
                         'No test CSV, pricing markup, row predictions or model checkpoints'])
        return store.path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run'])
    parser.add_argument('--run-id')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--market-csv', type=Path, default=DEFAULT_DATA/'btc_market_history.csv')
    parser.add_argument('--orders-csv', type=Path, default=DEFAULT_DATA/'pp_transactions_training.csv')
    parser.add_argument('--output-root', type=Path, default=PROJECT/'outputs/model_studies')
    args = parser.parse_args(argv)
    if args.action == 'run' and not args.run_id:
        parser.error('run requires --run-id')
    guard, started = install_development_guard(args.data_dir), perf_counter()
    base, config = load_model_config(), load_study_config()
    source = study_snapshot(args.data_dir)
    external_stats = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in (args.market_csv, args.orders_csv)}
    aux_code = {str(p.relative_to(PROJECT)): file_digest(p) for p in sorted((PROJECT/'pp_audit').glob('*.py'))}
    aux_code[str(CONFIG_PATH.relative_to(PROJECT))] = file_digest(CONFIG_PATH)
    pool = load_validation_pool(variant_config(base, 'original', 100), args.data_dir)
    cutoff = '2026-07-01T00:00:00Z'
    market, market_info = read_market_prefix(args.market_csv, cutoff)
    events, event_info = read_order_prefix(args.orders_csv, cutoff)
    pool, feature_info, feature_statistics = augment_market_activity(pool, market, events)
    del market, events
    def source_check():
        for path, expected in external_stats.items():
            current = path.stat()
            if (current.st_size, current.st_mtime_ns) != expected:
                raise ValueError('Market/order source changed during study')
        if study_snapshot(args.data_dir) != source or any(file_digest(PROJECT/p) != h for p, h in aux_code.items()):
            raise ValueError('Development code/data/config/environment changed during study')
    source_check()
    provenance = {'development': source, 'market_prefix': market_info, 'order_prefix': event_info,
                  'features': feature_info, 'auxiliary_code': aux_code}
    if args.action == 'plan':
        print(json.dumps(plan_study(pool, base, config) | {'source': provenance}, indent=2))
        return 0
    path = run_activity_study(args.output_root, args.run_id, base, config, pool, provenance, resume=args.resume,
        source_check=source_check, dataset_dir=args.data_dir, feature_statistics=feature_statistics,
        progress=lambda line: print(line, flush=True))
    execution = {'elapsed_seconds_including_data_loading': perf_counter()-started,
        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == 'darwin' else 1024),
        'test_csv_read': False, 'blocked_test_opens': guard['blocked_test_opens'],
        'opened_development_files': sorted(guard['opened_development_files'])}
    atomic_json(path/'provenance/execution.json', execution)
    archive = path/'provenance/source_snapshot.zip'
    archive_files = set(source['code_sha256']) | set(aux_code) | set(source.get('preprocessing_code', {}))
    archive_files.update(str(p.relative_to(PROJECT)) for p in (PROJECT/'pp_data').glob('*.py'))
    archive_files.update(str(p.relative_to(PROJECT)) for p in (PROJECT/'config').glob('*.json'))
    archive_files.update(str(p.relative_to(PROJECT)) for p in (PROJECT/'tests').glob('*.py'))
    with ZipFile(archive, 'w', ZIP_DEFLATED) as z:
        for name in sorted(archive_files):
            z.write(PROJECT/name, name)
    source_check()
    manifest_path = path/'provenance/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'].update({str(p.relative_to(path)): file_digest(p) for p in [path/'provenance/execution.json', archive]})
    atomic_json(manifest_path, manifest)
    print(json.dumps({'output': str(path), **execution}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
