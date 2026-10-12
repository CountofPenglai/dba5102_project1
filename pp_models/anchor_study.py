"""Validation-only anchor comparison: python -m pp_models.anchor_study plan|run."""
import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
from pp_data.config import DEFAULT_OUTPUT, PROJECT
from .anchor_models import INPUTS, MARKET_TERMS, build_anchor_model
from .config import load_model_config, resolve_config
from .data import coverage, daily_batches, period_bounds
from .directional_study import install_development_guard
from .evaluation import evaluate
from .runs import fingerprint
from .scoped_data import load_validation_pool
from .study import study_snapshot
from .study_storage import atomic_json, file_digest, study_workspace

CONFIG_PATH = PROJECT/'config/anchor_comparison_v1.json'
ANCHORS = ('official', 'grouped')


def load_study_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_study(config)
    return config


def validate_study(config):
    if (set(config) != {'version', 'periods', 'lambdas', 'rmse_tolerance'}
            or config['version'] != 'anchor_comparison_v1' or config['lambdas'] != [.1, 1., 10.]
            or config['rmse_tolerance'] != .01 or set(config['periods']) != {'tuning', 'followup'}):
        raise ValueError('Unsupported anchor study configuration')
    first = period_bounds('validation', config['periods']['tuning'])
    second = period_bounds('validation', config['periods']['followup'])
    if first[1] != second[0] or any(t != t.floor('D') for t in first+second):
        raise ValueError('Study phases must be contiguous UTC day boundaries')


def anchor_config(base, anchor, correction, lam=None):
    resolved = resolve_config(base, cohort='benchmark', use_base=True, models=['gam'])
    if (base['window_days'] != 30 or anchor not in ANCHORS or correction not in ('raw', 'scaled', 'dynamic')
            or (correction != 'dynamic' and lam is not None)
            or (correction == 'dynamic' and (isinstance(lam, bool) or lam not in (.1, 1., 10.)))):
        raise ValueError('Invalid anchor configuration or fixed history window')
    resolved.update(anchor=anchor, correction=correction, correction_lambda=lam,
        selected_models=[anchor+'_'+correction], features=list(INPUTS), daily_direction_metrics=True,
        baseline_smoothing=30)
    return resolved


def choose_correction(candidates, tolerance=.01):
    best = min(c['score'] for c in candidates)
    eligible = [c for c in candidates if c['score'] <= best*(1+tolerance)]
    # Prefer the nested intercept-only model, then the strongest coefficient shrinkage.
    return min(eligible, key=lambda c: (c['resolved']['correction'] != 'scaled',
        -(c['resolved'].get('correction_lambda') or 0)))


def plan_study(pool, base, config):
    validate_study(config)
    if not pool.metadata.benchmark_eligible.all():
        raise ValueError('Anchor comparison requires the same benchmark common pool')
    resolved = anchor_config(base, 'official', 'raw')
    periods = {phase: coverage(pool, 'validation', resolved, bounds) for phase, bounds in config['periods'].items()}
    if any(p['insufficient_history_days'] or p['min_training_rows'] < 2 for p in periods.values()):
        raise ValueError('Insufficient history for anchor study')
    for phase, bounds in config['periods'].items():
        batches = list(daily_batches(pool, 'validation', resolved['window_days'], bounds))
        periods[phase]['prediction_ids_sha256'] = fingerprint([str(i) for _, _, ids in batches for i in ids])
        periods[phase]['training_ids_by_day_sha256'] = fingerprint(
            {day.isoformat(): [str(i) for i in ids] for day, ids, _ in batches})
    return {'study': 'official_vs_direction_tenor_anchor', 'periods': periods, 'anchors': list(ANCHORS),
        'corrections': ['raw', 'scaled', 'dynamic'], 'market_terms': MARKET_TERMS, 'tuning_candidates': 10,
        'max_daily_evaluations': 10*periods['tuning']['prediction_days']+6*periods['followup']['prediction_days'],
        'max_nonlinear_fits': 6*periods['tuning']['prediction_days']+2*periods['followup']['prediction_days'],
        'selection': 'tuning RMSE within 1% of minimum; prefer scaled then larger lambda; freeze before followup',
        'fixed_parameters': {'window_days': 30, 'group_smoothing': 30, 'max_nfev': 200, 'solver_tolerance': 1e-8},
        'group_correction_training_anchor': 'leave-one-out within current legal history, including global shrinkage mean',
        'test_used': False, 'margin_used': False, 'frequency_markup_used': False, 'threads': 1}


def _tables(records, selected, phase):
    summaries, days = [], []
    for record in records.values():
        if record['phase'] != phase:
            continue
        cfg = record['resolved']
        chosen = cfg == selected[cfg['anchor']]
        tags = dict(anchor=cfg['anchor'], correction=cfg['correction'],
            lam=float('nan') if cfg['correction_lambda'] is None else float(cfg['correction_lambda']),
            phase=phase, selected=chosen, candidate_id=record['id'])
        summaries.append(record['summary'].assign(**tags))
        days.append(record['daily'].assign(**tags))
    summary, daily = pd.concat(summaries, ignore_index=True), pd.concat(days, ignore_index=True)
    keys = ['cohort', 'group', 'value']
    ref = summary.loc[summary.model.eq('official_raw'), keys+['rmse_per_btc']].rename(
        columns={'rmse_per_btc': 'official_raw_rmse'})
    summary = summary.merge(ref, on=keys, how='left', validate='many_to_one')
    summary['rmse_change_vs_official_raw'] = summary.rmse_per_btc / summary.official_raw_rmse.replace(0, float('nan')) - 1
    return summary.drop(columns='official_raw_rmse'), daily


def run_anchor_study(root, run_id, base, config, pool, source, resume=False, evaluator=evaluate,
                     progress=None, source_check=None, dataset_dir=DEFAULT_OUTPUT):
    plan = plan_study(pool, base, config)
    identity = {'base': base, 'study': config, 'source': source, 'plan': plan}
    with study_workspace(root, run_id, identity, resume, dataset_dir) as store:
        records, selected, references = {}, {}, []
        atomic_json(store.path/'provenance/plan.json', plan)
        def trial(cfg, phase):
            if source_check:
                source_check()
            result = store.trial(cfg, phase, config['periods'][phase], lambda: evaluator(pool, 'validation', cfg,
                model_factory=build_anchor_model, period=config['periods'][phase]))
            overall = result['summary'].query("cohort == 'benchmark' and group == 'overall'")
            if int(overall.orders.iloc[0]) != plan['periods'][phase]['prediction_rows']:
                raise ValueError('Candidate scoring cohort changed')
            if source_check:
                source_check()
            records[result['id']] = result
            if progress:
                progress(f"{phase}: {cfg['selected_models'][0]} lambda={cfg['correction_lambda']} RMSE={result['score']:.4f}")
            return result
        for anchor in ANCHORS:
            raw, scaled = (anchor_config(base, anchor, mode) for mode in ('raw', 'scaled'))
            references.extend([raw, scaled])
            trial(raw, 'tuning')
            candidates = [trial(scaled, 'tuning')]
            candidates.extend(trial(anchor_config(base, anchor, 'dynamic', lam), 'tuning') for lam in config['lambdas'])
            selected[anchor] = choose_correction(candidates, config['rmse_tolerance'])['resolved']
        # Publish the complete tuning decision before evaluating any followup outcome.
        atomic_json(store.path/'provenance/selected_models.json', selected)
        for cfg in references:
            trial(cfg, 'followup')
        for cfg in selected.values():
            if cfg['correction'] == 'dynamic':
                trial(cfg, 'followup')
        paths = [store.path/'provenance/plan.json', store.path/'provenance/selected_models.json']
        (store.path/'metrics').mkdir(exist_ok=True)
        for phase in config['periods']:
            summary, daily = _tables(records, selected, phase)
            for suffix, frame in [('', summary), ('_daily', daily)]:
                path = store.path/f'metrics/{phase}{suffix}.csv'
                frame.to_csv(path, index=False, float_format='%.17g')
                paths.append(path)
        store.status('complete', completed_candidates=len(records),
            files={str(p.relative_to(store.path)): file_digest(p) for p in paths},
            limitations=['June is repeatedly observed exploratory validation, not independent confirmation',
                         'Base availability restricts the common sample; quote matching timestamps are unavailable',
                         'Risk calibration is not optimal commercial pricing or a causal profit estimate',
                         'No test CSV, pricing margin, row predictions or model checkpoints'])
        return store.path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run'])
    parser.add_argument('--run-id')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--output-root', type=Path, default=PROJECT/'outputs/model_studies')
    args = parser.parse_args(argv)
    if args.action == 'run' and not args.run_id:
        parser.error('run requires --run-id')
    guard, started = install_development_guard(args.data_dir), perf_counter()
    base, config = load_model_config(), load_study_config()
    source = study_snapshot(args.data_dir)
    config_digest = file_digest(CONFIG_PATH)
    pool = load_validation_pool(anchor_config(base, 'official', 'raw'), args.data_dir)
    def source_check():
        if study_snapshot(args.data_dir) != source or file_digest(CONFIG_PATH) != config_digest:
            raise ValueError('Development code/data/config/environment changed during study')
    source_check()
    provenance = {'development': source, 'study_config_sha256': config_digest}
    if args.action == 'plan':
        print(json.dumps(plan_study(pool, base, config) | {'source': provenance}, indent=2))
        return 0
    path = run_anchor_study(args.output_root, args.run_id, base, config, pool, provenance, resume=args.resume,
        source_check=source_check, dataset_dir=args.data_dir, progress=lambda line: print(line, flush=True))
    execution = {'elapsed_seconds_including_data_loading': perf_counter()-started,
        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == 'darwin' else 1024),
        'test_csv_read': False, 'blocked_test_opens': guard['blocked_test_opens'],
        'opened_development_files': sorted(guard['opened_development_files'])}
    atomic_json(path/'provenance/execution.json', execution)
    archive = path/'provenance/source_snapshot.zip'
    names = set(source['code_sha256'])
    for directory, pattern in [('pp_data', '*.py'), ('config', '*.json'), ('tests', '*.py')]:
        names.update(str(p.relative_to(PROJECT)) for p in (PROJECT/directory).glob(pattern))
    with ZipFile(archive, 'w', ZIP_DEFLATED) as stream:
        for name in sorted(names):
            stream.write(PROJECT/name, name)
    source_check()
    manifest_path = path/'provenance/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'].update({str(p.relative_to(path)): file_digest(p) for p in [path/'provenance/execution.json', archive]})
    atomic_json(manifest_path, manifest)
    print(json.dumps({'output': str(path), **execution}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
