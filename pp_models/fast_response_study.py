"""Official Base x 15-minute Put response: validation-only plan|run entry point."""
import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
from pp_data.config import DEFAULT_DATA, DEFAULT_OUTPUT, PROJECT
from .anchor_models import build_anchor_model
from .anchor_study import anchor_config, plan_study as anchor_plan, validate_study as validate_anchor_study
from .config import load_model_config, resolve_config
from .directional_features import read_market_prefix
from .directional_study import install_development_guard
from .evaluation import evaluate
from .fast_response_features import augment_pool
from .fast_response_model import INPUTS, FastResponseModel
from .scoped_data import load_validation_pool
from .study import study_snapshot
from .study_storage import atomic_json, file_digest, study_workspace

CONFIG_PATH = PROJECT/'config/fast_response_v1.json'


def load_study_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_study(config)
    return config


def validate_study(config):
    if config.get('version') != 'fast_response_v1':
        raise ValueError('Unsupported response study version')
    validate_anchor_study(config | {'version': 'anchor_comparison_v1'})


def response_config(base, mode, lam=None):
    if (mode not in ('raw', 'scaled', 'fast') or base['window_days'] != 30
            or (mode != 'fast' and lam is not None)
            or (mode == 'fast' and (isinstance(lam, bool) or lam not in (.1, 1., 10.)))):
        raise ValueError('Invalid fixed response configuration')
    if mode == 'fast':
        cfg = resolve_config(base, cohort='benchmark', use_base=True, models=['gam'])
        cfg.update(selected_models=['official_fast'], features=list(INPUTS), daily_direction_metrics=True)
    else:
        cfg = anchor_config(base, 'official', mode)
    cfg.update(response_mode=mode, response_lambda=lam, fast_response_metrics=True)
    return cfg


def build_response_model(name, resolved):
    if name == 'official_fast':
        if (resolved['features'] != INPUTS or resolved['selected_models'] != [name]
                or resolved['cohort'] != 'benchmark' or resolved['use_base'] is not True
                or resolved['response_mode'] != 'fast'):
            raise ValueError('Invalid response model factory configuration')
        return FastResponseModel(resolved['response_lambda'])
    return build_anchor_model(name, resolved)


def plan_study(pool, base, config):
    validate_study(config)
    periods = anchor_plan(pool, base, config | {'version': 'anchor_comparison_v1'})['periods']
    if not {'fast_response_state', 'fast_response_phase'} <= set(pool.X):
        raise ValueError('Response features missing')
    return {'study': 'official_base_fast_response_v1', 'periods': periods,
        'methods': ['official_raw', 'official_scaled', 'official_fast'], 'tuning_candidates': 5,
        'daily_evaluations': 5*periods['tuning']['prediction_days']+3*periods['followup']['prediction_days'],
        'selection': 'fast family tuning RMSE within 1% of minimum; larger lambda; frozen before followup',
        'selection_does_not_imply_beating_raw': True, 'fixed_parameters': {
            'window_days': 30, 'interval_minutes': 15, 'scale_lookback_bars': 96,
            'scale_floor': .0001, 'noise_threshold_sigma': .1, 'recovery_factor_per_bar': .5},
        'coefficient': 'Put: 1 + beta * state; Call: 1; beta >= 0; daily fit on settled Put history',
        'test_used': False, 'margin_used': False, 'frequency_markup_used': False, 'threads': 1}


def _tables(records, selected, phase):
    summaries, days = [], []
    for record in records.values():
        if record['phase'] != phase:
            continue
        cfg = record['resolved']
        tags = dict(phase=phase, response_mode=cfg['response_mode'],
            lam=float('nan') if cfg['response_lambda'] is None else float(cfg['response_lambda']),
            selected=cfg == selected, candidate_id=record['id'])
        summaries.append(record['summary'].assign(**tags))
        days.append(record['daily'].assign(**tags))
    return pd.concat(summaries, ignore_index=True), pd.concat(days, ignore_index=True)


def run_response_study(root, run_id, base, config, pool, source, resume=False, evaluator=evaluate,
                       progress=None, source_check=None, dataset_dir=DEFAULT_OUTPUT):
    plan = plan_study(pool, base, config)
    identity = {'base': base, 'study': config, 'source': source, 'plan': plan}
    with study_workspace(root, run_id, identity, resume, dataset_dir) as store:
        records = {}
        atomic_json(store.path/'provenance/plan.json', plan)
        def trial(cfg, phase):
            if source_check: source_check()
            result = store.trial(cfg, phase, config['periods'][phase], lambda: evaluator(pool, 'validation', cfg,
                model_factory=build_response_model, period=config['periods'][phase]))
            overall = result['summary'].query("cohort == 'benchmark' and group == 'overall'")
            if int(overall.orders.iloc[0]) != plan['periods'][phase]['prediction_rows']:
                raise ValueError('Candidate scoring cohort changed')
            if source_check: source_check()
            records[result['id']] = result
            if progress:
                progress(f"{phase}: {cfg['selected_models'][0]} lambda={cfg['response_lambda']} RMSE={result['score']:.4f}")
            return result
        controls = [response_config(base, mode) for mode in ('raw', 'scaled')]
        for cfg in controls: trial(cfg, 'tuning')
        candidates = [trial(response_config(base, 'fast', lam), 'tuning') for lam in config['lambdas']]
        best = min(c['score'] for c in candidates)
        selected = max((c for c in candidates if c['score'] <= best*(1+config['rmse_tolerance'])),
                       key=lambda c: c['resolved']['response_lambda'])['resolved']
        atomic_json(store.path/'provenance/selected_models.json', {'fast': selected})
        for cfg in controls+[selected]: trial(cfg, 'followup')
        paths = [store.path/'provenance/plan.json', store.path/'provenance/selected_models.json']
        (store.path/'metrics').mkdir(exist_ok=True)
        for phase in config['periods']:
            for suffix, frame in zip(('', '_daily'), _tables(records, selected, phase)):
                path = store.path/f'metrics/{phase}{suffix}.csv'
                frame.to_csv(path, index=False, float_format='%.17g'); paths.append(path)
        store.status('complete', completed_candidates=len(records),
            files={str(p.relative_to(store.path)): file_digest(p) for p in paths},
            limitations=['June is repeatedly observed exploratory validation, not independent confirmation',
                         'Only completed 15-minute market bars; daily coefficients use as-of settled labels',
                         'Historical quote timestamp uncertainty remains; no causal pricing/profit claim',
                         'No test CSV, pricing margin, row predictions or model checkpoints'])
        return store.path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'run'])
    parser.add_argument('--run-id'); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--market-csv', type=Path, default=DEFAULT_DATA/'btc_market_history.csv')
    parser.add_argument('--output-root', type=Path, default=PROJECT/'outputs/model_studies')
    args = parser.parse_args(argv)
    if args.action == 'run' and not args.run_id: parser.error('run requires --run-id')
    guard, started = install_development_guard(args.data_dir), perf_counter()
    base, config = load_model_config(), load_study_config()
    source = study_snapshot(args.data_dir)
    original = args.market_csv.stat()
    aux_code = {str(p.relative_to(PROJECT)): file_digest(p) for p in sorted((PROJECT/'pp_audit').glob('*.py'))}
    aux_code[str(CONFIG_PATH.relative_to(PROJECT))] = file_digest(CONFIG_PATH)
    pool = load_validation_pool(response_config(base, 'raw'), args.data_dir)
    market, market_info = read_market_prefix(args.market_csv, '2026-07-01T00:00:00Z')
    pool, feature_info = augment_pool(pool, market)
    del market
    def source_check():
        current = args.market_csv.stat()
        if (current.st_size, current.st_mtime_ns) != (original.st_size, original.st_mtime_ns):
            raise ValueError('Market source changed during study')
        if study_snapshot(args.data_dir) != source or any(file_digest(PROJECT/p) != h for p, h in aux_code.items()):
            raise ValueError('Development code/data/config/environment changed during study')
    source_check()
    provenance = {'development': source, 'market_prefix': market_info, 'features': feature_info, 'auxiliary_code': aux_code}
    if args.action == 'plan':
        print(json.dumps(plan_study(pool, base, config) | {'source': provenance}, indent=2)); return 0
    path = run_response_study(args.output_root, args.run_id, base, config, pool, provenance,
        resume=args.resume, source_check=source_check, dataset_dir=args.data_dir,
        progress=lambda line: print(line, flush=True))
    execution = {'elapsed_seconds_including_data_loading': perf_counter()-started,
        'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform == 'darwin' else 1024),
        'test_csv_read': False, 'blocked_test_opens': guard['blocked_test_opens'],
        'opened_development_files': sorted(guard['opened_development_files'])}
    atomic_json(path/'provenance/execution.json', execution)
    archive = path/'provenance/source_snapshot.zip'
    names = set(source['code_sha256']) | set(aux_code)
    for directory, pattern in [('pp_data', '*.py'), ('config', '*.json'), ('tests', '*.py')]:
        names.update(str(p.relative_to(PROJECT)) for p in (PROJECT/directory).glob(pattern))
    with ZipFile(archive, 'w', ZIP_DEFLATED) as stream:
        for name in sorted(names): stream.write(PROJECT/name, name)
    source_check()
    manifest_path = path/'provenance/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'].update({str(p.relative_to(path)): file_digest(p) for p in [path/'provenance/execution.json', archive]})
    atomic_json(manifest_path, manifest)
    print(json.dumps({'output': str(path), **execution}, indent=2)); return 0


if __name__ == '__main__':
    raise SystemExit(main())
