"""Official Base x five-minute accelerated response, validation-only plan|run."""
import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import pandas as pd
from pp_data.config import DEFAULT_DATA, DEFAULT_OUTPUT, PROJECT
from pp_data.io import ModelSplit
from .accelerated_response import AcceleratedResponseModel, augment_pool
from .anchor_models import build_anchor_model
from .anchor_study import anchor_config, plan_study as anchor_plan, validate_study as validate_anchor_study
from .config import load_model_config, resolve_config
from .directional_features import read_market_prefix
from .directional_study import install_development_guard
from .evaluation import evaluate
from .fast_response_model import INPUTS, FastResponseModel
from .scoped_data import load_validation_pool
from .study import study_snapshot
from .study_storage import atomic_json, file_digest, study_workspace

CONFIG_PATH = PROJECT/'config/accelerated_response_v1.json'
VARIANTS = ('linear5', 'power5')


def _anchor_settings(config):
    return config | {'version': 'anchor_comparison_v1', 'lambdas': [.1, 1., 10.], 'rmse_tolerance': .01}


def load_study_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_study(config)
    return config


def validate_study(config):
    if (set(config) != {'version', 'periods', 'lambdas', 'rmse_tolerance'}
            or config['version'] != 'accelerated_response_v1' or config['lambdas'] != [0., .01, .1]
            or any(isinstance(v, bool) for v in config['lambdas']) or config['rmse_tolerance'] != 0):
        raise ValueError('Unsupported accelerated study configuration')
    validate_anchor_study(_anchor_settings(config))


def variant_config(base, mode, lam=None):
    if (base['window_days'] != 30 or mode not in ('raw', 'scaled', 'fast15', *VARIANTS)
            or (mode in ('raw', 'scaled') and lam is not None)
            or (mode == 'fast15' and lam != 10.)
            or (mode in VARIANTS and (isinstance(lam, bool) or lam not in (0., .01, .1)))):
        raise ValueError('Invalid accelerated response configuration')
    if mode in ('raw', 'scaled'):
        cfg = anchor_config(base, 'official', mode)
    else:
        cfg = resolve_config(base, cohort='benchmark', use_base=True, models=['gam'])
        cfg.update(selected_models=['official_'+mode], features=list(INPUTS), daily_direction_metrics=True)
    cfg.update(response_mode=mode, response_lambda=lam, fast_response_metrics=True,
               response_max_multiplier=4. if mode in VARIANTS else None)
    return cfg


def build_variant(name, resolved):
    mode = resolved['response_mode']
    if mode in ('raw', 'scaled'):
        return build_anchor_model(name, resolved)
    if (resolved['features'] != INPUTS or resolved['selected_models'] != [name] or name != 'official_'+mode
            or resolved['cohort'] != 'benchmark' or resolved['use_base'] is not True):
        raise ValueError('Invalid accelerated model factory configuration')
    if mode == 'fast15' and resolved['response_lambda'] == 10.:
        return FastResponseModel(10.)
    if mode in VARIANTS:
        return AcceleratedResponseModel(resolved['response_lambda'])
    raise ValueError('Unknown response variant')


def evaluate_variant(pool, split, resolved, period=None):
    if split != 'validation':
        raise ValueError('Accelerated response study is validation-only')
    mode = resolved['response_mode']
    # Controls use the same 5-minute stage labels as the new experiments.
    if mode != 'fast15':
        variant = mode if mode in VARIANTS else 'power5'
        X = pool.X.assign(fast_response_state=pool.X['response_'+variant],
                          fast_response_phase=pool.X['phase_'+variant])
        pool = ModelSplit(X, pool.y, pool.metadata)
    return evaluate(pool, split, resolved, model_factory=build_variant, period=period)


def plan_study(pool, base, config):
    validate_study(config)
    periods = anchor_plan(pool, base, _anchor_settings(config))['periods']
    required = {'fast_response_state', 'fast_response_phase', 'response_linear5', 'response_power5',
                'phase_linear5', 'phase_power5'}
    if not required <= set(pool.X):
        raise ValueError('Accelerated response features missing')
    return {'study': 'accelerated_response_v1', 'periods': periods, 'tuning_candidates': 9,
        'max_daily_evaluations': 9*periods['tuning']['prediction_days']+6*periods['followup']['prediction_days'],
        'selection': 'minimum tuning RMSE in each 5m family; exact ties prefer larger lambda; freeze before followup',
        'fixed_followup_sensitivity': 'power5 lambda=0, deduplicated if selected',
        'fixed_parameters': {'window_days': 30, 'interval_minutes': 5, 'scale_lookback_bars': 288,
            'scale_floor': .0001, 'noise_threshold_sigma': .1, 'pressure_cap': 3.,
            'powers': [1, 2], 'recovery_factor_per_bar': .5, 'maximum_multiplier': 4.},
        'legacy_reference': '15m saturating state, lambda=10 from previous frozen run',
        'test_used': False, 'margin_used': False, 'frequency_markup_used': False, 'threads': 1}


def run_accelerated_study(root, run_id, base, config, pool, source, resume=False, evaluator=evaluate_variant,
                          progress=None, source_check=None, dataset_dir=DEFAULT_OUTPUT):
    plan = plan_study(pool, base, config)
    identity = {'base': base, 'study': config, 'source': source, 'plan': plan}
    with study_workspace(root, run_id, identity, resume, dataset_dir) as store:
        records, selected = {}, {}
        atomic_json(store.path/'provenance/plan.json', plan)
        def trial(cfg, phase):
            if source_check: source_check()
            result = store.trial(cfg, phase, config['periods'][phase],
                lambda: evaluator(pool, 'validation', cfg, period=config['periods'][phase]))
            overall = result['summary'].query("cohort == 'benchmark' and group == 'overall'")
            if int(overall.orders.iloc[0]) != plan['periods'][phase]['prediction_rows']:
                raise ValueError('Candidate scoring cohort changed')
            if source_check: source_check()
            records[result['id']] = result
            if progress:
                progress(f"{phase}: {cfg['response_mode']} lambda={cfg['response_lambda']} RMSE={result['score']:.4f}")
            return result
        controls = [variant_config(base, mode) for mode in ('raw', 'scaled')]+[variant_config(base, 'fast15', 10.)]
        for cfg in controls: trial(cfg, 'tuning')
        for variant in VARIANTS:
            candidates = [trial(variant_config(base, variant, lam), 'tuning') for lam in config['lambdas']]
            selected[variant] = min(candidates, key=lambda c: (c['score'], -c['resolved']['response_lambda']))['resolved']
        atomic_json(store.path/'provenance/selected_models.json', selected)
        for cfg in controls+list(selected.values()): trial(cfg, 'followup')
        if selected['power5']['response_lambda'] != 0.:
            trial(variant_config(base, 'power5', 0.), 'followup')
        paths = [store.path/'provenance/plan.json', store.path/'provenance/selected_models.json']
        (store.path/'metrics').mkdir(exist_ok=True)
        for phase in config['periods']:
            summaries, days = [], []
            for record in records.values():
                if record['phase'] != phase: continue
                cfg = record['resolved']; mode = cfg['response_mode']
                tags = dict(phase=phase, response_mode=mode, lam=float('nan') if cfg['response_lambda'] is None else float(cfg['response_lambda']),
                    selected=cfg == selected.get(mode), fixed_unregularized=mode == 'power5' and cfg['response_lambda'] == 0.,
                    candidate_id=record['id'])
                summaries.append(record['summary'].assign(**tags)); days.append(record['daily'].assign(**tags))
            for suffix, frames in [('', summaries), ('_daily', days)]:
                path = store.path/f'metrics/{phase}{suffix}.csv'
                pd.concat(frames, ignore_index=True).to_csv(path, index=False, float_format='%.17g'); paths.append(path)
        store.status('complete', completed_candidates=len(records),
            files={str(p.relative_to(store.path)): file_digest(p) for p in paths},
            limitations=['Repeatedly observed June is exploratory, not independent confirmation',
                '5m versus old 15m changes accumulation, recovery and selection as well as cadence',
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
    pool = load_validation_pool(variant_config(base, 'raw'), args.data_dir)
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
    path = run_accelerated_study(args.output_root, args.run_id, base, config, pool, provenance,
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
