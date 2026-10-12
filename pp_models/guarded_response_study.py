"""Calm penalty then daily upward limit: validation-only plan|run."""
import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from pp_data.config import DEFAULT_DATA, DEFAULT_OUTPUT, PROJECT
from pp_data.io import ModelSplit
from .accelerated_response import augment_pool, AcceleratedResponseModel
from .accelerated_response_study import variant_config as reference_config
from .anchor_models import build_anchor_model
from .anchor_study import plan_study as anchor_plan, validate_study as validate_anchor_study
from .config import load_model_config, resolve_config
from .data import daily_batches
from .directional_features import read_market_prefix
from .directional_study import install_development_guard
from .evaluation import evaluate
from .guarded_response import INPUTS, GuardedResponseModel
from .scoped_data import load_validation_pool
from .study import study_snapshot
from .study_storage import atomic_json, file_digest, study_workspace

CONFIG_PATH = PROJECT/'config/guarded_response_v1.json'
BUDGETS = {'overall_rmse_vs_raw': 1.01, 'calm_mae_vs_raw': 1.02, 'tail_underprediction_vs_power5': 1.05}


def _anchor_settings(config):
    return {'version': 'anchor_comparison_v1', 'periods': config['periods'],
            'lambdas': [.1, 1., 10.], 'rmse_tolerance': .01}


def load_study_config(path=CONFIG_PATH):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_study(config)
    return config


def validate_study(config):
    if (set(config) != {'version', 'periods', 'calm_penalties', 'upward_steps', 'budgets'}
            or config['version'] != 'guarded_response_v1' or config['calm_penalties'] != [.01, .1, 1.]
            or any(isinstance(v, bool) for v in config['calm_penalties'])
            or config['upward_steps'] != [.05, .1, .2] or config['budgets'] != BUDGETS):
        raise ValueError('Unsupported guarded study configuration')
    validate_anchor_study(_anchor_settings(config))


def variant_config(base, mode, penalty=None, step=None, start=None):
    if (mode not in ('raw', 'power5', 'calm', 'calm_rate') or base['window_days'] != 30
            or (mode in ('raw', 'power5') and (penalty is not None or step is not None or start is not None))
            or (mode in ('calm', 'calm_rate') and (isinstance(penalty, bool) or penalty not in (.01, .1, 1.)))
            or (mode != 'calm_rate' and (step is not None or start is not None))
            or (mode == 'calm_rate' and (isinstance(step, bool) or step not in (.05, .1, .2) or start is None))):
        raise ValueError('Invalid guarded response configuration')
    if mode in ('raw', 'power5'):
        cfg = reference_config(base, mode, 0. if mode == 'power5' else None)
    else:
        cfg = resolve_config(base, cohort='benchmark', use_base=True, models=['gam'])
        cfg.update(selected_models=['official_'+mode], features=list(INPUTS), daily_direction_metrics=True,
                   fast_response_metrics=True, response_max_multiplier=4.)
    cfg.update(response_mode=mode, calm_penalty=penalty, upward_step=step, sequential_start=start,
               guarded_response_metrics=True)
    return cfg


def build_variant(name, resolved):
    mode = resolved['response_mode']
    if mode == 'raw': return build_anchor_model(name, resolved)
    if (name != 'official_'+mode or resolved['selected_models'] != [name]
            or resolved['cohort'] != 'benchmark' or resolved['use_base'] is not True):
        raise ValueError('Invalid guarded model factory')
    if mode == 'power5':
        from .fast_response_model import INPUTS as BASE_INPUTS
        if resolved['features'] != BASE_INPUTS: raise ValueError('Invalid reference features')
        return AcceleratedResponseModel(0.)
    if mode in ('calm', 'calm_rate') and resolved['features'] == INPUTS:
        return GuardedResponseModel(resolved['calm_penalty'], resolved['upward_step'])
    raise ValueError('Invalid guarded feature schema or mode')


def evaluate_variant(pool, split, resolved, period=None):
    if split != 'validation': raise ValueError('Guarded study is validation-only')
    X = pool.X.assign(fast_response_state=pool.X.response_power5, fast_response_phase=pool.X.phase_power5)
    return evaluate(ModelSplit(X, pool.y, pool.metadata), split, resolved, model_factory=build_variant, period=period)


def plan_study(pool, base, config):
    validate_study(config)
    periods = anchor_plan(pool, base, _anchor_settings(config))['periods']
    if not {'response_power5', 'phase_power5', 'realized_vol_60m', 'log_return_60m'} <= set(pool.X):
        raise ValueError('Guarded response features missing')
    start = next(daily_batches(pool, 'validation', 30, config['periods']['tuning']))[0].isoformat()
    return {'study': 'guarded_response_v1', 'periods': periods, 'tuning_candidates': 8,
        'daily_scored_evaluations': 8*periods['tuning']['prediction_days']+4*periods['followup']['prediction_days'],
        'sequential_initialization_utc': start,
        'stateful_protocol': 'fit every calendar day from common initialization; replay before later phase, score only requested phase',
        'selection': 'pass all fixed tuning budgets then minimum RMSE; if none passes, mark exploratory fallback and use minimum RMSE',
        'calm_definition': 'Put vol60 <= historical Put q80 AND downside60/vol60 <= historical Put q80',
        'tail_definition': 'evaluation only: Put payout per BTC > historical Put payout q90',
        'fixed_parameters': {'window_days': 30, 'maximum_multiplier': 4., 'existing_5m_squared_signal_unchanged': True},
        'budgets': config['budgets'], 'test_used': False, 'margin_used': False, 'threads': 1}


def budget_checks(record, raw, power, budgets):
    def metric(result, group, value, field):
        rows = result['summary'].query("cohort == 'benchmark'")
        rows = rows.loc[rows.group.eq(group) & rows.value.astype(str).eq(value)]
        if len(rows) != 1 or field not in rows: return None
        value = float(rows[field].iloc[0])
        return value if np.isfinite(value) else None
    specs = [('overall_rmse_vs_raw', raw, 'overall', 'all', 'rmse_per_btc'),
             ('calm_mae_vs_raw', raw, 'pricing_risk_group', 'calm', 'mae_per_btc'),
             ('tail_underprediction_vs_power5', power, 'payout_tail_group', 'tail', 'gross_underprediction_usdc')]
    checks = {}
    for name, reference, group, value, field in specs:
        actual, baseline = metric(record, group, value, field), metric(reference, group, value, field)
        limit = None if baseline is None else baseline*budgets[name]
        passed = actual is not None and limit is not None and actual <= limit+1e-12
        checks[name] = {'observed': actual, 'limit': limit, 'passed': bool(passed)}
    return checks


def choose_guarded(candidates, raw, power, budgets):
    checks = {c['id']: budget_checks(c, raw, power, budgets) for c in candidates}
    eligible = [c for c in candidates if all(v['passed'] for v in checks[c['id']].values())]
    chosen = min(eligible or candidates, key=lambda c: (c['score'], -c['resolved']['calm_penalty'],
                 c['resolved']['upward_step'] if c['resolved']['upward_step'] is not None else float('inf')))
    return chosen, {'selected_id': chosen['id'], 'selected_passes_all_budgets': bool(eligible),
                    'candidate_checks': checks}


def run_guarded_study(root, run_id, base, config, pool, source, resume=False, evaluator=evaluate_variant,
                      progress=None, source_check=None, dataset_dir=DEFAULT_OUTPUT):
    plan = plan_study(pool, base, config)
    with study_workspace(root, run_id, {'base': base, 'study': config, 'source': source, 'plan': plan}, resume, dataset_dir) as store:
        records, selected, decisions = {}, {}, {}
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
                progress(f"{phase}: {cfg['response_mode']} penalty={cfg['calm_penalty']} step={cfg['upward_step']} RMSE={result['score']:.4f}")
            return result
        controls = [variant_config(base, mode) for mode in ('raw', 'power5')]
        raw, power = [trial(cfg, 'tuning') for cfg in controls]
        candidates = [trial(variant_config(base, 'calm', penalty), 'tuning') for penalty in config['calm_penalties']]
        choice, decisions['calm'] = choose_guarded(candidates, raw, power, config['budgets'])
        selected['calm'] = choice['resolved']
        candidates = [trial(variant_config(base, 'calm_rate', selected['calm']['calm_penalty'], step,
            plan['sequential_initialization_utc']), 'tuning') for step in config['upward_steps']]
        choice, decisions['calm_rate'] = choose_guarded(candidates, raw, power, config['budgets'])
        selected['calm_rate'] = choice['resolved']
        atomic_json(store.path/'provenance/selected_models.json', selected)
        atomic_json(store.path/'provenance/selection_checks.json', decisions)
        for cfg in controls+list(selected.values()): trial(cfg, 'followup')
        paths = [store.path/'provenance'/name for name in ('plan.json', 'selected_models.json', 'selection_checks.json')]
        (store.path/'metrics').mkdir(exist_ok=True)
        for phase in config['periods']:
            summaries, days = [], []
            for record in records.values():
                if record['phase'] != phase: continue
                cfg = record['resolved']; mode = cfg['response_mode']
                tags = dict(phase=phase, response_mode=mode,
                    calm_penalty=float('nan') if cfg['calm_penalty'] is None else float(cfg['calm_penalty']),
                    upward_step=float('nan') if cfg['upward_step'] is None else float(cfg['upward_step']),
                    selected=cfg == selected.get(mode), candidate_id=record['id'])
                summaries.append(record['summary'].assign(**tags)); days.append(record['daily'].assign(**tags))
            for suffix, frames in [('', summaries), ('_daily', days)]:
                path = store.path/f'metrics/{phase}{suffix}.csv'
                pd.concat(frames, ignore_index=True).to_csv(path, index=False, float_format='%.17g'); paths.append(path)
        store.status('complete', completed_candidates=len(records),
            files={str(p.relative_to(store.path)): file_digest(p) for p in paths},
            limitations=['June repeatedly observed: exploratory only; numeric budgets are research assumptions, not official requirements',
                'Calm penalty shrinks the shared amplitude and can also reduce stress response; no separate gating change',
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
    path = run_guarded_study(args.output_root, args.run_id, base, config, pool, provenance,
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
