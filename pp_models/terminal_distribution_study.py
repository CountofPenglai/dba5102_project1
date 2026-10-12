"""Validation-only terminal distribution experiment: plan|run, no test prices."""
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
from .anchor_study import plan_study as anchor_plan, validate_study as validate_anchor
from .config import load_model_config, resolve_config
from .data import daily_batches
from .directional_features import read_market_prefix
from .directional_study import install_development_guard
from .evaluation import evaluate
from .scoped_data import load_validation_pool
from .study import study_snapshot
from .study_storage import atomic_json, file_digest, study_workspace
from .terminal_distribution import TerminalMarket, augment_pool, evaluate_market
from .terminal_payout import INPUTS, NAMES, build_terminal_model

CONFIG_PATH=PROJECT/'config/terminal_distribution_v1.json'
MARKET_SETTINGS={'horizons_minutes':[30,60,120,240,480],'window_days':30,'origin_stride_minutes':30,
    'signal_stride_minutes':5,'ewma_half_life_minutes':60,'continuous_warmup_minutes':1440,
    'sigma_floor':.000001,'minimum_history_origins':100,'return_center':'zero_log_return'}


def _anchor_settings(config):
    return {'version':'anchor_comparison_v1','periods':config['periods'],'lambdas':[.1,1.,10.],'rmse_tolerance':.01}


def validate_study(config):
    if (set(config)!={'version','periods','lambdas','market'} or config['version']!='terminal_distribution_v1'
            or config['lambdas']!=[.01,.1,1.] or any(isinstance(x,bool) for x in config['lambdas'])
            or config['market']!=MARKET_SETTINGS):
        raise ValueError('Unsupported terminal distribution study')
    validate_anchor(_anchor_settings(config))


def load_study_config(path=CONFIG_PATH):
    config=json.loads(Path(path).read_text(encoding='utf-8'));validate_study(config)
    return config


def variant_config(base,mode,lam=None):
    if (base['window_days']!=30 or mode not in NAMES
            or (mode in ('raw','scaled') and lam is not None)
            or (mode in ('static','dynamic') and (isinstance(lam,bool) or lam not in (.01,.1,1.)))):
        raise ValueError('Invalid terminal variant')
    resolved=resolve_config(base,cohort='benchmark',use_base=True,models=['gam'])
    resolved.update(selected_models=[NAMES[mode]],features=list(INPUTS),daily_direction_metrics=True,
                    distribution_mode=mode,distribution_lambda=lam)
    return resolved


def evaluate_variant(pool,split,resolved,period=None):
    if split!='validation':raise ValueError('Terminal study is validation-only')
    return evaluate(pool,split,resolved,model_factory=build_terminal_model,period=period)


def plan_study(pool,base,config):
    validate_study(config)
    periods=anchor_plan(pool,base,_anchor_settings(config))['periods']
    for phase,bounds in config['periods'].items():
        ids=[idx for _,_,today in daily_batches(pool,'validation',30,bounds) for idx in today]
        X=pool.X.loc[ids]
        periods[phase]['supported_put_rows']=int((X.option_type.eq(1)&X.remaining_seconds.le(28800)).sum())
        periods[phase]['unchanged_call_rows']=int(X.option_type.eq(0).sum())
        periods[phase]['unsupported_put_rows']=int((X.option_type.eq(1)&X.remaining_seconds.gt(28800)).sum())
    return {'study':'terminal_distribution_v1','periods':periods,'tuning_candidates':8,
        'daily_scored_evaluations':8*periods['tuning']['prediction_days']+4*periods['followup']['prediction_days'],
        'selection':'minimum tuning RMSE within each distribution family; exact ties prefer larger lambda; compare scaled separately',
        'market':config['market'],'target':'actual payout per BTC; terminal intrinsic payout is only a proxy',
        'market_intervals':'80/90/95 percent; scores in terminal log-return basis points',
        'horizon_adjustment':'nearest covering grid H, multiply centered returns by sqrt(remaining_minutes/H)',
        'historical_features':'reconstructed from each order origin day, with all market labels complete before that day',
        'test_used':False,'margin_used':False,'training_performed':False,'threads':1}


def run_terminal_study(root,run_id,base,config,pool,source,resume=False,evaluator=evaluate_variant,
                       market=None,progress=None,source_check=None,dataset_dir=DEFAULT_OUTPUT):
    plan=plan_study(pool,base,config)
    with study_workspace(root,run_id,{'base':base,'study':config,'source':source,'plan':plan},resume,dataset_dir) as store:
        records,selected={},{}
        atomic_json(store.path/'provenance/plan.json',plan)
        def trial(cfg,phase):
            if source_check:source_check()
            result=store.trial(cfg,phase,config['periods'][phase],
                lambda:evaluator(pool,'validation',cfg,period=config['periods'][phase]))
            overall=result['summary'].query("cohort == 'benchmark' and group == 'overall'")
            if int(overall.orders.iloc[0])!=plan['periods'][phase]['prediction_rows']:
                raise ValueError('Terminal candidate changed scoring cohort')
            if source_check:source_check()
            records[result['id']]=result
            if progress:progress(f"{phase}: {cfg['distribution_mode']} lambda={cfg['distribution_lambda']} RMSE={result['score']:.4f}")
            return result
        controls=[variant_config(base,m) for m in ('raw','scaled')]
        for cfg in controls:trial(cfg,'tuning')
        for mode in ('static','dynamic'):
            candidates=[trial(variant_config(base,mode,lam),'tuning') for lam in config['lambdas']]
            winner=min(candidates,key=lambda c:(c['score'],-c['resolved']['distribution_lambda']))
            selected[mode]=winner['resolved']
        atomic_json(store.path/'provenance/selected_models.json',selected)
        for cfg in controls+list(selected.values()):trial(cfg,'followup')
        paths=[store.path/'provenance/plan.json',store.path/'provenance/selected_models.json']
        (store.path/'metrics').mkdir(exist_ok=True)
        for phase in config['periods']:
            summaries,days=[],[]
            for record in records.values():
                if record['phase']!=phase:continue
                cfg=record['resolved'];mode=cfg['distribution_mode']
                tags=dict(phase=phase,distribution_mode=mode,
                    distribution_lambda=float('nan') if cfg['distribution_lambda'] is None else cfg['distribution_lambda'],
                    selected=cfg==selected.get(mode),candidate_id=record['id'])
                summaries.append(record['summary'].assign(**tags));days.append(record['daily'].assign(**tags))
            for suffix,frames in [('',summaries),('_daily',days)]:
                path=store.path/f'metrics/{phase}{suffix}.csv'
                pd.concat(frames,ignore_index=True).to_csv(path,index=False,float_format='%.17g');paths.append(path)
        if market is not None:
            phase_days={phase:[d for d,_,_ in daily_batches(pool,'validation',30,bounds)] for phase,bounds in config['periods'].items()}
            summary,daily=evaluate_market(market,phase_days,config['periods'])
            for name,frame in [('market_summary',summary),('market_daily',daily)]:
                path=store.path/f'metrics/{name}.csv';frame.to_csv(path,index=False,float_format='%.17g');paths.append(path)
        if source_check:source_check()
        store.status('complete',completed_candidates=len(records),
            files={str(p.relative_to(store.path)):file_digest(p) for p in paths},
            limitations=['Repeatedly observed June validation is exploratory; no independent generalization claim',
                'Empirical zero-centered horizon laws can omit unseen tails; overlapping origins are not independent events',
                'Square-root remaining-time scaling is an approximation; >8h Put and all Call use raw Base',
                'Terminal payoff proxy does not model voluntary early close, transaction fees or official settlement detail',
                'No margin, demand/profit claim, test CSV or July prices; only aggregate statistics are saved'])
        return store.path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','run']);parser.add_argument('--run-id')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--market-csv',type=Path,default=DEFAULT_DATA/'btc_market_history.csv')
    parser.add_argument('--output-root',type=Path,default=PROJECT/'outputs/model_studies')
    args=parser.parse_args(argv)
    if args.action=='run' and not args.run_id:parser.error('run requires --run-id')
    guard,started=install_development_guard(args.data_dir),perf_counter()
    base,config=load_model_config(),load_study_config()
    source=study_snapshot(args.data_dir)
    aux={str(p.relative_to(PROJECT)):file_digest(p) for p in sorted((PROJECT/'pp_audit').glob('*.py'))}
    aux[str(CONFIG_PATH.relative_to(PROJECT))]=file_digest(CONFIG_PATH)
    pool=load_validation_pool(variant_config(base,'raw'),args.data_dir)
    if args.action=='plan':
        print(json.dumps(plan_study(pool,base,config),indent=2));return 0
    original=args.market_csv.stat()
    usable,market_info=read_market_prefix(args.market_csv,'2026-07-01T00:00:00Z')
    market=TerminalMarket(usable);del usable
    pool,feature_info=augment_pool(pool,market)
    def source_check():
        current=args.market_csv.stat()
        if (current.st_size,current.st_mtime_ns)!=(original.st_size,original.st_mtime_ns):
            raise ValueError('Market source changed during terminal study')
        if study_snapshot(args.data_dir)!=source or any(file_digest(PROJECT/p)!=h for p,h in aux.items()):
            raise ValueError('Code/data/config/environment changed during terminal study')
    source_check()
    provenance={'development':source,'market_prefix':market_info,'features':feature_info,'auxiliary_code':aux}
    path=run_terminal_study(args.output_root,args.run_id,base,config,pool,provenance,resume=args.resume,
        market=market,source_check=source_check,dataset_dir=args.data_dir,progress=lambda s:print(s,flush=True))
    execution={'elapsed_seconds_including_data_loading':perf_counter()-started,
        'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        'test_csv_read':False,'blocked_test_opens':guard['blocked_test_opens'],
        'opened_development_files':sorted(guard['opened_development_files'])}
    atomic_json(path/'provenance/execution.json',execution)
    archive=path/'provenance/source_snapshot.zip'
    names=set(source['code_sha256'])|set(aux)
    for directory,pattern in [('pp_data','*.py'),('config','*.json'),('tests','*.py')]:
        names.update(str(p.relative_to(PROJECT)) for p in (PROJECT/directory).glob(pattern))
    with ZipFile(archive,'w',ZIP_DEFLATED) as stream:
        for name in sorted(names):stream.write(PROJECT/name,name)
    source_check()
    manifest_path=path/'provenance/manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['files'].update({str(p.relative_to(path)):file_digest(p) for p in [path/'provenance/execution.json',archive]})
    atomic_json(manifest_path,manifest)
    print(json.dumps({'output':str(path),**execution},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
