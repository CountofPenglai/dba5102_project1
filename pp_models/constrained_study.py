"""Validation-only contract/tail/flow ablation: python -m pp_models.constrained_study plan|run."""
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
from .constrained_features import CalibratedMarket, augment_constraints, evaluate_calibrated_market
from .constrained_pricing import INPUTS, MODES, build_constrained_model
from .data import daily_batches
from .directional_features import read_market_prefix
from .directional_study import install_development_guard
from .evaluation import evaluate, key_metrics
from .market_activity_features import read_order_prefix
from .scoped_data import load_validation_pool
from .study import study_snapshot
from .study_storage import atomic_json, file_digest, study_workspace
from .terminal_distribution import augment_pool as augment_terminal
from .terminal_distribution_study import variant_config as terminal_config
from .terminal_payout import build_terminal_model

CONFIG_PATH = PROJECT/'config/constrained_pricing_v1.json'
FAMILIES = ('contract','tails','flow')
FIXED = {'version':'constrained_pricing_v1','lambdas':[.1,1,10], 'flow_penalty_multiplier':10,
    'recent_tail_days':7,'tail_shrinkage_origins':200,'minimum_recent_origins':100,
    'tail_ratio_limits':[.5,2], 'feature_z_limits':[-3,3],'coefficient_limits':[-1,1],
    'proxy_coefficient_limits':[0,1],
    'selection':'minimum tuning overall RMSE; exact ties prefer larger lambda'}


def _anchor_settings(config):
    return {'version':'anchor_comparison_v1','periods':config['periods'],'lambdas':[.1,1.,10.],'rmse_tolerance':.01}


def validate_study(config):
    if {k:v for k,v in config.items() if k!='periods'} != FIXED:
        raise ValueError('Unsupported constrained study configuration')
    validate_anchor(_anchor_settings(config))


def load_study_config(path=CONFIG_PATH):
    config=json.loads(Path(path).read_text(encoding='utf-8'));validate_study(config)
    return config


def variant_config(base, mode, lam=None):
    if mode=='static_reference':
        if lam is not None: raise ValueError('Static reference is frozen')
        result=terminal_config(base,'static',.01)
        result.update(constraint_mode=mode,constraint_lambda=None)
        return result
    if (mode not in MODES or base['window_days']!=30
            or (mode in FAMILIES and (isinstance(lam,bool) or lam not in (.1,1.,10.)))
            or (mode not in FAMILIES and lam is not None)):
        raise ValueError('Invalid constrained variant')
    result=resolve_config(base,cohort='benchmark',use_base=True,models=['gam'])
    result.update(selected_models=[f'constrained_{mode}'],features=list(INPUTS),daily_direction_metrics=True,
                  constraint_mode=mode,constraint_lambda=lam)
    return result


def evaluate_variant(pool, split, resolved, period=None):
    if split!='validation': raise ValueError('Constrained study is validation-only')
    predictions=[]
    factory=build_terminal_model if resolved['constraint_mode']=='static_reference' else build_constrained_model
    class Recorder:
        def __init__(self,model): self.model=model
        def fit(self,X,y): self.model.fit(X,y);return self
        def predict_raw(self,X):
            p=self.model.predict_raw(X);predictions.append(pd.Series(p,index=X.index));return p
        def diagnostics(self): return self.model.diagnostics()
    summary,daily=evaluate(pool,split,resolved,model_factory=lambda name,cfg:Recorder(factory(name,cfg)),period=period)
    prediction=pd.concat(predictions);rows=[];daily_extra=[]
    for day,train,ids in daily_batches(pool,'validation',30,period):
        for side in (0,1):
            train_side=train[pool.X.loc[train,'option_type'].eq(side)]
            for tenor,today in pool.X.loc[ids].loc[lambda x:x.option_type.eq(side)].groupby('product_duration_seconds'):
                cell=train_side[pool.X.loc[train_side,'product_duration_seconds'].eq(tenor)]
                chosen=cell if len(cell)>=30 else train_side
                threshold=float(pool.y.loc[chosen].quantile(.9)) if len(chosen)>=30 else np.nan
                r=pd.DataFrame({'day':day.isoformat(),'actual':pool.y.loc[today.index],
                    'prediction':prediction.loc[today.index],'quantity':today.quantity_btc})
                r['value']=f'{side}_'+np.where(r.actual.gt(threshold),'tail','body') if np.isfinite(threshold) else f'{side}_unavailable'
                rows.append(r)
    observations=pd.concat(rows)
    model=resolved['selected_models'][0]
    extra=[]
    for value,frame in observations.groupby('value'):
        extra.append(dict(model=model,cohort='benchmark',group='direction_payout_tail',value=value,
                          **key_metrics(frame.actual,frame.prediction,frame.quantity)))
    for (day,value),frame in observations.groupby(['day','value']):
        daily_extra.append(dict(model=model,cohort='benchmark',group='direction_payout_tail',value=value,
            day_utc=day,**key_metrics(frame.actual,frame.prediction,frame.quantity)))
    return pd.concat([summary,pd.DataFrame(extra)],ignore_index=True),pd.concat([daily,pd.DataFrame(daily_extra)],ignore_index=True)


def plan_study(pool, base, config):
    validate_study(config)
    periods=anchor_plan(pool,base,_anchor_settings(config))['periods']
    return dict(study='constrained_pricing_v1',periods=periods,tuning_candidates=12,followup_candidates=6,
        daily_scored_evaluations=12*periods['tuning']['prediction_days']+6*periods['followup']['prediction_days'],
        controls=['raw','scaled_both_directions','prior_static_put_lambda_0.01'],families=list(FAMILIES),
        selection=FIXED['selection'],test_used=False,margin_used=False,threads=1,training_performed=False,
        target='actual payout per BTC; terminal payoff only an input proxy',
        market_method='30d static/dynamic plus 7d separately shrunk tail scales; no guaranteed coverage',
        uncertainty='same repeatedly observed June validation; not independent confirmation')


def run_study(root,run_id,base,config,pool,source,resume=False,evaluator=evaluate_variant,
              market=None,progress=None,source_check=None,dataset_dir=DEFAULT_OUTPUT):
    plan=plan_study(pool,base,config)
    with study_workspace(root,run_id,{'base':base,'study':config,'source':source,'plan':plan},resume,dataset_dir) as store:
        records,selected={},{}
        atomic_json(store.path/'provenance/plan.json',plan)
        def trial(cfg,phase):
            if source_check:source_check()
            result=store.trial(cfg,phase,config['periods'][phase],lambda:evaluator(pool,'validation',cfg,period=config['periods'][phase]))
            overall=result['summary'].query("cohort == 'benchmark' and group == 'overall'")
            if int(overall.orders.iloc[0])!=plan['periods'][phase]['prediction_rows']:
                raise ValueError('Constrained candidate changed scoring cohort')
            if source_check:source_check()
            records[result['id']]=result
            if progress:progress(f"{phase}: {cfg['constraint_mode']} lambda={cfg['constraint_lambda']} RMSE={result['score']:.4f}")
            return result
        controls=[variant_config(base,m) for m in ('raw','scaled','static_reference')]
        for cfg in controls:trial(cfg,'tuning')
        for mode in FAMILIES:
            candidates=[trial(variant_config(base,mode,lam),'tuning') for lam in config['lambdas']]
            selected[mode]=min(candidates,key=lambda c:(c['score'],-c['resolved']['constraint_lambda']))['resolved']
        atomic_json(store.path/'provenance/selected_models.json',selected)
        for cfg in controls+list(selected.values()):trial(cfg,'followup')
        paths=[store.path/'provenance/plan.json',store.path/'provenance/selected_models.json']
        (store.path/'metrics').mkdir(exist_ok=True)
        for phase in config['periods']:
            summaries,days=[],[]
            for record in records.values():
                if record['phase']!=phase:continue
                cfg=record['resolved'];mode=cfg['constraint_mode']
                tags=dict(phase=phase,variant=mode,lam=np.nan if cfg['constraint_lambda'] is None else cfg['constraint_lambda'],
                          selected=cfg==selected.get(mode),candidate_id=record['id'])
                summaries.append(record['summary'].assign(**tags));days.append(record['daily'].assign(**tags))
            for suffix,frames in [('',summaries),('_daily',days)]:
                path=store.path/f'metrics/{phase}{suffix}.csv'
                pd.concat(frames,ignore_index=True).to_csv(path,index=False,float_format='%.17g');paths.append(path)
        if market is not None:
            phase_days={phase:[d for d,_,_ in daily_batches(pool,'validation',30,b)] for phase,b in config['periods'].items()}
            summary,daily=evaluate_calibrated_market(market,phase_days,config['periods'])
            for name,frame in [('market_summary',summary),('market_daily',daily)]:
                path=store.path/f'metrics/{name}.csv';frame.to_csv(path,index=False,float_format='%.17g');paths.append(path)
        if source_check:source_check()
        store.status('complete',completed_candidates=len(records),files={str(p.relative_to(store.path)):file_digest(p) for p in paths},
            limitations=['Repeated June exploration; selection is not independent confirmation',
                'Both directions calibrated; compare against both-direction scalar, not only raw',
                'Empirical tails and sqrt remaining-time scaling are approximations; early close not modeled',
                'Tail-map ratios and feature/coefficient bounds are statistical controls, not business risk limits',
                'No margin, demand/profit optimization, test CSV or July prices; aggregate outputs only'])
        return store.path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','run']);parser.add_argument('--run-id');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--market-csv',type=Path,default=DEFAULT_DATA/'btc_market_history.csv')
    parser.add_argument('--orders-csv',type=Path,default=DEFAULT_DATA/'pp_transactions_training.csv')
    parser.add_argument('--output-root',type=Path,default=PROJECT/'outputs/model_studies')
    args=parser.parse_args(argv)
    if args.action=='run' and not args.run_id:parser.error('run requires --run-id')
    guard,started=install_development_guard(args.data_dir),perf_counter()
    base,config=load_model_config(),load_study_config();source=study_snapshot(args.data_dir)
    paths=[CONFIG_PATH,PROJECT/'scripts/payout_signal_analysis.py']+list((PROJECT/'pp_audit').glob('*.py'))
    aux={str(p.relative_to(PROJECT)):file_digest(p) for p in paths}
    pool=load_validation_pool(variant_config(base,'raw'),args.data_dir)
    if args.action=='plan':print(json.dumps(plan_study(pool,base,config),indent=2));return 0
    originals={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in (args.market_csv,args.orders_csv)}
    usable,market_info=read_market_prefix(args.market_csv,'2026-07-01T00:00:00Z')
    events,order_info=read_order_prefix(args.orders_csv,'2026-07-01T00:00:00Z')
    market=CalibratedMarket(usable)
    pool,reference_info=augment_terminal(pool,market)
    pool,feature_info=augment_constraints(pool,market,usable,events)
    del usable,events
    def source_check():
        if any((p.stat().st_size,p.stat().st_mtime_ns)!=stamp for p,stamp in originals.items()):
            raise ValueError('Raw source changed during study')
        if study_snapshot(args.data_dir)!=source or any(file_digest(PROJECT/p)!=h for p,h in aux.items()):
            raise ValueError('Code/data/config/environment changed during constrained study')
    source_check()
    provenance={'development':source,'market_prefix':market_info,'order_prefix':order_info,
                'features':feature_info,'reference_features':reference_info,'auxiliary_code':aux}
    path=run_study(args.output_root,args.run_id,base,config,pool,provenance,resume=args.resume,
        market=market,source_check=source_check,dataset_dir=args.data_dir,progress=lambda s:print(s,flush=True))
    execution={'elapsed_seconds_including_data_loading':perf_counter()-started,
        'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
        'test_csv_read':False,'blocked_test_opens':guard['blocked_test_opens'],
        'opened_development_files':sorted(guard['opened_development_files'])}
    atomic_json(path/'provenance/execution.json',execution)
    archive=path/'provenance/source_snapshot.zip';names=set(source['code_sha256'])|set(aux)
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
