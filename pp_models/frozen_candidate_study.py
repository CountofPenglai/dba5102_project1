"""Fixed existing payout recipes under monthly fits and no within-month labels."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT, DEFAULT_DATA, DEFAULT_OUTPUT
from pp_data.features import BASE_FEATURE
from .base_multiplier_study import AccessGuard, history, verify_reference
from .calibration_models import CalibrationModel
from .data import load_pool
from .directional_features import augment_pool as directional_pool, read_market_prefix
from .put_final_io import read_test_market_prefix
from .runs import run_destination, snapshot
from .scoped_data import load_validation_pool, development_snapshot
from .study_storage import atomic_json, file_digest
from .terminal_distribution import TerminalMarket, augment_pool as terminal_pool
from .terminal_payout import TerminalPayoutModel, INPUTS as TERMINAL_INPUTS
from .two_stage_components import AmountRefinement, INPUTS as AMOUNT_INPUTS
from .two_stage_severity import TwoStageSeverity, SEVERITY_INPUTS
from .two_stage_study import probability_frame, tail_context

CONFIG = PROJECT/'config/frozen_candidate_v1.json'
PLAN = PROJECT/'docs/plans/2026-10-14-frozen-candidate-review.md'
MODELS = ('official_raw','h0','contract30','g1a','static')
CANDIDATES = MODELS[2:]
SETTINGS = dict(version='frozen_candidate_v1', models=list(MODELS), history_days=30,
    development_start='2026-04-01T00:00:00Z', development_end='2026-07-01T00:00:00Z',
    pressure_end='2026-07-31T17:11:48Z', amount_strength=30, amount_bounds=[.5,1.5],
    gamma_alpha=1., static_lambda=.01, monthly_label_feedback=False,
    static_proxy_update='daily_past_completed_market_only', tail_history='all_settled_at_month_start',
    tail_quantile=.9, tail_minimum_history=30, risk_tolerance=1.02,
    bootstrap_blocks=[3,7,14], bootstrap_repeats=2000, mae_hard_gate=False,
    dev_market_sha256='21aa8f91a3b393452c93360cefc16b39255d64219ba3b67c3474b912ee5ff38b',
    july_market_sha256='cbf5d491d167ac377beb844dc9794e4f5e9986f72859035426f7789e67924b50')


def load_settings():
    result=json.loads(CONFIG.read_text())
    if result!=SETTINGS: raise ValueError('Fixed review settings changed; use a new version')
    return result


def canonical_identity(root=DEFAULT_OUTPUT,include_test=False,expected_manifest=None):
    root=Path(root);digest=file_digest(root/'manifest.json')
    if expected_manifest is not None and digest!=expected_manifest:
        raise ValueError('Canonical manifest changed')
    result={'manifest_sha256':digest,'test_artifacts':{}}
    if include_test:
        manifest=json.loads((root/'manifest.json').read_text())
        for kind in ('features','targets','metadata'):
            name=f'datasets/test/{kind}.csv';actual=file_digest(root/name)
            if actual!=manifest['artifacts'][name]['sha256']:raise ValueError('July artifact changed')
            result['test_artifacts'][name]=actual
    return result


def serialize_models(h0, amount, gamma, static):
    result={'h0':{},'contract30':{},'g1a':{},'static':{'a':static.a,'b':static.b,
        'training_rows':static.training_rows}, 'amount_support':amount.details,
        'gamma_support':gamma.diagnostics()}
    for side in (0,1):
        h=h0.parameters[side]
        hp={k:float(h[k]) for k in ('probability','severity')}
        if 'model' in h:
            hp.update(mean=h['scaler'].mean_.tolist(),scale=h['scaler'].scale_.tolist(),
                      coef=h['model'].coef_[0].tolist(),intercept=float(h['model'].intercept_[0]))
        result['h0'][str(side)]=hp
        a=amount.parameters[side]
        result['contract30'][str(side)]={} if not a else dict(beta=a['beta'].tolist(),
            mean=a['features'].scaler.mean_.tolist(),scale=a['features'].scaler.scale_.tolist())
        g=gamma.parameters[side]
        gp={k:g[k] for k in ('mode','coefficient','fallback_reason')}
        if g['mode']=='gamma':
            gp.update(mean=g['mean'].tolist(),scale=g['scale'].tolist(),
                      coef=g['model'].coef_.tolist(),intercept=float(g['model'].intercept_))
        result['g1a'][str(side)]=gp
    return result


@contextmanager
def single_cpu():
    # In this sandbox macOS physical-core discovery warns once; the original
    # Gamma recipe treats warnings as numerical failures. Declare the intended
    # CPU budget, rather than silencing numerical/convergence warnings.
    previous=os.environ.get('LOKY_MAX_CPU_COUNT')
    os.environ['LOKY_MAX_CPU_COUNT']='1'
    try:
        with threadpool_limits(limits=1):yield
    finally:
        if previous is None:os.environ.pop('LOKY_MAX_CPU_COUNT',None)
        else:os.environ['LOKY_MAX_CPU_COUNT']=previous


@single_cpu()
def predict_period(pool,start,end):
    """Only training masks inspect closed/target; queries use creation times alone."""
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if (start.tzinfo is None or end.tzinfo is None or start>=end or start.day!=1
            or start!=start.floor('D') or not pool.X.index.is_unique
            or not pool.X.index.equals(pool.y.index) or not pool.X.index.equals(pool.metadata.index)):
        raise ValueError('Invalid monthly pool or period')
    created,closed=pool.metadata.created_at_utc,pool.metadata.closed_at_utc
    outputs,states,cohorts=[],{},[]
    for origin in pd.date_range(start,end,freq='MS',inclusive='left'):
        stop=min(end,origin+pd.offsets.MonthBegin(1))
        query=pool.X.index[created.ge(origin)&created.lt(stop)]
        if len(query)==0: raise ValueError('Empty monthly prediction cohort')
        train=history(pool,origin,30)
        if len(train)<100: raise ValueError('Insufficient original H0 training support')
        tx,qx=pool.X.loc[train],pool.X.loc[query]
        h0=CalibrationModel('hurdle_base').fit(probability_frame(tx),pool.y.loc[train],created.loc[train])
        _,s0=h0.predict_components(probability_frame(tx))
        amount=AmountRefinement('contract',30).fit(tx[AMOUNT_INPUTS],pool.y.loc[train],
            created.loc[train],pd.Series(s0,index=train))
        gamma=TwoStageSeverity('g1a').fit(tx[SEVERITY_INPUTS],pool.y.loc[train],created.loc[train])
        static=TerminalPayoutModel('static',.01).fit(tx[TERMINAL_INPUTS],pool.y.loc[train])
        probability,severity=h0.predict_components(probability_frame(qx))
        predictions={'official_raw':qx[BASE_FEATURE].to_numpy(float),'h0':probability*severity,
            'contract30':probability*amount.predict_severity(qx[AMOUNT_INPUTS],pd.Series(severity,index=query)),
            'g1a':probability*gamma.predict_severity(qx[SEVERITY_INPUTS]),
            'static':static.predict_raw(qx[TERMINAL_INPUTS])}
        threshold,source=tail_context(pool,history(pool,origin),query)
        common=pd.DataFrame({'day':created.loc[query].dt.floor('D'),
            'month':created.loc[query].dt.strftime('%Y-%m'),'side':qx.option_type.astype(int),
            'tenor':qx.product_duration_seconds.astype(int),'quantity':qx.quantity_btc,
            'tail_threshold':threshold,'tail_source':source,'origin':origin.isoformat()},index=query)
        for name,pred in predictions.items():
            if not np.isfinite(pred).all() or (pred<0).any(): raise ValueError('Invalid candidate prediction')
            outputs.append(common.assign(model=name,prediction=pred,multiplier=pred/qx[BASE_FEATURE],
                probability=probability if name in ('h0','contract30','g1a') else np.nan))
        states[origin.isoformat()]=serialize_models(h0,amount,gamma,static)
        cohorts.append({'origin':origin.isoformat(),'training_orders':len(train),'query_orders':len(query),
            'earliest_training_create':created.loc[train].min().isoformat(),
            'latest_training_close':closed.loc[train].max().isoformat(),
            'tail_history_orders':len(history(pool,origin)),'monthly_fit_objects':4,
            'within_month_label_updates':0})
    return pd.concat(outputs),states,pd.DataFrame(cohorts)


def score(pool,predicted,end):
    base=predicted.loc[predicted.model.eq('official_raw')]
    if not base.index.is_unique: raise ValueError('Duplicate prediction IDs')
    eligible=pool.metadata.closed_at_utc.reindex(base.index).lt(pd.Timestamp(end))
    ids=base.index[eligible]
    result=predicted.loc[predicted.index.isin(ids)].copy()
    result['actual']=pool.y.reindex(result.index).to_numpy()
    if not np.isfinite(result.actual).all() or result.actual.lt(0).any():raise ValueError('Invalid scoring labels')
    result['tail_group']=np.where(result.tail_threshold.isna(),'unavailable',
        np.where(result.actual.gt(result.tail_threshold),'tail','body'))
    for name in MODELS:
        p=result.loc[result.model.eq(name)]
        if not p.index.equals(ids):raise ValueError('Candidate prediction cohorts differ')
    return result,dict(query_orders=len(base),scored_orders=len(ids),labels_unavailable=int((~eligible).sum()))


def prepare(pool,market):
    directional,dd=directional_pool(pool,market)
    augmented,td=terminal_pool(directional,TerminalMarket(market))
    return augmented,{'directional':dd,'terminal':td}


def write_tables(root,phase,tables):
    dest=root/'metrics'/phase;dest.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():table.to_csv(dest/(name+'.csv'),index=False,float_format='%.17g')


def source_paths():
    # Loaded project dependencies plus fixed specification and verification entry.
    paths={CONFIG,PLAN,Path(__file__),PROJECT/'scripts/verify_frozen_candidates.py'}
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            p=Path(filename).resolve()
            if PROJECT in p.parents and p.suffix=='.py' and not p.is_relative_to(PROJECT/'.venv'):
                paths.add(p)
    paths.update((PROJECT/'tests').glob('test_frozen_candidate*.py'))
    paths.update([PROJECT/'requirements.txt',PROJECT/'requirements-models.txt'])
    return sorted(paths)


def run(run_id):
    from .frozen_candidate_statistics import tables
    settings=load_settings();guard=AccessGuard();sys.addaudithook(guard.check)
    root=run_destination(PROJECT/'outputs/model_studies',run_id)
    root.mkdir(parents=True);(root/'provenance').mkdir()
    start_time=perf_counter();stamps={};phase_info={}
    atomic_json(root/'provenance/config.json',settings)
    sources={str(p.relative_to(PROJECT)):file_digest(p) for p in source_paths()}
    with ZipFile(root/'provenance/source.zip','w',ZIP_DEFLATED) as z:
        for name in sources:z.write(PROJECT/name,name)
    protected={str(p.relative_to(PROJECT)):file_digest(p) for pattern in
        ('two_stage_components_independent_20261014','rolling_calibration_apr_jun_20261009','put_final_july_20261014')
        for p in (PROJECT/'outputs/model_studies'/pattern).rglob('*') if p.is_file()}
    identity,_=development_snapshot()
    canonical=canonical_identity()
    atomic_json(root/'provenance/preflight.json',{'sources':sources,'protected':protected,
        'development_data':identity,'canonical':canonical})
    try:
        with threadpool_limits(limits=1):
            for phase,start,end in [('development',settings['development_start'],settings['development_end']),
                                    ('pressure',settings['development_end'],settings['pressure_end'])]:
                t=perf_counter()
                if phase=='development':
                    pool=load_validation_pool({'cohort':'benchmark','use_base':True})
                    market,market_info=read_market_prefix(DEFAULT_DATA/'btc_market_history.csv',end)
                    expected=settings['dev_market_sha256']
                else:
                    if not (root/'provenance/development_complete.json').exists():raise RuntimeError('Development not saved')
                    guard.allow_pressure=True
                    july_identity=canonical_identity(include_test=True,expected_manifest=canonical['manifest_sha256'])
                    pool=load_pool('test',{'cohort':'benchmark','use_base':True})
                    market,market_info=read_test_market_prefix(DEFAULT_DATA/'btc_market_history.csv')
                    expected=settings['july_market_sha256']
                if market_info['prefix_sha256']!=expected:raise ValueError('Old market prefix changed')
                pool,feature_info=prepare(pool,market)
                predicted,states,cohorts=predict_period(pool,start,end)
                frame,coverage=score(pool,predicted,end)
                verify_reference(frame,'development' if phase=='development' else 'test')
                computed=tables(frame,start,end)
                computed['cohorts']=cohorts
                write_tables(root,phase,computed)
                atomic_json(root/f'provenance/{phase}_parameters.json',states)
                stamps[phase]=datetime.now(timezone.utc).isoformat()
                phase_info[phase]={'coverage':coverage,'market':market_info,'features':feature_info,
                    'seconds':perf_counter()-t,'origins':len(states),'fit_objects':4*len(states),
                    'base_h0_frozen_reference_match':True}
                if phase=='pressure':phase_info[phase]['canonical']=july_identity
                atomic_json(root/f'provenance/{phase}_complete.json',phase_info[phase])
                print(json.dumps({'phase':phase,'orders':coverage['scored_orders'],
                                  'seconds':phase_info[phase]['seconds']},ensure_ascii=False),flush=True)
        if development_snapshot()[0]!=identity:raise ValueError('Development data changed during run')
        if canonical_identity(include_test=True,expected_manifest=canonical['manifest_sha256'])!=july_identity:
            raise ValueError('July artifacts changed during run')
        for name,digest in {**sources,**protected}.items():
            if file_digest(PROJECT/name)!=digest:raise ValueError(f'Protected source/output changed: {name}')
        artifacts={str(p.relative_to(root)):file_digest(p) for p in root.rglob('*') if p.is_file()}
        atomic_json(root/'provenance/manifest.json',dict(status='complete',producer='frozen_candidate_v1',
            phases=phase_info,completion_utc=stamps,artifacts=artifacts,source_sha256=sources,
            environment=snapshot()['environment'],elapsed_seconds=perf_counter()-start_time,
            max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            official_validation_open_attempts=guard.official_attempts,premature_july_open_attempts=guard.premature_test_attempts,
            july_csv_opens=guard.test_opens,protected_files=len(protected),selected_model=None))
    except Exception as error:
        atomic_json(root/'provenance/failure.json',{'type':type(error).__name__,'message':str(error)})
        raise
    return root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','run'])
    parser.add_argument('--run-id',default='frozen_candidates_20261014')
    args=parser.parse_args()
    if args.command=='plan':print(json.dumps(load_settings(),ensure_ascii=False,indent=2))
    else:print(run(args.run_id))


if __name__=='__main__':main()
