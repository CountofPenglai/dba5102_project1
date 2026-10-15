"""Seven-day updates with strictly forward, settled residual calibration."""
import argparse
import json
import resource
import sys
from time import perf_counter
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT,DEFAULT_DATA
from pp_data.features import BASE_FEATURE
from .base_multiplier_study import history,AccessGuard
from .base_residual_components import CONTRACT,ResidualModel,contract_features
from .calibration_models import CalibrationModel
from .frozen_candidate_study import canonical_identity,write_tables,SETTINGS as REF_SETTINGS
from .scoped_data import load_validation_pool,development_snapshot
from .data import load_pool
from .directional_features import read_market_prefix
from .put_final_io import read_test_market_prefix
from .two_stage_study import probability_frame,tail_context
from .runs import run_destination,snapshot
from .study_storage import atomic_json,file_digest
from .weekly_residual import Extras,OVERLAYS
from .weekly_residual_features import features as signal_features

REFERENCE=PROJECT/'outputs/model_studies/base_residual_components_20261015_v2'
CONFIG=PROJECT/'config/weekly_residual_v1.json'
PLAN=PROJECT/'docs/plans/2026-10-15-weekly-residual.md'
MODELS=('official_raw','h0','contract_monthly','contract_weekly')+OVERLAYS
SETTINGS=dict(version='weekly_residual_v1',models=list(MODELS),update_days=7,history_days=30,
    warmup_start='2026-03-04T00:00:00Z',weekly_label_feedback=True,overlay_history='settled_forward_predictions_30d',
    overlay_smoothing_orders=100,overlay_min_orders=100,overlay_min_days=10,signal_min_orders=30,signal_min_days=5,
    level_cap=.05,jump_cap=.05,streak_caps=[.05,.1],streak_minutes=5,streak_sigma_bars=288,
    streak_min_run=3,streak_min_bar_sigma=.25,streak_run_sigma=2.,streak_recovery=.5,streak_expiry_bars=12,
    jump_window=60,jump_scale_window=240,jump_sigma=3.,jump_horizon_cap_hours=4.,
    tail_threshold_update='monthly_all_settled_history_q90',parameter_search=False,
    reference_run=str(REFERENCE.relative_to(PROJECT)),
    **{k:REF_SETTINGS[k] for k in ('development_start','development_end','pressure_end','dev_market_sha256','july_market_sha256')})


def state_prediction(x,f,state):
    m=ResidualModel('contract');m.parameters=state['parameters']
    return m.predict(f,x[BASE_FEATURE],x.option_type)[0]


def h0_state(model):
    result={}
    for side,p in model.parameters.items():
        v={k:float(p[k]) for k in ('probability','severity')}
        if 'model' in p:v.update(mean=p['scaler'].mean_.tolist(),scale=p['scaler'].scale_.tolist(),
            coef=p['model'].coef_[0].tolist(),intercept=float(p['model'].intercept_[0]))
        result[str(side)]=v
    return result


@threadpool_limits.wrap(limits=1)
def predict_period(pool,signals,start,end,monthly):
    start,end=pd.Timestamp(start),pd.Timestamp(end);warmup=pd.Timestamp(SETTINGS['warmup_start'])
    if (start.tzinfo is None or end.tzinfo is None or start>=end or start<warmup
        or (start-warmup)%pd.Timedelta(days=7)!=pd.Timedelta(0)
        or not pool.X.index.is_unique or any(not pool.X.index.equals(v.index) for v in (pool.y,pool.metadata,signals))):
        raise ValueError('Invalid weekly pool or grid')
    f=contract_features(pool.X);created=pool.metadata.created_at_utc;closed=pool.metadata.closed_at_utc
    frames=[];states={};cache=[]
    for origin in pd.date_range(warmup,end,freq='7D',inclusive='left'):
        stop=min(origin+pd.Timedelta(days=7),end)
        q=pool.X.index[created.ge(origin)&created.lt(stop)];train=history(pool,origin,30)
        model=ResidualModel('contract').fit(f.loc[train],pool.X.loc[train,BASE_FEATURE],pool.y.loc[train],
            pool.X.loc[train,'option_type'],created.loc[train])
        if not len(q):continue
        x=pool.X.loc[q];h,hs=model.predict(f.loc[q],x[BASE_FEATURE],x.option_type)
        query=pd.DataFrame(dict(h=h,jump=signals.loc[q,'jump'],streak=signals.loc[q,'streak'],side=x.option_type,
            quantity=x.quantity_btc,created=created.loc[q],forecast_origin=origin),index=q)
        if origin>=start:
            previous=pd.concat(cache) if cache else query.iloc[:0]
            keep=previous.created.ge(origin-pd.Timedelta(days=30))&closed.reindex(previous.index).lt(origin)
            past=previous.loc[keep].copy();past['actual']=pool.y.reindex(past.index)
            if not ((past.forecast_origin<=past.created)&(past.created<origin)).all():raise ValueError('Non-forward calibration')
            extra=Extras().fit(past);pred=extra.predict(query)
            h0=CalibrationModel('hurdle_base').fit(probability_frame(pool.X.loc[train]),pool.y.loc[train],created.loc[train])
            pred.update(official_raw=x[BASE_FEATURE],contract_weekly=h,h0=h0.predict_raw(probability_frame(x)))
            monthly_pred=pd.Series(index=q,dtype=float);threshold=pd.Series(index=q,dtype=float);source=pd.Series(index=q,dtype=str)
            for month in sorted(created.loc[q].dt.strftime('%Y-%m').unique()):
                mo=pd.Timestamp(month+'-01',tz='UTC');ids=q[created.loc[q].dt.strftime('%Y-%m').eq(month)]
                monthly_pred.loc[ids]=state_prediction(pool.X.loc[ids],f.loc[ids],monthly[mo.isoformat()]['models']['contract_residual'])
                tv,sv=tail_context(pool,history(pool,mo),ids);threshold.loc[ids]=tv;source.loc[ids]=sv
            pred['contract_monthly']=monthly_pred
            common=pd.DataFrame(dict(day=created.loc[q].dt.floor('D'),month=created.loc[q].dt.strftime('%Y-%m'),
                side=x.option_type.astype(int),tenor=x.product_duration_seconds.astype(int),quantity=x.quantity_btc,
                tail_threshold=threshold,tail_source=source,origin=origin.isoformat(),h=h,
                jump=signals.loc[q,'jump'],streak=signals.loc[q,'streak'],
                jump_available=signals.loc[q,'jump_available'],streak_available=signals.loc[q,'streak_available']),index=q)
            for name in MODELS:
                values=np.asarray(pred[name]);support=np.ones(len(q),bool)
                if name=='contract_weekly':support=hs.to_numpy()
                elif name.startswith('level_'):support[:]=extra.state['level_supported']
                elif name in OVERLAYS:
                    key=name.removeprefix('uniform_')
                    support=x.option_type.map({s:extra.state[key][str(s)]['supported'] for s in (0,1)}).to_numpy()
                if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid predictions')
                frames.append(common.assign(model=name,prediction=values,multiplier=values/x[BASE_FEATURE],
                    supported=support,extra_amount=values-h))
            states[origin.isoformat()]=dict(stop=stop.isoformat(),history_orders=len(train),query_orders=len(q),
                latest_training_close=closed.loc[train].max().isoformat(),oos_orders=len(past),
                latest_oos_close=closed.loc[past.index].max().isoformat(),oos_strict_forward=True,
                contract=model.state(),h0=h0_state(h0),extras=extra.state)
        # No outcome stored or inspected for the current block; labels are attached only at a later origin.
        cache.append(query)
    return pd.concat(frames),states


def score(pool,predicted,end):
    ref=predicted.loc[predicted.model.eq('official_raw')]
    if not ref.index.is_unique:raise ValueError('Duplicate quotes')
    known=pool.metadata.closed_at_utc.reindex(ref.index).lt(pd.Timestamp(end));ids=ref.index[known]
    frame=predicted.loc[predicted.index.isin(ids)].copy();frame['actual']=pool.y.reindex(frame.index).to_numpy()
    frame['tail_group']=np.where(frame.tail_threshold.isna(),'unavailable',np.where(frame.actual>frame.tail_threshold,'tail','body'))
    for name in MODELS:
        if not frame.loc[frame.model.eq(name)].index.equals(ids):raise ValueError('Unequal cohorts')
    return frame,dict(query_orders=len(ref),scored_orders=len(ids),labels_unavailable=int((~known).sum()))


def run(run_id):
    from .weekly_residual_statistics import tables
    settings=json.loads(CONFIG.read_text())
    if settings!=SETTINGS:raise ValueError('Fixed settings changed')
    guard=AccessGuard();sys.addaudithook(guard.check)
    root=run_destination(PROJECT/'outputs/model_studies',run_id);root.mkdir(parents=True);(root/'provenance').mkdir()
    began=perf_counter();canonical=canonical_identity();development,_=development_snapshot()
    reference_manifest=json.loads((REFERENCE/'provenance/manifest.json').read_text())
    if reference_manifest['status']!='complete':raise ValueError('Incomplete monthly reference')
    for name,digest in reference_manifest['artifacts'].items():
        if file_digest(REFERENCE/name)!=digest:raise ValueError('Reference modified')
    if canonical!=json.loads((REFERENCE/'provenance/preflight.json').read_text())['canonical']:raise ValueError('Canonical changed')
    protected={str(p.relative_to(PROJECT)):file_digest(p) for p in REFERENCE.rglob('*') if p.is_file()}
    paths={CONFIG,PLAN,PROJECT/'scripts/verify_weekly_residual.py'}
    for folder in ('pp_models','pp_data'):paths.update((PROJECT/folder).glob('*.py'))
    paths.update((PROJECT/'tests').glob('test_weekly_residual*.py'))
    sources={str(p.relative_to(PROJECT)):file_digest(p) for p in sorted(paths)}
    with ZipFile(root/'provenance/source.zip','w',ZIP_DEFLATED) as z:
        for name in sources:z.write(PROJECT/name,name)
    atomic_json(root/'provenance/config.json',settings)
    atomic_json(root/'provenance/preflight.json',dict(sources=sources,protected=protected,canonical=canonical,development_data=development))
    phases={}
    try:
        for phase,start,end in [('development',settings['development_start'],settings['development_end']),
                                ('pressure',settings['development_end'],settings['pressure_end'])]:
            clock=perf_counter()
            if phase=='development':
                pool=load_validation_pool({'cohort':'benchmark','use_base':True});raw,info=read_market_prefix(DEFAULT_DATA/'btc_market_history.csv',end)
                expected=settings['dev_market_sha256']
            else:
                if not (root/'provenance/development_complete.json').exists():raise RuntimeError('Development incomplete')
                guard.allow_pressure=True;july=canonical_identity(include_test=True,expected_manifest=canonical['manifest_sha256'])
                if july!=reference_manifest['phases']['pressure']['canonical']:raise ValueError('July changed')
                pool=load_pool('test',{'cohort':'benchmark','use_base':True});raw,info=read_test_market_prefix(DEFAULT_DATA/'btc_market_history.csv')
                expected=settings['july_market_sha256']
            if info['prefix_sha256']!=expected:raise ValueError('Market prefix changed')
            signals=signal_features(pool.X,pool.metadata.created_at_utc,raw)
            references=json.loads((REFERENCE/f'provenance/{phase}_parameters.json').read_text())
            predicted,states=predict_period(pool,signals,start,end,references);scored,coverage=score(pool,predicted,end)
            result=tables(scored,start,end)
            write_tables(root,phase,result);atomic_json(root/f'provenance/{phase}_parameters.json',states)
            phases[phase]=dict(coverage=coverage,market=info,seconds=perf_counter()-clock,update_origins=len(states),
                weekly_label_feedback=True,within_week_updates=0,monthly_reference_fits=0,
                historical_contract_refits=int((pd.Timestamp(start)-pd.Timestamp(settings['warmup_start'])).days/7),
                missing_jump_signal_orders=int((~scored.loc[scored.model.eq('official_raw'),'jump_available']).sum()),
                missing_streak_signal_orders=int((~scored.loc[scored.model.eq('official_raw'),'streak_available']).sum()))
            if phase=='pressure':phases[phase]['canonical']=july
            atomic_json(root/f'provenance/{phase}_complete.json',phases[phase]);print(json.dumps({'phase':phase,**phases[phase]}),flush=True)
        if development_snapshot()[0]!=development or canonical_identity(include_test=True,expected_manifest=canonical['manifest_sha256'])!=july:
            raise ValueError('Canonical changed')
        for name,digest in {**sources,**protected}.items():
            if file_digest(PROJECT/name)!=digest:raise ValueError('Protected file changed: '+name)
        artifacts={str(p.relative_to(root)):file_digest(p) for p in root.rglob('*') if p.is_file()}
        atomic_json(root/'provenance/manifest.json',dict(status='complete',producer=settings['version'],phases=phases,artifacts=artifacts,
            elapsed_seconds=perf_counter()-began,environment=snapshot()['environment'],max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            official_validation_open_attempts=guard.official_attempts,premature_test_attempts=guard.premature_test_attempts))
    except Exception as e:
        atomic_json(root/'provenance/failure.json',dict(error=repr(e),phases=phases));raise
    return root


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);args=parser.parse_args();print(run(args.run_id))

if __name__=='__main__':main()
