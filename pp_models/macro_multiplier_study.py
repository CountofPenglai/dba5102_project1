"""Monthly no-feedback announcement correction; old July opens only after selection."""
import argparse
from datetime import datetime, timezone
import json
import resource
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT, DEFAULT_DATA
from pp_data.features import BASE_FEATURE
from .base_multiplier_study import AccessGuard, history, verify_reference, write_tables, protected_hashes
from .calibration_models import CalibrationModel
from .data import load_pool
from .directional_features import augment_pool, read_market_prefix
from .macro_multiplier import MacroMultiplier
from .macro_multiplier_features import build_features, BASIS
from .macro_multiplier_statistics import tables
from .put_final_io import read_test_market_prefix
from .runs import run_destination, snapshot
from .scoped_data import load_validation_pool, development_snapshot
from .study_storage import atomic_json, file_digest
from .two_stage_study import probability_frame, tail_context, aggregate

CONFIG=PROJECT/'config/macro_multiplier_v1.json'


def h0_parameters(model):
    rows=[]
    for side,p in model.parameters.items():
        row={'side':side,'probability':p['probability'],'severity':p['severity']}
        if 'model' in p:
            row.update(mean=p['scaler'].mean_.tolist(),scale=p['scaler'].scale_.tolist(),
                       coef=p['model'].coef_[0].tolist(),intercept=float(p['model'].intercept_[0]))
        rows.append(row)
    return rows


def pool_features(pool, calendar):
    return build_features(pd.DataFrame({'created':pool.metadata.created_at_utc,
        'remaining_seconds':pool.X.remaining_seconds,'option_type':pool.X.option_type}),calendar)


def predict_period(pool, calendar, start, end, specs):
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if (start.tzinfo is None or end.tzinfo is None or start>=end or start!=start.floor('D')
            or not pool.X.index.equals(pool.y.index) or not pool.X.index.equals(pool.metadata.index)
            or not pool.X.index.is_unique):raise ValueError('Invalid no-feedback period or pool')
    created,closed=pool.metadata.created_at_utc,pool.metadata.closed_at_utc
    features=pool_features(pool,calendar)
    # Query selection uses creation time only; settlement belongs to later scoring.
    ids=pool.X.index[created.ge(start)&created.lt(end)]
    if not len(ids):raise ValueError('Empty evaluation period')
    outputs,fits,cohorts=[],[],[]
    for origin in pd.date_range(start,end,freq='MS',inclusive='left'):
        stop=min(end,origin+pd.offsets.MonthBegin(1))
        q=ids[created.loc[ids].ge(origin)&created.loc[ids].lt(stop)]
        if not len(q):continue
        train=history(pool,origin); ht=history(pool,origin,30)
        if len(ht)<100:raise ValueError('Insufficient original H0 history')
        h0=CalibrationModel('hurdle_base').fit(probability_frame(pool.X.loc[ht]),pool.y.loc[ht],created.loc[ht])
        fits.append({'model':'h0','origin':origin.isoformat(),'parameters':h0_parameters(h0)})
        threshold,source=tail_context(pool,train,q)
        f=features.loc[q]; elapsed=(created.loc[q]-f.event_time).dt.total_seconds()/60
        phase=np.select([elapsed.lt(-60),elapsed.lt(0),elapsed.lt(15),elapsed.lt(60),elapsed.lt(240)],
                        ['pre_far','pre60','post0_15','post15_60','post60_240'],default='outside')
        common=pd.DataFrame({'day':created.loc[q].dt.floor('D'),'month':created.loc[q].dt.strftime('%Y-%m'),
            'side':pool.X.loc[q,'option_type'].astype(int),'tenor':pool.X.loc[q,'product_duration_seconds'].astype(int),
            'quantity':pool.X.loc[q,'quantity_btc'],'tail_threshold':threshold,'tail_source':source,
            'origin':origin.isoformat(),'latest_training_close':closed.loc[train].max().isoformat(),
            'event_id':f.event_id,'calendar_status':f.calendar_status,'phase':phase,
            'evaluation_end':end.isoformat()},index=q)
        base=pool.X.loc[q,BASE_FEATURE]
        outputs.append(common.assign(model='official_raw',prediction=base,multiplier=1.,supported=True,
            fallback_reason='reference',shared_adjustment=0.,direction_adjustment=0.))
        hp=h0.predict_raw(probability_frame(pool.X.loc[q]))
        outputs.append(common.assign(model='h0',prediction=hp,multiplier=hp/base,supported=True,
            fallback_reason='reference',shared_adjustment=0.,direction_adjustment=0.))
        for name,strength in specs.items():
            model=MacroMultiplier(strength).fit(features.loc[train],pool.X.loc[train,BASE_FEATURE],pool.y.loc[train],origin)
            d=model.predict_diagnostics(f)
            outputs.append(common.assign(model=name,prediction=base*d.multiplier,**{c:d[c] for c in d}))
            fits.append({'model':name,**model.diagnostics()})
        for role,h in [('multiplier_and_tail',train),('h0',ht)]:
            cohorts.append({'origin':origin.isoformat(),'role':role,'history_orders':len(h),'query_orders':len(q),
                'history_earliest_create':created.loc[h].min().isoformat(),'history_latest_close':closed.loc[h].max().isoformat(),
                'no_feedback':True})
    return pd.concat(outputs),fits,pd.DataFrame(cohorts)


def score(pool, predicted):
    closed=pool.metadata.closed_at_utc.reindex(predicted.index)
    settled=closed.lt(pd.to_datetime(predicted.evaluation_end,utc=True)).to_numpy()
    f=predicted.loc[settled].copy();f['actual']=pool.y.reindex(f.index).to_numpy()
    if not np.isfinite(f.actual).all():raise ValueError('Scoring needs finite historical labels')
    f['tail_group']=np.where(f.tail_threshold.isna(),'unavailable',np.where(f.actual.gt(f.tail_threshold),'tail','body'))
    for _,g in f.groupby('model'):
        if not g.index.is_unique:raise ValueError('Duplicate scoring IDs')
    return f


def choose(macro,specs):
    m=macro.set_index('model');eligible=[n for n in specs if m.loc[n,'macro_mae_per_btc']<=m.loc['official_raw','macro_mae_per_btc']]
    name=min(eligible or specs,key=lambda n:(m.loc[n,'macro_rmse_per_btc'],-specs[n]))
    return {'grid_name':name,'strength':specs[name],'mae_vs_base_pass':bool(eligible),'scope':'2026-04_to_2026-06_only'}


def run(run_id):
    guard=AccessGuard();sys.addaudithook(guard.check)
    settings=json.loads(CONFIG.read_text())
    if (settings['strengths']!=[1.,10.,100.] or settings['direction_penalty_ratio']!=10
            or settings['shared_support']!={'events':5,'orders':30,'months':2}
            or settings['direction_support']!={'events':8,'orders':60,'months':3}
            or [settings[k] for k in ['total_cap','direction_cap','pre_minutes','post_minutes','half_life_minutes']]!=[.05,.02,60,240,60]):
        raise ValueError('Fixed experiment settings changed; version a new study')
    calendar_path=PROJECT/settings['calendar']
    if file_digest(calendar_path)!=settings['calendar_sha256']:raise ValueError('Changed calendar evidence')
    calendar=json.loads(calendar_path.read_text())
    root=run_destination(PROJECT/'outputs/model_studies',run_id);root.mkdir(parents=True);(root/'provenance').mkdir()
    started=perf_counter();protected=protected_hashes()
    for name in ('base_multiplier_20261014',):
        for p in (PROJECT/'outputs/model_studies'/name).rglob('*'):
            if p.is_file():protected[str(p.relative_to(PROJECT))]=file_digest(p)
    sources=list((PROJECT/'pp_models').glob('*.py'))+list((PROJECT/'pp_data').glob('*.py'))+list((PROJECT/'pp_calendar').glob('*.py'))
    sources += [CONFIG,calendar_path,PROJECT/'docs/plans/2026-10-14-macro-multiplier-execution.md']
    hashes={str(p.relative_to(PROJECT)):file_digest(p) for p in sources}
    with ZipFile(root/'provenance/source_code.zip','w',ZIP_DEFLATED) as z:
        for p in sources:z.write(p,str(p.relative_to(PROJECT)))
    atomic_json(root/'provenance/started.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':hashes,'protected_sha256':protected,'environment':snapshot()['environment']})
    atomic_json(root/'provenance/settings.json',settings)
    identity,_=development_snapshot()
    market,dev_market=read_market_prefix(DEFAULT_DATA/'btc_market_history.csv',settings['development_end'])
    if dev_market['prefix_sha256']!=settings['dev_market_sha256']:raise ValueError('Changed development market prefix')
    pool,_=augment_pool(load_validation_pool({'cohort':'benchmark','use_base':True}),market)
    specs={f'announcement_l{x:g}':x for x in settings['strengths']}
    print('Development: three frozen monthly announcement configurations',flush=True)
    pred,fits,cohorts=predict_period(pool,calendar,settings['development_start'],settings['development_end'],specs)
    dev=score(pool,pred);summary,_,macro=aggregate(dev)
    checks=verify_reference(dev,'development')
    write_tables(root,'development_grid',{'summary':summary,'macro_monthly':macro})
    selection=choose(macro,specs)
    atomic_json(root/'provenance/selection.json',{'selected':selection,'selected_at_utc':datetime.now(timezone.utc).isoformat(),
        'old_july_opened':False,'new_official_validation_opened':False,'development_identity':identity,'development_market':dev_market})
    lock=file_digest(root/'provenance/selection.json')
    chosen=dev.loc[dev.model.isin(['official_raw','h0',selection['grid_name']])].copy()
    chosen['model']=chosen.model.replace({selection['grid_name']:'announcement'})
    write_tables(root,'development',tables(chosen,settings['development_start'],settings['development_end'],development_eligible=selection['mae_vs_base_pass']))
    atomic_json(root/'provenance/development_fits.json',fits);cohorts.to_csv(root/'provenance/development_cohorts.csv',index=False)
    guard.allow_pressure=True
    atomic_json(root/'provenance/pressure_started.json',{'utc':datetime.now(timezone.utc).isoformat(),'selection_sha256':lock})
    print('Selection locked; old July pressure review, no new validation',flush=True)
    market,july_market=read_test_market_prefix(DEFAULT_DATA/'btc_market_history.csv')
    if july_market['prefix_sha256']!=settings['july_market_sha256']:raise ValueError('Changed old July market prefix')
    full,_=augment_pool(load_pool('test',{'cohort':'benchmark','use_base':True}),market)
    pred,jfits,jcohorts=predict_period(full,calendar,settings['development_end'],settings['pressure_end'],{'announcement':selection['strength']})
    july=score(full,pred);jchecks=verify_reference(july,'test')
    write_tables(root,'july_pressure',tables(july,settings['development_end'],settings['pressure_end'],development_eligible=selection['mae_vs_base_pass']))
    atomic_json(root/'provenance/pressure_fits.json',jfits);jcohorts.to_csv(root/'provenance/pressure_cohorts.csv',index=False)
    threshold=pd.concat([f[['origin','side','tenor','tail_threshold','tail_source']].drop_duplicates().assign(period=p)
                         for p,f in [('development',chosen),('july_pressure',july)]])
    threshold.to_csv(root/'provenance/tail_thresholds.csv',index=False)
    for name,digest in (protected|hashes).items():
        if file_digest(PROJECT/name)!=digest:raise ValueError('Source or protected artifact changed: '+name)
    if file_digest(root/'provenance/selection.json')!=lock:raise ValueError('Selection changed after July release')
    execution={'elapsed_seconds':perf_counter()-started,'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),'guard':vars(guard),
        'development_reference_groups':checks,'july_reference_groups':jchecks,'july_market':july_market,
        'development_orders':int(chosen.model.eq('official_raw').sum()),'july_orders':int(july.model.eq('official_raw').sum()),
        'protected_files':len(protected),'selection_unchanged':True}
    atomic_json(root/'provenance/execution.json',execution)
    atomic_json(root/'provenance/manifest.json',{'status':'complete','files':{str(p.relative_to(root)):file_digest(p) for p in root.rglob('*') if p.is_file()}})
    print(json.dumps({'output':str(root),'selection':selection,**execution},ensure_ascii=False),flush=True)
    return root


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True)
    with threadpool_limits(limits=1):run(parser.parse_args().run_id)
