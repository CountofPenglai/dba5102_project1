"""Run development selection, then authorized old-July pressure checks. No new validation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT, DEFAULT_OUTPUT, DEFAULT_DATA
from pp_data.features import BASE_FEATURE
from .base_multiplier import BaseMultiplier, INPUTS, FAMILIES
from .base_multiplier_statistics import tables
from .calibration_models import CalibrationModel
from .data import load_pool
from .directional_features import augment_pool, read_market_prefix
from .put_final_io import read_test_market_prefix
from .runs import run_destination, snapshot
from .scoped_data import load_validation_pool, development_snapshot
from .study_storage import atomic_json, file_digest
from .two_stage_study import probability_frame, tail_context, aggregate

CONFIG=PROJECT/'config/base_multiplier_v1.json'
OLD_RUN=PROJECT/'outputs/model_studies/put_final_july_20261014'


def load_settings():
    settings=json.loads(CONFIG.read_text())
    expected={'version':'base_multiplier_v1','families':list(FAMILIES),'lambdas':[.1,1.,10.],
        'primary_delta':.1,'sensitivity_delta':.2,'minimum_put_orders':100,'minimum_put_days':10,
        'minimum_tenor_orders':30,'minimum_tenor_days':5,'standardized_clip':3.,
        'development_start':'2026-04-01T00:00:00Z','development_end':'2026-07-01T00:00:00Z',
        'pressure_end':'2026-07-31T17:11:48Z','long_freeze_start':'2026-05-01T00:00:00Z',
        'h0_history_days':30,'tail_quantile':.9,'tail_minimum_history':30,'risk_tolerance':1.02,
        'bootstrap_blocks':[3,7,14],'bootstrap_repeats':2000,'minimum_valid_repeats':1900,
        'selection':'min_macro_rmse_among_mae_le_base_else_failed_best_rmse_tie_stronger_lambda',
        'dev_market_sha256':'21aa8f91a3b393452c93360cefc16b39255d64219ba3b67c3474b912ee5ff38b',
        'july_market_sha256':'cbf5d491d167ac377beb844dc9794e4f5e9986f72859035426f7789e67924b50'}
    if settings!=expected:raise ValueError('Multiplier fixed settings changed; version a new study instead')
    return settings


class AccessGuard:
    def __init__(self):
        self.allow_pressure=False
        self.official_attempts=0
        self.premature_test_attempts=0
        self.test_opens=0

    def check(self,event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        path=Path(args[0].decode() if isinstance(args[0],bytes) else args[0])
        if path.name=='pp_transactions_validation.csv':
            self.official_attempts+=1
            raise PermissionError('New official validation is forbidden')
        if '/datasets/test/' in path.as_posix() and path.suffix=='.csv':
            if not self.allow_pressure:
                self.premature_test_attempts+=1
                raise PermissionError('Old July is locked until development selection is written')
            self.test_opens+=1


def history(pool,origin,days=None):
    created,closed=pool.metadata.created_at_utc,pool.metadata.closed_at_utc
    mask=created.lt(origin)&closed.lt(origin)
    if days is not None:mask &= created.ge(origin-pd.Timedelta(days=days))
    return pool.X.index[mask]


def predict_period(pool,start,end,specs,continuous=False):
    """Forecast without consulting any target at/after each fitted origin."""
    start,end=pd.Timestamp(start),pd.Timestamp(end)
    if (start.tzinfo is None or end.tzinfo is None or start>=end
            or start!=start.floor('D') or not pool.X.index.equals(pool.y.index)
            or not pool.X.index.equals(pool.metadata.index) or not pool.X.index.is_unique):
        raise ValueError('Invalid evaluation period or pool alignment')
    created,closed=pool.metadata.created_at_utc,pool.metadata.closed_at_utc
    eligible=created.ge(start)&created.lt(end)&closed.lt(end)
    ids=pool.X.index[eligible]
    if len(ids)==0:raise ValueError('Empty prediction period')
    origins=[start] if continuous else list(pd.date_range(start,end,freq='MS',inclusive='left'))
    outputs,fits,cohorts=[],[],[]
    for origin in origins:
        stop=end if continuous else min(end,origin+pd.offsets.MonthBegin(1))
        query=ids[created.loc[ids].ge(origin)&created.loc[ids].lt(stop)]
        if len(query)==0:continue
        train=history(pool,origin)
        htrain=history(pool,origin,30)
        if len(htrain)<100:raise ValueError('Insufficient original H0 history')
        h0=CalibrationModel('hurdle_base').fit(probability_frame(pool.X.loc[htrain]),pool.y.loc[htrain],created.loc[htrain])
        threshold,source=tail_context(pool,train,query)
        common=pd.DataFrame({'day':created.loc[query].dt.floor('D'),
            'month':created.loc[query].dt.strftime('%Y-%m'),'side':pool.X.loc[query,'option_type'].astype(int),
            'tenor':pool.X.loc[query,'product_duration_seconds'].astype(int),'quantity':pool.X.loc[query,'quantity_btc'],
            'tail_threshold':threshold,'tail_source':source,'origin':origin.isoformat(),
            'latest_training_close':closed.loc[train].max().isoformat()},index=query)
        base=pool.X.loc[query,BASE_FEATURE]
        outputs.append(common.assign(model='official_raw',prediction=base,multiplier=1.,supported=True,fallback_reason='reference'))
        hp=h0.predict_raw(probability_frame(pool.X.loc[query]))
        outputs.append(common.assign(model='h0',prediction=hp,multiplier=hp/base,supported=True,fallback_reason='reference'))
        for name,(family,strength,delta) in specs.items():
            model=BaseMultiplier(family,strength,delta).fit(pool.X.loc[train,INPUTS],pool.X.loc[train,BASE_FEATURE],
                                                         pool.y.loc[train],created.loc[train])
            diag=model.predict_diagnostics(pool.X.loc[query,INPUTS])
            outputs.append(common.assign(model=name,prediction=base*diag.multiplier,
                multiplier=diag.multiplier,supported=diag.supported,fallback_reason=diag.fallback_reason))
            fits.append({'origin':origin.isoformat(),'model':name,**model.diagnostics()})
        for name,hist in [('multiplier_and_tail',train),('h0',htrain)]:
            cohorts.append({'origin':origin.isoformat(),'role':name,'history_orders':len(hist),'query_orders':len(query),
                'history_earliest_create':created.loc[hist].min().isoformat(),
                'history_latest_close':closed.loc[hist].max().isoformat(),
                'no_feedback':True})
    return pd.concat(outputs),fits,pd.DataFrame(cohorts)


def score(pool,predicted):
    frame=predicted.copy()
    frame['actual']=pool.y.reindex(frame.index).to_numpy()
    if not np.isfinite(frame.actual).all():raise ValueError('Scoring requires finite historical targets')
    frame['tail_group']=np.where(frame.tail_threshold.isna(),'unavailable',
                                np.where(frame.actual.gt(frame.tail_threshold),'tail','body'))
    for model in frame.model.unique():
        part=frame.loc[frame.model.eq(model)]
        if not part.index.is_unique:raise ValueError('Duplicate scoring IDs per model')
        if model not in ('h0','official_raw'):
            np.testing.assert_array_equal(part.loc[part.side.eq(0),'multiplier'],1.)
    return frame


def choose_specs(macro,specs):
    lookup=macro.set_index('model');base_mae=lookup.loc['official_raw','macro_mae_per_btc']
    selected={}
    for family in FAMILIES:
        names=[n for n,s in specs.items() if s[0]==family]
        if not names:continue
        eligible=[n for n in names if lookup.loc[n,'macro_mae_per_btc']<=base_mae]
        name=min(eligible or names,key=lambda n:(lookup.loc[n,'macro_rmse_per_btc'],-specs[n][1]))
        selected[family]={'grid_name':name,'strength':specs[name][1],'delta':specs[name][2],
            'mae_vs_base_pass':bool(eligible),'selection_scope':'2026-04_to_2026-06_only'}
    return selected


def write_tables(root,label,values):
    dest=root/'metrics'/label;dest.mkdir(parents=True,exist_ok=True)
    for name,table in values.items():table.to_csv(dest/(name+'.csv'),index=False,float_format='%.12g')


def multiplier_summary(frame):
    rows=[]
    for (month,model,side),p in frame.groupby(['month','model','side']):
        if model in ('h0','official_raw'):continue
        m=p.multiplier
        row={'month':month,'model':model,'side':side,'orders':len(p),'supported_orders':int(p.supported.sum()),
             'mean_multiplier':m.mean(),'min_multiplier':m.min(),'p10':m.quantile(.1),
             'median':m.median(),'p90':m.quantile(.9),'max_multiplier':m.max(),
             'raised_orders':int(m.gt(1+1e-10).sum()),'lowered_orders':int(m.lt(1-1e-10).sum())}
        row.update({f'fallback_{reason}':int(p.fallback_reason.eq(reason).sum())
                    for reason in ('global_support','tenor_support','outside_training_state','call_identity')})
        rows.append(row)
    return pd.DataFrame(rows)


def verify_reference(frame,period):
    """Existing frozen Base/H0 predictions must reproduce, independent of new tail grouping."""
    expected=pd.read_csv(OLD_RUN/f'metrics/{period}_frozen/summary.csv',dtype={'value':str})
    summary,_,_=aggregate(frame)
    groups=('overall','direction')
    actual=summary.loc[summary.model.isin(['official_raw','h0'])&summary.group.isin(groups)]
    ref=expected.loc[expected.model.isin(['official_raw','h0'])&expected.group.isin(groups)]
    keys=['month','model','group','value']
    columns=['orders','rmse_per_btc','mae_per_btc','actual_total_usdc','predicted_total_usdc',
             'total_bias_usdc','gross_underprediction_usdc','gross_overprediction_usdc']
    a,b=actual.set_index(keys).sort_index(),ref.set_index(keys).sort_index()
    if not a.index.equals(b.index):raise ValueError('Reference comparison cohort mismatch')
    np.testing.assert_allclose(a[columns],b[columns],rtol=1e-9,atol=1e-7)
    return len(a)


def protected_hashes():
    # Reading hash manifests does not open July datasets or new official inputs.
    paths=[DEFAULT_OUTPUT/'manifest.json']
    paths+=list(OLD_RUN.rglob('*'))
    paths+=list((PROJECT/'outputs/diagnostics/base_regimes_20261014').rglob('*'))
    return {str(p.relative_to(PROJECT)):file_digest(p) for p in paths if p.is_file()}


def run(run_id):
    guard=AccessGuard();sys.addaudithook(guard.check)
    settings=load_settings()
    root=run_destination(PROJECT/'outputs/model_studies',run_id)
    root.mkdir(parents=True);(root/'provenance').mkdir()
    started=perf_counter()
    protected=protected_hashes();identity=snapshot()
    source_paths=list((PROJECT/'pp_models').glob('*.py'))+list((PROJECT/'pp_data').glob('*.py'))+list((PROJECT/'pp_audit').glob('*.py'))
    source_paths += [CONFIG,PROJECT/'docs/plans/2026-10-14-base-multiplier-execution.md',PROJECT/'requirements.txt',PROJECT/'requirements-models.txt']
    sources={str(p.relative_to(PROJECT)):file_digest(p) for p in source_paths}
    with ZipFile(root/'provenance/source_code.zip','w',ZIP_DEFLATED) as z:
        for p in source_paths:z.write(p,str(p.relative_to(PROJECT)))
    atomic_json(root/'provenance/settings.json',settings)
    atomic_json(root/'provenance/started.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':sources,'environment':identity['environment'],'protected_sha256':protected})
    dev_identity,_=development_snapshot()
    market_path=DEFAULT_DATA/'btc_market_history.csv'
    market,dev_market=read_market_prefix(market_path,settings['development_end'])
    if dev_market['prefix_sha256']!=settings['dev_market_sha256']:raise ValueError('Changed development market prefix')
    pool,feature_info=augment_pool(load_validation_pool({'cohort':'benchmark','use_base':True}),market)
    specs={f'{f}_l{s:g}':(f,s,.1) for f in settings['families'] for s in settings['lambdas']}
    print('Development: 9 fixed multiplier configurations, monthly frozen fits',flush=True)
    pred,fits,cohorts=predict_period(pool,settings['development_start'],settings['development_end'],specs)
    dev=score(pool,pred);summary,_,macro=aggregate(dev)
    reference_dev=verify_reference(dev,'development')
    write_tables(root,'development_grid',{'summary':summary,'macro_monthly':macro})
    selected=choose_specs(macro,specs)
    selection={'selected':selected,'selected_at_utc':datetime.now(timezone.utc).isoformat(),
        'old_july_opened':False,'new_official_validation_opened':False,'development_identity':dev_identity,
        'development_market':dev_market,'development_feature_check':feature_info}
    atomic_json(root/'provenance/selection.json',selection)
    lock=file_digest(root/'provenance/selection.json')
    chosen={f:(f,s['strength'],.1) for f,s in selected.items()}
    eligibility={f:s['mae_vs_base_pass'] for f,s in selected.items()}
    wide_eligibility={f+'_wide':False for f in selected} # Sensitivity never promotes itself.
    chosen_pred=pred.loc[pred.model.isin(['official_raw','h0',*[s['grid_name'] for s in selected.values()]])].copy()
    chosen_pred['model']=chosen_pred.model.replace({s['grid_name']:f for f,s in selected.items()})
    selected_dev=score(pool,chosen_pred)
    write_tables(root,'development',tables(selected_dev,settings['development_start'],settings['development_end'],list(chosen),development_eligibility=eligibility))
    write_tables(root,'development',{'multipliers':multiplier_summary(selected_dev)})
    sensitivity={f+'_wide':(f,s['strength'],.2) for f,s in selected.items()}
    sens_pred,sens_fits,sens_cohorts=predict_period(pool,settings['development_start'],settings['development_end'],sensitivity)
    sens=score(pool,sens_pred)
    write_tables(root,'development_sensitivity',tables(sens,settings['development_start'],settings['development_end'],list(sensitivity),False,wide_eligibility))
    write_tables(root,'development_sensitivity',{'multipliers':multiplier_summary(sens)})
    atomic_json(root/'provenance/development_fits.json',fits+sens_fits)
    cohorts.to_csv(root/'provenance/development_cohorts.csv',index=False)
    # Release only after selection is durable. No selection is run below this line.
    if file_digest(root/'provenance/selection.json')!=lock:raise ValueError('Selection changed')
    guard.allow_pressure=True
    atomic_json(root/'provenance/pressure_started.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'selection_sha256':lock,'reason':'User-authorized old July pressure review; not independent test'})
    print('Selection locked; reading old July only for pressure review',flush=True)
    market,july_market=read_test_market_prefix(market_path)
    if july_market['prefix_sha256']!=settings['july_market_sha256']:raise ValueError('Changed old-July market prefix')
    full,full_feature_info=augment_pool(load_pool('test',{'cohort':'benchmark','use_base':True}),market)
    test_pred,test_fits,test_cohorts=predict_period(full,settings['development_end'],settings['pressure_end'],chosen|sensitivity)
    test=score(full,test_pred)
    reference_july=verify_reference(test,'test')
    primary=test.loc[test.model.isin(['official_raw','h0',*chosen])]
    write_tables(root,'july_pressure',tables(primary,settings['development_end'],settings['pressure_end'],list(chosen),development_eligibility=eligibility))
    write_tables(root,'july_pressure',{'multipliers':multiplier_summary(primary)})
    wider=test.loc[test.model.isin(['official_raw','h0',*sensitivity])]
    write_tables(root,'july_sensitivity',tables(wider,settings['development_end'],settings['pressure_end'],list(sensitivity),False,wide_eligibility))
    write_tables(root,'july_sensitivity',{'multipliers':multiplier_summary(wider)})
    print('Continuous May-to-July freeze; no refitting inside this period',flush=True)
    long_pred,long_fits,long_cohorts=predict_period(full,settings['long_freeze_start'],settings['pressure_end'],chosen,True)
    long=score(full,long_pred)
    write_tables(root,'continuous_freeze',tables(long,settings['long_freeze_start'],settings['pressure_end'],list(chosen),development_eligibility=eligibility))
    write_tables(root,'continuous_freeze',{'multipliers':multiplier_summary(long)})
    atomic_json(root/'provenance/pressure_fits.json',test_fits+long_fits)
    pd.concat([test_cohorts.assign(period='july'),long_cohorts.assign(period='continuous')]).to_csv(root/'provenance/pressure_cohorts.csv',index=False)
    all_periods={'development':selected_dev,'july_pressure':primary,'continuous_freeze':long}
    threshold_rows=[]
    for label,fr in all_periods.items():
        unique=fr[['origin','side','tenor','tail_threshold','tail_source']].drop_duplicates()
        threshold_rows.append(unique.assign(period=label))
    pd.concat(threshold_rows).to_csv(root/'provenance/tail_thresholds.csv',index=False)
    if protected_hashes()!=protected:raise ValueError('Historical artifacts changed')
    if {n:file_digest(PROJECT/n) for n in sources}!=sources:raise ValueError('Source changed during run')
    if file_digest(root/'provenance/selection.json')!=lock:raise ValueError('Selection changed during pressure checks')
    atomic_json(root/'provenance/execution.json',{'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'elapsed_seconds':perf_counter()-started,'max_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
        'guard':vars(guard),'development_reference_groups_matched':reference_dev,
        'july_reference_groups_matched':reference_july,'july_market':july_market,'july_feature_check':full_feature_info,
        'protected_artifacts_unchanged':True,'selection_unchanged':True,
        'development_orders':int(selected_dev.model.eq('official_raw').sum()),
        'july_orders':int(primary.model.eq('official_raw').sum()),
        'continuous_orders':int(long.model.eq('official_raw').sum())})
    files={str(p.relative_to(root)):file_digest(p) for p in root.rglob('*') if p.is_file()}
    atomic_json(root/'provenance/manifest.json',{'status':'complete','files':files})
    print(json.dumps({'output':str(root),'selected':selected,'elapsed_seconds':perf_counter()-started},ensure_ascii=False),flush=True)
    return root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.run_id)


if __name__=='__main__':main()
