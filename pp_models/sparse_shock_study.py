"""Fixed Apr--Jun sparse-shock overlays: python -m pp_models.sparse_shock_study plan|run."""
import argparse
import json
import resource
import sys
from time import perf_counter
import warnings
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT, DEFAULT_DATA, DEFAULT_OUTPUT
from pp_data.features import BASE_FEATURE
from .calibration_models import CalibrationModel, MODEL_INPUTS
from .directional_features import read_market_prefix
from .directional_study import install_development_guard
from .evaluation import key_metrics
from .rolling_protocol import load_protocol, load_rolling_pool, rolling_batches, protocol_coverage, PROTOCOL_PATH
from .rolling_study import payout_aggregates, tail_context, recent_loss_state, settlement_buckets
from .runs import fingerprint, run_destination
from .sparse_shock import (CONFIG_PATH, load_settings, market_states, order_states,
                           fit_amplitude, market_diagnostics)
from .study import study_snapshot
from .study_storage import atomic_json, file_digest

ANCHORS={'official_raw':'official','hurdle_base':'hurdle'}
METHODS=['official_raw','official_shock','official_shock_fixed','hurdle_base','hurdle_shock','hurdle_shock_fixed']


def pricing_inputs(pool):
    # These unused compatibility fields do not enter either selected anchor.
    X=pool.X.copy()
    for name in ['terminal_supported','terminal_static_proxy','terminal_dynamic_proxy']:
        if name not in X:X[name]=0.
    return X[MODEL_INPUTS]


def budget_reference(frame):
    """Ex-post equal-cash reference, explicitly nondeployable, no labels in scaling."""
    rows=[]
    for period in ['all']+sorted(frame.month.unique()):
        f=frame if period=='all' else frame[frame.month.eq(period)]
        for (model,side),part in f.groupby(['model','side']):
            if model in ANCHORS:continue
            base=float(np.dot(part.anchor,part.quantity))
            adjusted=float(np.dot(part.prediction,part.quantity))
            ratio=adjusted/base if base>0 else 1.
            rows.append({'month':period,'overlay':model,'direction':side,'uniform_coefficient':ratio,
                'diagnostic_only':True,**key_metrics(part.actual,part.anchor*ratio,part.quantity)})
    return pd.DataFrame(rows)


def trigger_tables(frame):
    diagnostic=[];slices=[]
    for period in ['all']+sorted(frame.month.unique()):
        selected=frame if period=='all' else frame[frame.month.eq(period)]
        for (model,side),f in selected.groupby(['model','side']):
            gate=f.effective_state.gt(0);changed=f.prediction.gt(f.anchor+1e-10);tail=f.tail_group.eq('tail')
            cash=f.actual*f.quantity;tail_cash=float(cash[tail].sum())
            diagnostic.append({'month':period,'model':model,'direction':side,'orders':len(f),
                'market_state_orders':int(f.state.gt(0).sum()),'gate_orders':int(gate.sum()),
                'adjusted_orders':int(changed.sum()),'adjusted_fraction':float(changed.mean()),
                'gate_events':int(f.loc[gate,'event_id'].nunique()),'tail_orders':int(tail.sum()),
                'gate_tail_orders':int((gate&tail).sum()),'adjusted_tail_orders':int((changed&tail).sum()),
                'gate_tail_precision':float(tail[gate].mean()) if gate.any() else np.nan,
                'gate_tail_recall':float(gate[tail].mean()) if tail.any() else np.nan,
                'gate_tail_cash_recall':float(cash[gate&tail].sum()/tail_cash) if tail_cash else np.nan,
                'mean_multiplier':float(f.multiplier.mean()),'maximum_multiplier':float(f.multiplier.max()),
                'added_predicted_cash':float(np.dot(f.prediction-f.anchor,f.quantity))})
            stage=np.where(f.state.eq(0),'quiet',np.where(f.effective_state.eq(0),'contract_filtered',f.phase))
            for value in sorted(set(stage)):
                part=f[stage==value]
                slices.append({'month':period,'model':model,'direction':side,'stage':value,
                    **key_metrics(part.actual,part.prediction,part.quantity)})
    return pd.DataFrame(diagnostic),pd.DataFrame(slices)


def evaluate(pool,features,protocol,c,progress=None):
    if (not pool.X.index.equals(features.index) or not features.index.is_unique
            or not np.isfinite(features.effective_state).all()):raise ValueError('Misaligned shock features')
    batches={day:(h,q) for day,h,q in rolling_batches(pool,protocol)}
    start,end=pd.Timestamp(protocol['validation_start']),pd.Timestamp(protocol['validation_end'])
    replay_start=start-pd.Timedelta(days=protocol['history_days'])
    X=pricing_inputs(pool);created=pool.metadata.created_at_utc;closed=pool.metadata.closed_at_utc
    ledger={name:pd.Series(np.nan,index=X.index) for name in ANCHORS}
    frames=[];parameters=[];cohorts=[];replay=[];scored=[]
    for day in pd.date_range(replay_start,end,freq='D',inclusive='left'):
        q=X.index[created.ge(day)&created.lt(day+pd.Timedelta(days=1))]
        if not len(q):continue
        h=X.index[created.ge(day-pd.Timedelta(days=protocol['history_days']))&created.lt(day)&closed.lt(day)]
        if len(h)<protocol['minimum_history_orders']:raise ValueError(f'Insufficient base replay history at {day}')
        for name in ANCHORS:
            with threadpool_limits(limits=1),warnings.catch_warnings():
                warnings.simplefilter('error')
                base=CalibrationModel(name).fit(X.loc[h],pool.y.loc[h]).predict_raw(X.loc[q])
            ledger[name].loc[q]=base
        replay.append({'day':day.isoformat(),'training_orders':len(h),'predicted_orders':len(q),
            'latest_training_close':closed.loc[h].max().isoformat(),
            'train_ids_sha256':fingerprint(list(map(str,h))),'prediction_ids_sha256':fingerprint(list(map(str,q)))})
        if day not in batches:continue
        expected_h,scoring=batches[day]
        if not h.equals(expected_h):raise ValueError('Base and overlay history differ')
        q=scoring;scored.extend(map(str,q))
        if any(ledger[name].loc[h].isna().any() for name in ANCHORS):
            raise ValueError('Missing out-of-sample historical anchor forecasts')
        threshold,source=tail_context(pool,h,q)
        common=pd.DataFrame({'day':day.isoformat(),'month':day.strftime('%Y-%m'),'actual':pool.y.loc[q],
            'quantity':pool.X.loc[q,'quantity_btc'],'side':X.loc[q,'option_type'].astype(int),
            'tenor':X.loc[q,'product_duration_seconds'],'settlement':settlement_buckets(pool.metadata.loc[q]),
            'recent_loss':recent_loss_state(pool,h,q,day).map({True:'recent_high_loss',False:'other'}),
            'tail_group':np.where(threshold.isna(),'unavailable',np.where(pool.y.loc[q].gt(threshold),'tail','body'))},index=q)
        for field in ['state','effective_state','event_id','phase']:common[field]=features.loc[q,field]
        cohorts.append({'day':day.isoformat(),'month':day.strftime('%Y-%m'),'training_orders':len(h),'scoring_orders':len(q),
            'train_ids_sha256':fingerprint(list(map(str,h))),'score_ids_sha256':fingerprint(list(map(str,q))),
            'latest_training_close':closed.loc[h].max().isoformat(),
            'tail_threshold_fallbacks':int(source.eq('direction_fallback').sum()),
            'tail_threshold_unavailable':int(source.eq('unavailable').sum())})
        for name,prefix in ANCHORS.items():
            prediction=ledger[name].loc[q]
            frames.append(common.assign(model=name,anchor=prediction,prediction=prediction,multiplier=1.))
            coefficients={}
            for side in [0,1]:
                ids=h[X.loc[h,'option_type'].eq(side)]
                amplitude,detail=fit_amplitude(ledger[name].loc[ids],features.loc[ids,'effective_state'],
                                               pool.y.loc[ids],features.loc[ids,'event_id'],c)
                coefficients[side]=amplitude
                parameters.append({'day':day.isoformat(),'month':day.strftime('%Y-%m'),'anchor':name,
                    'direction':side,'amplitude':amplitude,**detail})
            learned=1+common.side.map(coefficients)*common.effective_state
            fixed=1+c['fixed_probe_amplitude']*common.effective_state
            frames.append(common.assign(model=prefix+'_shock',anchor=prediction,prediction=prediction*learned,multiplier=learned))
            frames.append(common.assign(model=prefix+'_shock_fixed',anchor=prediction,prediction=prediction*fixed,multiplier=fixed))
        if progress and (day.day==1 or day.day%10==0):progress(f'{day.date()}: {len(h)} settled history; {len(q)} scored')
    frame=pd.concat(frames);tables=payout_aggregates(frame)
    tables['trigger_diagnostics'],tables['trigger_slices']=trigger_tables(frame)
    tables['budget_matched_diagnostic']=budget_reference(frame)
    tables['daily_parameters']=pd.DataFrame(parameters);tables['daily_cohorts']=pd.DataFrame(cohorts)
    tables['anchor_replay_cohorts']=pd.DataFrame(replay)
    return tables,{'models':METHODS,'unique_scoring_orders':len(scored),'scoring_days':len(batches),
        'scoring_ids_sha256':fingerprint(scored),'anchor_replay_days':len(replay),
        'anchor_daily_evaluations':len(replay)*2,'scored_model_days':len(batches)*len(METHODS),
        'parameters_tuned':False,'fixed_probe':'diagnostic only, not selected after outcomes'}


def check_anchors(tables):
    """Exact common-cohort reproduction of both prior Apr--Jun anchors."""
    old=PROJECT/'outputs/model_studies/rolling_calibration_apr_jun_20261009'
    manifest=json.loads((old/'provenance/manifest.json').read_text())
    name='metrics/summary.csv'
    if file_digest(old/name)!=manifest['files'][name]:raise ValueError('Prior comparator file changed')
    previous=pd.read_csv(old/name,float_precision='round_trip',dtype={'value':str})
    keys=['month','model','group','value']
    left=tables['summary'][tables['summary'].model.isin(ANCHORS)].set_index(keys).sort_index()
    right=previous[previous.model.isin(ANCHORS)].set_index(keys).sort_index()
    if not left.index.equals(right.index):raise ValueError('Anchor cohorts or reporting groups changed')
    np.testing.assert_allclose(left.to_numpy(float),right[left.columns].to_numpy(float),rtol=1e-10,atol=1e-8,equal_nan=True)
    return {'rows':len(left),'prior_summary_sha256':file_digest(old/name),'status':'passed'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','run'])
    parser.add_argument('--run-id');args=parser.parse_args(argv)
    if args.action=='run' and not args.run_id:parser.error('run requires --run-id')
    started=perf_counter();guard=install_development_guard(DEFAULT_OUTPUT)
    protocol,c=load_protocol(),load_settings();pool=load_rolling_pool();coverage=protocol_coverage(pool,protocol)
    source=study_snapshot(DEFAULT_OUTPUT)
    hashes={str(p.relative_to(PROJECT)):file_digest(p) for p in [CONFIG_PATH,PROTOCOL_PATH]}
    plan={'protocol':protocol,'settings':c,'methods':METHODS,'common_coverage':coverage.to_dict('records'),
        'training_performed':False,'test_used':False,'parameter_search':False}
    if args.action=='plan':print(json.dumps(plan,indent=2));return 0
    root=run_destination(PROJECT/'outputs/model_studies',args.run_id);root.mkdir(parents=True,exist_ok=False)
    for name in ['metrics','reports','provenance']:(root/name).mkdir()
    atomic_json(root/'provenance/plan.json',plan);atomic_json(root/'provenance/manifest.json',{'status':'running'})
    try:
        market,market_info=read_market_prefix(DEFAULT_DATA/'btc_market_history.csv',protocol['validation_end'])
        states=market_states(market,c);features=order_states(pool,states,c)
        tables,execution=evaluate(pool,features,protocol,c,progress=lambda s:print(s,flush=True))
        tables['market_state']=market_diagnostics(states,protocol);tables['coverage']=coverage
        reference=check_anchors(tables)
        if study_snapshot(DEFAULT_OUTPUT)!=source or any(file_digest(PROJECT/k)!=v for k,v in hashes.items()):
            raise ValueError('Source changed during sparse shock study')
        for name,f in tables.items():f.to_csv(root/f'metrics/{name}.csv',index=False,float_format='%.17g')
        provenance={'source':source,'config_sha256':hashes,'market_prefix':market_info,'execution':execution,
            'anchor_reproduction':reference,'elapsed_seconds':perf_counter()-started,
            'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=='darwin' else 1024),
            'test_csv_read':False,'blocked_test_opens':guard['blocked_test_opens'],
            'opened_development_files':sorted(guard['opened_development_files']),
            'limitations':['Exploratory retrospective Apr--Jun replay; not independent untouched holdout',
                'Thresholds and contract weighting are fixed heuristics, not validated risk probabilities',
                'Price state can only alter new predictions after observable evidence',
                'Equal-cash uniform reference is ex-post diagnostic, not deployable',
                'Low event support explicitly falls back to no learned uplift',
                'No demand, margin, July, or model promotion']}
        atomic_json(root/'provenance/source.json',provenance)
        with ZipFile(root/'provenance/source_snapshot.zip','w',ZIP_DEFLATED) as z:
            files=set(source['code_sha256'])|set(hashes)|{'tests/test_sparse_shock.py','tests/test_sparse_shock_study.py',
                'docs/plans/2026-10-09-sparse-shock.md'}
            files.update(str(p.relative_to(PROJECT)) for d in ['pp_data','pp_audit'] for p in (PROJECT/d).glob('*.py'))
            for name in sorted(files):z.write(PROJECT/name,name)
        paths=list((root/'metrics').iterdir())+[root/'provenance/plan.json',root/'provenance/source.json',root/'provenance/source_snapshot.zip']
        atomic_json(root/'provenance/manifest.json',{'status':'complete','files':{str(p.relative_to(root)):file_digest(p) for p in paths}})
        print(json.dumps({'output':str(root),**execution,'elapsed_seconds':provenance['elapsed_seconds']},indent=2))
    except BaseException as e:
        atomic_json(root/'provenance/manifest.json',{'status':'failed','error':f'{type(e).__name__}: {e}'});raise
    return 0


if __name__=='__main__':raise SystemExit(main())
