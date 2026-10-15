"""Replay saved coefficients using independent calendar integrals; never retrain."""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.integrate import quad
from threadpoolctl import threadpool_limits
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pp_data.config import PROJECT, DEFAULT_DATA
from pp_data.features import BASE_FEATURE
from pp_models.data import load_pool
from pp_models.directional_features import augment_pool
from pp_models.put_final_io import read_test_market_prefix
from pp_models.study_storage import file_digest, atomic_json
from pp_models.put_final_statistics import draw_weights
from scripts.verify_base_multiplier import manual_metrics

BASIS=['pre_exposure','cross_release','post_exposure']


def independent_features(pool, calendar):
    groups=[]
    for e in calendar['events']:
        if e['kind']=='press_conference':continue
        t=pd.Timestamp(e['event_time_utc']);parts=[e]
        if e['family']=='FOMC':
            parts += [r for r in calendar['events'] if r['kind']=='press_conference'
                      and pd.Timestamp(r['event_time_utc']).date()==t.date()]
        offsets=sorted((pd.Timestamp(r['event_time_utc'])-t).total_seconds()/60 for r in parts)
        known=max(pd.Timestamp(r['known_by_utc']) for r in parts)
        groups.append((e['event_id'],t,offsets,known))
    coverage=[(pd.Timestamp(c['start'],tz='America/New_York'),pd.Timestamp(c['end_exclusive'],tz='America/New_York'))
              for c in calendar['coverage'].values()]
    def integral(lo,hi,offsets):
        if hi<=lo:return 0.
        def kernel(x):return max([2.**(-(x-a)/60) for a in offsets if a<=x] or [0.])
        return quad(kernel,lo,hi,points=[a for a in offsets if lo<a<hi],epsabs=1e-11,epsrel=1e-11)[0]
    result=[]
    for idx in pool.X.index:
        t=pool.metadata.loc[idx,'created_at_utc'];end=t+pd.Timedelta(seconds=float(pool.X.loc[idx,'remaining_seconds']))
        row={'option_type':int(pool.X.loc[idx,'option_type']),'pre_exposure':0.,'cross_release':0.,'post_exposure':0.,
             'event_id':'','event_time':pd.NaT,'event_month':'','calendar_status':'no_identified_event'}
        if not all(t-pd.Timedelta(hours=4)>=lo and end<=hi for lo,hi in coverage):
            row['calendar_status']='outside_coverage';result.append(row);continue
        relevant=[g for g in groups if t<g[1]+pd.Timedelta(hours=4) and end>g[1]-pd.Timedelta(hours=1)]
        if len(relevant)!=1:
            if relevant:row['calendar_status']='multiple_events'
            result.append(row);continue
        event,anchor,offsets,known=relevant[0]
        row.update(event_id=event,event_time=anchor,event_month=anchor.strftime('%Y-%m'))
        if t<known:row['calendar_status']='unknown_evidence';result.append(row);continue
        lo=(t-anchor).total_seconds()/60;hi=(end-anchor).total_seconds()/60
        row['pre_exposure']=max(min(hi,0)-max(lo,-60),0)/60
        row['cross_release']=float(any(lo<a<=hi for a in offsets))
        row['post_exposure']=integral(max(lo,0),min(hi,240),offsets)/integral(0,240,offsets)
        row['calendar_status']='known_event';result.append(row)
    f=pd.DataFrame(result,index=pool.X.index);f['event_time']=pd.to_datetime(f.event_time,utc=True)
    return f


def manual_announcement(f,fit):
    values=f[BASIS].to_numpy(float)
    common=values*np.array(fit['shared_enabled'])
    parts=[values*np.array(fit['direction_enabled'][s])*f.option_type.eq(s).to_numpy()[:,None] for s in (0,1)]
    phi=np.column_stack([common,*parts]);coef=np.array(fit['coefficients'])
    return 1+phi@coef,phi


def manual_h0(X,fit):
    answer=np.zeros(len(X))
    for p in fit['parameters']:
        side=X.option_type.eq(p['side']).to_numpy();g=X.loc[side]
        if not len(g):continue
        prob=np.full(len(g),p['probability'])
        if 'coef' in p:
            aligned=(1-2*g.option_type)*g.log_moneyness/np.maximum(g.realized_vol_60m*np.sqrt(g.remaining_seconds/3600),1e-6)
            raw=np.column_stack([np.log(g[BASE_FEATURE]),np.log(g.remaining_seconds),aligned.clip(-5,5)])
            z=np.clip((raw-np.array(p['mean']))/np.array(p['scale']),-3,3)
            prob=1/(1+np.exp(-(z@np.array(p['coef'])+p['intercept'])))
        answer[side]=prob*g[BASE_FEATURE]*p['severity']
    return answer


def verify(root):
    counts={'official_attempts':0,'model_fit_predict_calls':0}
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            if Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).name=='pp_transactions_validation.csv':
                counts['official_attempts']+=1;raise PermissionError('Official validation forbidden')
    def profile(frame,event,arg):
        if (event=='call' and frame.f_code.co_name in ('fit','partial_fit','predict_raw','predict_diagnostics','predict_multiplier')
                and frame.f_globals.get('__name__','').startswith(('pp_models','sklearn'))):
            counts['model_fit_predict_calls']+=1;raise PermissionError('Independent replay must not call models')
    sys.addaudithook(audit);sys.setprofile(profile)
    manifest=json.loads((root/'provenance/manifest.json').read_text())
    for name,sha in manifest['files'].items():assert file_digest(root/name)==sha,name
    started=json.loads((root/'provenance/started.json').read_text())
    for name,sha in (started['source_sha256']|started['protected_sha256']).items():assert file_digest(PROJECT/name)==sha,name
    settings=json.loads((root/'provenance/settings.json').read_text())
    calendar=json.loads((PROJECT/settings['calendar']).read_text())
    assert file_digest(PROJECT/settings['calendar'])==settings['calendar_sha256']
    selection=json.loads((root/'provenance/selection.json').read_text());selected=selection['selected']
    pressure=json.loads((root/'provenance/pressure_started.json').read_text())
    assert pd.Timestamp(selection['selected_at_utc'])<pd.Timestamp(pressure['utc'])
    assert pressure['selection_sha256']==file_digest(root/'provenance/selection.json')
    market,info=read_test_market_prefix(DEFAULT_DATA/'btc_market_history.csv')
    assert info['prefix_sha256']==settings['july_market_sha256']
    pool,_=augment_pool(load_pool('test',{'cohort':'benchmark','use_base':True}),market)
    X,y,meta=pool.X,pool.y,pool.metadata
    f=independent_features(pool,calendar)
    # Compare the public feature function with a separate numerical integration implementation.
    from pp_models.macro_multiplier_features import build_features
    prod=build_features(pd.DataFrame({'created':meta.created_at_utc,'remaining_seconds':X.remaining_seconds,'option_type':X.option_type}),calendar)
    for col in BASIS:np.testing.assert_allclose(f[col],prod[col],atol=1e-12,rtol=1e-10)
    pd.testing.assert_frame_equal(f.drop(columns=BASIS),prod.drop(columns=BASIS),check_dtype=False)
    fits=json.loads((root/'provenance/development_fits.json').read_text())+json.loads((root/'provenance/pressure_fits.json').read_text())
    training=[];fit_count=0
    for fit in fits:
        if fit['model']=='h0':continue
        origin=pd.Timestamp(fit['origin'])
        ids=X.index[meta.created_at_utc.lt(origin)&meta.closed_at_utc.lt(origin)&f.event_time.lt(origin)
                    &f.calendar_status.eq('known_event')&f[BASIS].gt(0).any(axis=1)]
        h=f.loc[ids];assert len(h)==fit['training_orders'] and h.event_id.nunique()==fit['training_events']
        for label,side,minimum in [('shared',None,(30,5,2)),('call',0,(60,8,3)),('put',1,(60,8,3))]:
            g=h if side is None else h.loc[h.option_type.eq(side)]
            for j,basis in enumerate(BASIS):
                cell=g.loc[g[basis].gt(0)];n,events,months=len(cell),cell.event_id.nunique(),cell.event_month.nunique()
                saved=fit['support'][label][j]
                assert (n,events,months)==(saved['orders'],saved['events'],saved['months'])
                enabled=bool(n>=minimum[0] and events>=minimum[1] and months>=minimum[2])
                assert saved['enabled']==enabled
        coef=np.array(fit['coefficients']);corners=np.array(list(itertools.product([0.,1.],repeat=3)))
        assert np.max(abs(corners@coef[:3]))<=.05+1e-9
        for s in (0,1):
            assert np.max(abs(corners@coef[3+3*s:6+3*s]))<=.02+1e-9
            assert np.max(abs(corners@(coef[:3]+coef[3+3*s:6+3*s])))<=.05+1e-9
        w=1/h.event_id.map(h.event_id.value_counts()).to_numpy(float);w/=w.sum()
        scale=max(np.sqrt(w@y.loc[ids].to_numpy()**2),1e-6)
        np.testing.assert_allclose(scale,fit['loss_scale'],rtol=1e-10)
        multiplier,phi=manual_announcement(h,fit)
        penalty=fit['strength']*np.r_[np.ones(3),np.full(6,10.)]
        objective=w@((X.loc[ids,BASE_FEATURE]*multiplier-y.loc[ids])/scale).to_numpy()**2+penalty@(coef**2)
        null=w@((X.loc[ids,BASE_FEATURE]-y.loc[ids])/scale).to_numpy()**2
        np.testing.assert_allclose([objective,null],[fit['objective'],fit['null_objective']],rtol=1e-10,atol=1e-10)
        assert objective<=null+1e-8
        fit_count+=1
        if fit['model'] in (selected['grid_name'],'announcement'):
            for (event,side),g in h.groupby(['event_id','option_type']):
                v=g.index;b=X.loc[v,BASE_FEATURE];actual=y.loc[v]
                training.append({'origin':origin.isoformat(),'event_id':event,'side':side,'orders':len(g),
                    'base_mean':b.mean(),'actual_mean':actual.mean(),'bias_mean':(b-actual).mean(),
                    'positive_fraction':actual.gt(0).mean()})
    replay={};checked=0
    for label,start,end in [('development','2026-04-01T00:00Z','2026-07-01T00:00Z'),
                            ('july_pressure','2026-07-01T00:00Z',settings['pressure_end'])]:
        start,end=pd.Timestamp(start),pd.Timestamp(end);outputs=[]
        ids=X.index[meta.created_at_utc.ge(start)&meta.created_at_utc.lt(end)&meta.closed_at_utc.lt(end)]
        for origin in pd.date_range(start,end,freq='MS',inclusive='left'):
            q=ids[meta.loc[ids,'created_at_utc'].ge(origin)&meta.loc[ids,'created_at_utc'].lt(origin+pd.offsets.MonthBegin(1))]
            hist=X.index[meta.created_at_utc.lt(origin)&meta.closed_at_utc.lt(origin)]
            threshold=pd.Series(np.nan,index=q)
            for (side,tenor),g in X.loc[q].groupby(['option_type','product_duration_seconds']):
                sid=hist[X.loc[hist,'option_type'].eq(side)];cell=sid[X.loc[sid,'product_duration_seconds'].eq(tenor)]
                chosen=cell if len(cell)>=30 else sid
                if len(chosen)>=30:threshold.loc[g.index]=np.quantile(y.loc[chosen],.9)
            common=pd.DataFrame({'day':meta.loc[q,'created_at_utc'].dt.floor('D'),
                'month':meta.loc[q,'created_at_utc'].dt.strftime('%Y-%m'),'side':X.loc[q,'option_type'].astype(int),
                'tenor':X.loc[q,'product_duration_seconds'].astype(int),'quantity':X.loc[q,'quantity_btc'],
                'actual':y.loc[q],'event_id':f.loc[q,'event_id'],
                'tail_group':np.where(threshold.isna(),'unavailable',np.where(y.loc[q].gt(threshold),'tail','body'))},index=q)
            outputs.append(common.assign(model='official_raw',prediction=X.loc[q,BASE_FEATURE]))
            local=[fit for fit in fits if pd.Timestamp(fit['origin'])==origin]
            hf=next(fit for fit in local if fit['model']=='h0')
            outputs.append(common.assign(model='h0',prediction=manual_h0(X.loc[q],hf)))
            names=[selected['grid_name']] if label=='development' else ['announcement']
            for name in names:
                fit=next(fit for fit in local if fit['model']==name)
                multiplier,_=manual_announcement(f.loc[q],fit)
                outputs.append(common.assign(model='announcement',prediction=X.loc[q,BASE_FEATURE]*multiplier))
        frame=pd.concat(outputs);replay[label]=frame
        summary=pd.read_csv(root/f'metrics/{label}/summary.csv',dtype={'value':str})
        for row in summary.itertuples():
            g=frame.loc[frame.model.eq(row.model)]
            if row.month!='all':g=g.loc[g.month.eq(row.month)]
            if row.group=='direction':g=g.loc[g.side.eq(int(row.value))]
            elif row.group=='direction_tail':
                side,tail=row.value.split('|');g=g.loc[g.side.eq(int(side))&g.tail_group.eq(tail)]
            elif row.group=='tenor':g=g.loc[g.tenor.eq(int(row.value))]
            for key,value in manual_metrics(g).items():np.testing.assert_allclose(value,getattr(row,key),rtol=1e-9,atol=1e-7)
            checked+=1
        macro=pd.read_csv(root/f'metrics/{label}/macro_monthly.csv').set_index('model')
        for model,g in frame.groupby('model'):
            monthly=[manual_metrics(s) for _,s in g.groupby('month')]
            np.testing.assert_allclose([np.sqrt(np.mean([m['rmse_per_btc']**2 for m in monthly])),np.mean([m['mae_per_btc'] for m in monthly])],
                macro.loc[model,['macro_rmse_per_btc','macro_mae_per_btc']].to_numpy(float),rtol=1e-10)
    interval_count=0
    for label,frame in replay.items():
        start='2026-04-01T00:00Z' if label=='development' else '2026-07-01T00:00Z'
        end='2026-07-01T00:00Z' if label=='development' else settings['pressure_end']
        days=pd.date_range(pd.Timestamp(start),pd.Timestamp(end).ceil('D'),inclusive='left');weights=draw_weights(days,7)
        ci=pd.read_csv(root/f'metrics/{label}/paired_intervals.csv')
        for scope,side in [('all',None),('call',0),('put',1)]:
            part=frame if side is None else frame.loc[frame.side.eq(side)]
            losses={}
            for model,g in part.groupby('model'):
                e=g.prediction-g.actual
                daily=pd.DataFrame({'day':g.day,'n':1,'sse':e**2,'sae':abs(e)}).groupby('day').sum().reindex(days,fill_value=0)
                rates=[]
                for month in sorted(set(days.strftime('%Y-%m'))):
                    mask=days.strftime('%Y-%m')==month;t=weights[:,mask]@daily.loc[mask].to_numpy()
                    rates.append(t[:,1:]/t[:,0,None])
                avg=np.mean(rates,axis=0);losses[model]=np.column_stack([np.sqrt(avg[:,0]),avg[:,1]])
            for reference in ['official_raw','h0']:
                rows=ci.loc[ci.scope.eq(scope)&ci.reference.eq(reference)&ci.block_days.eq(7)].set_index('metric')
                for j,metric in enumerate(['rmse_delta','mae_delta']):
                    values=losses['announcement'][:,j]-losses[reference][:,j]
                    np.testing.assert_allclose(np.quantile(values,[.025,.975]),rows.loc[metric,['lower_95','upper_95']].to_numpy(float),rtol=1e-7,atol=1e-7)
                    interval_count+=1
    sys.setprofile(None)
    assert counts=={'official_attempts':0,'model_fit_predict_calls':0}
    for name,sha in started['protected_sha256'].items():assert file_digest(PROJECT/name)==sha,name
    pd.DataFrame(training).to_csv(root/'metrics/training_event_diagnostics.csv',index=False)
    result={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'calendar_exposure_rows':len(f),
        'multiplier_fits_recomputed':fit_count,'summary_rows_replayed_including_h0':checked,
        'paired_7day_intervals_recomputed':interval_count,'protected_files':len(started['protected_sha256']),
        'source_output_hashes_valid':True,'selection_before_july':True,'guards':counts,'verifier_sha256':file_digest(Path(__file__))}
    atomic_json(root/'provenance/verification.json',result)
    for name in ['provenance/verification.json','metrics/training_event_diagnostics.csv']:manifest['files'][name]=file_digest(root/name)
    atomic_json(root/'provenance/manifest.json',manifest);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True)
    with threadpool_limits(limits=1):verify(parser.parse_args().run_dir)
