"""Independent coefficient replay and arithmetic checks; never fit a model."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pp_data.config import PROJECT,DEFAULT_DATA
from pp_data.features import BASE_FEATURE
from pp_models.data import load_pool
from pp_models.directional_features import augment_pool
from pp_models.put_final_io import read_test_market_prefix
from pp_models.study_storage import file_digest,atomic_json


def manual_multiplier(X,fit):
    arrays={}
    if fit['family']!='constant':
        arrays['log_remaining']=np.log(X.remaining_seconds.to_numpy())
        arrays['position']=-X.log_moneyness.to_numpy()/np.maximum(
            X.total_vol_240m.to_numpy()*np.sqrt(X.remaining_seconds.to_numpy()/60),1e-6)
    if fit['family']=='market':
        arrays['log_down_vol']=np.log(np.maximum(X.down_vol_240m.to_numpy(),1e-12))
        arrays['adverse_trend']=-X.log_return_60m.to_numpy()/np.maximum(X.total_vol_240m.to_numpy()*np.sqrt(60),1e-6)
    supported=X.option_type.eq(1).to_numpy() & X.product_duration_seconds.isin(fit['supported_tenors']).to_numpy() & fit['global_support']
    adjustment=np.full(len(X),fit['coefficients']['intercept'])
    for name,values in arrays.items():
        supported &= np.isfinite(values)&(values>=fit['lower'][name])&(values<=fit['upper'][name])
        adjustment += fit['coefficients'][name]*np.clip((values-fit['mean'][name])/fit['std'][name],-3,3)
    adjustment[~supported]=0.
    return 1+np.clip(adjustment,-fit['delta'],fit['delta']),arrays,adjustment


def manual_metrics(frame):
    e=frame.prediction-frame.actual;q=frame.quantity
    return {'orders':len(frame),'rmse_per_btc':float(np.sqrt(np.mean(e**2))),
        'mae_per_btc':float(np.mean(abs(e))),'actual_total_usdc':float((frame.actual*q).sum()),
        'predicted_total_usdc':float((frame.prediction*q).sum()),'total_bias_usdc':float((e*q).sum()),
        'gross_overprediction_usdc':float((e.clip(lower=0)*q).sum()),
        'gross_underprediction_usdc':float(((-e).clip(lower=0)*q).sum()),
        'absolute_total_bias_usdc':float(abs((e*q).sum()))}


def verify(root):
    counts={'official_attempts':0,'model_fit_calls':0}
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            name=Path(args[0].decode() if isinstance(args[0],bytes) else args[0]).name
            if name=='pp_transactions_validation.csv':
                counts['official_attempts']+=1;raise PermissionError('Official validation forbidden')
    def profile(frame,event,arg):
        if event=='call' and frame.f_code.co_name in ('fit','partial_fit') and frame.f_globals.get('__name__','').startswith(('pp_models','sklearn')):
            counts['model_fit_calls']+=1;raise PermissionError('Verifier must not train')
    sys.addaudithook(audit);sys.setprofile(profile)
    manifest=json.loads((root/'provenance/manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if file_digest(root/name)!=digest:raise ValueError(f'Artifact checksum mismatch {name}')
    started=json.loads((root/'provenance/started.json').read_text())
    for name,digest in (started['source_sha256']|started['protected_sha256']).items():
        if file_digest(PROJECT/name)!=digest:raise ValueError(f'Source/protected checksum mismatch {name}')
    selected=json.loads((root/'provenance/selection.json').read_text())
    pressure=json.loads((root/'provenance/pressure_started.json').read_text())
    assert pd.Timestamp(selected['selected_at_utc'])<pd.Timestamp(pressure['utc'])
    assert pressure['selection_sha256']==file_digest(root/'provenance/selection.json')
    fits=json.loads((root/'provenance/development_fits.json').read_text())+json.loads((root/'provenance/pressure_fits.json').read_text())
    market,info=read_test_market_prefix(DEFAULT_DATA/'btc_market_history.csv')
    execution=json.loads((root/'provenance/execution.json').read_text())
    assert info['prefix_sha256']==execution['july_market']['prefix_sha256']
    pool,_=augment_pool(load_pool('test',{'cohort':'benchmark','use_base':True}),market)
    X,y,meta=pool.X,pool.y,pool.metadata
    # Check training statistics and the actual objective without calling model methods.
    for fit in fits:
        origin=pd.Timestamp(fit['origin'])
        ids=X.index[meta.created_at_utc.lt(origin)&meta.closed_at_utc.lt(origin)&X.option_type.eq(1)]
        m,raw,g=manual_multiplier(X.loc[ids],fit)
        assert len(ids)==fit['put_training_orders']
        np.testing.assert_allclose(np.sqrt(np.mean(y.loc[ids]**2)),fit['loss_scale'])
        for name,v in raw.items():
            for key,want in [('mean',np.mean(v)),('std',np.std(v) or 1.),('lower',min(v)),('upper',max(v))]:
                np.testing.assert_allclose(want,fit[key][name],rtol=1e-12,atol=1e-12)
        own_tenors=[float(t) for t,sub in X.loc[ids].groupby('product_duration_seconds')
                    if len(sub)>=30 and meta.loc[sub.index,'created_at_utc'].dt.floor('D').nunique()>=5]
        assert own_tenors==fit['supported_tenors']
        loss=np.mean(((X.loc[ids,BASE_FEATURE]*m-y.loc[ids])/fit['loss_scale'])**2)
        loss+=fit['strength']*sum(c*c for c in fit['coefficients'].values())
        np.testing.assert_allclose(loss,fit['objective'],rtol=1e-10,atol=1e-12)
        assert abs(g).max()<=fit['delta']+1e-8
    periods=[('development','2026-04-01','2026-07-01',False,False),
             ('development_sensitivity','2026-04-01','2026-07-01',False,True),
             ('july_pressure','2026-07-01','2026-07-31T17:11:48',False,False),
             ('july_sensitivity','2026-07-01','2026-07-31T17:11:48',False,True),
             ('continuous_freeze','2026-05-01','2026-07-31T17:11:48',True,False)]
    checked=0
    replay={}
    for label,st,en,continuous,wide in periods:
        start,end=pd.Timestamp(st,tz='UTC'),pd.Timestamp(en,tz='UTC')
        ids=X.index[meta.created_at_utc.ge(start)&meta.created_at_utc.lt(end)&meta.closed_at_utc.lt(end)]
        frames=[]
        for origin in ([start] if continuous else pd.date_range(start,end,freq='MS',inclusive='left')):
            stop=end if continuous else min(end,origin+pd.offsets.MonthBegin(1))
            q=ids[meta.created_at_utc.loc[ids].ge(origin)&meta.created_at_utc.loc[ids].lt(stop)]
            h=X.index[meta.created_at_utc.lt(origin)&meta.closed_at_utc.lt(origin)]
            actual=y.loc[q];threshold=pd.Series(np.nan,index=q)
            for (side,tenor),sub in X.loc[q].groupby(['option_type','product_duration_seconds']):
                cell=h[X.loc[h,'option_type'].eq(side)&X.loc[h,'product_duration_seconds'].eq(tenor)]
                if len(cell)<30:cell=h[X.loc[h,'option_type'].eq(side)]
                if len(cell)>=30:threshold.loc[sub.index]=np.quantile(y.loc[cell],.9)
            common=pd.DataFrame({'actual':actual,'quantity':X.loc[q,'quantity_btc'],
                'month':meta.created_at_utc.loc[q].dt.strftime('%Y-%m'),'day':meta.created_at_utc.loc[q].dt.floor('D'),
                'side':X.loc[q,'option_type'].astype(int),'tenor':X.loc[q,'product_duration_seconds'].astype(int),
                'tail_group':np.where(threshold.isna(),'unavailable',np.where(actual>threshold,'tail','body'))},index=q)
            frames.append(common.assign(model='official_raw',prediction=X.loc[q,BASE_FEATURE]))
            for family,spec in selected['selected'].items():
                name=family+'_wide' if wide else family
                fitname=name if continuous or label.startswith('july') or wide else spec['grid_name']
                matching=[f for f in fits if f['model']==fitname and pd.Timestamp(f['origin'])==origin]
                assert len(matching)==1,(label,fitname,origin,len(matching))
                m,_,_=manual_multiplier(X.loc[q],matching[0])
                np.testing.assert_array_equal(m[X.loc[q,'option_type'].eq(0)],1.)
                frames.append(common.assign(model=name,prediction=X.loc[q,BASE_FEATURE]*m))
        frame=pd.concat(frames);replay[label]=frame
        summary=pd.read_csv(root/f'metrics/{label}/summary.csv',dtype={'value':str})
        for row in summary.itertuples():
            if row.model=='h0':continue
            sub=frame.loc[frame.model.eq(row.model)]
            if row.month!='all':sub=sub.loc[sub.month.eq(row.month)]
            if row.group=='direction':sub=sub.loc[sub.side.eq(int(row.value))]
            if row.group=='direction_tail':
                side,tail=row.value.split('|');sub=sub.loc[sub.side.eq(int(side))&sub.tail_group.eq(tail)]
            if row.group=='tenor':sub=sub.loc[sub.tenor.eq(int(row.value))]
            for key,value in manual_metrics(sub).items():
                np.testing.assert_allclose(value,getattr(row,key),rtol=1e-9,atol=1e-7)
            checked+=1
        # Independently check month-equal primary ranking (not pooled errors).
        macro=pd.read_csv(root/f'metrics/{label}/macro_monthly.csv').set_index('model')
        for model,sub in frame.groupby('model'):
            vals=[manual_metrics(p) for _,p in sub.groupby('month')]
            np.testing.assert_allclose(np.sqrt(np.mean([v['rmse_per_btc']**2 for v in vals])),macro.loc[model,'macro_rmse_per_btc'],rtol=1e-10)
            np.testing.assert_allclose(np.mean([v['mae_per_btc'] for v in vals]),macro.loc[model,'macro_mae_per_btc'],rtol=1e-10)
    # A separate vectorized 7-day bootstrap calculation for Put market vs Base.
    from pp_models.put_final_statistics import draw_weights
    intervals_checked=0
    for label,st,en,_,_ in [p for p in periods if p[0] in ('development','july_pressure','continuous_freeze')]:
        f=replay[label];p=f.loc[f.side.eq(1)&f.model.eq('market')];b=f.loc[f.side.eq(1)&f.model.eq('official_raw')]
        days=pd.date_range(pd.Timestamp(st,tz='UTC'),pd.Timestamp(en,tz='UTC').ceil('D'),inclusive='left')
        w=draw_weights(days,7);losses=[]
        for pred in (p,b):
            e=pred.prediction-pred.actual
            daily=pd.DataFrame({'day':pred.day,'n':1,'sse':e**2,'sae':abs(e)}).groupby('day').sum().reindex(days,fill_value=0)
            rates=[]
            for month in sorted(set(days.strftime('%Y-%m'))):
                mask=days.strftime('%Y-%m')==month;t=w[:,mask]@daily.loc[mask].to_numpy()
                rates.append(t[:,1:]/t[:,0,None])
            avg=np.mean(rates,axis=0);losses.append(np.column_stack([np.sqrt(avg[:,0]),avg[:,1]]))
        ci=pd.read_csv(root/f'metrics/{label}/paired_intervals.csv').query("scope == 'put' and candidate == 'market' and reference == 'official_raw' and block_days == 7").set_index('metric')
        for i,metric in enumerate(['rmse_delta','mae_delta']):
            values=losses[0][:,i]-losses[1][:,i]
            np.testing.assert_allclose(np.quantile(values,[.025,.975]),ci.loc[metric,['lower_95','upper_95']].to_numpy(float),rtol=1e-8,atol=1e-8)
            intervals_checked+=1
    sys.setprofile(None)
    result={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'coefficient_fits_checked':len(fits),
        'summary_rows_independently_replayed':checked,'bootstrap_intervals_independently_checked':intervals_checked,
        'source_and_protected_hashes_unchanged':True,'selection_precedes_july':True,
        'guards':counts,'verifier_sha256':file_digest(Path(__file__))}
    atomic_json(root/'provenance/verification.json',result)
    manifest['files']['provenance/verification.json']=file_digest(root/'provenance/verification.json')
    atomic_json(root/'provenance/manifest.json',manifest)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True)
    verify(p.parse_args().run_dir)
