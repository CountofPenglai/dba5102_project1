"""Reconcile aggregates, verify input alignment, independently replay two fit days."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from pp_data.config import DEFAULT_DATA, DEFAULT_OUTPUT, PROJECT
from pp_models.directional_features import read_market_prefix
from pp_models.directional_study import install_development_guard
from pp_models.study_storage import file_digest, atomic_json
from scripts.complementary_signals import read_external, market_frame, joined_signals, directional_signals, GROUPS


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--external',type=Path,required=True);args=parser.parse_args();p=args.run_dir
    guard=install_development_guard(DEFAULT_OUTPUT)
    manifest=json.loads((p/'provenance/manifest.json').read_text())
    for name,digest in manifest['files'].items():assert file_digest(p/name)==digest,name
    source=json.loads((p/'provenance/source.json').read_text())
    prior=json.loads((PROJECT/'outputs/diagnostics/payout_signals_20261009/provenance/source.json').read_text())
    assert source['order_coverage']==prior['coverage']
    s=pd.read_csv(p/'metrics/market_summary.csv');d=pd.read_csv(p/'metrics/market_daily.csv')
    checks=0
    for _,r in s[s.period.isin(['all_june','early','late'])].iterrows():
        part=d[d.lag_minutes.eq(r.lag_minutes)&d.direction.eq(r.direction)&d.horizon.eq(r.horizon)&d.model.eq(r.model)]
        if r.period=='early':part=part[part.day.lt('2026-06-21')]
        if r.period=='late':part=part[part.day.ge('2026-06-21')]
        for name in ['orders','tail_orders','alerts','true_positives']:assert part[name].sum()==r[name]
        for name in ['brier','log_loss']:
            np.testing.assert_allclose(np.average(part[name],weights=part.orders),r[name],rtol=1e-9)
        checks+=1
    od=pd.read_csv(p/'metrics/order_daily.csv');oa=pd.read_csv(p/'metrics/order_alerts.csv')
    order_checks=0
    for _,r in oa[oa.period.isin(['all_june','early','late'])&oa.signal_group.eq('high')&oa.feature.isin(od.feature.unique())].iterrows():
        part=od[od.direction.eq(r.direction)&od.feature.eq(r.feature)]
        if r.period in ['early','late']:part=part[part.phase.eq(r.period)]
        for col in ['orders','tail_orders','alerts']:assert part[col].sum()==r[col]
        assert part.hits.sum()==r.true_positives;order_checks+=1
    market,_=read_market_prefix(DEFAULT_DATA/'btc_market_history.csv','2026-07-01T00:00:00Z')
    alignment_manifest=args.external/'alignment_check/manifest.json'
    if not alignment_manifest.exists():alignment_manifest=args.external/'alignment_check/verification.json'
    alignment=json.loads(alignment_manifest.read_text())
    for row in alignment:
        path=args.external/'alignment_check'/f"BTCUSDC-1m-{row['date']}.zip"
        assert file_digest(path)==row['sha256']
        with ZipFile(path) as z:f=pd.read_csv(z.open(z.namelist()[0]))
        f['time']=pd.to_datetime(f.open_time,unit='ms',utc=True)
        assert len(f)==1440 and f.time.is_unique and f.time.dt.strftime('%Y-%m-%d').eq(row['date']).all()
        expected=market.set_index('open_time_utc').loc[f.time]
        for a,b in [('close','close'),('quote_volume','quote_asset_volume'),('count','number_of_trades'),('taker_buy_quote_volume','taker_buy_quote_asset_volume')]:
            np.testing.assert_array_equal(f[a].to_numpy(),expected[b].to_numpy())
    external,_,_=read_external(args.external);q=pd.Series(pd.date_range('2026-05-01','2026-07-01',inclusive='left',freq='30min',tz='UTC'))
    raw=joined_signals(q,market_frame(market),external,5);replays=0;errors=[]
    for date,side,horizon in [('2026-06-01',0,60),('2026-06-24',1,240)]:
        day=pd.Timestamp(date,tz='UTC');X=directional_signals(raw,side);valid=X.notna().all(axis=1)
        # Independent exact close lookup, without forward_returns/history_mask/StandardScaler.
        indexed=market.set_index('close_time_utc').close
        starts=pd.DatetimeIndex(q-pd.Timedelta(milliseconds=1));ends=starts+pd.Timedelta(minutes=horizon)
        move=(1 if side==0 else -1)*np.log(indexed.reindex(ends).to_numpy()/indexed.reindex(starts).to_numpy())
        hist=(q>=day-pd.Timedelta(days=30))&(q<day)&(ends<day)&valid&np.isfinite(move)
        today=(q>=day)&(q<day+pd.Timedelta(days=1))&valid&np.isfinite(move)
        threshold=np.quantile(move[hist],.9);target=move[hist]>threshold;actual=move[today]>threshold
        for name,features in GROUPS.items():
            a=X.loc[hist,features].to_numpy();b=X.loc[today,features].to_numpy();mean=a.mean(axis=0);sd=a.std(axis=0);sd[sd==0]=1
            model=LogisticRegression(C=.1,solver='lbfgs',max_iter=500,tol=1e-8).fit(np.clip((a-mean)/sd,-5,5),target)
            prob=model.predict_proba(np.clip((b-mean)/sd,-5,5))[:,1]
            expected=d[d.lag_minutes.eq(5)&d.direction.eq(side)&d.horizon.eq(horizon)&d.model.eq(name)&d.day.str.startswith(date)].iloc[0]
            error=abs(float(np.mean((prob-actual)**2))-expected.brier);assert error<1e-8
            errors.append(error);replays+=1
    result={'status':'passed','market_aggregate_groups':checks,'order_aggregate_groups':order_checks,
        'canonical_order_cohort_unchanged':True,'independent_daily_fit_replays':replays,'max_brier_difference':max(errors),
        'public_provider_alignment_rows':len(alignment)*1440,'alignment_columns':4,
        'original_manifest_files_checked':len(manifest['files']),'external_inputs_checked':len(source['external_manifest']['files']),
        'test_csv_read':False,'blocked_test_opens':guard['blocked_test_opens'],
        'code_sha256':file_digest(Path(__file__))}
    atomic_json(p/'provenance/verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
