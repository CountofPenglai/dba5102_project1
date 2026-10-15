"""Small signed residuals conditional on Base; no future settlement/path inputs."""
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from .directional_features import directional_features
from .macro_multiplier_features import build_features as calendar_features, BASIS

CONTRACT=['log_base','log_remaining','base_scaled_position']
MARKET=['risk_to_base','risk_change_60_240','risk_change_240_1440','aligned_trend','activity_change']
REACTION=['reaction_signed','reaction_abs','reaction_variance']


def contract_features(x):
    b=x[BASE_FEATURE];sign=1-2*x.option_type
    return pd.DataFrame({'log_base':np.log(b),'log_remaining':np.log(x.remaining_seconds),
        'base_scaled_position':sign*(x.market_price_usdc_per_btc-x.strike_price_usdc_per_btc)/b},index=x.index)


def market_features(x,created,market):
    d=directional_features(created,market,windows=(60,240,1440));side=x.option_type
    risk={w:np.where(side.eq(0),d[f'up_vol_{w}m'],d[f'down_vol_{w}m']) for w in (60,240,1440)}
    h=market.sort_values('close_time_utc');right=h[['close_time_utc']].copy()
    contiguous=h.open_time_utc.diff(240).eq(pd.Timedelta(minutes=240))
    right['activity']=((h.number_of_trades.rolling(60).sum()+1)/(h.number_of_trades.rolling(240).sum()/4+1)).where(contiguous)
    left=pd.DataFrame({'created':created,'pos':range(len(created))})
    joined=pd.merge_asof(left.sort_values('created'),right,left_on='created',right_on='close_time_utc',
        direction='backward',tolerance=pd.Timedelta(seconds=60)).set_index('pos').reindex(range(len(created)))
    result=contract_features(x)
    result['risk_to_base']=np.log1p(x.market_price_usdc_per_btc*risk[60]*np.sqrt(x.remaining_seconds/60)/x[BASE_FEATURE])
    result['risk_change_60_240']=np.log((risk[60]+1e-8)/(risk[240]+1e-8))
    result['risk_change_240_1440']=np.log((risk[240]+1e-8)/(risk[1440]+1e-8))
    result['aligned_trend']=(1-2*side)*x.log_return_60m/np.maximum(x.realized_vol_60m,1e-6)
    result['activity_change']=np.log(joined.activity.to_numpy())
    if not np.isfinite(result).all().all():raise ValueError('Missing causal market feature; no silent order deletion')
    return result


def event_features(quotes,market,calendar):
    result=calendar_features(quotes,calendar)
    for name in REACTION:result[name]=0.
    result['reaction_status']='not_yet_observable'
    # End-of-minute timestamp is deliberately rounded up for event information availability.
    p=pd.Series(market.close.to_numpy(float),index=pd.DatetimeIndex(market.open_time_utc)+pd.Timedelta(minutes=1))
    cache={}
    for eid in result.loc[result.calendar_status.eq('known_event'),'event_id'].unique():
        event=next(e for e in calendar['events'] if e['event_id']==eid)
        anchor=pd.Timestamp(event['event_time_utc']);anchors=[anchor]
        if event['family']=='FOMC':
            anchors += [pd.Timestamp(e['event_time_utc']) for e in calendar['events'] if e['family']=='FOMC'
                and e['kind']=='press_conference' and pd.Timestamp(e['event_time_utc']).date()==anchor.date()]
        for a in anchors:
            times=pd.date_range(a-pd.Timedelta(minutes=60),a+pd.Timedelta(minutes=5),freq='min')
            v=p.reindex(times).to_numpy();valid=np.isfinite(v).all() and (v>0).all()
            if not valid:cache[a]=None;continue
            r=np.diff(np.log(v));sigma=max(np.sqrt(np.mean(r[:60]**2)*5),1e-6)
            jump=np.clip(r[60:].sum()/sigma,-3,3);variance=np.clip(np.sum(r[60:]**2)/sigma**2,0,3)
            cache[a]=np.array([jump,abs(jump),variance])
        for oid in result.index[result.event_id.eq(eid)&result.calendar_status.eq('known_event')]:
            t=quotes.loc[oid,'created'];eligible=[a for a in anchors if a+pd.Timedelta(minutes=5)<=t]
            if not eligible:continue
            a=max(eligible)
            if cache[a] is None:
                result.loc[oid,'reaction_status']='missing_market'
                continue
            result.loc[oid,'reaction_status']='available'
            decay=2**(-float((t-a).total_seconds())/3600)
            result.loc[oid,REACTION]=cache[a]*decay
    return result


def _aligned(features,*values):
    if not features.index.is_unique or any(not features.index.equals(v.index) for v in values):
        raise ValueError('Misaligned inputs')


class ResidualModel:
    def __init__(self,family):
        if family not in ('contract','market'):raise ValueError('Unregistered family')
        self.family=family;self.columns=CONTRACT+(MARKET if family=='market' else [])

    def fit(self,features,base,y,side,created):
        self._validate(features);_aligned(features,base,y,side,created)
        if (created.dt.tz is None or not side.isin([0,1]).all() or not np.isfinite(base).all() or base.le(0).any()
                or not np.isfinite(y).all() or y.lt(0).any()):raise ValueError('Invalid training inputs')
        self.parameters={}
        for s in (0,1):
            ids=features.index[side.eq(s)&np.isfinite(features).all(axis=1)]
            raw=features.loc[ids];n=len(ids);days=created.loc[ids].dt.floor('D').nunique()
            mean=raw.mean().fillna(0);scale=raw.std(ddof=0).replace(0,1).fillna(1)
            state=dict(orders=n,days=int(days),supported=bool(n>=100 and days>=10),
                mean=mean.tolist(),scale=scale.tolist(),coef=[0.]*(len(self.columns)+1))
            if state['supported']:
                z=np.column_stack([np.ones(n),np.clip((raw-mean)/scale,-3,3)])
                b=base.loc[ids].to_numpy();normal=np.sqrt(np.mean(b*b))
                a=z*b[:,None]/normal;target=(y.loc[ids]-b).to_numpy()/normal
                penalty=np.r_[np.full(4,10.),np.full(len(self.columns)-3,100.)]
                beta=np.linalg.solve(a.T@a/n+np.diag(penalty),a.T@target/n)
                state.update(coef=beta.tolist(),loss_scale=float(normal),
                    null_objective=float(np.mean(target**2)),objective=float(np.mean((target-a@beta)**2)+np.dot(penalty,beta**2)))
            self.parameters[str(s)]=state
        return self

    def _validate(self,f):
        if list(f)!=self.columns:raise ValueError('Residual feature whitelist violated')

    def predict(self,f,base,side):
        self._validate(f);_aligned(f,base,side)
        if not side.isin([0,1]).all() or not np.isfinite(base).all() or base.le(0).any():raise ValueError('Invalid quotes')
        out=base.copy();supported=pd.Series(False,index=f.index)
        for s in (0,1):
            par=self.parameters[str(s)]
            ids=f.index[side.eq(s)&np.isfinite(f).all(axis=1)]
            if not par['supported'] or not len(ids):continue
            z=np.column_stack([np.ones(len(ids)),np.clip((f.loc[ids]-par['mean'])/par['scale'],-3,3)])
            out.loc[ids]=base.loc[ids]*(1+np.clip(z@par['coef'],-.1,.1));supported.loc[ids]=True
        return out,supported

    def state(self):return dict(family=self.family,columns=self.columns,parameters=self.parameters)


class EventResidual:
    def __init__(self,family):
        if family not in ('calendar','reaction'):raise ValueError('Unregistered event family')
        self.family=family;self.columns=BASIS+(REACTION if family=='reaction' else [])

    def _valid(self,f):
        valid=f.calendar_status.eq('known_event')
        if self.family=='reaction':
            if 'reaction_status' not in f or not f.reaction_status.isin(['available','not_yet_observable','missing_market']).all():
                raise ValueError('Reaction availability status required')
            valid &= f.reaction_status.ne('missing_market')
        return valid

    def _design(self,f):
        values=f[self.columns].to_numpy(float)
        valid=self._valid(f).to_numpy()
        return np.column_stack([values*self.enabled[0]*valid[:,None],
            *[values*self.enabled[s+1]*f.option_type.eq(s).to_numpy()[:,None]*valid[:,None] for s in (0,1)]])

    def fit(self,f,base,y,origin):
        _aligned(f,base,y);origin=pd.Timestamp(origin)
        if origin.tzinfo is None or not np.isfinite(f[self.columns]).all().all() or not f.option_type.isin([0,1]).all():
            raise ValueError('Invalid event features')
        if not np.isfinite(base).all() or base.le(0).any() or not np.isfinite(y).all() or y.lt(0).any():raise ValueError('Invalid outcomes')
        keep=self._valid(f)&f.event_time.lt(origin)&f[self.columns].ne(0).any(axis=1)
        x=f.loc[keep];self.enabled=[];self.support=[]
        for scope in ('shared',0,1):
            part=x if scope=='shared' else x.loc[x.option_type.eq(scope)]
            en=[]
            for c in self.columns:
                p=part.loc[part[c].ne(0)];n,events,months=len(p),p.event_id.nunique(),p.event_month.nunique()
                required=(30,5,2) if scope=='shared' else (60,8,3)
                ok=all(a>=b for a,b in zip((n,events,months),required));en.append(ok)
                self.support.append(dict(scope=str(scope),feature=c,orders=n,events=events,months=months,enabled=ok))
            self.enabled.append(en)
        self.enabled=np.array(self.enabled,bool);self.enabled[1:] &= self.enabled[0]
        self.coef=np.zeros(3*len(self.columns));self.orders=len(x);self.events=x.event_id.nunique()
        self.fit_performed=bool(self.enabled.any())
        if self.fit_performed:
            # Each event receives equal aggregate weight, regardless of order crowding.
            w=1/x.event_id.map(x.event_id.value_counts()).to_numpy(float);w/=w.sum()
            b=base.loc[x.index].to_numpy();scale=np.sqrt(np.dot(w,b*b));a=self._design(x)*b[:,None]/scale
            target=(y.loc[x.index].to_numpy()-b)/scale
            penalty=np.r_[np.full(len(self.columns),100.),np.full(2*len(self.columns),1000.)]
            self.coef=np.linalg.solve(a.T@(w[:,None]*a)+np.diag(penalty),a.T@(w*target))
        return self

    def predict(self,f,base):
        _aligned(f,base)
        if not np.isfinite(f[self.columns]).all().all() or not f.option_type.isin([0,1]).all() or not np.isfinite(base).all() or base.le(0).any():
            raise ValueError('Invalid event quote')
        phi=self._design(f);supported=pd.Series(np.any(phi!=0,axis=1),index=f.index)
        return base*(1+np.clip(phi@self.coef,-.05,.05)),supported

    def state(self):return dict(family=self.family,columns=self.columns,coef=self.coef.tolist(),enabled=self.enabled.tolist(),
        support=self.support,orders=self.orders,events=int(self.events),fit_performed=self.fit_performed)
