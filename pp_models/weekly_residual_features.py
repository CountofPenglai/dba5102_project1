"""Causal jump and conservative trend signals; no outcome or settlement inputs."""
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE


def streak_state(aligned_returns, sigma):
    """Three meaningful same-direction bars, then a two-sigma run gate."""
    if len(aligned_returns)!=len(sigma):raise ValueError('Misaligned bars')
    out=[];run=0;pressure=0.;state=0.;quiet=0
    for r,s in zip(aligned_returns,sigma):
        if not np.isfinite(r) or not np.isfinite(s) or s<=0:
            run=0;pressure=0.;state=0.;quiet=0;out.append(0.);continue
        if r/s>.25:run+=1;pressure+=r/s
        else:run=0;pressure=0.
        trigger=run>=3 and pressure>2*np.sqrt(run)
        if trigger:
            state=max(state,float(np.clip(pressure/(2*np.sqrt(run))-1,0,1)));quiet=0
        else:
            quiet+=1;state=state*.5 if quiet<12 else 0.
        out.append(state)
    return np.array(out)


def _match(created,right,time,tolerance):
    left=pd.DataFrame({'created':created.array,'position':range(len(created))})
    return pd.merge_asof(left.sort_values('created'),right.sort_values(time),left_on='created',right_on=time,
        direction='backward',tolerance=pd.Timedelta(tolerance)).set_index('position').reindex(range(len(created)))


def features(x,created,market):
    if (not x.index.equals(created.index) or not x.index.is_unique or created.dt.tz is None
        or created.isna().any() or not x.option_type.isin([0,1]).all()
        or not np.isfinite(x[[BASE_FEATURE,'remaining_seconds','market_price_usdc_per_btc']]).all().all()
        or x[[BASE_FEATURE,'remaining_seconds','market_price_usdc_per_btc']].le(0).any().any()):
        raise ValueError('Invalid pricing-time inputs')
    m=market.sort_values('open_time_utc').copy()
    if (m.empty or m.open_time_utc.dt.tz is None or m.open_time_utc.duplicated().any()
        or not m.open_time_utc.eq(m.open_time_utc.dt.floor('min')).all()
        or not m.close_time_utc.eq(m.open_time_utc+pd.Timedelta(milliseconds=59999)).all()
        or not np.isfinite(m.close).all() or m.close.le(0).any()):raise ValueError('Invalid minute market')
    # Reindex explicitly: gaps must invalidate rolling windows, not compress time.
    t=pd.date_range(m.open_time_utc.min(),m.open_time_utc.max(),freq='min')
    p=m.set_index('open_time_utc').close.reindex(t);r=np.log(p).diff();r2=r*r
    sigma=np.sqrt(r2.shift(1).rolling(240).mean()).clip(lower=1e-6)
    rv=r2.rolling(60).sum();bv=(np.pi/2)*60/59*(r.abs()*r.abs().shift(1)).rolling(59).sum()
    j=(rv-bv).clip(lower=0)
    large=r.abs()>3*sigma
    # A missing scale is unknown, rather than a negative jump observation.
    valid=sigma.notna()&r.notna()
    up=r2.where(large&r.gt(0),0.).where(valid).rolling(60).sum()
    down=r2.where(large&r.lt(0),0.).where(valid).rolling(60).sum()
    share_up=(up/(up+down)).where(up+down>0,0.);share_down=(down/(up+down)).where(up+down>0,0.)
    right=pd.DataFrame({'at':t+pd.Timedelta(minutes=1),'up':j*share_up,'down':j*share_down,
        'ok':rv.notna()&bv.notna()&up.notna()&down.notna()})
    a=_match(created,right,'at','59.999999999s')
    # Only complete five-minute buckets have a price; never fill an incomplete bucket.
    bars=p.resample('5min').agg(['last','count']);price=bars['last'].where(bars['count'].eq(5))
    br=np.log(price).diff();bs=np.sqrt(br.pow(2).shift(1).rolling(288).mean()).clip(lower=.0001)
    trends=pd.DataFrame({'bar_at':bars.index+pd.Timedelta(minutes=5),
        'call_streak':streak_state(br.to_numpy(),bs.to_numpy()),
        'put_streak':streak_state(-br.to_numpy(),bs.to_numpy()),'ok_bar':br.notna()&bs.notna()})
    b=_match(created,trends,'bar_at','299.999999999s')
    jump_ok=a.ok.eq(True).to_numpy();streak_ok=b.ok_bar.eq(True).to_numpy()
    side=x.option_type.to_numpy();direction_j=np.where(side==0,a.up,a.down)
    ratio=x.market_price_usdc_per_btc.to_numpy()*np.sqrt(direction_j*np.minimum(x.remaining_seconds.to_numpy()/3600,4))/x[BASE_FEATURE].to_numpy()
    jump=np.where(jump_ok,np.clip(ratio-1,0,1),0.)
    streak=np.where(streak_ok,np.where(side==0,b.call_streak,b.put_streak),0.)
    out=pd.DataFrame({'jump':jump,'streak':streak,'jump_available':jump_ok,'streak_available':streak_ok,
        'available':jump_ok&streak_ok},index=x.index)
    if not np.isfinite(out[['jump','streak']]).all().all():raise ValueError('Nonfinite signals')
    return out
