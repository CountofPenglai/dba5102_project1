"""Strongly shrunk overlays trained on settled forward residuals only."""
import numpy as np
import pandas as pd

OVERLAYS=('level_add','level_scale','jump','streak05','streak10','uniform_jump','uniform_streak05','uniform_streak10')
SPECS={'jump':('jump',.05),'streak05':('streak',.05),'streak10':('streak',.10)}


def validate(f,training=False):
    columns=['h','jump','streak','side','quantity']+(['actual'] if training else [])
    if (not f.index.is_unique or not np.isfinite(f[columns]).all().all() or f.h.le(0).any()
        or f.quantity.le(0).any() or not f.side.isin([0,1]).all()
        or not f[['jump','streak']].ge(0).all().all() or not f[['jump','streak']].le(1).all().all()
        or (training and (f.actual.lt(0).any() or f.created.dt.tz is None or f.created.isna().any()))):
        raise ValueError('Invalid overlay data')


class Extras:
    def fit(self,f):
        validate(f,True);n=len(f);days=f.created.dt.floor('D').nunique()
        self.state={'orders':n,'days':int(days),'level_supported':bool(n>=100 and days>=10),'additive':0.,'scale':0.}
        h=f.h.to_numpy();residual=(f.actual-f.h).to_numpy()
        if self.state['level_supported']:
            self.state['additive']=float(np.clip(residual.sum()/(n+100),-.05*np.median(h),.05*np.median(h)))
            self.state['scale']=float(np.clip(h@residual/(h@h+100*np.mean(h*h)),-.05,.05))
        for name,(signal,cap) in SPECS.items():
            self.state[name]={}
            for s in (0,1):
                p=f.loc[f.side.eq(s)];active=p.loc[p[signal]>0];nn=len(p);dd=p.created.dt.floor('D').nunique()
                ok=nn>=100 and dd>=10 and len(active)>=30 and active.created.dt.floor('D').nunique()>=5
                beta=0.;uniform=0.
                if ok:
                    h=p.h.to_numpy();g=p[signal].to_numpy();z=h*g;r=(p.actual-p.h).to_numpy()
                    beta=float(np.clip(z@r/(z@z+100*np.mean(h*h)),0,cap))
                    uniform=float(np.sum(p.quantity*h*beta*g)/np.sum(p.quantity*h))
                self.state[name][str(s)]=dict(orders=nn,days=int(dd),active_orders=len(active),
                    active_days=int(active.created.dt.floor('D').nunique()),supported=bool(ok),beta=beta,uniform=uniform,cap=cap)
        return self

    def predict(self,f):
        validate(f);h=f.h
        out={'level_add':(h+self.state['additive']).clip(lower=0),'level_scale':h*(1+self.state['scale'])}
        for name,(signal,cap) in SPECS.items():
            coef=f.side.map({s:self.state[name][str(s)]['beta'] for s in (0,1)})
            uniform=f.side.map({s:self.state[name][str(s)]['uniform'] for s in (0,1)})
            out[name]=h*(1+coef*f[signal]);out['uniform_'+name]=h*(1+uniform)
        return out
