"""Small directional peak-payout classifier with explicit support fallback."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from pp_data.config import PROJECT
from .high_risk_features import FEATURE_NAMES

CONFIG_PATH=PROJECT/'config/high_risk_gbdt_v1.json'


def load_settings():
    c=json.loads(CONFIG_PATH.read_text())
    if (c['version']!='high_risk_gbdt_v1' or c['protocol']!='config/validation_protocol_v2.json'
            or c['targets']!=['high_risk_unit_q975','high_risk_unit_q950']
            or c['minimum_orders']!=100 or c['minimum_positives']!=10
            or c['minimum_negatives']!=20 or c['minimum_positive_days']!=3
            or c['alarm_probability']!=.2 or c['alarm_multiplier']!=1.5
            or c['diagnostic_probabilities']!=[.1,.2,.3,.5]
            or c['parameter_search'] is not False or c['use_test'] is not False
            or c['classifier']!={'learning_rate':.05,'max_iter':80,'max_leaf_nodes':4,'max_depth':2,
                'min_samples_leaf':30,'l2_regularization':10.,'max_bins':64,
                'early_stopping':False,'random_state':20261009}):
        raise ValueError('Unsupported fixed high-risk classifier study settings')
    return c


def _check_x(x):
    if (not isinstance(x,pd.DataFrame) or not x.index.is_unique or list(x)!=FEATURE_NAMES
            or np.isinf(x.to_numpy(float)).any() or not x.option_type.isin([0,1]).all()
            or x.option_type.nunique()>1):
        raise ValueError('Classifier requires one direction and the pricing-time whitelist')


class PeakClassifier:
    def __init__(self,settings):
        self.settings=settings;self.fitted=False

    def fit(self,x,y,created):
        _check_x(x)
        if (not isinstance(y,pd.Series) or not x.index.equals(y.index) or not y.isin([0,1]).all()
                or not x.index.equals(created.index) or created.isna().any() or created.dt.tz is None):
            raise ValueError('Misaligned binary targets or creation dates')
        c=self.settings;n=len(y);positive=int(y.sum());negative=n-positive
        days=int(created[y.eq(1)].dt.floor('D').nunique())
        self.side=int(x.option_type.iloc[0]) if n else None
        self.prior=(positive+.5)/(n+1)
        reasons=[]
        for value,key in [(n,'orders'),(positive,'positives'),(negative,'negatives'),(days,'positive_days')]:
            if value<c['minimum_'+key]:reasons.append('insufficient_'+key)
        self.trained=not reasons;self.model=None
        if self.trained:
            self.model=HistGradientBoostingClassifier(**c['classifier'])
            with threadpool_limits(limits=1):self.model.fit(x,y.astype(int))
        self.detail={'training_orders':n,'positive_orders':positive,'negative_orders':negative,
            'positive_days':days,'historical_probability':self.prior,'trained':self.trained,
            'fallback_reason':'|'.join(reasons) if reasons else 'none'}
        self.fitted=True;return self

    def predict(self,x):
        if not self.fitted:raise ValueError('Classifier not fitted')
        _check_x(x)
        if self.side is not None and len(x) and not x.option_type.eq(self.side).all():
            raise ValueError('Prediction direction differs from training')
        if x.empty:return np.array([],dtype=float)
        if not self.trained:return np.full(len(x),self.prior)
        with threadpool_limits(limits=1):p=self.model.predict_proba(x)[:,1]
        if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid risk probability')
        return p


def risk_multiplier(probability,trained,settings):
    p=np.asarray(probability,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Expected finite probabilities in [0,1]')
    return np.where(np.asarray(trained,dtype=bool)&(p>=settings['alarm_probability']),settings['alarm_multiplier'],1.)
