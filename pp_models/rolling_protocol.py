"""Active rolling evaluation roles over immutable v1 development artifacts.

The storage split is retained for reproduction; v2 evaluation uses Apr--Jun.
All new rolling studies share this one versioned protocol, never per-run dates.
"""
import json
import pandas as pd
from pp_data.config import PROJECT
from .scoped_data import load_validation_pool
from .runs import fingerprint

PROTOCOL_PATH=PROJECT/'config/validation_protocol_v2.json'


def validate_protocol(c):
    if (c.get('version')!='rolling_validation_v2' or c.get('timezone')!='UTC'
        or c.get('input_cohort')!='immutable_preprocessing_v1_development_union'
        or c.get('warmup_start')!='2026-02-01T00:00:00Z'
        or c.get('validation_start')!='2026-04-01T00:00:00Z'
        or c.get('validation_end')!='2026-07-01T00:00:00Z'
        or c.get('scoring_label_cutoff')!=c.get('validation_end')
        or c.get('report_months')!=['2026-04','2026-05','2026-06']
        or isinstance(c.get('history_days'),bool) or c.get('history_days')!=30
        or c.get('minimum_history_orders')!=100):
        raise ValueError('Unsupported rolling protocol; July is reserved')


def load_protocol():
    config=json.loads(PROTOCOL_PATH.read_text());validate_protocol(config);return config


def load_rolling_pool(cohort='benchmark'):
    return load_validation_pool({'cohort':cohort,'use_base':cohort=='benchmark'})


def rolling_batches(pool,config):
    validate_protocol(config)
    if (not pool.X.index.is_unique or not pool.X.index.equals(pool.y.index)
        or not pool.X.index.equals(pool.metadata.index)):raise ValueError('Misaligned rolling pool')
    created,closed=pool.metadata.created_at_utc,pool.metadata.closed_at_utc
    start,end,warmup=map(pd.Timestamp,[config['validation_start'],config['validation_end'],config['warmup_start']])
    if (created.dt.tz is None or closed.dt.tz is None or created.isna().any() or closed.isna().any()
        or created.lt(warmup).any() or created.ge(end).any() or closed.lt(created).any()):
        raise ValueError('Invalid pre-July development pool')
    eligible=created.ge(start)&created.lt(end)&closed.lt(end)
    for day in pd.date_range(start,end,freq='D',inclusive='left'):
        today=eligible&created.ge(day)&created.lt(day+pd.Timedelta(days=1))
        if not today.any():continue
        train=created.ge(day-pd.Timedelta(days=config['history_days']))&created.lt(day)&closed.lt(day)
        yield day,pool.X.index[train],pool.X.index[today]


def protocol_coverage(pool,config):
    batches=list(rolling_batches(pool,config));rows=[]
    for month in config['report_months']:
        sub=[(d,h,q) for d,h,q in batches if d.strftime('%Y-%m')==month]
        ids=[str(i) for _,_,q in sub for i in q]
        rows.append({'month':month,'scoring_orders':len(ids),'scoring_days':len(sub),
            'minimum_training_orders':min((len(h) for _,h,_ in sub),default=0),
            'maximum_training_orders':max((len(h) for _,h,_ in sub),default=0),
            'scoring_ids_sha256':fingerprint(ids)})
    return pd.DataFrame(rows)
