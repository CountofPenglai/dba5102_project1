"""One causal rolling schedule for every research recipe.

Data stays in a flat pool. Roles are assigned at each update, never by permanent
train/validation directories or random splits. Prediction windows do not read labels.
"""
from dataclasses import dataclass
import copy

import pandas as pd
from .config import load_protocol


def validate_protocol(protocol):
    """Validate temporal settings without hard-coding dates in model branches."""
    protocol = copy.deepcopy(load_protocol() if protocol is None else protocol)
    try:
        start, end, first = (pd.Timestamp(protocol[k]) for k in
                             ('validation_start', 'validation_end', 'training_start'))
        if (any(pd.isna(t) or t.tzinfo is None for t in (start, end, first))
                or not first < start < end or start != start.floor('D')):
            raise ValueError('Invalid timezone-aware rolling dates')
        for key in ('history_days', 'update_days'):
            if type(protocol[key]) is not int or protocol[key] <= 0:
                raise ValueError('Rolling day counts must be positive integers')
    except (KeyError, TypeError) as error:
        raise ValueError('Incomplete rolling protocol') from error
    return protocol


@dataclass(frozen=True)
class QuoteWindow:
    origin: pd.Timestamp
    stop: pd.Timestamp
    query_ids: pd.Index
    is_validation: bool


@dataclass(frozen=True)
class RollingWindow(QuoteWindow):
    history_ids: pd.Index


def quote_windows(created, protocol=None, *, warmup_days=0):
    """Yield nonempty [origin, stop) windows using creation timestamps only.

    Warm-up must align with the validation clock, so month boundaries never reset
    the update schedule. Empty quote weeks do not trigger unused model fitting.
    """
    protocol = validate_protocol(protocol)
    if (not isinstance(created, pd.Series) or not created.index.is_unique
            or created.isna().any() or not isinstance(created.dtype, pd.DatetimeTZDtype)):
        raise ValueError('Quote times must be unique-indexed and timezone-aware')
    start, end = pd.Timestamp(protocol['validation_start']), pd.Timestamp(protocol['validation_end'])
    step = protocol['update_days']
    if (type(warmup_days) is not int or warmup_days < 0 or warmup_days % step
            or start-pd.Timedelta(days=warmup_days) < pd.Timestamp(protocol['training_start'])):
        raise ValueError('Warm-up must lie within training and align with the update clock')
    for origin in pd.date_range(start-pd.Timedelta(days=warmup_days), end,
                                freq=pd.Timedelta(days=step), inclusive='left'):
        stop = min(origin+pd.Timedelta(days=step), end)
        ids = created.index[created.ge(origin) & created.lt(stop)]
        if len(ids):
            yield QuoteWindow(origin, stop, ids, bool(origin >= start))


def history_ids(pool, origin, protocol, window=True):
    """Creation-window membership AND settlement strictly before the update."""
    origin = pd.Timestamp(origin)
    earliest = pd.Timestamp(protocol['training_start'])
    if window:
        earliest = max(earliest, origin-pd.Timedelta(days=protocol['history_days']))
    created = pool.metadata.created_at_utc
    closed = pd.to_datetime(pool.metadata.closed_at_utc, utc=True)
    return pool.X.index[created.ge(earliest) & created.lt(origin) & closed.lt(origin)]


class RollingSplitter:
    """Bind the common clock and causal historical selection to one aligned pool."""
    def __init__(self, pool, protocol=None):
        self.protocol = validate_protocol(protocol)
        if (not pool.X.index.is_unique or not pool.X.index.equals(pool.y.index)
                or not pool.X.index.equals(pool.metadata.index)):
            raise ValueError('Features, targets and metadata must have identical unique indices')
        for name in ('created_at_utc', 'closed_at_utc'):
            dates = pool.metadata[name]
            if (not isinstance(dates.dtype, pd.DatetimeTZDtype)
                    and (name == 'created_at_utc' or dates.notna().any())):
                raise ValueError('Pool timestamps must be timezone-aware')
        if pool.metadata.created_at_utc.isna().any():
            raise ValueError('Missing order creation time')
        self.pool = pool

    def history(self, origin, *, expanding=False):
        return history_ids(self.pool, origin, self.protocol, window=not expanding)

    def windows(self, *, warmup_days=0):
        for q in quote_windows(self.pool.metadata.created_at_utc, self.protocol, warmup_days=warmup_days):
            yield RollingWindow(q.origin, q.stop, q.query_ids, q.is_validation, self.history(q.origin))

    def describe(self, window):
        """Stable provenance fields retained by existing report parameter archives."""
        h = window.history_ids
        created, closed = self.pool.metadata.created_at_utc, self.pool.metadata.closed_at_utc
        return dict(origin=window.origin.isoformat(), stop=window.stop.isoformat(),
            history_orders=len(h), query_orders=len(window.query_ids),
            earliest_training_create=created.loc[h].min().isoformat() if len(h) else None,
            latest_training_close=closed.loc[h].max().isoformat() if len(h) else None)
