"""One authoritative weekly protocol and causal historical/tail selection."""
import numpy as np
import pandas as pd
from pp_data.config import PROTOCOL_PATH, load_protocol
PROTOCOL = load_protocol()

def history_ids(pool, origin, protocol, window=True):
    created = pool.metadata.created_at_utc; closed = pool.metadata.closed_at_utc
    earliest = pd.Timestamp(protocol['training_start'])
    if window: earliest = max(earliest, origin-pd.Timedelta(days=protocol['history_days']))
    return pool.X.index[created.ge(earliest) & created.lt(origin) & closed.lt(origin)]


def tail_context(pool, history, today):
    h, q = pool.X.loc[history], pool.X.loc[today]
    threshold = pd.Series(np.nan, index=today, dtype=float)
    source = pd.Series("unavailable", index=today, dtype=object)
    for (side, tenor), part in q.groupby(["option_type", "product_duration_seconds"]):
        side_ids = h.index[h.option_type.eq(side)]
        cell = h.index[h.option_type.eq(side) & h.product_duration_seconds.eq(tenor)]
        ids = cell if len(cell) >= 30 else side_ids
        if len(ids) >= 30:
            threshold.loc[part.index] = pool.y.loc[ids].quantile(.9)
            source.loc[part.index] = "direction_tenor" if len(cell) >= 30 else "direction_fallback"
    return threshold, source
