"""One authoritative weekly protocol and causal historical/tail selection."""
import numpy as np
import pandas as pd
from pp_data.config import PROTOCOL_PATH, load_protocol
from pp_data.splitting import history_ids
PROTOCOL = load_protocol()


def tail_context(pool, history, today):
    """Historical q90 thresholds are scoring context, never prediction inputs."""
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


def prediction_context(pool, window, splitter):
    """Build common scoring metadata; monthly tails never enter model inputs."""
    q = window.query_ids
    created = pool.metadata.created_at_utc
    x = pool.X.loc[q]
    threshold = pd.Series(index=q, dtype=float)
    source = pd.Series(index=q, dtype=object)
    months = created.loc[q].dt.strftime('%Y-%m')
    for month in sorted(months.unique()):
        month_start = pd.Timestamp(month+'-01', tz='UTC')
        ids = q[months.eq(month)]
        values, origins = tail_context(pool, splitter.history(month_start, expanding=True), ids)
        threshold.loc[ids], source.loc[ids] = values, origins
    return pd.DataFrame(dict(day=created.loc[q].dt.floor('D'), month=months,
        side=x.option_type.astype(int), tenor=x.product_duration_seconds.astype(int), quantity=x.quantity_btc,
        tail_threshold=threshold, tail_source=source, origin=window.origin.isoformat()), index=q)
