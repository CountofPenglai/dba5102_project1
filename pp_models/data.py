"""Canonical data access and strict daily, as-of settlement filtering."""
import pandas as pd
from pp_data import load_split
from pp_data.config import load_config
from pp_data.io import ModelSplit


def load_pool(split, resolved, data_dir=None):
    if split not in ("validation", "test"):
        raise ValueError("Evaluate only validation or test on the fixed date partitions")
    if split == "validation":
        from .scoped_data import load_validation_pool
        return load_validation_pool(resolved, data_dir)
    parts = [load_split(name, cohort=resolved["cohort"], use_base=resolved["use_base"], output_dir=data_dir)
             for name in (("train", "validation") if split == "validation" else ("train", "validation", "test"))]
    pool = ModelSplit(pd.concat([p.X for p in parts]), pd.concat([p.y for p in parts]),
                      pd.concat([p.metadata for p in parts]))
    if not pool.X.index.is_unique:
        raise ValueError("Duplicate IDs across canonical splits")
    order = pool.metadata.sort_values("created_at_utc", kind="stable").index
    return ModelSplit(pool.X.loc[order], pool.y.loc[order], pool.metadata.loc[order])


def period_bounds(split, period=None):
    bounds = load_config()["splits"][split]
    outer = tuple(pd.Timestamp(bounds[k]) for k in ("start", "end"))
    start, end = outer if period is None else tuple(pd.Timestamp(t) for t in period)
    if (start.tzinfo is None or end.tzinfo is None or not outer[0] <= start < end <= outer[1]):
        raise ValueError("Evaluation period must be timezone-aware and inside the fixed split")
    return start, end


def daily_batches(pool, split, window_days, period=None):
    if split not in ("validation", "test"):
        raise ValueError("Expected validation or test")
    if not pool.X.index.equals(pool.y.index) or not pool.X.index.equals(pool.metadata.index):
        raise ValueError("Pool IDs are not aligned")
    if not pool.X.index.is_unique:
        raise ValueError("Pool IDs must be unique")
    if isinstance(window_days, bool) or not isinstance(window_days, int) or window_days <= 0:
        raise ValueError("window_days must be a positive integer")
    start, end = period_bounds(split, period)
    created, closed = pool.metadata.created_at_utc, pool.metadata.closed_at_utc
    eligible = created.ge(start) & created.lt(end) & closed.lt(end)
    if not eligible.any():
        raise ValueError("No prediction orders in requested fixed split")
    day_for_order = created.dt.floor("D")
    for day in sorted(day_for_order[eligible].unique()):
        history = created.ge(day - pd.Timedelta(days=window_days)) & created.lt(day) & closed.lt(day)
        today = eligible & day_for_order.eq(day)
        yield pd.Timestamp(day), pool.X.index[history], pool.X.index[today]


def coverage(pool, split, resolved, period=None):
    rows = [{"day_utc": str(day), "training_rows": len(train), "prediction_rows": len(test)}
            for day, train, test in daily_batches(pool, split, resolved["window_days"], period)]
    start, end = period_bounds(split, period)
    in_period = pool.metadata.created_at_utc.ge(start) & pool.metadata.created_at_utc.lt(end)
    return {"prediction_days": len(rows), "prediction_rows": sum(r["prediction_rows"] for r in rows),
            "labels_unavailable_at_period_end": int((in_period & pool.metadata.closed_at_utc.ge(end)).sum()),
            "min_training_rows": min(r["training_rows"] for r in rows),
            "max_training_rows": max(r["training_rows"] for r in rows),
            "insufficient_history_days": sum(r["training_rows"] < resolved["min_train_rows"] for r in rows)}
