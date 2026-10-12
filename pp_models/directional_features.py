"""Versioned supplemental features; canonical preprocessing artifacts stay immutable."""
import csv
from datetime import datetime
import hashlib
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
from pp_audit.market import audit_market
from pp_data.config import WINDOWS, load_config
from pp_data.io import ModelSplit


def read_market_prefix(path, cutoff):
    """Read a chronological CSV prefix, inspecting only the first boundary timestamp.

    The original provider file is chronological. Abort on observed disorder. No
    future tail is scanned or hashed; prices on the boundary row are never parsed.
    """
    end = pd.Timestamp(cutoff)
    validation_end = pd.Timestamp(load_config()["splits"]["validation"]["end"])
    if end.tzinfo is None or end > validation_end:
        raise ValueError("Market cutoff must be timezone-aware and within validation")
    payload, digest, count, previous, boundary = bytearray(), hashlib.sha256(), 0, None, None
    with Path(path).open("rb") as stream:
        header = stream.readline()
        columns = next(csv.reader([header.decode("utf-8-sig")]))
        time_index = columns.index("open_time_utc")
        payload.extend(header)
        digest.update(header)
        for line in stream:
            fields = next(csv.reader([line.decode("utf-8")]))
            stamp = datetime.fromisoformat(fields[time_index].replace("Z", "+00:00"))
            if stamp.tzinfo is None or (previous is not None and stamp < previous):
                raise ValueError("Market prefix is not chronological/timezone-aware")
            if stamp >= end:
                boundary = stamp.isoformat()
                break
            previous = stamp
            payload.extend(line)
            digest.update(line)
            count += 1
    raw = pd.read_csv(BytesIO(payload))
    audit = audit_market(raw)
    usable = audit.usable
    if usable.empty or not usable.close_time_utc.lt(end).all():
        raise ValueError("No valid completed market prefix before cutoff")
    return usable, {"source_path": str(Path(path).resolve()), "cutoff_exclusive": end.isoformat(),
                    "prefix_sha256": digest.hexdigest(), "prefix_rows": count,
                    "usable_rows": len(usable), "boundary_timestamp_only": boundary,
                    "last_usable_close_utc": usable.close_time_utc.max().isoformat(),
                    "prefix_ordering_assumption": "chronological provider file; stop at first boundary"}


def directional_features(created, usable, windows=WINDOWS):
    """Root mean square signed returns; both signs divide by the entire window."""
    if (created.empty or not created.index.is_unique or created.isna().any()
            or created.dt.tz is None or not windows
            or any(isinstance(w, bool) or not isinstance(w, int) or w <= 0 for w in windows)):
        raise ValueError("Invalid order times or windows")
    history = usable.sort_values("close_time_utc").reset_index(drop=True)
    if (history.empty or history.open_time_utc.duplicated().any()
            or not np.isfinite(history.close).all() or history.close.le(0).any()):
        raise ValueError("Expected unique audited market candles")
    returns = np.log(history.close).diff()
    right = history[["close_time_utc"]].copy()
    names = []
    for w in windows:
        contiguous = history.open_time_utc.diff(w).eq(pd.Timedelta(minutes=w))
        for label, series in (("up", returns.clip(lower=0)), ("down", returns.clip(upper=0)),
                              ("total", returns)):
            name = f"{label}_vol_{w}m"
            right[name] = np.sqrt(series.pow(2).rolling(w, min_periods=w).mean()).where(contiguous)
            names.append(name)
    left = pd.DataFrame({"created": created.array, "position": np.arange(len(created))})
    matched = pd.merge_asof(left.sort_values("created"), right, left_on="created", right_on="close_time_utc",
                           direction="backward", tolerance=pd.Timedelta(seconds=60), allow_exact_matches=True)
    matched = matched.set_index("position").reindex(range(len(created)))
    result = matched[names].set_axis(created.index)
    if not np.isfinite(result).all().all():
        raise ValueError("Missing contiguous direction features; canonical orders must not be silently dropped")
    result["feature_last_close_utc"] = pd.Series(matched.close_time_utc.array, index=created.index)
    return result


def augment_pool(pool, usable):
    if not pool.X.index.equals(pool.y.index) or not pool.X.index.equals(pool.metadata.index):
        raise ValueError("Misaligned canonical pool")
    extra = directional_features(pool.metadata.created_at_utc, usable)
    if not extra.feature_last_close_utc.equals(pool.metadata.last_candle_close_utc.rename("feature_last_close_utc")):
        raise ValueError("Supplemental candle timestamps differ from canonical features")
    extra = extra.drop(columns="feature_last_close_utc")
    max_error = 0.
    for w in WINDOWS:
        rebuilt = extra[f"total_vol_{w}m"] * np.sqrt(w)
        canonical = pool.X[f"realized_vol_{w}m"]
        max_error = max(max_error, float(np.abs(rebuilt - canonical).max()))
        if not np.allclose(rebuilt, canonical, rtol=1e-9, atol=1e-12):
            raise ValueError("Supplemental total volatility differs from canonical features")
    if set(extra) & set(pool.X):
        raise ValueError("Supplemental columns collide with canonical features")
    X = pd.concat([pool.X, extra], axis=1)
    return ModelSplit(X, pool.y.copy(), pool.metadata.copy()), {
        "version": "directional_features_v1", "orders": len(X), "windows_minutes": list(WINDOWS),
        "normalization": "sqrt(sum signed minute log returns squared / all window minutes)",
        "max_canonical_volatility_absolute_error": max_error,
        "last_close_matches_canonical": True, "added_features": list(extra)}
