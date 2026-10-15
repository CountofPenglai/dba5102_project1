"""Shared synthetic fixture; extracted without importing an unrelated study."""
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from pp_data.io import ModelSplit


def fixture():
    created = pd.Series(pd.date_range("2026-02-01", "2026-04-04", freq="3h", tz="UTC", inclusive="left"))
    n = len(created)
    X = pd.DataFrame({"option_type": np.arange(n) % 2,
        BASE_FEATURE: np.tile([80., 100., 120.], n // 3 + 1)[:n],
        "remaining_seconds": 3600., "product_duration_seconds": 3600.,
        "realized_vol_60m": .02, "log_moneyness": 0., "quantity_btc": 1.,
        "up_vol_60m": np.where(np.arange(n) % 7 < 4, .01, .03), "up_vol_240m": .02,
        "down_vol_60m": np.where(np.arange(n) % 5 < 3, .01, .03), "down_vol_240m": .02})
    metadata = pd.DataFrame({"created_at_utc": created,
        "closed_at_utc": created + pd.Timedelta(hours=1),
        "expiration_at_utc": created + pd.Timedelta(hours=1)})
    return ModelSplit(X, pd.Series(np.where(np.arange(n) % 3 == 0, 0., 120.), index=X.index), metadata)
