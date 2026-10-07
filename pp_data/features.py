"""Feature whitelist derived solely from contract inputs and completed candles."""

import numpy as np
import pandas as pd
from .config import WINDOWS

MARKET_FEATURES = ["market_price_usdc_per_btc"] + [
    name for n in WINDOWS for name in (f"log_return_{n}m", f"realized_vol_{n}m")
]
RISK_FEATURES = [
    "option_type", "product_duration_seconds", "remaining_seconds", "quantity_btc",
    "strike_price_usdc_per_btc", "market_price_usdc_per_btc", "log_moneyness",
] + MARKET_FEATURES[1:]
BASE_FEATURE = "api_base_price_per_btc_usdc"
FEATURE_COLUMNS = RISK_FEATURES + [BASE_FEATURE]


def build_features(orders: pd.DataFrame, usable_market: pd.DataFrame):
    """Return features and matched candle times. Outcomes are never read here.

    Market input must be audit_market(...).usable (unique, valid, sorted candles).
    N-minute returns need N+1 candles; realized vol is non-annualized sqrt(sum r²).
    """
    created = pd.to_datetime(orders.createdAt, utc=True, errors="coerce")
    expiry = pd.to_datetime(orders.expirationDate, utc=True, errors="coerce")
    x = pd.DataFrame(index=orders.index, columns=FEATURE_COLUMNS, dtype=float)
    for source, dest in {
        "optionType": "option_type", "productType": "product_duration_seconds",
        "quantity_BTC": "quantity_btc", "strikePrice": "strike_price_usdc_per_btc",
        "apiBasePrice_per_BTC_USDC": BASE_FEATURE,
    }.items():
        x[dest] = pd.to_numeric(orders[source], errors="coerce")
    x["remaining_seconds"] = (expiry - created).dt.total_seconds()
    last_close = pd.Series(pd.NaT, index=orders.index, dtype="datetime64[ns, UTC]")
    if not usable_market.empty and created.notna().any():
        history = usable_market.sort_values("close_time_utc").reset_index(drop=True)
        right = history[["close_time_utc"]].copy()
        right["market_price_usdc_per_btc"] = history.close
        log_price = np.log(history.close)
        returns = log_price.diff()
        for n in WINDOWS:
            # N+1 distinct minute starts with exact span establish a contiguous window.
            continuous = history.open_time_utc.diff(n).eq(pd.Timedelta(minutes=n))
            right[f"log_return_{n}m"] = (log_price - log_price.shift(n)).where(continuous)
            right[f"realized_vol_{n}m"] = np.sqrt(
                returns.pow(2).rolling(n, min_periods=n).sum()
            ).where(continuous)
        left = pd.DataFrame({"created": created.to_numpy(), "position": np.arange(len(orders))})
        joined = pd.merge_asof(
            left.dropna(subset="created").sort_values("created"), right,
            left_on="created", right_on="close_time_utc", direction="backward",
            tolerance=pd.Timedelta(seconds=60), allow_exact_matches=True,
        ).set_index("position").reindex(range(len(orders)))
        x[MARKET_FEATURES] = joined[MARKET_FEATURES].to_numpy()
        last_close = pd.Series(joined.close_time_utc.array, index=orders.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        x["log_moneyness"] = np.log(x.market_price_usdc_per_btc) - np.log(x.strike_price_usdc_per_btc)
    return x, last_close
