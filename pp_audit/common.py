"""Shared parsing and compact aggregate metric representation."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


ORDER_DATES = ["createdAt", "closedAt", "expirationDate"]
ORDER_NUMBERS = [
    "premiumAmount_USDC", "optionType", "productType", "quantity_BTC",
    "strikePrice", "realizedProfit_USDC", "orderPrice_per_BTC_USDC",
    "apiBasePrice_per_BTC_USDC", "markup_per_BTC_USDC",
]
ORDER_COLUMNS = ORDER_DATES + ORDER_NUMBERS + ["optionType_label"]
MARKET_DATES = ["open_time_utc", "close_time_utc"]
MARKET_NUMBERS = [
    "open", "high", "low", "close", "volume", "quote_asset_volume",
    "number_of_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
]
MARKET_COLUMNS = MARKET_DATES + MARKET_NUMBERS + ["symbol", "interval"]


def parse_frame(raw, required, dates, numbers):
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if raw.empty:
        raise ValueError("Input dataset is empty")
    frame = raw.reset_index(drop=True).copy()
    # Treat whitespace-only CSV cells as missing, not malformed values.
    frame = frame.replace(r"^\s*$", np.nan, regex=True)
    source_missing = frame[required].isna()
    bad_numeric = pd.DataFrame(False, index=frame.index, columns=numbers)
    bad_dates = pd.DataFrame(False, index=frame.index, columns=dates)
    for name in numbers:
        parsed = pd.to_numeric(frame[name], errors="coerce")
        bad_numeric[name] = ~source_missing[name] & ~np.isfinite(parsed)
        frame[name] = parsed.where(np.isfinite(parsed))
    for name in dates:
        parsed = pd.to_datetime(frame[name], format="mixed", errors="coerce", utc=True)
        bad_dates[name] = ~source_missing[name] & parsed.isna()
        frame[name] = parsed
    return frame, source_missing, bad_numeric, bad_dates


def ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return None
    return float(numerator / denominator)


def total(series):
    return series.sum(min_count=1)


@dataclass
class Metrics:
    records: list = field(default_factory=list)

    def add(self, name, value, unit, definition, denominator=None):
        self.records.append(dict(
            metric=name, value=value, unit=unit, denominator=denominator,
            rate=ratio(value, denominator) if denominator is not None else None,
            definition=definition,
        ))

    def count(self, name, mask, definition, denominator=None):
        self.add(name, int(mask.sum()), "rows", definition,
                 len(mask) if denominator is None else denominator)

    def frame(self):
        return pd.DataFrame(self.records, columns=[
            "metric", "value", "unit", "denominator", "rate", "definition",
        ])
