"""Minute candle quality and usable history audit."""

from dataclasses import dataclass
import pandas as pd
from .common import MARKET_COLUMNS, MARKET_DATES, MARKET_NUMBERS, Metrics, parse_frame


@dataclass
class MarketAudit:
    metrics: pd.DataFrame
    rows: pd.DataFrame
    usable: pd.DataFrame


def audit_market(raw: pd.DataFrame) -> MarketAudit:
    d, missing, bad_num, bad_dates = parse_frame(
        raw, MARKET_COLUMNS, MARKET_DATES, MARKET_NUMBERS,
    )
    open_valid = d.open_time_utc.notna()
    duplicates = open_valid & d.open_time_utc.duplicated()
    ambiguous = open_valid & d.open_time_utc.duplicated(keep=False)
    on_grid = open_valid & d.open_time_utc.eq(d.open_time_utc.dt.floor("min"))
    expected_close = d.open_time_utc + pd.Timedelta(milliseconds=59999)
    close_valid = (d.close_time_utc - expected_close).abs().le(pd.Timedelta(milliseconds=1))
    price_cols = ["open", "high", "low", "close"]
    price_complete = d[price_cols].notna().all(axis=1)
    ohlc_bad = price_complete & (
        d[price_cols].le(0).any(axis=1)
        | d.high.lt(d[["open", "close", "low"]].max(axis=1))
        | d.low.gt(d[["open", "close", "high"]].min(axis=1))
    )
    volumes = ["volume", "quote_asset_volume", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
    volume_bad = (
        d[volumes].lt(0).any(axis=1) | d.number_of_trades.lt(0)
        | d.number_of_trades.mod(1).ne(0)
        | (d.taker_buy_base_asset_volume - d.volume).gt(1e-6)
        | (d.taker_buy_quote_asset_volume - d.quote_asset_volume).gt(1e-6)
    )
    instrument_bad = ~d.symbol.eq("BTCUSDC") | ~d.interval.eq("1m")
    usable_mask = (
        d[MARKET_COLUMNS].notna().all(axis=1) & on_grid & close_valid
        & ~ambiguous & ~ohlc_bad & ~volume_bad & ~instrument_bad
    )
    usable = d.loc[usable_mask].sort_values("close_time_utc").reset_index(drop=True)
    aligned = d.loc[on_grid, "open_time_utc"].drop_duplicates().sort_values()
    gaps = aligned.diff().dt.total_seconds().div(60).sub(1).clip(lower=0)
    missing_minutes = int(gaps.sum()) if len(aligned) else None
    m = Metrics()
    m.add("rows", len(d), "rows", "原始一分钟行情数量")
    m.add("first_open_utc", d.open_time_utc.min(), "UTC", "原始可解析行情最早开始时间")
    m.add("last_close_utc", d.close_time_utc.max(), "UTC", "原始可解析行情最晚结束时间，不等于全部可用")
    checks = [
        ("missing_value_rows", missing.any(axis=1), "任意必需字段原始缺失"),
        ("numeric_invalid_rows", bad_num.any(axis=1), "非空数值不可解析或非有限"),
        ("timestamp_invalid_rows", bad_dates.any(axis=1), "非空时间不可解析"),
        ("duplicate_open_time_rows", duplicates, "重复开始时间的多余记录"),
        ("ambiguous_open_time_rows", ambiguous, "重复开始时间的全部记录，均不用于历史窗口"),
        ("off_grid_or_missing_open_rows", ~on_grid, "开始时间缺失或不在整分钟"),
        ("invalid_close_time_rows", ~close_valid, "结束时间缺失或不符合一分钟结束时刻，容差1毫秒"),
        ("invalid_ohlc_rows", ohlc_bad, "完整OHLC中的非正价格或高低价关系错误"),
        ("invalid_volume_or_trade_rows", volume_bad, "负成交量、非整数成交笔数、或主动买入超过总量"),
        ("unexpected_instrument_rows", instrument_bad, "标的不是BTCUSDC或周期不是1m"),
        ("usable_rows", usable_mask, "通过全部质量检查且不存在重复时刻的行情"),
    ]
    for name, mask, definition in checks:
        m.count(name, mask, definition)
    m.add("missing_minutes", missing_minutes, "minutes", "原始整分钟时间网格内缺失的分钟数；不包含窗口外和有记录但无效的分钟")
    m.add("largest_gap_minutes", int(gaps.max()) if gaps.notna().any() else (0 if len(aligned) else None), "minutes", "原始整分钟时间网格最大连续缺口")
    m.add("backward_time_steps", int(d.open_time_utc.diff().lt(pd.Timedelta(0)).sum()), "steps", "源文件相邻可解析时间倒序的次数")
    return MarketAudit(m.frame(), d, usable)
