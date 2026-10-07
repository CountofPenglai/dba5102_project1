import pandas as pd


def order(**changes):
    row = dict(
        premiumAmount_USDC=10.0,
        createdAt="2026-04-30 23:58:30",
        closedAt="2026-05-01 00:28:12",
        expirationDate="2026-05-01 00:28:10",
        optionType=0,
        productType=1800,
        quantity_BTC=1.0,
        strikePrice=60000.0,
        realizedProfit_USDC=0.0,
        optionType_label="Call",
        orderPrice_per_BTC_USDC=10.0,
        apiBasePrice_per_BTC_USDC=8.0,
        markup_per_BTC_USDC=2.0,
    )
    return row | changes


def candles(start="2026-04-30 00:00:00", periods=1440):
    opens = pd.date_range(start, periods=periods, freq="min", tz="UTC")
    return pd.DataFrame(dict(
        symbol="BTCUSDC", interval="1m", open_time_utc=opens,
        close_time_utc=opens + pd.Timedelta(milliseconds=59999),
        open=60000.0, high=60010.0, low=59990.0, close=60001.0,
        volume=2.0, quote_asset_volume=120000.0, number_of_trades=10,
        taker_buy_base_asset_volume=1.0, taker_buy_quote_asset_volume=60000.0,
    ))


def metric(result, name):
    frame = result.metrics if hasattr(result, "metrics") else result
    selected = frame.loc[frame.metric == name, "value"]
    if len(selected) != 1:
        raise AssertionError(f"Expected one metric {name}, got {len(selected)}")
    return selected.iloc[0]
