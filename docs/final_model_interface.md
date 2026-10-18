# Final model interface / 最终模型接口

Updated: 2026-10-18. The official brief accepts a pricing multiplier or premium per observation; it does not prescribe a Python function signature. This repository provides both a Python API and a CSV CLI.

## Model and parameter state

`pp_pricing.FinalPricingModel` implements the reported `blend_c040_p015`:

```text
H0 = probability_of_positive_payout * Base * positive_payout_coefficient
prediction = alpha * H0 + (1-alpha) * Base
pricing_multiplier = prediction / Base
alpha = 0.40 for Call (0), 0.15 for Put (1)
```

The artifact [pp_pricing/model.json](../pp_pricing/model.json) contains the existing **2026-07-31 00:00 UTC** weekly state, exported from `report_announcement_v1_20261016`. Its 2,131 historical orders were created in the prior 30 days; the latest training settlement was 2026-07-30 23:51:36 UTC. This export performs no refitting and uses no official validation data. It is not an all-history model.

The backtest refits every seven days when settled labels are available. The delivered artifact remains frozen for an unlabeled evaluation file, while quote-time inputs change per order. It rejects quotes earlier than its fitting timestamp. Use the original weekly states to reproduce May–July results, not this last snapshot for the whole period. Frozen future performance has not been measured by the rolling report.

No announcement overlay, hourly amount correction, time decay, residual model, or dynamic weight selection is enabled in this final interface.

## Installation and command line

Python 3.10+; report and delivery checks use Python 3.12. For inference, only NumPy and pandas are required:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pp_pricing --help
.venv/bin/python -m pp_pricing --features examples/pricing_features.csv --output outputs/example_predictions.csv
```

Run from the repository root. On Windows use `.venv\Scripts\python.exe`. The example is synthetic. Output must not already exist.

For authorized official-format evaluation files, the evaluator can use:

```sh
.venv/bin/python -m pp_pricing --orders /path/to/orders.csv --market /path/to/btc_market_history.csv --output outputs/predictions.csv
```

This is an inference command for the evaluator, not authorization to access the withheld file during research. The report runner's official-validation guard remains enabled.

## Input contract

### Official-format orders

| Field | Meaning |
|---|---|
| `createdAt` | Quote/creation timestamp; official timezone-naive timestamps are interpreted as UTC |
| `expirationDate` | Scheduled expiry, known when pricing |
| `optionType` | 0 = Call, 1 = Put |
| `productType` | Nominal duration in seconds |
| `quantity_BTC` | Protected BTC quantity; retained by the common feature builder, not an H0 predictor |
| `strikePrice` | Contract strike in USDC/BTC |
| `apiBasePrice_per_BTC_USDC` | Positive official Base per BTC |
| `order_id` (optional) | Unique nonempty identifier; otherwise output uses zero-based `source_row` |

`closedAt`, `realizedProfit_USDC`, paid premium, order price, and markup may be absent. The adapter selects only the pricing fields above, so any provided outcomes cannot affect predictions. No outcome-based filtering is performed.

The market CSV uses the supplied BTCUSDC 1-minute schema. `pp_audit.market` checks candle quality; `pp_data.features` joins the latest completed candle backward, at most 60 seconds old. The model needs at least 61 consecutive valid candles to compute its 60-minute volatility. Extra future candles do not enter an order's features. The shared builder also computes longer windows, but these are not required or used by H0.

### Prepared features

| Column | Definition |
|---|---|
| `option_type` | 0 = Call, 1 = Put |
| `api_base_price_per_btc_usdc` | Positive official Base, USDC/BTC |
| `remaining_seconds` | Scheduled expiry minus creation, in seconds; strictly positive |
| `product_duration_seconds` | Nominal positive duration; validated but not one of the three logistic regressors |
| `realized_vol_60m` | Square root of the sum of squared minute log returns over the prior 60 minutes; not annualized |
| `log_moneyness` | log(latest completed market price / strike) |
| `created_at_utc` | Required for CLI temporal eligibility |
| `order_id` (optional) | Unique nonempty identifier |

The Python `predict` method accepts exactly the six numerical feature columns in the order above, plus an aligned `created_at` Series. Prepared-feature callers are responsible for constructing their features without future data. The official-order adapter enforces completed-candle alignment automatically.

Missing Base, invalid directions, nonpositive durations, missing/gapped market history, duplicate identifiers, or quotes before model fitting cause a clear error. The interface does not silently drop rows, invent Base, or substitute zero volatility. Resolve those inputs before resubmission.

## Output contract

The CSV preserves input order and contains the identifier, UTC creation time, `base_per_btc_usdc`, `positive_payout_probability`, `conditional_payout_per_btc_usdc`, `h0_per_btc_usdc`, `alpha_h0`, `prediction_per_btc_usdc`, `pricing_multiplier`, and `model_fitted_at_utc`.

The Python official-order adapter also retains `order_id` as the result index and rejects missing/duplicate supplied IDs. Read IDs with `pd.read_csv(path, converters={'order_id': str})` to preserve leading zeros and literal identifiers such as `NA` or `NULL`. The CLI uses the same conversion; empty IDs remain invalid. Mixed ISO/SQL timestamp formats and fractional seconds are normalized to UTC before feature construction.

For official multiplier evaluation, use **`pricing_multiplier`**. For per-BTC price evaluation, use **`prediction_per_btc_usdc`**. A whole-order amount is this per-BTC value times `quantity_BTC`; the model does not add the historical platform markup or estimate demand conversion. Prediction is trained against expected payout; the official profitability/conversion evaluation remains separate.

```python
import pandas as pd
from pp_pricing import FinalPricingModel
from pp_models.pricing_inputs import MODEL_INPUTS

model = FinalPricingModel.load()
quotes = pd.read_csv('examples/pricing_features.csv', index_col='order_id', converters={'order_id': str})
result = model.predict(quotes[MODEL_INPUTS], created_at=quotes.created_at_utc)
# Or: result = model.predict_orders(official_orders, btc_minute_candles)
```

## Re-export and evidence

Re-export from an existing completed run, into a new file:

```sh
.venv/bin/python scripts/export_final_model.py --run-dir outputs/model_studies/report_announcement_v1_20261016 --output outputs/reexported_model.json
```

The exporter verifies the parameter-file SHA256, completed-run identity, and historical time bounds. It selects the latest existing state and does not train. Importing `pp_pricing` does not load scikit-learn or the research runner. JSON parameters avoid pickle execution and scikit-learn serialization dependencies.

`scripts/verify_delivery.py` also checks the delivered JSON's exact SHA256 against the recorded export, so changes to coefficients are detected without private data. A later authorized model version requires a new audited artifact and evidence record.

See [delivery checks](reports/2026-10-18-delivery-readiness.md), [model design](model_design.md), and [validation protocol](validation_protocol_v3.md) for the implementation/report mapping.

## 中文交付要点

- 官方测试入口为 `python -m pp_pricing`，输入原始报价与BTC行情，或已处理特征；输出逐笔倍率与每BTC预测值。
- 最终模型仍是Call 40%／Put 15%的H0与Base混合，公告只是独立研究分支。
- 随代码交付2026年7月31日零时的既有周更参数，未做全量重新训练。无标签环境保持参数冻结；这与报告中的有标签周更验证不同。
- 不需要最终赔付、实际结束时间或成交价；保留输入顺序，不静默删除缺失或异常订单。
- Base按官方“创建时可观察”的定义使用；提供方未给出的报价匹配时间细节属于来源说明边界，不是已发现的数据泄漏。
