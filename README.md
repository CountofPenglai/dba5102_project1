# PP Pricing — H0 and Directional Base Blending

Updated: 2026-10-18. [中文说明](#中文说明)

This project estimates expected payouts for Bitcoin Perpetual Protect orders and returns a pricing multiplier relative to the official Base. The selected model combines a simple two-stage H0 model with Base: **40% H0 for Call and 15% H0 for Put**.

**Official-test entry point: `python -m pp_pricing`.** It loads the included numeric model artifact and predicts without labels or training. `python -m pp_models` is the separate research/report reproduction entry point.

## Quick start: inference

Python 3.10+; verified report environment: Python 3.12. Run from the repository root.

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pp_pricing --features examples/pricing_features.csv --output outputs/example_predictions.csv
```

On Windows, use `.venv\Scripts\python.exe`. The example contains synthetic rows. Choose a new output path on each run; existing files are never overwritten.

For an evaluator's authorized official-format file:

```sh
.venv/bin/python -m pp_pricing --orders /path/to/orders.csv --market /path/to/btc_market_history.csv --output outputs/predictions.csv
```

Use `pricing_multiplier` for multiplier submission, or `prediction_per_btc_usdc` for per-BTC price submission. Output order is preserved. If `order_id` is absent, the CSV includes zero-based `source_row`. Whole-order amounts require multiplication by BTC quantity.

```python
import pandas as pd
from pp_pricing import FinalPricingModel

model = FinalPricingModel.load()
# Both frames contain pricing-time information only; payout labels are not needed.
result = model.predict_orders(orders=pd.read_csv('/path/to/orders.csv', converters={'order_id': str}),
                              market=pd.read_csv('/path/to/btc_market_history.csv'))
```

The [interface guide](docs/final_model_interface.md) defines input columns, units, feature formulas, error handling, Python feature input, and artifact export. Missing Base or insufficient valid market history raises an error; rows are not silently dropped or imputed. No official validation data was read to prepare this delivery.

## Final model

```text
H0 = P(positive payout) × Base × positive-payout coefficient
Prediction = alpha × H0 + (1 − alpha) × Base
Multiplier = Prediction / Base = 1 + alpha × (p × c − 1)
alpha: Call = 0.40; Put = 0.15
```

Call and Put have separate logistic probability models and separate positive-payout amount coefficients. Probability uses log Base, log remaining time, and volatility-scaled directional moneyness. Amount is a zero-intercept least-squares multiple of Base fitted on positive claims. See [model details](docs/model_design.md).

The included [model.json](pp_pricing/model.json) is the **existing 2026-07-31 00:00 UTC weekly state**, trained on 2,131 eligible historical orders. It is exported without refitting. Frozen inference needs no outcome feedback and rejects orders earlier than this timestamp. This is different from the report's weekly updating backtest: the report's scores are not scores for one frozen model on future data. Announcement adjustments and other experimental branches are not enabled in the final model.

## Evidence and validation

One protocol: April training warm-up, May–July rolling validation, seven-day updates starting May 1, and a trailing 30-day creation window with settlement strictly before each update. The common validation cohort contains 4,640 orders: May 726, June 1,373, July 2,541. July is seen validation, not an independent test set.

| Same May–July cohort; USDC/BTC | Month-equal RMSE | Month-equal MAE | Pooled RMSE |
|---|---:|---:|---:|
| Official Base | 293.564338 | 184.439853 | **295.150078** |
| H0 | 298.010152 | 186.263090 | 303.531640 |
| H0/Base, Call 50% / Put 25% H0 | 293.114750 | 183.652670 | 296.046896 |
| **Final blend, Call 40% / Put 15% H0** | **293.048025** | **183.558057** | 295.614043 |

Month-equal RMSE is the square root of the mean of the three monthly MSEs. Pooled RMSE weights orders equally. H0 improves both RMSE and MAE over Base in May and June; directional blending limits July's deterioration. The selected blend has a small month-equal improvement, but does not beat Base on pooled RMSE or all risk constraints. Its 95% improvement interval crosses zero; Call absolute net amount bias increases 10.81%. Unseen-test superiority and profitability have not been established.

[Research results](docs/reports/2026-10-15-report-unified-rerun.md) · [Published statistics](docs/results/README.md) · [Data audit](docs/reports/2026-10-16-report-data-audit.md) · [Delivery verification](docs/reports/2026-10-18-delivery-readiness.md) · [Final code review](docs/reports/2026-10-18-final-code-review.md)

## Repository layout

```text
pp_pricing/     Final frozen inference API/CLI and portable model.json
examples/       Synthetic pricing inputs; no private orders
pp_models/      H0, shared transform, report recipes, scoring and verification
pp_data/        Pricing features, targets and metadata in one validated flat pool
pp_audit/       Raw order/market quality checks
pp_calendar/    Announcement evidence, timezone and version handling
config/         The single validation protocol and report recipe settings
scripts/        Artifact export, data migration, calendar and result verification
external_data/  Public announcement sources and frozen calendars
tests/          Synthetic/unit/integration tests
docs/           Report, interface/data/model guides and compact evidence
outputs/        Local generated data and complete runs; excluded from Git
```

The research catalog retains 97 report-relevant recipes, including negative results that explain the final choice. Obsolete GAM/EBM, early GBDT, distribution, residual-model frameworks and old executable data splits were already removed. The legacy migration tool is retained only to reproduce the historical report cohort. Private raw data is not distributed.

All eight [active configuration files](config/README.md) remain required. Superseded result/configuration snapshots have been removed; the older numerical reference now contains only two tables for 96 unchanged recipes. Local complete runs retain the current report source and the latest shared-training verification. See the [results index](docs/results/README.md).

Research branches now share `RollingSplitter` (time windows and available history), `TrainingSession` (one H0 fit per weekly origin), common prediction metadata, and `ForwardPredictionHistory` (settled forward errors). The unified runner passes one session to all branches; standalone calls remain supported. See the [training architecture and reusable API](docs/training_architecture.md).

## Reproduce and verify research

Research and tests also require scikit-learn, SciPy and threadpoolctl:

```sh
.venv/bin/python -m pip install -r requirements-models.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/verify_delivery.py
.venv/bin/python -m pp_models --help
```

Data setup is documented in the [data guide](docs/data_pipeline.md). Reuse an authorized flat dataset, or explicitly migrate the historical report pool:

```sh
.venv/bin/python scripts/migrate_report_data.py --source-dir /path/to/old/preprocessing --output-dir outputs/preprocessing
```

To rerun all report recipes and independently verify their results:

```sh
.venv/bin/python -m pp_models --run-id report_local --data-dir outputs/preprocessing --market-path /path/to/btc_market_history.csv
.venv/bin/python scripts/verify_report_unified.py outputs/model_studies/report_local
```

The run ID must be new. The report runner preserves the 4,640-order cohort, uses only the old market prefix, and blocks access to the new official validation file. It archives data/config/source identities, parameters and metrics. Frozen published results are not overwritten. A native raw-data rebuild without the documented cohort reference may include additional orders and must not be presented as the same report experiment.

To re-export the latest already-fitted state from an existing verified run:

```sh
.venv/bin/python scripts/export_final_model.py --run-dir outputs/model_studies/report_announcement_v1_20261016 --output outputs/reexported_model.json
```

## External announcements and handover

The announcement branch demonstrates **external data collection and expansion of available inputs**. Its versioned CPI, US employment and FOMC calendar records public sources, timezones, schedule revisions and historical knowledge times. The current frozen source package was checked on 2026-10-14: 82 schedule records representing 65 main events. It is not automatically refreshed.

The independent research overlay adds up to 1% per preannounced event, strengthens as release approaches (24-hour pre-release half-life), fades to zero within 60 minutes after release, and caps the total at 5%. Unknown schedules contribute nothing; the FOMC statement and press conference count as one event. Call and Put use the same rule. This is a separate comparison, not part of the delivered blend; its small RMSE change is not evidence of reliable predictive gain.

```sh
.venv/bin/python scripts/update_macro_calendar.py --source external_data/macro_events/sources/2026-10-14/verified_schedule_history.json --check-only
```

[Report Word](docs/deliverables/研究报告大纲_H0与Base混合模型.docx) · [Handover index](docs/research_document_index.md) · [Announcement data](external_data/macro_events/README.md) · [Validation protocol](docs/validation_protocol_v3.md)

---

## 中文说明

更新：2026-10-18。本项目估计BTC保险订单的每BTC预期赔付，并相对官方Base输出定价倍率。最终选用 **Call 40% H0 + 60% Base、Put 15% H0 + 85% Base**。

### 官方测试入口

**`python -m pp_pricing` 是最终模型预测入口；`python -m pp_models` 是报告研究复现入口。** 前者无需赔付标签，不会启动训练或97配置实验。

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pp_pricing --features examples/pricing_features.csv --output outputs/example_predictions.csv
```

官方拿到授权报价文件与行情后，可以运行：

```sh
.venv/bin/python -m pp_pricing --orders /path/to/orders.csv --market /path/to/btc_market_history.csv --output outputs/predictions.csv
```

输出中`pricing_multiplier`为提交倍率，`prediction_per_btc_usdc`为每BTC价格/赔付估计；整单金额需乘BTC数量。保留原顺序，无订单ID时使用从0开始的`source_row`。输出文件存在时拒绝覆盖。字段、单位、Python接口及异常处理详见[最终模型接口](docs/final_model_interface.md)。示例是合成数据，不含真实订单。

### 模型与参数

H0分别为Call和Put估计“是否赔付”的逻辑回归概率，以及“赔付时有多大”的Base金额系数。概率输入为log(Base)、log(剩余期限)和经过60分钟波动归一化的方向价格位置；金额系数只在正赔付历史上拟合。再将H0向Base有限收缩，得到最终倍率。

交付参数是报告**2026-07-31零时**的最后一次既有周更结果，使用此前30日创建、更新前已结算的2,131笔订单，未重新训练或改为全量训练。无标签测试期间参数冻结，逐笔报价仍随当前合同和行情改变。不能把这个快照用于更早订单，也不能把报告中的周更成绩当成冻结模型的未来成绩。公告、小时响应、时间衰减、残差修正均不叠加到最终模型。

### 统一验证与结论

唯一口径为4月训练、5–7月滚动验证，从5月1日起连续每7日更新，使用此前30日创建且更新前已结算的历史。共同验证4,640单，5/6/7月分别726/1,373/2,541单。7月属于已见验证；新官方验证未用于本轮交付整理。

上方结果表同时列出月等权与逐单口径。H0在5、6月的RMSE和MAE均优于Base，分方向混合缩小7月退化。最终混合的月等权RMSE为293.048025、MAE为183.558057；Base分别为293.564338、184.439853。月等权RMSE是三个月MSE均值再开方。

最终混合的全期逐单RMSE为295.614043，仍高于Base的295.150078；改善95%区间跨0，Call净金额偏差绝对值扩大10.81%。它体现了简单H0及分方向效益的价值，但不能保证在测试期保持优势，也不能把赔付误差改善直接等同于利润提升。完整证据见[统一结果](docs/reports/2026-10-15-report-unified-rerun.md)。

### 架构、复现与交接

项目结构及完整命令见上方英文部分。`pp_pricing`负责交付预测，`pp_models`负责报告模型与评分，`pp_data`负责平面数据池，`pp_audit`负责质量检查，`pp_calendar`负责公告证据，`scripts`负责导出和核验，`docs`保留报告及关键统计。`outputs`仅本地生成，不提交私有数据。

[配置索引](config/README.md)说明仍在使用的8份配置；旧运行的配置快照和重复结果已清理。旧数值参照仅保留96个未改配方所需的两张表，本地完整运行只保留当前报告来源和最新架构核验。

当前保留97个报告相关配置，包含解释研究取舍所需的负结果。无关旧模型、旧框架与旧可执行分割已清除；数据迁移桥仅用于保持原报告队列。原始资料重建时应按[数据流程](docs/data_pipeline.md)提供队列参照，不能把扩大样本后的结果冒称为原报告复现。

研究代码现统一使用`RollingSplitter`划分时间窗口、`TrainingSession`复用同周H0、公共评价上下文，以及只释放已结算标签的历史预测缓存。各分支保留自己的模型差异，不再复制公共训练流程。调用示例见[训练架构说明](docs/training_architecture.md)。

```sh
.venv/bin/python -m pip install -r requirements-models.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/verify_delivery.py
```

公告分支主要体现数据搜集和维度拓展：维护CPI、非农、FOMC的官方来源、时区、预告证据与版本；固定规则的微小效果只作接入检验。当前日历最后核对于2026-10-14，没有自动更新任务。没有当时已公开证据的公告不能作为已知输入。

[商业汇报Word](docs/deliverables/研究报告大纲_H0与Base混合模型.docx) · [交接索引](docs/research_document_index.md) · [交付核验](docs/reports/2026-10-18-delivery-readiness.md) · [数据核对](docs/reports/2026-10-16-report-data-audit.md)
