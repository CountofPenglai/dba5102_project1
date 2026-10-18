# PP Pricing — H0 and Base Blending

## English

Estimate expected payouts for Bitcoin Perpetual Protect orders and return a pricing multiplier relative to the official **Base**. The final model blends Base with **H0**, a simple two-stage model that estimates payout probability and payout size.

### Model and entry points

```text
H0 = P(positive payout) × Base × positive-payout coefficient
Prediction = alpha × H0 + (1 − alpha) × Base
Pricing multiplier = Prediction / Base
alpha: Call = 0.40; Put = 0.15
```

Call and Put use separate logistic probability models and amount coefficients. Inputs are Base, remaining contract duration and volatility-scaled directional moneyness.

| Entry point | Purpose |
|---|---|
| `python -m pp_pricing` | **Final inference / official evaluation** using the included frozen model; no labels or training required |
| `pp_pricing.FinalPricingModel` | Python API: `load()`, `predict()` for features, `predict_orders()` for orders and market data |
| `python -m pp_models` | Reproduce the report's rolling training, model comparisons and metrics |
| `python -m pp_data` | Build the validated research dataset from authorized raw data |

The included model is the existing **2026-07-31 00:00 UTC** weekly snapshot and accepts orders at or after that time. Research uses April for training warm-up and May–July for rolling validation, updating every seven days from the preceding 30 days of orders already settled at the update time. Report results describe this rolling process; they do not establish frozen-model performance on unseen data. The announcement branch remains a separate experiment.

### Quick start

Python 3.10+ (verified on 3.12). Run from the repository root:

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pp_pricing --features examples/pricing_features.csv --output outputs/example_predictions.csv
```

For official-format orders and BTC minute candles:

```sh
.venv/bin/python -m pp_pricing --orders /path/to/orders.csv --market /path/to/btc_market_history.csv --output outputs/predictions.csv
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`. Examples are synthetic; private raw data is not included. Use a new output path for each run.

Outputs preserve input order. **`pricing_multiplier`** is the multiplier output; **`prediction_per_btc_usdc`** is the expected payout per BTC. Multiply the latter by BTC quantity for a whole-order amount. Input fields, units and API examples are in the [interface guide](docs/final_model_interface.md).

### Project structure

```text
pp_pricing/     Final inference API/CLI and frozen model.json
pp_models/      H0, shared training, model comparisons and evaluation
pp_data/        Preprocessing, validated dataset and rolling time splits
pp_audit/       Data quality checks
pp_calendar/    External announcement evidence and calendar handling
config/        Validation protocol and research settings
scripts/       Model export, data migration and verification
examples/      Synthetic inputs
tests/         Automated checks
docs/          Report, technical guides and key results
external_data/ Public announcement sources
outputs/       Local generated data and runs; excluded from Git
```

Research branches share `RollingSplitter` for time splits, `TrainingSession` for weekly H0 fitting, and `ForwardPredictionHistory` for settled prediction errors. See the [training architecture](docs/training_architecture.md).

### Research and verification

Prepare an authorized dataset using the [data guide](docs/data_pipeline.md). Reproducing the report requires its documented cohort. Use a new run ID:

```sh
.venv/bin/python -m pip install -r requirements-models.txt
.venv/bin/python -m pp_models --run-id report_local --data-dir outputs/preprocessing --market-path /path/to/btc_market_history.csv
.venv/bin/python scripts/verify_report_unified.py outputs/model_studies/report_local --data-dir outputs/preprocessing
```

Check the code and included delivery artifacts without starting research training:

```sh
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/verify_delivery.py
```

More detail: [Model design](docs/model_design.md) · [Validation protocol](docs/validation_protocol_v3.md) · [Results](docs/results/README.md) · [Report and handover](docs/research_document_index.md).

---

## 中文说明

本项目估计比特币 Perpetual Protect 订单的预期赔付，并相对官方 **Base** 输出定价倍率。最终模型将 Base 与简单的两阶段 **H0** 模型混合；H0 分别估计赔付概率和赔付金额。

### 模型与关键入口

```text
H0 = 正赔付概率 × Base × 正赔付金额系数
预测值 = alpha × H0 + (1 − alpha) × Base
定价倍率 = 预测值 / Base
alpha：Call = 0.40；Put = 0.15
```

Call、Put 分别使用逻辑回归概率模型和金额系数。输入为 Base、剩余保障期限及波动归一化后的方向价格位置。

| 入口 | 用途 |
|---|---|
| `python -m pp_pricing` | **最终预测／官方评测**：加载仓库内的冻结模型，无需标签或训练 |
| `pp_pricing.FinalPricingModel` | Python 接口：`load()` 加载，`predict()` 接收特征，`predict_orders()` 接收订单与行情 |
| `python -m pp_models` | 复现报告中的滚动训练、模型比较与评价指标 |
| `python -m pp_data` | 从授权原始数据构建经过校验的研究数据集 |

交付参数是 **2026-07-31 00:00 UTC** 的既有周更快照，用于该时刻及之后的订单。研究采用4月训练、5–7月滚动验证，每7日用此前30日创建且更新前已结算的订单重新训练。报告成绩描述的是滚动过程，不能直接证明冻结模型在新数据上的表现。公告分支保留为独立实验。

### 快速开始

需要 Python 3.10+，已在 3.12 上验证。从仓库根目录运行：

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pp_pricing --features examples/pricing_features.csv --output outputs/example_predictions.csv
```

输入官方格式订单与 BTC 分钟行情：

```sh
.venv/bin/python -m pp_pricing --orders /path/to/orders.csv --market /path/to/btc_market_history.csv --output outputs/predictions.csv
```

Windows 下将 `.venv/bin/python` 替换为 `.venv\Scripts\python.exe`。示例为合成数据，仓库不包含私有原始数据。每次运行须选择新的输出路径。

输出保持输入顺序。**`pricing_multiplier`** 是定价倍率；**`prediction_per_btc_usdc`** 是每 BTC 预期赔付，乘以 BTC 数量可得整单金额。输入字段、单位和 Python 示例见[最终模型接口](docs/final_model_interface.md)。

### 项目架构

```text
pp_pricing/     最终预测接口、命令行与冻结参数 model.json
pp_models/      H0、共享训练、模型比较与评价
pp_data/        预处理、校验后的数据集与滚动时间划分
pp_audit/       数据质量检查
pp_calendar/    外部公告证据与日历处理
config/        验证协议与研究配置
scripts/       模型导出、数据迁移与核验
examples/      合成输入示例
tests/         自动化检查
docs/          报告、技术说明与关键结果
external_data/ 公开公告来源
outputs/       本地生成数据与运行结果，不纳入 Git
```

研究分支统一使用 `RollingSplitter` 划分时间窗口、`TrainingSession` 复用每周 H0 训练，以及 `ForwardPredictionHistory` 管理已结算预测误差，详见[训练架构](docs/training_architecture.md)。

### 研究复现与核验

按[数据指南](docs/data_pipeline.md)准备授权数据。报告复现须保留文档规定的样本队列，并使用新的运行名称：

```sh
.venv/bin/python -m pip install -r requirements-models.txt
.venv/bin/python -m pp_models --run-id report_local --data-dir outputs/preprocessing --market-path /path/to/btc_market_history.csv
.venv/bin/python scripts/verify_report_unified.py outputs/model_studies/report_local --data-dir outputs/preprocessing
```

检查代码与交付产物，以下命令不会启动研究训练：

```sh
.venv/bin/python -m unittest discover -s tests
.venv/bin/python scripts/verify_delivery.py
```

[模型设计](docs/model_design.md) · [验证口径](docs/validation_protocol_v3.md) · [关键结果](docs/results/README.md) · [报告与交接索引](docs/research_document_index.md)
