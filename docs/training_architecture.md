# Shared rolling training / 统一滚动训练架构

Updated: 2026-10-18. This refactor changes code ownership and reuse, not model selection, input data or the reported validation protocol.

## One flow

```text
load_dataset → TrainingSession → RollingSplitter.windows
             → fit one H0 per origin → branch-specific predictions
             → prediction_context → common scoring and verification
```

| Module | Responsibility |
|---|---|
| `pp_data.config` | Load the single date protocol from `config/validation_protocol_v3.json` |
| `pp_data.splitting` | Validate temporal settings; produce ordered query/history IDs on one continuous update clock |
| `pp_models.training` | Fit/cache each weekly H0 and time-decay variant once per run; reject changed pools/protocols |
| `pp_models.validation` | Build shared day/month/direction/tenor/tail metadata; tail identity remains evaluation-only |
| `pp_models.forecast_history` | Store previous forward forecasts; attach actual labels only after settlement |
| `*_study.py` | Apply only that branch's blend, market gate, amount adjustment or weighting settings |
| `report_unified_study` | Share one session across branches, merge common cohorts, score and archive results |
| `pp_pricing` | Separate frozen inference for delivery; never imports the training session |

## Reusable Python API

```python
from pp_data.io import load_dataset
from pp_models.training import TrainingSession
from pp_models.h0 import probability_frame
from pp_models.validation import prediction_context

pool = load_dataset('outputs/preprocessing', cohort='benchmark', use_base=True)
session = TrainingSession(pool)  # Uses the sole protocol by default.

for window in session.splitter.windows():
    model = session.fit(window)
    quotes = probability_frame(pool.X.loc[window.query_ids])
    prediction = model.predict_raw(quotes)
    context = prediction_context(pool, window, session.splitter)
    # Combine prediction with context for common scoring; context is not a feature.
```

`window` contains `origin`, exclusive `stop`, `history_ids`, `query_ids`, and `is_validation`. History is created in `[max(training_start, origin-history_days), origin)` and **closed strictly before origin**. The query interval is `[origin, stop)`. `windows(warmup_days=28)` adds the existing April forward warm-up for adaptive mixtures while keeping the same seven-day clock.

An existing study remains callable alone. To share its H0 fits with another branch, pass the same session explicitly:

```python
from pp_models.h0_time_decay_study import predict_period as predict_decay
from pp_models.h0_blend_ratios_study import predict_period as predict_blends

protocol = session.protocol
decay_predictions, decay_states, weights = predict_decay(pool, protocol, training=session)
blend_predictions, blend_states = predict_blends(pool, protocol, training=session)
```

Create a fresh session for a new dataset or run. Pools and protocols must not change during a session; content fingerprints detect mutations. Returned fitted models are read-only to callers: use them for prediction/state export, never refit or modify their parameters. The cache is in memory and local to this one run; it cannot reuse a fit across different data sources or evaluation protocols.

The unweighted cached estimator is `TimeDecayH0(None)`, which delegates its fitting to the original `H0` and only adds weight diagnostics. Weighted 7/14-day variants keep their existing formulas. Hourly amount and market-state estimators remain branch-specific; only their shared H0 input is reused.

## Time and label boundaries

- The current protocol remains April training and May–July validation, with 30-day history and seven-day updates. Month boundaries never restart the weekly clock.
- Creation-only `quote_windows` also supplies the announcement schedule and never reads settlement or payout data.
- Monthly tail thresholds use the expanding historical pool from April, settled strictly before month start. They do not determine training eligibility or prediction inputs.
- `ForwardPredictionHistory.append` stores forecasts without labels. `settled(origin)` attaches only prior forecasts whose outcomes are known before that update; dynamic mixtures cannot use current-week or future errors.
- Insufficient historical support during warm-up is recorded and skipped; validation weeks with fewer than 100 eligible training orders fail rather than silently changing the model.
- Independent verification intentionally retains separate arithmetic and historical checks. Sharing the verifier's calculations with the training code would weaken the regression check.

## Entry points and provenance

Research CLI remains `python -m pp_models`; official frozen inference remains `python -m pp_pricing`. No new date-split option or parallel validation configuration is introduced.

New runs record actual fit counts in `provenance/manifest.json` and `metrics/validation/training_reuse.csv`. They distinguish logical requests, actual H0 fits, and cache hits. Historical run manifests retain their original, accurate pre-refactor fit counts. The archived parameter-state layout remains compatible with the export and verification scripts.

## 中文说明

此次将数据划分、H0训练、评价上下文和已结算预测历史分为四个公共模块。研究分支不再各自生成日期窗口、筛选训练历史、计算月初尾部参考或重复拟合同一周的H0。

使用方式：加载统一数据池，创建一次`TrainingSession`，由`session.splitter.windows()`取周窗口，通过`session.fit(window)`训练或复用模型。多个研究分支显式传入同一会话，就会共用H0；单独运行某一分支仍可自动创建独立会话。

主验证口径、模型公式、权重和交付模型不变。重跑只是验证架构等价性，不是新的选参实验。完整回归记录见[本轮核验](reports/2026-10-18-shared-training.md)。
