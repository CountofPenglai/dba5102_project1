# 当前报告关键统计

本目录只保留当前报告所需的聚合统计，不含原始数据、逐单预测或重复的历史实验结果。

`final_code_review_20261018.json`是最终代码检查摘要，记录110项测试、独立报告回放和无私有数据的交付检查；它不新增训练结果或替换报告来源。

`shared_training_verification_20261018.json`记录统一划分和训练模块重构的等价核验：原有97配置统计和参数不变，H0系列拟合由78次减少到39次。它是架构核验摘要，不是新的选参结果；说明见[本轮重构记录](../reports/2026-10-18-shared-training.md)。

`report_announcement_v1_20261016/`是正式报告来源：97个配置，4月训练、5–7月每7日滚动验证、同一4,640单。当前Word和Markdown使用这一结果；公告规则无需训练，其他模型口径保持不变。

| 文件 | 内容 |
|---|---|
| `validation/model_comparison.csv` | 全部97配置的月等权和全期逐单误差 |
| `validation/fixed_grid.csv` | 去重后的86种固定比例 |
| `validation/summary.csv`、`macro_monthly.csv` | 正文16个代表配置的总体/月份/方向/期限/普通尾部与月等权结果 |
| `validation/risk_constraints.csv`、`monthly_risk_constraints.csv` | 方向尾部、累计少估和净金额风险 |
| `validation/intervals.csv`、`influence.csv` | 成对块重采样与删除连续周诊断 |
| `validation/announcement_events.csv`、`announcement_routing.csv` | 逐事件贡献、实际触发及未知证据路由 |
| `validation/training_cohorts.csv` | 合法历史样本量及身份指纹 |
| `independent_verification.json` | 原统一运行的独立核验摘要 |

筛选规则、原始与保存后的SHA256在[source_manifest.json](source_manifest.json)中。其他聚合文件用于比较、月度差异和参考复现；不是单独的新实验。

这些精选表不能替代完整运行目录。当前`python -m pp_models`读取flat v2数据，生产者为`report_unified_v3`；核验脚本验证当前97配置与固定公告规则。完整本地运行位于`outputs/model_studies/report_announcement_v1_20261016/`。

上一轮[最小数值参照](report_unified_v3_20261015_r2/README.md)仅保留`summary.csv`和`model_comparison.csv`，供96个未改配方的回归检查使用。已移除三档旧公告记录、过时配置/协议快照、重复诊断以及旧清理核验摘要。当前公告没有参数学习；其微小RMSE改善不能解释为稳定预测能力。研究结论见[统一研究结果](../reports/2026-10-15-report-unified-rerun.md)。

本地完整运行保留两套：`report_announcement_v1_20261016`是报告正式来源，`report_shared_training_v1_20261018`是最新架构等价核验。两套已替代的旧完整运行已删除；当前报告配置快照和来源清单作为证据保留，不是另一套执行配置。清理范围及复核见[清理记录](../reports/2026-10-18-obsolete-results-cleanup.md)。
