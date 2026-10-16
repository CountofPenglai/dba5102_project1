# 当前报告关键统计

本目录只保留当前报告所需的聚合统计，不含原始数据、逐单预测或重复的历史实验结果。

`report_announcement_v1_20261016/`是正式报告来源：97个配置，4月训练、5–7月每7日滚动验证、同一4,640单。当前Word和Markdown使用这一结果；公告规则无需训练，其他模型口径保持不变。

| 文件 | 内容 |
|---|---|
| `validation/model_comparison.csv` | 全部97配置的月等权和全期逐单误差 |
| `validation/fixed_grid.csv` | 去重后的86种固定比例 |
| `validation/summary.csv`、`macro_monthly.csv` | 正文16个代表配置的总体/月份/方向/期限/普通尾部与月等权结果 |
| `validation/risk_constraints.csv`、`monthly_risk_constraints.csv` | 方向尾部、累计少估和净金额风险 |
| `validation/intervals.csv`、`influence.csv` | 成对块重采样与删除连续周诊断 |
| `validation/announcement_support.csv`、`announcement_routing.csv` | 公告独立事件支持与回退 |
| `validation/training_cohorts.csv` | 合法历史样本量及身份指纹 |
| `independent_verification.json` | 原统一运行的独立核验摘要 |

筛选规则、原始与保存后的SHA256在[source_manifest.json](source_manifest.json)中。其他聚合文件用于比较、月度差异和参考复现；不是单独的新实验。

这些精选表不能替代完整运行目录。当前`python -m pp_models`读取flat v2数据，生产者为`report_unified_v3`；核验脚本验证当前97配置与固定公告规则。完整本地运行位于`outputs/model_studies/report_announcement_v1_20261016/`。

上一轮`report_unified_v3_20261015_r2/`仅用于96个未改配方的数值回归参照，其三档旧公告结果不代表当前实现。清理复算摘要`architecture_cleanup_verification.json`也是历史记录。当前公告没有参数学习；其微小RMSE改善不能解释为稳定预测能力。研究结论见[统一研究结果](../reports/2026-10-15-report-unified-rerun.md)。
