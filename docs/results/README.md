# 当前报告关键统计

本目录只保留当前报告所需的聚合统计，不含原始数据、逐单预测或重复的历史实验结果。

`report_unified_v3_20261015_r2/`是正式报告来源：99个既有配置，4月训练、5–7月每7日滚动验证、同一4,640单。当前Word和Markdown使用这一结果，架构重构不改写其来源事实。

| 文件 | 内容 |
|---|---|
| `validation/model_comparison.csv` | 全部99配置的月等权和全期逐单误差 |
| `validation/fixed_grid.csv` | 去重后的86种固定比例 |
| `validation/summary.csv`、`macro_monthly.csv` | 正文18个报告配置的总体/月份/方向/期限/普通尾部与月等权结果 |
| `validation/risk_constraints.csv`、`monthly_risk_constraints.csv` | 方向尾部、累计少估和净金额风险 |
| `validation/intervals.csv`、`influence.csv` | 成对块重采样与删除连续周诊断 |
| `validation/announcement_support.csv`、`announcement_routing.csv` | 公告独立事件支持与回退 |
| `validation/training_cohorts.csv` | 合法历史样本量及身份指纹 |
| `independent_verification.json` | 原统一运行的独立核验摘要 |

筛选规则、原始与保存后的SHA256在[source_manifest.json](source_manifest.json)中。其他聚合文件用于比较、月度差异和参考复现；不是单独的新实验。

这些精选表不能替代完整运行目录。当前`python -m pp_models`读取flat v2数据，生产者为`report_unified_v2`；其核验脚本仅验证这一格式。原r2完整运行保留在本地`outputs/model_studies/report_unified_v3_20261015_r2/`，需使用清理前提交`7d66a5f`或其源码快照核验。旧结果与旧代码不再复制在当前目录。

清理复算的核验摘要另存[architecture_cleanup_verification.json](architecture_cleanup_verification.json)，只记录队列不变与数值一致的证据，不重复发布一整套相同结果。主要研究结论见[统一研究结果](../reports/2026-10-15-report-unified-rerun.md)。
