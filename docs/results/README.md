# 当前结果与历史归档

本目录随代码提交，保留选中研究的关键聚合，供无需私有数据的汇报和交接使用。当前报告已按v3重新训练和评价，采用`report_unified_v3_20261015_r2`。其余目录保留历史实验，避免改写原协议含义。

| 子目录 | 内容与日期口径 |
|---|---|
| `report_unified_v3_20261015_r2/validation/` | **当前：99个既有配置统一重跑；4月训练、5–7月周更验证，包括公告** |
| `h0_blend_fine_20261015/validation/` | 63组细网格＋参考，5–7月v3 |
| `h0_blend_ratios_20261015/validation/` | 25组固定＋2组动态，5–7月v3 |
| `h0_state_fallback_20261015/validation/` | 固定50/50、状态回退、历史权重及组合，5–7月v3 |
| `h0_time_decay_20261015/validation/` | H0、7/14日半衰期，5–7月v3 |
| `h0_hourly_amount_20261015/validation/` | 小时水平、变化和门控，5–7月v3 |
| `macro_multiplier_20261014_v2/` | 公告倍率；旧4–6月开发、7月压力口径 |
| `macro_events_20261014_v2/`、`macro_phases_20261014_v2/` | 公告覆盖、事件支持及市场/订单分组统计 |

`macro_monthly.csv`与`summary.csv`同时保留月等权和逐单、方向、尾部口径；`fixed_grid.csv`列全网格；风险条件独立保存。历史目录的`intervals.csv`仅保留全体、RMSE差、7日块；当前统一目录保留报告配置对Base/H0的各方向/尾部分组RMSE和MAE的3/7/14日块区间。当前`model_comparison.csv`列99个配置，`fixed_grid.csv`列86种唯一固定比例，其他表仅保留报告中的18个配置；不保存逐单预测。筛选规则与原/新SHA256逐文件记录在[source_manifest.json](source_manifest.json)。

这些是精选表，不含逐单预测、原始订单/行情、训练模型或源码ZIP，也不是可供原核验脚本直接重放的完整运行目录。当前完整运行在main工作树的`outputs/model_studies/report_unified_v3_20261015_r2/`；原历史完整档案仍在研究目录。当前统一复算只需授权输入与源码，不依赖旧模型运行；历史入口仍要求原前置依赖。

当前最优月等权点Call40%/Put15%尚未通过全期逐单RMSE及完整风险条件；历史区间不抵消多轮开发选择偏差。主要解释见[统一重跑报告](../reports/2026-10-15-report-unified-rerun.md)。
