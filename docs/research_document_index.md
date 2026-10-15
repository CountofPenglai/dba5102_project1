# 主分支研究与交接索引

更新：2026-10-15。本索引只汇总本次选中内容；此前研究分支的Word大纲、完整探索和未提交文件继续保留在原目录，本次不整分支搬入。

| 内容 | 方案/数据 | 结果/入口 |
|---|---|---|
| 混合主线 | [粗网格计划](plans/2026-10-15-h0-blend-ratios.md)、[细网格计划](plans/2026-10-15-h0-blend-fine.md) | [细网格报告](reports/2026-10-15-h0-blend-fine.md)、`pp_models.h0_blend_fine_study` |
| 动态权重和状态回退对照 | [计划](plans/2026-10-15-h0-state-fallback.md) | [报告](reports/2026-10-15-h0-state-fallback.md)、`pp_models.h0_state_fallback_study` |
| 当前验证角色 | [协议说明](validation_protocol_v3.md)、[固定配置](../config/validation_protocol_v3.json) | 4月训练、5–7月已见滚动验证、7日更新 |
| 官方宏观公告 | [日历及来源](../external_data/macro_events/README.md)、[跨公告计划](plans/2026-10-14-macro-events.md)、[阶段计划](plans/2026-10-14-macro-phase-audit.md) | [跨公告报告](reports/2026-10-14-macro-events.md)、[阶段报告](reports/2026-10-14-macro-phases.md) |
| 公告倍率研究记录 | [执行设置](plans/2026-10-14-macro-multiplier-execution.md) | [报告](reports/2026-10-14-macro-multiplier.md)、`pp_models.macro_multiplier_study`；未采用 |
| 额外1：H0时间衰减 | [计划](plans/2026-10-15-h0-time-decay.md) | [报告](reports/2026-10-15-h0-time-decay.md)、`pp_models.h0_time_decay_study`；无稳定整体优势 |
| 额外2：小时金额响应 | [计划](plans/2026-10-15-h0-hourly-amount.md) | [报告](reports/2026-10-15-h0-hourly-amount.md)、`pp_models.h0_hourly_amount_study`；无稳定整体优势 |
| 版本化关键结果 | [结果索引](results/README.md) | [统计来源与哈希](results/source_manifest.json) |
| 合并追溯 | [集成计划](plans/2026-10-15-main-integration.md) | [文件选择清单](integration/h0_main_selection.json)、[集成核验](integration/verification.md) |

当前最佳开发比例是Call40%/Put15%的H0权重，月等权RMSE293.0480，但全期逐单295.6140高于Base295.1501，完整风险条件未通过。不能将它写成稳定全面超过Base或正式上线。

## 运行交接

H0运行依赖链：时间衰减研究生成同协议Base/H0参考 → 状态回退 → 粗网格 → 细网格。小时金额响应的前置运行只依赖第一步。状态回退和小时金额响应还需要默认`DEFAULT_DATA`位置的原始市场文件，并校验历史前缀；预处理的`--data-dir`不会传递给这些入口。具体位置、哈希、命令和固定运行目录名见[README](../README.md)。独立核验脚本只重放参数，不重新拟合。

公共日历维护的`--check-only`可在无订单、无模型产物的干净克隆中执行。宏观历史订单分析、四阶段分析和倍率实验保留旧口径，需要已获授权的canonical/市场数据及原完整参考档案，**不属于只克隆仓库即可重放的历史实验**：

- `scripts/analyze_macro_events.py`需要`outputs/model_studies/put_final_july_20261014/`的Base历史汇总，默认输出`outputs/diagnostics/macro_events_20261014_v2/`。
- `scripts/analyze_macro_phases.py`还需要上一项完整诊断目录。它固定使用`snapshots/20261014T123002Z.json`，不随当前日历静默变化。
- `pp_models.macro_multiplier_study`需要旧Put最终模型参考运行及其配置指定的公开日历快照；保持4–6月开发、7月压力复查，与当前v3成绩分开。
- 原完整运行的核验依赖其源码/配置指纹和前置文件；使用原归档或在匹配源码环境重放。`docs/results/`仅为结果阅读和来源核对，不满足这些完整依赖。

公共日历最后核对于2026-10-14；未来日程可能变更。任何新官方验证/未知测试前先维护来源包，未知不编码为无事件。当前合并不打开新官方验证，也不安排自动训练、推送或部署。
