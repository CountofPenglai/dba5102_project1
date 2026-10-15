# 主分支研究与交接索引

更新：2026-10-15。本索引汇总主分支选中内容及当前汇报文档；此前研究分支的历史Word、完整探索和未提交文件继续保留在原目录。

## 商业汇报文档

[研究报告大纲 Word](deliverables/研究报告大纲_H0与Base混合模型.docx)围绕H0与Base分方向混合组织七章：业务目标、数据与公平比较、核心模型、研究路径、研究结果、商业建议、研究局限。本轮99个既有配置统一重跑，当前结果来自`report_unified_v3_20261015_r2`（main@9c2389b加统一入口），包括公告模型；统一使用4月训练、5–7月滚动验证及7日更新口径。

大纲并列展示月等权与全期逐单成绩，保留未采用探索、普通/尾部代价、金额风险和区间不确定性。40%/15%是开发候选，不能表述为全面超过Base或已通过独立测试。文档内结果链接指向仓库中的本轮统计，实现链接保留证据版本；此次已重跑训练与评价，未读取新官方验证集。旧研究分支的《未来研究计划》和《具体实现方案》保留其历史含义，不自动视为当前执行安排。

## 研究证据与运行入口

| 内容 | 方案/数据 | 结果/入口 |
|---|---|---|
| 当前统一重跑 | [计划](plans/2026-10-15-report-unified-rerun.md) | [报告](reports/2026-10-15-report-unified-rerun.md)、`pp_models.report_unified_study` |
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

当前推荐使用README中的统一入口，显式传入canonical与市场路径；输出全部模型本轮统计，不要求旧H0参考运行。独立核验用`scripts/verify_report_unified.py`。

以下为历史单项H0运行依赖链：时间衰减研究生成同协议Base/H0参考 → 状态回退 → 粗网格 → 细网格。小时金额响应的前置运行只依赖第一步。状态回退和小时金额响应还需要默认`DEFAULT_DATA`位置的原始市场文件，并校验历史前缀；预处理的`--data-dir`不会传递给这些入口。具体位置、哈希、命令和固定运行目录名见[README](../README.md)。独立核验脚本只重放参数，不重新拟合。

公共日历维护的`--check-only`可在无订单、无模型产物的干净克隆中执行。宏观历史订单分析、四阶段分析和倍率实验保留旧口径，需要已获授权的canonical/市场数据及原完整参考档案，**不属于只克隆仓库即可重放的历史实验**：

- `scripts/analyze_macro_events.py`需要`outputs/model_studies/put_final_july_20261014/`的Base历史汇总，默认输出`outputs/diagnostics/macro_events_20261014_v2/`。
- `scripts/analyze_macro_phases.py`还需要上一项完整诊断目录。它固定使用`snapshots/20261014T123002Z.json`，不随当前日历静默变化。
- `pp_models.macro_multiplier_study`需要旧Put最终模型参考运行及其配置指定的公开日历快照；保持4–6月开发、7月压力复查，与当前v3成绩分开。
- 原完整运行的核验依赖其源码/配置指纹和前置文件；使用原归档或在匹配源码环境重放。`docs/results/`仅为结果阅读和来源核对，不满足这些完整依赖。

公共日历最后核对于2026-10-14；未来日程可能变更。任何新官方验证/未知测试前先维护来源包，未知不编码为无事件。当前合并不打开新官方验证，也不安排自动训练、推送或部署。
