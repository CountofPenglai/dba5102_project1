# Active configuration / 当前有效配置

All eight JSON files in this directory are used by the current implementation. Version suffixes identify the format or recipe; they do not mean the file is obsolete.

| File | Purpose / 用途 |
|---|---|
| `preprocessing_v2.json` | Flat preprocessing pool and quote-time feature settings / 平面数据池与报价特征 |
| `validation_protocol_v3.json` | Sole research date protocol / 唯一训练与验证日期口径 |
| `h0_time_decay_v1.json` | H0 and time-decay comparison / H0及时间衰减对照 |
| `h0_hourly_amount_v1.json` | Hourly amount response comparison / 小时金额响应对照 |
| `h0_state_fallback_v1.json` | Market-state and historical-error fallback / 市场状态与历史误差回退 |
| `h0_blend_ratios_v1.json` | Coarse fixed and dynamic blend comparisons / 粗粒度固定及动态混合 |
| `h0_blend_fine_v1.json` | Fine fixed-weight comparison / 细粒度固定比例比较 |
| `announcement_overlay_v1.json` | Current additive preannounced-event rule / 当前独立小额公告规则 |

The study recipe JSON files are checked against the implementation before a report run. They preserve the reported experiment settings; editing them alone does not change model behavior. Frozen official-test inference uses [the packaged model](../pp_pricing/model.json), not the experiment grid.

本目录没有可执行的旧验证划分。结果目录中保留的当前`config.json`和`protocol.json`是报告运行快照，不是另一套可编辑配置。已替代运行的配置快照已移除；旧数值参照只保留回归核验所需的两张表，见[结果索引](../docs/results/README.md)。
