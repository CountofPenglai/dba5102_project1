# Final delivery review / 最终交付核对

日期：2026-10-18。范围为当前main工作树的代码、入口、英文注释、结构与双语README；保留上一轮报告数据核对的修订。本轮没有重新选择模型、扩大训练池或读取新的官方验证集。

## 交付内容

- `python -m pp_pricing`：最终模型独立预测入口，提供Python API及CSV CLI。
- `pp_pricing/model.json`：从已完成的报告运行导出7月31日零时的既有H0参数；Call40%／Put15%混合，无重新拟合。
- `pp_models/pricing_inputs.py`：抽出训练与预测共用的字段白名单、输入检查与特征变换。H0核心和历史时间边界补充英文说明，计算公式保持不变。
- `scripts/export_final_model.py`：检查来源哈希和历史标签时间后显式导出，拒绝覆盖。
- `scripts/verify_delivery.py`：无需私有数据即可检查交付证据；提供本地历史池时可重放全部报告周次。
- README改为英文在前、中文在后；[接口说明](../final_model_interface.md)列出字段、单位、错误处理及官方测试命令。

原研究配置保留其报告用途。旧GAM/EBM、早期分类/残差框架与旧执行分割已在前轮移除；本轮没有发现需要再次删除的可执行旧架构。显式数据迁移桥和旧统计回归参照仍是报告复现依赖。更新了“唯一执行入口”的表述，并将过时的`announcement_support.csv`索引改为当前`announcement_events.csv`。

## 汇报内容与代码对应

| 汇报内容 | 当前实现与核对 |
|---|---|
| 两阶段H0：是否赔付×赔付金额 | `pp_models/h0.py`，Call/Put分开；逻辑回归C=0.1，正赔付无截距金额回归 |
| 输入Base、剩余期限、标准化价格位置 | `pricing_inputs.py`共享3项概率变换；名义期限只做接口验证，不作为第4个回归项 |
| H0与Base分方向混合 | `pp_pricing/model.py`，Call alpha=0.40、Put alpha=0.15，输出价格及等价倍率 |
| 4月训练、5–7月验证、7日更新 | 唯一`validation_protocol_v3.json`及`validation.history_ids`；30日创建窗、更新前严格已结算 |
| 最新完整分钟行情、60秒以内 | `pp_data.features.build_features`后向匹配；测试改变未来K线不影响报价 |
| 实际赔付和提前终止不进入报价 | H0精确字段白名单；交付适配器不读取`closedAt`和`realizedProfit_USDC` |
| 月等权与逐单误差不同 | `fixed_blend_statistics.py`及当前`model_comparison.csv`；交付重放四项指标一致 |
| H0在5、6月的RMSE及MAE优于Base | `verify_delivery.py`直接核对两月整体结果 |
| 公告数据拓展与独立小额规则 | `pp_calendar`及`announcement_overlay.py`；不擅自叠加到最终混合 |
| 尾部、方向、金额与不确定性 | 原风险、月度、重采样结果保留；独立核验确认汇总与原始运行一致 |

零赔付属于概率模型的有效负样本，不是被过滤的噪声。每7日更新两阶段参数，不等于报价每7日才变化。现有小时尺度实验未证明“小时重训必然放大噪声”，代码也没有实现或声称这一结论。上述表述与此前确认的讲稿一致。

## 数值与证据核对

机器记录：[delivery_verification_20261018.json](../results/delivery_verification_20261018.json)。

- 19个公开结果文件SHA256与来源清单一致；Word报告与上一轮完整数据审计时的SHA256一致。
- README的12个主要误差数值与当前统计一致。
- 新入口使用原12组参数重放4,640单，与独立H0算术重放的最大逐笔差异为2.274×10⁻¹³ USDC/BTC。
- 最终混合：月等权RMSE **293.04802481408905**，月等权MAE **183.5580565930617**，逐单RMSE **295.6140432392129**，逐单MAE **185.80382025125272**，均复现报告。
- 原统一核验器重新通过：97配置日汇总、1,164行日统计、87个独立重放模型、4,611行分组统计、86组混合RMSE恒等式、4,640笔公告及2,024行风险统计。
- 上述真实数据核验无模型拟合，官方验证打开次数为0。测试中的小规模合成拟合只用于确认训练类与推理算术一致。

## 运行检查

- 最终完整测试：97项通过，其中原有86项、交付新增11项。
- 干净目录检查：仅复制Git文件与本轮新增交付文件，不复制本地`outputs`或私有数据；成功加载默认模型，完成2笔合成特征输入及2笔无赔付标签的官方格式输入。
- 输入顺序与带前导零的订单ID均保留；已有输出文件拒绝覆盖。
- 导入和加载预测模型不加载scikit-learn或研究运行器；预测只需基础依赖。
- 43个新增/修改文档中的本地链接有效，`git diff --check`通过。

独立代码审查指出的3项接口问题已通过先复现失败、再修正的方式处理：公开核验增加交付模型本体哈希检查；原始订单适配先规范化混合时间格式；Python与CLI共享订单ID保留和校验。没有改变模型参数或预测公式。未来泛化与官方实际测试不属于本轮代码验收；不使用的BTC数量未新增业务规则，整单金额转换仍由调用方负责。

复现公开证据与完整本地重放：

```sh
python scripts/verify_delivery.py
python scripts/verify_delivery.py --run-dir outputs/model_studies/report_announcement_v1_20261016 --data-dir outputs/preprocessing
python scripts/verify_report_unified.py outputs/model_studies/report_announcement_v1_20261016
python -m unittest discover -s tests
```

## 交付边界

研究回测使用合法已结算标签每7日更新；交付默认加载最后一次参数，不依赖新测试标签。**冻结模型可运行，不等于其未来效果已经被原滚动报告验证。** 无标签反馈时不会假装继续训练，也未恢复已撤回的全量训练计划。

Base按官方“创建时可观察”的说明使用，匹配原始时间未提供属于数据提供方的来源说明边界。不能因此把研究表述成已经存在前视泄漏。

这次整理不修改已有模型成绩，不宣称已通过独立官方测试或已经提高利润。README保留H0在5、6月的优势、分方向效益与最终混合的实际适用范围。此轮尚未提交或推送。
