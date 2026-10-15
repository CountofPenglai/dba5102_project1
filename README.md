# PP定价研究：H0与官方Base的有限混合

更新：2026-10-15。主分支收录 **H0/Base混合、4月训练与5–7月滚动验证、官方宏观公告日历与分析**；额外保留H0时间衰减、小时金额响应两项探索。目标为估计BTC保险订单的每BTC预期赔付。原始订单与行情不随仓库发布。

**当前结论：** Call `40% H0 + 60% Base`、Put `15% H0 + 85% Base` 是细网格中的月等权误差最优点，邻近比例近乎持平；全期逐单RMSE和完整风险约束仍未超过Base，不是已经部署的最终模型。

| 同一5–7月队列 | 月等权RMSE | 月等权MAE | 全期逐单RMSE |
|---|---:|---:|---:|
| Base | 293.564338 | 184.439853 | **295.150078** |
| H0 | 298.010152 | 186.263090 | 303.531640 |
| 粗网格最优Call50%/Put25% | 293.114750 | 183.652670 | 296.046896 |
| 细网格最优Call40%/Put15% | **293.048025** | **183.558057** | 295.614043 |

误差单位USDC/BTC。月等权RMSE先对三个单月MSE等权平均再开方，全期逐单则按订单数加权，两种口径必须并列。最优相对Base的改善区间仍跨0，Call净金额偏差扩大10.81%。见[细网格报告](docs/reports/2026-10-15-h0-blend-fine.md)和[版本化关键统计](docs/results/README.md)。

## 模型与合并范围

H0为两阶段模型：分Call/Put估计正赔付概率，再用正赔付订单拟合相对Base的金额系数，输出`概率 × Base × 金额系数`。概率输入为Base、剩余期限及按波动标准化的方向价格位置；实现为`pp_models/calibration_models.py`中的`hurdle_base`。最终赔付仅作已结算训练目标和事后评价，不作定价输入。

| 内容 | 价值与状态 | 依据 |
|---|---|---|
| 固定/动态混合与状态回退 | 固定混合减轻跨期退化；动态权重未胜出；不按月份拼赢家 | [粗网格](docs/reports/2026-10-15-h0-blend-ratios.md)、[回退](docs/reports/2026-10-15-h0-state-fallback.md)、[细网格](docs/reports/2026-10-15-h0-blend-fine.md) |
| 验证协议v3 | 4月训练、5–7月滚动验证，30日已结算历史、每7日更新 | [说明](docs/validation_protocol_v3.md)、[配置](config/validation_protocol_v3.json) |
| 官方宏观公告 | CPI、非农、FOMC日历、历史预告证据、跨公告/阶段分析及公告倍率研究记录 | [数据集](external_data/macro_events/README.md)、[阶段报告](docs/reports/2026-10-14-macro-phases.md)、[倍率结果](docs/reports/2026-10-14-macro-multiplier.md) |
| **额外1：H0时间衰减** | 检验近期加权能否减少滞后；无稳定整体增量，保留分方向更新速度诊断 | [报告](docs/reports/2026-10-15-h0-time-decay.md) |
| **额外2：小时金额响应** | 检验波动水平、变化速度和保守门控；局部尾部收益伴随普通订单代价 | [报告](docs/reports/2026-10-15-h0-hourly-amount.md) |

选中入口所需的共享加载、评估、统计及历史模型代码同时纳入。部分共享函数仍在旧研究模块中，这些模块作为依赖保留，不代表额外采用其研究方案。范围及来源哈希见[集成清单](docs/integration/h0_main_selection.json)，其他探索保留在研究分支。

## 当前验证与数据边界

- 训练起点 **2026-04-01 UTC**；验证为 **2026-05-01至2026-07-31 17:11:48 UTC**，右端不含。同队列4,640单，5/6/7月分别726/1,373/2,541单。
- 从5月1日起连续每7日更新，每次只用此前30日创建、更新之前已结算的订单；周内冻结。7月1–2日沿用6月26日参数。
- 验证标签结算后可用于之后的合法更新，不能影响已有预测。当前依赖标签反馈，官方无标签部署兼容性尚未验证。
- [预处理v1](config/preprocessing_v1.json)中的train/validation/test仍为不可变存储分区；v3从其并集中按时间限定角色。**旧test目录的7月现在是已见验证数据**，不重新分区、不补回原来排除的跨边界订单。
- 新官方验证文件`pp_transactions_validation.csv`禁止读取。未启动全量训练或未知官方测试。
- RMSE优先，MAE监控代价；方向/期限、普通/尾部、现金低估另查。尾部阈值只用月初可用历史，真实尾部身份仅供评价。

详见[协议v3](docs/validation_protocol_v3.md)。5–7月已经反复用于开发，选参结果不是独立盲测。

## 环境和运行

Python 3.10+；已记录实验环境为Python 3.12。测试使用合成样本，不需要提供方订单。

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-models.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m pp_audit --help
.venv/bin/python -m pp_data --help
.venv/bin/python -m pp_models.h0_blend_fine_study --help
```

审计和预处理使用`--data-dir`指定已获授权的原始资料；默认是仓库旁`docs/Jasper Vault Options Pricing Optimisation(Included Base Premium)/`。H0研究读取本仓库`outputs/preprocessing/`的经过哈希验证的canonical产物。完整生成数据、虚拟环境和运行档案由`.gitignore`排除。

以下顺序用于**明确重新开展实验**，安装/合并时不会自动运行。前置是通过`pp_data`生成兼容的canonical数据；输出目录必须不存在，不覆盖历史运行，各阶段校验前置参考和来源哈希。

状态回退与小时金额响应还必须保留原始`btc_market_history.csv`，位置固定为`<仓库父目录>/docs/Jasper Vault Options Pricing Optimisation(Included Base Premium)/btc_market_history.csv`，即`pp_data.config.DEFAULT_DATA`下的市场文件。预处理的自定义`--data-dir`不会传递给这些研究入口。它们只读取历史截止前的市场前缀，并要求前缀SHA256为`cbf5d491d167ac377beb844dc9794e4f5e9986f72859035426f7789e67924b50`；这是前缀哈希，不是整个可继续追加行情的文件哈希。独立工作树的仓库父目录可能不同，运行前须按该位置准备已授权市场文件，不能仅复制canonical目录。

```sh
.venv/bin/python -m pp_models.h0_time_decay_study --run-id h0_time_decay_20261015
.venv/bin/python -m pp_models.h0_state_fallback_study --run-id h0_state_fallback_20261015
.venv/bin/python -m pp_models.h0_blend_ratios_study --run-id h0_blend_ratios_20261015
.venv/bin/python -m pp_models.h0_blend_fine_study --run-id h0_blend_fine_20261015
.venv/bin/python scripts/verify_h0_blend_fine.py outputs/model_studies/h0_blend_fine_20261015

# 小时金额响应的前置运行仅为时间衰减；同时需要上述默认位置的市场文件。
.venv/bin/python -m pp_models.h0_hourly_amount_study --run-id h0_hourly_amount_20261015
```

`scripts/verify_h0_*.py`独立重放保存参数，不新增训练。核验原历史完整档案需对应源码快照、canonical和前置运行；`docs/results/`只有精选统计，不能替代完整运行目录。宏观历史分析/倍率还需要旧参考档案，见[交接索引](docs/research_document_index.md)。

旧`python -m pp_models study-plan/study-run`保留早期六月GAM/EBM/浅层树流程，**不会自动采用v3**，不是当前H0默认入口。旧说明见[模型文档](docs/model_training.md)。

## 公告数据维护

`external_data/macro_events/`最后核对时间为2026-10-14。82条日程含65个主要事件；CPI/非农覆盖2026–2027，FOMC延至2028年1月。使用纽约时区转UTC，保留`known_by_utc`、历史预告和改期来源；未知或尚未公开不能当作“无公告”。

```sh
.venv/bin/python scripts/update_macro_calendar.py --source external_data/macro_events/sources/2026-10-14/verified_schedule_history.json --check-only
```

新日程另建来源包，先检查后发布新快照，不覆盖历史版本。脚本不自动抓取或定时更新；未来训练/测试前仍需核对官方日期。详见[字段与维护流程](external_data/macro_events/README.md)。

## 项目组织与交接

`pp_audit/`负责审计，`pp_data/`负责固定存储与加载，`pp_models/`负责模型及统计，`pp_calendar/`负责公告时间和版本；`scripts/`存独立分析、维护与核验入口。`config/`存固定配置，`docs/reports/`存选中报告，`docs/results/`存关键统计，`docs/integration/`存合并来源，`outputs/`存不提交的本地完整产物。

[研究交接索引](docs/research_document_index.md) · [研究决策](docs/research_decisions.md) · [集成计划](docs/plans/2026-10-15-main-integration.md)。尚未建立需求模型，不将赔付预测改善等同于真实利润提升。
