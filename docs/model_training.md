# 三类赔付模型：入口与运行约定

> 历史版本说明：本文描述早期模型与验证安排。2026-10-15合入的H0/Base研究使用[验证协议v3](validation_protocol_v3.md)，已完成历史实验；当前入口和结果见[README](../README.md)。下文原日期与“未训练”状态只适用于该历史版本。


更新日期：2026-10-07。**代码已实现，本轮不进行真实模型拟合，不产生训练成绩。** 配置参数是供后续验证的起点，不是已选出的最优模型。

## 1. 模型与统一目标

| 名称 | 实现 | 初始复杂度 |
| --- | --- | --- |
| `gam` | pyGAM `LinearGAM`，原始单位平方误差，连续变量为带二阶导数惩罚的三次样条 | 每项 6 个样条基函数，惩罚强度 10；方向为带惩罚的 0/1 线性项 |
| `ebm` | InterpretML `ExplainableBoostingRegressor`，RMSE 目标 | 32 个主效应分箱、8 个交互分箱，500 轮，叶节点最少 20 条；两项指定交互 |
| `shallow_gbdt` | scikit-learn `HistGradientBoostingRegressor`，平方误差 | 最大深度 3、最多 8 个叶节点、150 轮、叶节点最少 20 条 |

标签统一为 `realizedProfit_USDC / quantity_BTC`，零值和大额真实赔付均保留，不额外加权，不改变标签。预测统一取 `max(raw_prediction, 0)`，汇总另报被修正的预测数量。

GAM 与 EBM 保持可拆解的函数结构，后续可从各自估计器提取贡献。本阶段不导出图表或解释文件。EBM 的两项交互为 `realized_vol_60m × remaining_seconds` 与 `option_type × log_return_60m`；复制配置并将 `interactions` 设为 `[]` 即可准备无交互对照，不能根据测试期结果更改。

GAM 不使用自动网格搜索；EBM 关闭内部验证、随机早停和外层分袋，固定训练轮数；浅层树也关闭随机早停。保守余量、两步赔付结构、需求和最优报价尚未实现。后续已补齐 B0/B1 固定均值估计器与有限调参流程，所有方法都尚未进行真实训练。

**当前研究流程：**使用 [验证期训练与调参计划](validation_training_plan.md) 的 `study-plan / study-run` 入口。6 月 1–20 日调参、21–30 日确认；只读取原训练期和验证期，不读取 7 月测试集 CSV。单配置入口仍保留，下文测试期示例是未来接口说明，本阶段不执行。新的 scoped loader 不修改既有预处理或样本名单。

## 2. 依赖与安全检查入口

在 `project1_program` 下运行。模型依赖独立于审计依赖：

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-models.txt
.venv/bin/python -m pp_models
.venv/bin/python -m pp_models check
```

无参数只显示帮助。`check` 检查配置、依赖、真实估计器构造、现有数据哈希与逐日样本覆盖，只向终端输出结果；不会调用 `fit`、创建训练目录或计算预测成绩。

```sh
# 共同样本但不含基础价，适合控制训练样本后比较
.venv/bin/python -m pp_models check --cohort benchmark
# 同一共同样本，加入基础价
.venv/bin/python -m pp_models check --cohort benchmark --use-base
# 单模型检查
.venv/bin/python -m pp_models check --models gam
```

`--dataset-dir` 指向 `pp_data` 产物目录，不是原始 CSV 目录；默认读取既有 `outputs/preprocessing`。没有修改分割日期的参数。

完整研究的安全规划检查为 `.venv/bin/python -m pp_models study-plan`。明确获准后才能使用 `study-run --run-id ID`；续跑加 `--resume`。搜索范围见 `config/validation_study_v1.json`，结果放在 `outputs/model_studies/`，与单配置运行分开保存。

## 3. 将来训练时的明确入口（本轮未执行）

```sh
# 6 月验证：按天训练与预测所有三个模型
.venv/bin/python -m pp_models train --split validation --run-id validation_v1

# 7 月测试：必须引用已完成且规则完全相同的验证运行
.venv/bin/python -m pp_models train --split test --run-id test_v1 \
  --validation-run outputs/model_runs/validation_v1
```

可用 `--models gam ebm` 选择子集；同一验证/测试配对必须保持相同模型清单与顺序。用 `--config PATH` 指定复制后修改的模型配置。窗口、特征、超参数、口径和基础价开关的更改都会使旧验证引用失效；代码、数据或依赖版本变化亦会拒绝沿用旧验证运行。

`train` 同时执行逐日拟合与评价，不是全量拼接后的一次拟合。运行 ID 必须唯一，既有目录即使为空也不覆盖。训练中出错不发布成功清单；未完成或被篡改的运行不能作为测试引用。该检查帮助固定规则，但不能代替研究者不依据测试成绩再次调参的纪律。

## 4. 共同输入与每日历史

初始七项：方向、剩余期限、`log_moneyness`、市场价格、60 分钟波动、1440 分钟波动、60 分钟收益。全部来自既有白名单。不直接使用历史成交保费、markup、结算时间、实际持有时长或事后行情。基础价只在共同样本上显式加入，不插补。

默认 `risk` 口径、不含基础价，保留所有 v1 风险样本。选择 `benchmark` 时，训练与评价都只使用共同样本；结果不能与更大训练池上的模型混作纯算法比较。

每天 UTC 00:00 为 D，使用 `[D−30天, D)` 创建且 `closedAt < D` 的 v1 订单，重新构造并拟合当天模型。当天输入仍按各笔订单的实际创建时刻构建；不在日内使用新结算标签重训。可以使用之前已结算的验证/测试订单，不补回 v1 排除的 24 条跨分区记录。

- 少于 `min_train_rows=30` 条合法历史时明确失败，不静默扩大窗口或换模型。
- 若历史标签全为同一个值，直接使用该常数作为平方误差解，避免全零标签导致 GAM 零尺度问题；逐日统计标记 `constant_target`。
- 单一方向窗口使用 GAM 二元线性项，避免预测时出现未见分类水平错误；没有另一方向的数据时不能据此宣称已学会方向差异。
- 常量特征仍保留，逐日记录其名称；这些条件下的真实数值拟合尚未执行验证。
- 所有编码、样条节点与树分箱由当日拟合历史决定；没有跨日全量预拟合变换。

## 5. 分类输出：只保留关键统计量

只有显式 `train` 成功后才会生成：

```text
outputs/model_runs/<run-id>/
  metrics/
    summary.csv         总体、方向、产品期限的关键预测指标
    daily.csv           各预测日指标、训练样本量与退化窗口标记
  provenance/
    config.json         解析后的完整实验配置
    manifest.json       配置、代码、数据、依赖版本、产物哈希与验证引用
```

指标包括订单数、单位赔付 RMSE/MAE、预测/实际订单总赔付、总额差值、预测/实际总额比、负预测修正数量。总额按订单数量换算；实际总额为零时比例留空。

风险口径运行同时报告全部风险样本与共同评价子集的成绩；后者仍使用风险历史池训练，不能误称为“仅共同样本训练”。共同口径运行只报告共同样本。逐日表记录正赔付训练数量、总训练数量、常量目标与常量特征。

不默认保存逐笔预测、模型文件或每日检查点，因此训练结束后不能直接从这些文件恢复估计器；需要按相同来源和配置重新拟合。该取舍遵循当前只保存关键统计量的要求。后续如需部署或解释导出，应单独增加持久化方案。

## 6. 验证边界

本次测试包含真实模型构造、配置/输入限制、时间边界、未来标签隔离、指标手算和运行来源核验；调度测试使用人工替身，未调用真实模型的 `fit`，也未进行合成数据上的真实训练。

真实拟合是否收敛、数值稳定性和预测效果，留待正式训练时检验。内部逐日更新成绩也不能直接等同于不回传新标签的官方批量评估。

运行全套检查：`.venv/bin/python -m unittest discover -s tests -v`。

方法资料：[GAM](https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html)、[EBM 参数](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html)、[梯度提升树](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)。
