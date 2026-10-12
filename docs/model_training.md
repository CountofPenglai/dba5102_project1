# 赔付模型：入口与运行约定

**高风险分类训练：**`python -m pp_models.high_risk_study plan|run --run-id ID`已完成q97.5与q95两组GBDT及固定1.5倍抬价试验，沿用32项特征、4–6月滚动协议。两组暂未改善整体赔付预测，不升级主模型。[训练入口与结果](high_risk_training.md)。`high_risk_data plan|audit`仍只生成数据审计。

**当前入口：**`python -m pp_models.rolling_study plan|run --run-id ID`，日期由 `config/validation_protocol_v2.json` 统一管理；采用 4–6 月逐日、30 天已结算历史，7 月保留。第 15 节为当前基础比较，第 16 节为稀疏冲击叠加；下方六月 GAM 等入口保留历史复现。[当前协议](rolling_validation.md)，[本轮报告](../outputs/model_studies/rolling_calibration_apr_jun_20261009/reports/evaluation.md)。

更新日期：2026-10-09。**GAM 首轮验证研究已完成，EBM和原浅层GBDT回归模型尚未训练；高风险GBDT分类器已独立完成首轮研究。** 默认配置仍是初始设置；本次选出的 GAM 参数单独保存在运行来源目录，不覆盖默认配置。结果见[研究记录第 13 节](research_decisions.md#13-gam-首轮运行与结果评价)和[本地评价报告](../outputs/model_studies/gam_validation_20261008/reports/evaluation.md)。

同日已完成多窗口 / 方向 GAM 的后续探索，入口见第 2 节；整体误差没有改善，部分方向金额校准更接近实际。结果见[研究记录第 15 节](research_decisions.md#15-多窗口与方向-gam-重跑结果2026-10-08)和[后续评价报告](../outputs/model_studies/gam_directional_20261008/reports/evaluation.md)。

## 1. 模型与统一目标

| 名称 | 实现 | 初始复杂度 |
| --- | --- | --- |
| `gam` | pyGAM `LinearGAM`，原始单位平方误差，连续变量为带二阶导数惩罚的三次样条 | 每项 6 个样条基函数，惩罚强度 10；方向为带惩罚的 0/1 线性项 |
| `ebm` | InterpretML `ExplainableBoostingRegressor`，RMSE 目标 | 32 个主效应分箱、8 个交互分箱，500 轮，叶节点最少 20 条；两项指定交互 |
| `shallow_gbdt` | scikit-learn `HistGradientBoostingRegressor`，平方误差 | 最大深度 3、最多 8 个叶节点、150 轮、叶节点最少 20 条 |

标签统一为 `realizedProfit_USDC / quantity_BTC`，零值和大额真实赔付均保留，不额外加权，不改变标签。预测统一取 `max(raw_prediction, 0)`，汇总另报被修正的预测数量。

GAM 与 EBM 保持可拆解的函数结构，后续可从各自估计器提取贡献。本阶段不导出图表或解释文件。EBM 的两项交互为 `realized_vol_60m × remaining_seconds` 与 `option_type × log_return_60m`；复制配置并将 `interactions` 设为 `[]` 即可准备无交互对照，不能根据测试期结果更改。

GAM 不使用自动网格搜索；EBM 关闭内部验证、随机早停和外层分袋，固定训练轮数；浅层树也关闭随机早停。此处为首轮三模型状态；保守余量、需求和最优报价尚未实现，两步赔付结构已在第 15 节独立实现。B0/B1 固定均值估计器与有限调参流程已实现，GAM 首轮与两种均值对照已运行。

**历史六月流程：**使用 [验证期训练与调参计划](validation_training_plan.md) 的 `study-plan / study-run` 入口。6 月 1–20 日调参、21–30 日确认；只读取原训练期和验证期，不读取 7 月测试集 CSV。单配置入口仍保留，下文测试期示例是未来接口说明，本阶段不执行。新的 scoped loader 不修改既有预处理或样本名单。

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

两个研究入口均接受 `--models gam` 等模型子集；省略时运行三模型，均值基准始终保留。只构造和运行所选模型，模型选择进入预算与运行身份，续跑时不能改变。

本次已执行 `.venv/bin/python -m pp_models study-run --models gam --run-id gam_validation_20261008`，完成 5 个 GAM 调参候选与 1 个确认候选，共 110 次拟合。确认参数为 lam=100、n_splines=4；它是按 1% 容忍度选择的较简单配置，并非调参 RMSE 最低者。运行数据与代码校验、实际耗时及分组评价都保存在该运行目录。

### 多窗口与方向 GAM 的后续探索入口

依据[研究记录第 14 节](research_decisions.md#14-风险方向与多观测窗口2026-10-08)，使用独立版本 `config/gam_directional_v1.json`，保留旧配置和预处理产物：

```sh
# 构建并核对补充特征及样本范围，不训练
.venv/bin/python -m pp_models.directional_study plan
# 实际执行四个 GAM 版本和 B0/B1；续跑使用同一 ID 并添加 --resume
.venv/bin/python -m pp_models.directional_study run --run-id gam_directional_20261008
```

`original` 重现首轮已选 GAM；`multiwindow` 补齐 1 / 4 / 24 小时收益与总波动；`directional` 增加 Put 与共享曲线的差异；`semivol` 将总波动替换为上行 / 下行波动。差异基函数仅在当日合法 Put 历史上中心化，防止顺带引入无惩罚的方向截距；预测复用训练中心。现价曲线保持共享，期限、moneyness、收益、波动允许方向差异。波动统一到每分钟尺度，原配置保留原尺度以精确复现。

新版本样条数固定为 4，各比较 lam=10/100/1000；最佳调参 RMSE 的 1% 内优先更强平滑。先保存所有版本的所选配置，再做后段评价，并保留 lam=100 固定参数对照。最多 270 次每日 GAM 拟合与 60 次均值更新。

此入口只允许验证期，打开测试产物会被拦截。源行情读取截止 7 月 1 日前的时间前缀，仅检查首个边界行时间以停止，不解析其价格或读取未来尾部。方向特征的行 ID、最后 K 线时刻与重算总波动须和原数据一致；不静默删行。

产物位于 `outputs/model_studies/<run-id>/`：`trials/` 保存候选统计；`metrics/tuning*.csv`、`followup*.csv` 保存汇总与逐日方向统计；`provenance/` 保存配置、来源、所选模型与执行信息。逐日表的总体、方向和共同子集行共享一次拟合的时间/自由度，计算拟合次数或耗时时应只取 `cohort=risk, group=overall`。不保存逐笔预测、特征全表和模型。

6 月后段已被用于形成此次假设，全部新结果属于后续探索，不能称为独立确认。保守余量和 7 月测试继续留在以后阶段。

### 价格动态与平台订单频率的顺序实验

```sh
.venv/bin/python -m pp_models.market_activity_study plan
.venv/bin/python -m pp_models.market_activity_study run --run-id gam_market_activity_20261008
```

独立配置 `config/gam_market_activity_v1.json`；不改变旧预处理或实验配置。以 D 半波动版为底座，E (`price_dynamics`) 增加六个价格动态特征，F (`platform_activity`) 再增加四个方向/窗口订单数。保留原 A、D、B0、B1；E/F 各比较 lambda=10/100/1000，按前 20 日 RMSE 最佳 1% 内偏好强平滑，并预先指定 lambda=1000 的 D→E→F 固定正则对照。最多 220 次 GAM 拟合与 60 次均值更新。

六个价格动态是：最近 60 分钟 RMS 减前一个不重叠小时 RMS、60/240 分钟 RMS 比、240 分钟内最大正/负分钟对数收益幅度、240 分钟有先后顺序的最大对数回撤/上涨。频率分别统计 `[下单时间−1/4小时, 下单时间)` 的 Call/Put 记录数；包含当时已经创建但未结算、未入 canonical 的订单，不利用结果、Base 或结算时间筛选。源无订单 ID，按提供记录计数；数据覆盖是否代表完整平台需求未知。

行情与订单原始文件都只读 7 月前时间前缀；首个边界行仅用于检查时间，不解析其行情或方向，未来尾部不遍历。补充特征只在内存里和原样本精确对齐。价格特征复核最后 K 线时刻；缺口或错位直接失败。

订单活跃度评价用当天合法训练行的过去 4 小时 Call+Put 数量 q80，高于或等于阈值为 high；仅用于分组诊断，没有强制加价或单调约束。输出继续按 `trials/metrics/provenance/reports` 分类，另有 `metrics/feature_statistics.csv` 的聚合分布与 `provenance/source_snapshot.zip` 的小型代码归档。`selected` 标识主比较配置；`fixed_regularization_control` 标识 lambda=1000 对照，一条记录可同时属于两者，不重复累计拟合数。

这两组实验仅检验新增信息对 GAM 赔付预测的作用，没有训练新回归基准或独立波动模型，也不检验涨价后的需求或真实利润。

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
- 常量特征仍保留，逐日记录其名称；首轮 GAM 没有出现常量输入或常量目标日，因此尚不能由本次结果验证这些退化情况。
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

指标包括订单数、单位赔付 RMSE/MAE、预测/实际订单总赔付、总额差值、预测/实际总额比、负预测修正数量。新运行额外保留高估金额累计和低估金额累计，二者相减等于净偏差；旧运行文件保持原样。总额按订单数量换算；实际总额为零时比例留空。

风险口径运行同时报告全部风险样本与共同评价子集的成绩；后者仍使用风险历史池训练，不能误称为“仅共同样本训练”。共同口径运行只报告共同样本。逐日表记录正赔付训练数量、总训练数量、常量目标与常量特征。

不默认保存逐笔预测、模型文件或每日检查点，因此训练结束后不能直接从这些文件恢复估计器；需要按相同来源和配置重新拟合。该取舍遵循当前只保存关键统计量的要求。后续如需部署或解释导出，应单独增加持久化方案。

## 6. 验证边界

自动测试包含真实模型构造、配置/输入限制、时间边界、未来标签隔离、指标手算和运行来源核验；调度测试使用人工替身，与正式模型实验分开。方向扩展另含人工数据上的真实 GAM 响应、惩罚和旧模型一致性测试；价格/频率扩展增加有序路径、左右时间边界、未结算事件与候选恢复测试。基础比较增加留一分组、原单位乘数、四项正则、零标签、共同历史和选参冻结检查。当前 140 项测试通过（包括两轮快速响应的 16 项、双约束的 8 项、终点分布的 10 项、信号探索的 9 项和业务约束的 10 项新增测试）。双约束覆盖历史阈值、幅度限制、无评分日更新、跨阶段重放、取舍标准及结果冻结。

GAM 正式运行的 110 次拟合均通过收敛检查，无拟合 / 预测告警；EBM 和树的真实拟合仍待执行。内部逐日更新成绩不能直接等同于不回传新标签的官方批量评估，也不能由误差改善直接推断真实利润改善。

多窗口 / 方向扩展另完成 270 次真实 GAM 拟合与 60 次均值更新，全部收敛、无告警。后段所选模型的负预测修正数为原模型 0、多窗口 6、方向差异 19、半方差 35；均按同一非负规则评分，不代表数值不收敛。

价格动态 / 平台频率扩展完成 200 次 GAM 拟合与 60 次均值更新，全部收敛、无告警；两组仍未超过原对照。后段 D/E/F 负预测修正数为 35/54/86。完整结果见 `outputs/model_studies/gam_market_activity_20261008/reports/evaluation.md`。

运行全套检查：`.venv/bin/python -m unittest discover -s tests -v`。

方法资料：[GAM](https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html)、[EBM 参数](https://interpret.ml/docs/python/api/ExplainableBoostingRegressor.html)、[梯度提升树](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)。

## 7. 官方基础价与方向×期限基础比较

独立入口与配置：

```sh
.venv/bin/python -m pp_models.anchor_study plan
.venv/bin/python -m pp_models.anchor_study run --run-id ID
# 相同配置、代码和数据下的中断续跑：追加 --resume
```

配置为 `config/anchor_comparison_v1.json`。不改变既有分割或预处理；只读 train/validation 六表，共同样本仅保留官方 Base 可用且符合原技术条件的订单。两种基础使用相同 30 天已结算历史和相同预测订单。前段实际 6 月 12–20 日共 491 条，后段 21–30 日 879 条；不能拿旧 risk 总体分数当作相同训练样本的结果。

每种基础分别报告原值 raw、统一非负乘数 scaled，以及四项动态修正候选 dynamic。官方 raw 原样输出 API Base；分组 raw 沿用 B1 的方向×标称期限均值，k=30 向整体均值收缩。分组修正层拟合时，基础采用当日合法历史内的留一值，全局收缩均值也剔除该行目标；预测使用全部合法历史，不使用当天赔付。

修正形式 `mu=b*exp(alpha+z@beta)`，四项输入是 log(1h 每分钟 RMS 波动)、log(1h/24h RMS 比)、方向对齐的 1h 收益和 log_moneyness。标准化只拟合当前历史。损失为原单位赔付平方误差除以训练标签平方均值，加 `lambda*sum(beta²)`；alpha 不惩罚。保留零赔付，不回归 `y/Base` 或其对数；统一乘数为 `sum(b*y)/sum(b²)`。

lambda 仅 0.1/1/10，候选包含 scaled。前段 RMSE 最低值 1% 内优先 scaled，再偏好更大 lambda；所有选择先保存，再计算后段结果。后段仍保留两种 raw/scaled 固定对照，选中 scaled 时不重复拟合动态版。零标签历史下修正模型输出零；官方 raw 仍输出原基础价。数值指数保护只防溢出，最终触发则失败，不作为隐含业务加价上限。

输出在 `outputs/model_studies/<ID>/`：`metrics` 总体、方向、期限、波动组与逐日关键统计；`trials` 各候选统计；`provenance` 选参、数据/代码哈希、代码快照和执行资源；`reports` 结果解读。逐日统计额外保留截距乘数、四个标准化系数、系数范数和求解器诊断。不保存逐笔预测或拟合模型，不推断需求溢价或最优商业价格。设计详见 [实施计划](plans/2026-10-08-anchor-comparison.md)。

## 8. 官方 Base × 15 分钟快速响应

```sh
.venv/bin/python -m pp_models.fast_response_study plan
.venv/bin/python -m pp_models.fast_response_study run --run-id ID
# 完全相同来源的续跑追加 --resume
```

配置 `config/fast_response_v1.json`；三项比较为官方 raw、官方每日统一系数 scaled、Put 快速响应 fast。仍采用上一节共同样本和两阶段边界，训练窗口固定 30 天。市场状态每 15 分钟更新，系数幅度 beta 每日零点用已结算 Put 历史拟合；它们不是同一个更新频率。

状态的输入只包括已结束的 UTC 15 分钟区间收益及此前 96 段收益的 RMS；RMS 最低 0.0001，过滤阈值为 0.1 倍 RMS。`d=max(-return/sigma-0.1,0)`；d>0 时 `s=s+(1-s)*(1-exp(-d))`，否则 `s=s/2`。第一次有效下跌即响应，连续下跌趋向饱和；没有强制等两段下跌，不在午夜重置状态。完整区间、连续历史或最新状态缺失时拒绝运行，不丢失评分订单。

Put 预测为 `Base*(1+beta*s)`，Call 原样输出 Base。beta 非负并用 lambda 收缩到零；损失 `SSE_put/sum(y_put²)+lambda*beta²`，闭式拟合，无迭代优化器。无 Put 历史或目标全零时 beta=0。lambda 仅 0.1/1/10，前段最优 RMSE 的 1% 内优先更大值，选型落盘后才做后段评价。此处只在 fast 族内选参，必须另外核对是否优于 raw/scaled，不能将“选中”解释为“改进”。

沿用 `metrics/`、`trials/`、`provenance/`、`reports/` 分类输出；额外保留 Put 首次下跌/连续下跌/恢复与 Call 对照的聚合指标、每日 beta、平均/最大实际乘数。响应状态在恢复期会逐步趋近零；阶段名称仅描述已观测行情，不等于未来是否发生尾部赔付。当前不做小时或 5 分钟搜索，不加商业余量，不保存逐单输出。精确约束见 [实施计划](plans/2026-10-08-fast-response.md)。

## 9. 5 分钟线性与平方累积响应

```sh
.venv/bin/python -m pp_models.accelerated_response_study plan
.venv/bin/python -m pp_models.accelerated_response_study run --run-id ID
```

版本化配置 `config/accelerated_response_v1.json`；共同样本、日期、日度已结算历史边界不变。5 分钟完整行情、lagged 288 段 RMS（24h），下跌强度 `d=max(-return/sigma-0.1,0)`。压力 `q=min(3,q+d)`，平方状态 `s=(q/3)^2`；未有效下跌时 q 除以 sqrt(2)，使额外倍率每 5 分钟减半。状态持续跨日，缺失区间必须重新预热，不能沿用陈旧状态。

Put 预测 `Base*(1+beta*s)`，Call 仍输出 Base；beta 每日拟合且限制 [0,3]，实际倍率最多 4。平方作用在累积压力上，连续相同强度的下跌会产生递增的额外倍率。线性对照 `s=q/3`，未有效下跌时 q 除以 2，额外倍率同样减半。两族均可拟合为 beta=0；4 倍仅为本轮固定探索上限，记录 `multiplier_at_cap_orders` 与 `response_amplitude_capped` 判断是否约束实际结果。

每族仅试 lambda=0/0.01/0.1，选前段 RMSE 最小者，不使用旧版 1% 无差别区间；仅完全相等时偏好更强收缩。后段先评价冻结配置，并按事前设定补充平方版 lambda=0（去重）。参考模型包含 raw、scaled、原 15 分钟 lambda=10，保证可核对旧结果。旧版到新版同时改变累积、恢复与选型规则，效果差异不能只归于频率。

输出仍是 metrics/trials/provenance/reports 的关键聚合记录，无测试数据、逐单预测或模型导出。精确规则与结果记入 [研究决策第 23 节](research_decisions.md#23-5-分钟平方累积响应2026-10-08)。

## 10. 低风险加价惩罚与每日上调限制

```sh
.venv/bin/python -m pp_models.guarded_response_study plan
.venv/bin/python -m pp_models.guarded_response_study run --run-id ID
```

配置 `config/guarded_response_v1.json`。保留原 5 分钟平方状态和原始 MSE 目标；新模型的输入只增加定价前一小时波动、收益，用于历史 calm 阈值和惩罚。calm Put 指当前波动与下跌强度都不超过当日合法训练 Put 各自 q80；下跌强度 `max(-log_return_60m,0)/max(realized_vol_60m,1e-8)`。训练目标为 `SSE_put/sum(y_put²)+eta*mean_calm((beta*s)²)`，eta 搜索 0.01/0.1/1；仍是共享幅度，因此附加惩罚也会间接压低 stress 响应。标签全部保留，无 Huber、标签截断或额外门控。

先按前段选 eta，再比较每日 beta 增加上限 0.05/0.1/0.2；可立即下调。`sequential_start` 为前段首个评分日，首次用合法历史拟合，后续每天更新，包括无评分订单的日子；后段评价必须从同一初始日重放，只有阶段内订单计分。`unscored_fit_days/seconds/warning_count` 是累计未计分拟合统计，每候选取最后一条 overall 日记录即可，不要在每天或分组行上重复相加。

前段选参取舍标准：总体 RMSE<=raw*1.01、calm MAE<=raw*1.02、tail 欠估金额<=原平方版*1.05。在满足者中最小化 RMSE；若无满足者，保存最优 RMSE 候选但标记探索性失败。选择与逐项检查保存于 provenance/selected_models.json 和 selection_checks.json，后段前冻结。tail 只在评价时由实际赔付超过当日合法训练 Put 的 q90 来分类；不能将该标签用作预测特征或门控。没有 Put 历史时分组 unavailable，缺失的评价组不自动通过取舍标准。

输出额外包含 `pricing_risk_group`、`payout_tail_group` 的汇总和日度统计，beta 约束前值、前日值和限制触发标记。峰值、低风险和恢复期按不同轴观察，不把“行情平稳”和“实际赔付较低”混用。所有容差均为本轮研究假设；无测试集读取，无商业利润解释。详见 [研究决策第 24 节](research_decisions.md#24-低风险加价惩罚与每日幅度上调限制2026-10-08)。

首次运行 `guarded_response_20261008` 选 eta=0.01、step=0.1；双约束后段 RMSE 相对 raw 仅降低 0.10%，MAE 升高 2.83%，且未通过 calm MAE 容差。当前保留研究候选，未升级主模型。详见 [完整评价报告](../outputs/model_studies/guarded_response_20261008/reports/evaluation.md)。

## 11. 终点收益分布初步实验

```sh
.venv/bin/python -m pp_models.terminal_distribution_study plan
.venv/bin/python -m pp_models.terminal_distribution_study run --run-id ID
# 同配置、代码和数据中断续跑：追加 --resume
```

配置 `config/terminal_distribution_v1.json`，独立模块 `terminal_distribution.py`（行情分布/代理/区间评分）、`terminal_payout.py`（非负赔付校准）、`terminal_distribution_study.py`（冻结调参/输出）。静态经验分布和固定 EWMA 过滤分布使用 30 天历史，0.5/1/2/4/8 小时收益全部在预测日前完整已知。动态尺度每 5 分钟更新；半衰期固定 60 分钟、1440 连续收益预热，缺口重启。

分布均减去历史中心，未来零对数收益中心不参与调参。订单特征按下单日重建；依剩余期限平方根缩放到期内在赔付代理，不使用提前关闭时间作输入。只调整 <=8h Put，Call 和超期限 Put 保持原值并评分。原值、Put 统一倍率为对照；两个分布族分别只搜 lambda=0.01/0.1/1，按前段最小 RMSE 选型、后段前冻结。模型 `a*Base+b*Proxy`，a,b 非负，b 的收缩以实际赔付平方和归一化。

`metrics/market_summary.csv` 和 `market_daily.csv` 为行情分布关键统计，单位为累计对数收益 bp：CRPS、80/90/95% 区间覆盖/宽度/评分及两侧越界。`scheduled_origins/excluded_origins` 仅 all 行报告，不与 high/normal 行重复相加。各期限分别评价，不将重叠起点当独立观测做显著性推断。订单汇总保存原有赔付统计和每日 a/b、平均/最大 Put 倍率。输出仍仅 metrics/trials/provenance/reports/figures，不保存逐笔预测、场景、特征或模型。

画图入口（独立绘图环境需 `requirements-plots.txt`）：`python scripts/plot_terminal_distribution.py --run-dir outputs/model_studies/ID`。仅读取已校验的聚合 CSV，生成 Put 金额及行情区间诊断图。此实验仅是分布路线初探，不能把到期代理称为精确提前结算模型；[完整冻结设计](plans/2026-10-09-terminal-distribution.md)。

首次运行 `terminal_distribution_20261009`：静态版后段 RMSE 比 raw 降 2.10%、MAE 升 5.18%；动态版未优于 Put 统一倍率。短期限行情分布略有改善，8h 两侧越界失衡，不能仅按总体覆盖率认定成功。[评价报告](../outputs/model_studies/terminal_distribution_20261009/reports/evaluation.md)。

## 12. 高额赔付信号探索

`.venv/bin/python -m scripts.payout_signal_analysis --run-id ID` 仅运行统计探索，不训练模型。41 项价格动态、市场活动与产品活动指标分 Put/Call 输出相关性、历史阈值下的尾部发生率及时间稳定性；使用全部 risk 订单，与 Base 共同样本的模型评价分开解释。输出位于 `outputs/diagnostics/ID/`。[固定分析口径](plans/2026-10-09-payout-signals.md)。

## 13. 合约状态、双侧尾部与强收缩订单流

```sh
.venv/bin/python -m pp_models.constrained_study plan
.venv/bin/python -m pp_models.constrained_study run --run-id ID
.venv/bin/python -m scripts.verify_constrained_study --run-dir outputs/model_studies/ID
# 绘图环境单独安装 requirements-plots.txt：
python scripts/plot_constrained_pricing.py --run-dir outputs/model_studies/ID
```

配置 `config/constrained_pricing_v1.json`；`constrained_features.py` 构建 5m 快照与双侧分布代理，`constrained_pricing.py` 拟合分方向 `Base*c*exp(Z*beta)`，`constrained_study.py` 执行小网格和分类输出。三族 contract/tails/flow，各 lambda=0.1/1/10；订单流惩罚为风险特征十倍。系数由前 30 天已结算订单按日学习，前段选择后段冻结，不加余量。`scaled` 同时校准两个方向；冻结静态参考仍只修正 Put。

特征含代理/Base、经过 moneyness 和剩余期限门控的动量及波动变化，flow 增加市场/平台 Call/平台 Put 各 60m 活跃度。缺失频率中位数填补；同方向历史标准化，截断 Z 为 [-3,3]，系数 [-1,1]、代理系数非负。动态修正仅覆盖 <=8h，其余订单保留、使用整体系数。参数约束为统计稳定性设计，不是业务承保限额或无套利保证。

双侧分布以 30d FHS 为底座，最近 7d 完整收益的 q05/q95 分别调整负/正收益，倍率比 [0.5,2]、log 空间 n/(n+200) 收缩；不足 100 起点回退。输出每期限两侧越界率、区间宽度/评分和 CRPS。没有名义覆盖率保证，也没有建立提前结算模型。

`direction_payout_tail` 使用预测日前同方向同期限历史 q90，n<30 回退方向，仍不足则 unavailable。这个结果切片只做事后评价，不能作为预测输入；不与全 risk 信号研究的高额计数混用。日度系数和数值求解回退记录仅在 overall 行合计，避免重复统计分组。

首次完整运行 `constrained_pricing_20261009_verified`：contract/tails/flow 分别选 1/10/1。flow 后段 RMSE 375.864、MAE 200.084，只比统一倍率改善 0.159% RMSE，仍落后于静态参照 372.364；Call 高额单误差更大。18 候选阶段记录、168 次日度评价，约 10.11 秒、434.1 MiB，140 测试通过。修复前失败记录保留在 `constrained_pricing_20261009`，原因是最优点附近浮点线搜索；有界一阶收敛检查后同配置完整重跑。当前保留研究，不升级主模型。[评价与图](../outputs/model_studies/constrained_pricing_20261009_verified/reports/evaluation.md)。

## 14. 三类互补信息探索

```sh
.venv/bin/python -m scripts.fetch_complementary_data --destination outputs/external_data/complementary_20261009
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.complementary_signals --external outputs/external_data/complementary_20261009 --run-id ID
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.verify_complementary_signals --run-dir outputs/diagnostics/ID --external outputs/external_data/complementary_20261009
```

官方公开输入约 34 MB，下载边界固定在七月前；无私有 API。盘口按完整分钟归一，OI 每 5m，资金费率只用已结算值，DVOL 等完整小时；加入 5m/60m 可用延迟假设。归档事件时间不能证明历史实时接收时间。DVOL 是 30 天 IV 代理、OI 是持仓数量，均不能当成精确的短期限 IV/杠杆率。

固定 C=0.1 Logistic 探针每天以过去 30d 已完成收益更新，比较背景、压力×盘口、波动×拥挤、波动×IV、全部；每 30m 评分未来 1/4/8h 上下尾部，不训练赔付定价模型。不调参、无逐笔输出。结果输出 Brier/log loss/AP、预警增量/重合、实际高额订单关联、逐日汇总及数据覆盖。`market_*` CSV 的 `orders` 指市场起点数。

首次运行 `complementary_signals_20261009`，1,800 次小模型拟合约 9.48 秒。三个组后段六类任务平均相对 Brier 各恶化 0.58%/2.06%/6.27%；有局部订单命中，尚无稳定整体增益，不升级模型。下载入口同时获取 `alignment_check/` 中三天公开分钟归档，核验入口将其与提供方数据逐列比较。旧运行保留原核对清单，新下载写 manifest。[报告](../outputs/diagnostics/complementary_signals_20261009/reports/analysis.md)。


## 15. 4–6 月滚动验证与赔付校准

```sh
.venv/bin/python -m pp_models.rolling_study plan
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pp_models.rolling_study run --run-id ID
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.verify_rolling_study --run-dir outputs/model_studies/ID
```

`rolling_protocol.py` 定义唯一 v2 日期角色及合法日度队列；`calibration_models.py` 实现四对照与三候选；`rolling_study.py` 负责特征、训练与分类评价。使用 v1 开发数据联集，不修改原特征/分割/排除名单，不读测试 CSV 或七月价格。配置为 `validation_protocol_v2.json` 和 `rolling_calibration_v1.json`，当前是固定候选探针，无调参搜索。

对照为官方原值、方向×期限平滑均值、分方向倍率、已冻结静态分布修正。候选为概率×正赔付强度的两步模型、倍率向 1 收缩、创建日等权后同样收缩。结算时间只用于合法历史筛选和预测后的诊断，绝不作为输入；三种方案均无商业余量或外部信号。

首轮 `rolling_calibration_apr_jun_20261009` 共 2,953 笔、72 天、504 次评价，4.40 秒、455.84 MiB。两步模型相对 Base 的月等权 RMSE/MAE 改善约 1.49%/6.36%，但高额 Call 更差；月等权 RMSE 对分组均值只领先约 0.05%。收缩降低部分事后高估同时压低峰值；日等权不继续推进。158 项测试通过，独立日度公式重放与旧对照复现一致。输出仅聚合统计、参数摘要与来源。完整边界和方法定义见 [滚动验证协议](rolling_validation.md)，详细结论见[评价报告](../outputs/model_studies/rolling_calibration_apr_jun_20261009/reports/evaluation.md)。


## 16. 稀疏冲击响应叠加

```sh
.venv/bin/python -m pp_models.sparse_shock_study plan
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pp_models.sparse_shock_study run --run-id ID
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.verify_sparse_shock_study --run-dir outputs/model_studies/ID
```

`sparse_shock.py` 管理价格状态、合约权重与事件收缩幅度；`sparse_shock_study.py` 负责基础预测的历史重放、两项叠加、固定幅度诊断及分类评价。配置 `sparse_shock_v1.json` 与统一 `validation_protocol_v2.json`；不修改旧基础估计器或预处理。

普通触发为15m方向收益的历史z分位q99、绝对收益0.3%、当前5m同向，连续两根确认；极端q99.9且0.6%单根启动。弱证据两根退出，30m强证据有效期、10m半衰期，状态<0.05归零。幅度由此前30天已结算的触发事件校准，至少3事件/10笔，向零收缩，最大总倍率3；两步历史基础预测从3/2逐日重放，避免训练内残差。

四个主版本为原Base、Base＋响应、原两步、两步＋响应；另两项固定A=0.5为诊断。事后等额统一加价对照只评价预算分配时机，不可部署、不参与排名。所有评分方法相同2953笔/72天，7月保留。

首轮 `sparse_shock_apr_jun_20261009` 正式运行4.79秒、450.31MiB，198次基础回放、432次评分日×方法评价。月等权RMSE：Base292.57→293.39，两步288.20→288.88；MAE也略恶化。稀疏性达成，但尾部命中不足，保持/恢复阶段增加高估；不升级模型。170测试通过，194条旧anchor切片复现，独立288幅度/1284日指标核对通过。[冻结方案](plans/2026-10-09-sparse-shock.md)，[完整报告](../outputs/model_studies/sparse_shock_apr_jun_20261009/reports/evaluation.md)。
