# PP 定价研究：两阶段赔付校准

更新时间：2026-10-13。

本项目研究如何在官方基础值（Base）之上，更准确地估计保险订单的预期实际赔付，为后续定价提供风险成本依据。当前主线为**两阶段赔付模型：先估计赔付概率，再估计发生赔付时的金额**。输出单位为 USDC/BTC；尚未建立需求模型，也不将预测误差改善等同于真实利润提升。

**当前状态：** 已完成数据审计、统一预处理、早期模型探索、4–6月滚动校准，以及稀疏冲击和高风险分类对照。后续五个研究分支的设计已确认，尚未实现或训练；7月继续保留。前五组完成后必须汇报并停止，正式冻结、部署兼容性回放和7月后期测试须由用户另行明确启动。

## 文档与交接入口

接手项目先读[研究文档专用索引](docs/research_document_index.md)。它汇集两份 Word 和具体实施方案，并说明版本差异、当前状态及执行边界。

| 文档 | 用途 |
| --- | --- |
| [研究报告大纲（Word）](docs/deliverables/研究报告大纲_两阶段赔付模型.docx) | 商业汇报框架、已有成果与未来结果展示位置 |
| [未来研究计划（Word）](docs/deliverables/未来研究计划_两阶段赔付模型.docx) | 用简洁语言解释研究问题、做法和交付 |
| [前五分支实施方案](docs/plans/2026-10-12-two-stage-implementation.md) | 当前开发阶段的实验设置、评价标准、拟实现文件及执行 prompt |
| [后期独立阶段方案](docs/plans/2026-10-12-two-stage-post-development.md) | 正式冻结、无新标签回放、7月两种模式及离线部署检查；等待单独启动 |
| [研究总计划](docs/research_plan_two_stage.md) / [研究决策记录](docs/research_decisions.md) | 主线依据、历史证据、讨论过程与已确认决策 |
| [滚动评价协议](docs/rolling_validation.md) / [模型使用说明](docs/model_training.md) | 当前数据口径与已有实验入口 |

Word 是逐分支细化前的沟通版本；实际实施以最新用户确认、详细实施方案和决策记录第36–42节为准。特别是四档收缩、三层概率对照及独立后期阶段，不能沿用 Word 中较早的简化设置或连续日程。文档中的 prompt 是未来执行模板，阅读文档不代表启动研究。

## 已实现的功能与已有实验

| 范围 | 实现与研究状态 | 主要入口 |
| --- | --- | --- |
| 数据基础 | 审计、固定预处理、特征与标签隔离、来源和产物校验已实现 | `pp_audit`、`pp_data` |
| 当前滚动校准 | 4–6月七个固定方法已完成，包括 Base、分组均值、方向倍率、静态分布、原两阶段模型和两种收缩对照 | `pp_models.rolling_study` |
| 稀疏冲击叠加 | 已分别叠加 Base 和两阶段模型；整体误差略升，未采用 | `pp_models.sparse_shock_study` |
| 高风险特征与分类 | 32项创建时特征、q97.5/q95标签及 GBDT 分类叠加已完成；未证明整体增益 | `pp_models.high_risk_data`、`pp_models.high_risk_study` |
| 早期回归框架 | GAM、EBM、浅层 GBDT 回归接口已实现；只实际开展了 GAM 研究，EBM及原浅层回归模型尚未训练 | `pp_models` 的 `check/train`、`study-plan/study-run` |
| 历史六月探索 | 方向波动、价格动态与订单频率、基础值比较、快速响应、终点分布和约束修正已实现并开展实验 | 下文历史入口表及模型使用说明 |
| 新一轮两阶段改进 | 五个分支仅完成设计，拟新增的 `two_stage_*` 模块和配置尚不存在 | 前五分支实施方案 |

高风险 GBDT **分类器**与早期三模型框架中的 GBDT **回归器**是不同实验，不能把分类器已训练理解为三模型均已完成。

### 当前两阶段模型 H0

已实现的 `hurdle_base` 在后续研究中记为 **H0**，实现见 [calibration_models.py](pp_models/calibration_models.py)：

`预期单位赔付 = P(赔付 > 0 | 创建时信息) × Base × 正赔付金额系数`

- Put和Call分别拟合。概率阶段使用强正则 Logistic（`C=0.1`），输入为 `log(Base)`、`log(合同剩余期限)` 和按历史波动标准化的方向价格位置。
- 金额系数仅从合法历史的同方向正赔付订单估计：`c = sum(Base × Y) / sum(Base²)`，其中 `Y` 为每BTC实际赔付。同一天、同一方向共用一个金额系数。
- 每天只使用当时已结算的历史更新；标准化和裁剪遵循同一信息边界。零赔付、单一类别或缺少方向历史有明确回退，并记录支持度。

这与计划中的合同条件化 Gamma 金额模型不同；不能把尚未实现的分支能力归给 H0。

### 已有结果与当前判断

以下为2026-10-09完成的4–6月共同样本结果，2,953笔订单、72个评分日。误差单位为 USDC/BTC，越低越好；总金额按订单BTC数量换算。

| 方法 | 月等权 RMSE | 月等权 MAE | 预测/实际总赔付 |
| --- | ---: | ---: | ---: |
| 官方 Base | 292.57 | 182.13 | 73.5% |
| 方向×期限平滑均值 | 288.34 | 174.40 | 76.3% |
| 分方向倍率 | 289.73 | 169.80 | 69.9% |
| 静态分布修正 | 288.91 | 179.18 | 85.4% |
| 原两阶段 H0 | 288.20 | 170.55 | 71.7% |

H0相对Base的月等权RMSE下降约1.49%、MAE下降约6.36%，三个单月均改善；但对分组均值的RMSE优势仅约0.05%，高额Call的RMSE反而增加6.64%，总金额低估也未消失。因此保留两阶段结构作为研究主线，尚未将H0认定为满足全部风险约束的最终替代模型。

稀疏冲击与高风险分类叠加均未带来整体改善，当前不继续通过扩大报警或普遍抬价推进。历史失败探索及原因保留在[决策记录](docs/research_decisions.md)；上述数值不能作为新分支已经成功的证据。

完整[滚动校准报告](outputs/model_studies/rolling_calibration_apr_jun_20261009/reports/evaluation.md)及其他运行结果位于本地 `outputs/`，不随 Git 发布；仓库内保留研究摘要与来源说明。

## 后续研究路线：已确认、待实施

主模型固定现有概率阶段、30天历史窗口和方向拆分，优先改进正赔付金额；概率替换只作为第四分支的归因对照。本阶段不增加商业风险余量、外部信号或需求估计。

| 分支 | 固定设计 | 要回答的问题 |
| --- | --- | --- |
| 1. 合同条件化金额 | H0；正赔付 `Y/Base` 均值对照G0；合同剩余期限与方向标准化价格位置的 Gamma log-link 模型G1a/G1b，`alpha=1/10` | 改善来自估计方法变化，还是合同信息增量？ |
| 2. 方向波动信号 | 只增加60/240分钟风险方向RMS的对数比；Put取下跌、Call取上涨；G2与同alpha的G1配对 | 在合同信息之外，波动变化是否带来额外收益？ |
| 3. 收缩与方向回退 | 固定原始候选R，Put/Call各按 `1 → 0.5 → 0.25 → 0` 保留修正；支持度不足回退Base | 能否以适度精度取舍通过风险约束，同时仍超过H0？ |
| 4. 概率与金额归因 | 方向原始频率、方向×期限平滑频率（固定`k=30`）、原Logistic，分别搭配原金额与保留金额 | 复杂概率和金额部分各自贡献多少？ |
| 5. 稳定性补充验证 | 月份×方向、排除单日/连续7日的评分影响、配对时间块区间；覆盖最终候选及所有拟主张正向增量的分支 | 改善是否依赖少数日期，证据有多强？ |

最终采用的必要条件是**月等权RMSE低于H0，MAE不高于H0**；只超过Base不足以替换现有模型。分方向还须检查全部MAE、普通订单MAE、高额RMSE、累计低估和绝对总金额偏差，最终组合再次核查整体指标。部分指标最多比Base恶化2%是已确认的研究工作容忍度，不是官方或商业规则；准确适用范围及尾部支持度见实施方案分支3。

第五组主设置为7日时间块、2,000次配对重采样，并固定检查3日/14日块。若点估计与风险条件通过，但95%改善区间仍跨0，可以保留候选，须写明“增益尚不确定”。不因结果不理想扩大参数范围，也不只保留有利分支。

**开发阶段完成后的交付是研究结果、候选或无合格候选结论及复现来源，然后停止。** 正式模型冻结、包括仅读4–6月的冻结参数部署兼容性回放，以及7月读取、拟合和评价，均属于[后期独立阶段](docs/plans/2026-10-12-two-stage-post-development.md)。该阶段以后单独启动，分别比较按日更新和无新标签的冻结参数模式，不根据7月成绩重新选模型。

## 当前数据与评价协议

日期统一为UTC，当前唯一开发评价协议为 [validation_protocol_v2.json](config/validation_protocol_v2.json)。4–6月是报告窗口，**不是随机交叉验证，也不是三个独立留出集**。

| 开发窗口 | Base共同评分订单 | 有评分订单的日期 |
| --- | ---: | ---: |
| 2026年4月 | 854 | 29 |
| 2026年5月 | 726 | 24 |
| 2026年6月 | 1,373 | 19 |
| 合计 | 2,953 | 72 |

- 预测日D只用创建于 `[D−30天,D)`、且 `closedAt < D` 的历史训练；前月已结算订单可以进入后续日期训练，当天新结算结果不用于当天更新。
- 评分期为 `[2026-04-01, 2026-07-01)`，评分标签要求7月1日前已知。风险口径共4,074笔，缺少Base的1,121笔不插补加入含Base模型比较；高风险分类可使用风险口径，金额叠加仍在相同共同样本比较。
- `macro RMSE = sqrt(mean(各月MSE))`，`macro MAE = mean(各月MAE)`；另报全部订单合并指标。不能把RMSE直接跨月取均值，或混用月等权与订单等权成绩。
- 高额组依据预测日前历史q90阈值事后划分，仅用于评价；不能将实际赔付、实际结束时间、实际持有期或未来行情作为定价输入。合法历史赔付可作为监督目标，结算时间用于确认标签当时是否可得。
- 4–6月已经被多轮查看，属于开发期。7月曾查看描述统计，保留为内部后期测试，不能称为完全未接触的官方盲测。

底层固定存储仍由 [preprocessing_v1.json](config/preprocessing_v1.json) 定义。v2只将原train/validation联集赋予滚动角色，不改变特征、ID、数据哈希或原先的排除规则：

| v1存储分区 | 创建时间（UTC，左闭右开） | 风险样本 | Base共同样本 |
| --- | --- | ---: | ---: |
| train | 2026-02-01 至 2026-06-01 | 2,253 | 2,020 |
| validation | 2026-06-01 至 2026-07-01 | 2,261 | 1,373 |
| test | 2026-07-01 至 2026-07-31 17:11:48 | 2,541 | 2,541 |

各分区还要求结算严格早于分区结束。原先24笔跨边界记录不补回；正常提前结算保留，缺失赔付不填零，缺失Base不插补。原六月实验的1,370笔评分与当前六月1,373笔不同，历史成绩不能直接混排，详见[滚动评价说明](docs/rolling_validation.md)。

## 环境准备与操作入口

所有命令在本 README 所在的项目根目录运行。基础和模型环境使用 Python 3.10+；已有 `.venv` 可直接复用。

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-models.txt
```

`requirements-models.txt`已包含基础依赖；仅做审计和预处理可只安装 `requirements.txt`。绘图另用 `requirements-plots.txt`，不需要为模型实验安装绘图库。

### 新环境的数据初始化

提供方原始资料和数据不随仓库发布。默认输入目录相对于项目根目录为 `../docs/Jasper Vault Options Pricing Optimisation(Included Base Premium)/`，包括 `pp_transactions_training.csv`、`btc_market_history.csv` 及可用的提供方元数据。自定义位置可在下列两个入口使用 `--data-dir`，输出可用 `--output-dir` 指定。

```sh
.venv/bin/python -m pp_audit
.venv/bin/python -m pp_data
```

审计只保存关键统计，预处理是唯一的模型数据生成入口，不接受随机比例或临时改切分日期。**这两个初始化命令读取原始全量输入，预处理会生成包括test在内的三个存储分区；它们不属于仅访问开发期的模型流程。** 已有校验通过的产物时直接复用，无需每次研究重建。当前滚动入口固定读取 `outputs/preprocessing/`，完整运行还读取默认输入目录的行情；它尚未提供自定义数据路径参数。

### 查看范围与检查接口

```sh
# 只显示接口帮助，不读取研究数据或启动训练
.venv/bin/python -m pp_models --help
.venv/bin/python -m pp_models.rolling_study --help

# 已有预处理产物时：核对4–6月范围与七个既有方法，不训练、不写结果
.venv/bin/python -m pp_models.rolling_study plan

# 只查看高风险分类设置，不启动训练
.venv/bin/python -m pp_models.high_risk_study plan
```

`rolling_study plan`仍会读取并校验开发期产物。它描述的是**已经实现的七方法校准实验**，不是后续五分支计划；后者目前只有文档，还没有可运行入口。

### 明确开展既有实验时

以下命令会实际运行既有研究，仅供以后明确复现时使用；每次使用新的run-id，不覆盖已完成结果。

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pp_models.rolling_study run --run-id rolling_reproduction_NEW_ID
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.verify_rolling_study --run-dir outputs/model_studies/rolling_reproduction_NEW_ID
```

`run`执行七个固定方法，没有参数搜索；核验脚本独立重建历史与模型计算，写入紧凑的核验结果。完成一次运行不会自动执行后续研究或7月测试。[高风险特征说明](docs/high_risk_features.md)和[分类训练说明](docs/high_risk_training.md)提供各自入口与输出。

### 人工样本测试

```sh
.venv/bin/python -m unittest discover -s tests -v
```

测试不依赖提供方数据。2026-10-12提交前全套189项通过；这是当时版本的核验记录，不代表尚未实现的五个分支已经完成。

## 历史研究入口

以下入口保留用于追溯。除明确标为4–6月的当前实验外，早期模型主要在六月研究范围运行，不能以其默认设置代替当前协议。详细参数、结果、来源与限制集中在[模型使用说明](docs/model_training.md)及[研究决策记录](docs/research_decisions.md)。

| 研究 | 模块或命令入口 | 范围/用途 |
| --- | --- | --- |
| 原三模型与均值对照 | `pp_models study-plan` / `study-run --models gam --run-id ID` | 旧六月调参/后段确认；省略`--models`会选择全部三个模型 |
| 多窗口方向GAM | `pp_models.directional_study` | 六月方向波动对照 |
| 价格动态与平台频率 | `pp_models.market_activity_study` | 六月顺序特征比较 |
| 官方/方向×期限基础比较 | `pp_models.anchor_study` | 六月基础值与乘数比较 |
| 15分钟、5分钟和受约束响应 | `pp_models.fast_response_study`、`accelerated_response_study`、`guarded_response_study` | 六月快速加价与恢复机制 |
| 终点分布、业务约束 | `pp_models.terminal_distribution_study`、`constrained_study` | 六月分布代理与有限修正 |
| 高额赔付关联、互补信号 | `scripts.payout_signal_analysis`、`scripts.complementary_signals` | 历史诊断；外部输入获取另见使用说明 |
| 稀疏冲击叠加 | `pp_models.sparse_shock_study` | 4–6月，共同评分名单与滚动校准一致 |
| 高风险数据及分类 | `pp_models.high_risk_data`、`pp_models.high_risk_study` | 4–6月，特征/标签审计与q97.5/q95分类对照 |

`pp_models.*_study`独立模块使用 `python -m 模块 plan` 或 `run --run-id ID`；旧三模型使用上表所列子命令，`high_risk_data`使用 `plan/audit`。各入口参数以 `--help`为准，不能假设所有入口都支持续跑或自定义路径。旧 `pp_models check/train`仍保留，但不是本轮五分支或独立后期方案的替代入口。

已有绘图脚本包括 [官方Base日总赔付](scripts/plot_official_base_payout.py)与[方向/期限分组图](scripts/plot_official_base_payout_stratified.py)。它们依赖指定历史运行及本地产物，默认针对旧六月实验，不是新的4–6月通用结果图。

## 项目结构与输出约定

```text
pp_audit/                  订单、行情、时间对齐与关键统计
pp_data/                   固定预处理、特征白名单、标签、元数据与加载校验
pp_models/
  calibration_models.py    当前七方法及原两阶段H0
  rolling_protocol.py      统一4–6月逐日前向角色
  rolling_study.py         当前滚动校准入口
  high_risk_*.py           高风险特征、标签、分类与评价
  sparse_shock*.py         稀疏冲击状态及叠加实验
  estimators.py / study.py  原三模型框架及旧六月研究
  *_study.py               其他历史实验的独立入口
config/                    预处理、v2日期协议及各已实现实验的固定配置
scripts/                   指标独立核验、信号诊断、数据获取与绘图
requirements*.txt          基础、模型与可选绘图依赖
tests/                     人工样本、时间边界、回退和指标核验
docs/
  research_document_index.md  工作交接入口
  deliverables/            随仓库发布的两份正式Word副本
  research_decisions.md    证据、处理方案与历史决策
  research_plan_two_stage.md  两阶段研究总路线
  plans/                   前五分支、后期独立阶段及历史实现方案
outputs/                   本地生成产物，不随Git发布
  data_audit/               orders / market / alignment关键统计
  preprocessing/           固定三分区、reports与manifest.json
  model_studies/<run-id>/   metrics / provenance / reports；部分旧研究另有trials
  model_runs/<run-id>/      原单配置训练入口的metrics / provenance
  feature_engineering/     高风险特征审计与标签支持度
  diagnostics/             信号分析、关键统计及figures
  external_data/           历史互补信号的独立外部输入
  documents/               文档生成、预览与原交付产物
```

审计仅导出聚合统计；统一预处理是需要保存逐笔特征、标签和元数据的基础产物。模型研究默认仅保存关键指标、配置/源码/数据来源、回退摘要和必要报告，不重复导出逐笔预测、数据集或大量模型副本。后期冻结模型的必要产物按独立方案另行处理。

`outputs/preprocessing/datasets/<split>/`中，`features.csv`为定价时输入，`targets.csv`为监督目标，`metadata.csv`为时间、实际保费、样本标记等辅助信息；按`order_id`对齐，不能将metadata整体拼入模型。14项基础特征包含一个可选Base；方向和其他探索特征由相应研究模块构建，不改写基础数据。

行情特征仅使用订单创建前已完成的连续分钟K线，历史变换只用合法训练样本拟合。加载器核对来源、代码、配置和产物哈希；不关闭校验以迁就不匹配的输出，也不自动改变样本名单。审计`key_metrics.csv`中的空值表示不可计算或不适用，不能当零；各指标定义随表保存。

## 资料与结论边界

Base来自提供方匹配结果，缺少逐笔匹配时刻，可能存在报价前后120秒内匹配带来的时点差异；代码对市场特征的时间约束不能消除该输入局限。官方留出订单及其未知赔付没有加入当前研究。

目前缺少完整报价曝光、未购买记录、需求弹性、对冲与手续费成本、正式评分规则和标签回传安排。原始订单还经过提供方负markup过滤。因此现阶段只能讨论已成交样本上的赔付估计与风险取舍，不能声称找到了最优保费、识别了客户行为因果关系或提高了真实利润。

仓库发布程序、配置、人工测试和研究文档；提供方原始资料、数据、`outputs/`、本地虚拟环境与凭据均不随Git发布。取得仓库不等于已具备复现实验的输入与历史输出，接手时应按文档索引核对已授权数据和对应来源版本。
