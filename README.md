# PP 定价研究：数据处理与模型框架

当前已实现数据审计、统一预处理，以及 GAM、EBM、浅层梯度提升树的统一训练框架。**2026-10-09 已完成4–6月滚动验证及高风险GBDT分类器的q97.5/q95对照；7月保留，EBM和原浅层GBDT回归模型尚未训练。** 审计仍只输出关键统计量；预处理生成固定分割数据、标签和来源清单；模型运行默认只保存关键指标和来源，不导出逐笔预测、模型文件或模拟收益报告。原始文件保留在 `../docs/`，程序不会修改它们。

研究中各问题的证据、已确定原则与待讨论方案统一记录在 [研究问题与处理方案](docs/research_decisions.md)。

**工作交接从[研究文档专用索引](docs/research_document_index.md)开始**：汇集研究报告大纲Word、未来研究计划Word和具体实施方案，注明版本差异、研究状态、代码入口与执行边界。前五组研究完成后必须汇报并停止；正式模型冻结、部署兼容性回放和7月后期测试须由用户另行启动。

2026-10-12研究主线已收敛为[两阶段实际赔付校准的完整计划](docs/research_plan_two_stage.md)：固定赔付概率，有限改进条件金额，检验向Base收缩及分方向回退，冻结后再做内部后期验证。该计划尚未执行新增训练，7月继续保留。

当前模型入口见 [模型使用说明](docs/model_training.md)，**当前唯一滚动评价协议**见 [4–6 月滚动验证](docs/rolling_validation.md)与 [v2 日期配置](config/validation_protocol_v2.json)。每日使用此前 30 天已结算订单，三个月共 2,953 笔共同样本、72 个评分日；底层 v1 存储和历史结果保留。

两步模型三个单月的 RMSE/MAE 均优于官方原值，但高额 Call 误差加大，尚不能升级主模型。结果见[本轮评价报告](outputs/model_studies/rolling_calibration_apr_jun_20261009/reports/evaluation.md)。旧 [GAM 训练计划](docs/validation_training_plan.md)和六月入口仅供历史复现，不代表当前评分范围。

此前已完成稀疏冲击机制叠加：分别在官方 Base 与两步模型上增加高门槛、5 分钟更新、快速衰减的倍率。只实际调整约 2% 的订单，但两组月等权 RMSE 均略恶化，当前不升级主模型。[结果与核验](outputs/model_studies/sparse_shock_apr_jun_20261009/reports/evaluation.md)；独立入口为 `.venv/bin/python -m pp_models.sparse_shock_study plan` / `run --run-id ID`，仍使用同一 v2 验证协议。

## 仓库范围

仓库包含程序、配置、人工测试与研究文档，两份正式Word交付副本位于 `docs/deliverables/`。提供方的原始资料和数据不随仓库发布；`outputs/` 下的生成数据与统计、本地虚拟环境和凭据文件均由 `.gitignore` 排除。文档中指向 `../docs/` 或 `../../docs/` 的提供方资料链接依赖原本的本地目录，不属于仓库文件。

新环境需自行准备已获授权的输入数据：可放在默认的相邻 `docs/Jasper Vault Options Pricing Optimisation(Included Base Premium)/` 目录，或在审计与预处理入口通过 `--data-dir` 指定。先生成预处理产物，再检查模型计划；仓库中的测试使用人工样本，不依赖提供方数据。

## 运行

高风险分类的数据准备入口：`.venv/bin/python -m pp_models.high_risk_data plan`，以及 `audit --run-id ID`。当前提供32项创建时特征、历史97.5分位主标签及95分位/总赔付辅助标签，只做数据审计，不启动分类器训练或抬价回测。[特征、标签与入口说明](docs/high_risk_features.md)。

高风险分类训练入口：`.venv/bin/python -m pp_models.high_risk_study plan` / `run --run-id ID`。已完成两组标签及固定抬价对照：q97.5在20%概率门槛下未报警，q95报警78笔、命中8笔；金额误差略升，暂不推广。[训练说明及报告](docs/high_risk_training.md)。

需要 Python 3.10+，依赖见 `requirements.txt`。

```sh
python -m pip install -r requirements.txt
python -m pp_audit
python -m pp_data
```

运行位置为本目录。两个入口均接受 `--data-dir`、`--output-dir`。`pp_audit --cutoff` 只改变审计查询时间，不改变任何模型数据分割；模型预处理只有 `python -m pp_data` 一个入口，使用下方唯一配置，不接受比例、随机种子或切分日期参数。时间均按 UTC 解析。

模型依赖位于 `requirements-models.txt`，建议在项目 `.venv` 中安装。安全检查为 `.venv/bin/python -m pp_models check`；无参数只显示帮助，不训练、不写文件。未来明确执行 `.venv/bin/python -m pp_models train --split validation --run-id validation_v1` 才会启动训练。本轮未执行该训练命令，完整测试使用 `.venv/bin/python -m unittest discover -s tests -v`。

历史六月三模型入口为 `.venv/bin/python -m pp_models study-plan`：只列出调参/确认范围、候选与资源预算，不训练。将来显式使用 `study-run --run-id ID` 才会执行完整验证期研究，`--resume` 支持中断续跑。该流程没有测试集参数，不访问测试集 CSV；旧单配置 `check/train` 入口仍保留，但本阶段不调用其测试期路径。

只运行 GAM 时使用 `.venv/bin/python -m pp_models study-run --models gam --run-id ID`；只检查对应计划使用 `study-plan --models gam`。两种均值对照始终保留。省略 `--models` 会选择全部三个模型；续跑必须保留原模型选择，不允许在同一运行中追加其他模型。已完成运行 `gam_validation_20261008` 的[本地评价报告](outputs/model_studies/gam_validation_20261008/reports/evaluation.md)随其他输出留在本地，不纳入 Git。

多窗口与方向 GAM 使用独立入口 `.venv/bin/python -m pp_models.directional_study plan` / `run --run-id ID`，不改变原预处理。已完成 `gam_directional_20261008`：三个新版本后段 RMSE 为 382.09 / 383.87 / 383.77，均未优于原 GAM 380.40 或 B1 374.08；方向金额校准有部分改善。保留原对照，不加保守余量，不读 7 月测试集。六月后段已被分析过，因此新结果属于后续探索，详见[评价报告](outputs/model_studies/gam_directional_20261008/reports/evaluation.md)与[研究记录第 15 节](docs/research_decisions.md#15-多窗口与方向-gam-重跑结果2026-10-08)。

官方 / 方向×期限基础比较使用 `.venv/bin/python -m pp_models.anchor_study plan` / `run --run-id ID`。已完成 `anchor_comparison_20261008`：相同 benchmark 历史与评分样本下，后段原官方 / 分组原值 / 分组统一系数 RMSE 为 380.34 / 372.71 / 371.12；前段原官方略好，分组后段优势集中在大赔付日。四项动态修正均按事前规则回退统一系数，尚未证明稳定增益。详见[评价报告](outputs/model_studies/anchor_comparison_20261008/reports/evaluation.md)及[研究记录第 20 节](docs/research_decisions.md#20-官方与方向期限基础的共同样本实验2026-10-08)。

官方基础值与实际赔付的每日折线图由 [绘图脚本](scripts/plot_official_base_payout.py) 生成。在安装了 `requirements-plots.txt` 的独立绘图环境中运行 `python scripts/plot_official_base_payout.py`。脚本校验既有研究的统计文件并核对日总额，只读汇总、不训练模型；PNG/SVG 保存至 `outputs/diagnostics/official_base_payout_20261008/figures/`，来源清单保存至 `provenance/`。

每 BTC、方向与期限分组图使用 `python scripts/plot_official_base_payout_stratified.py`。该脚本只读取原 validation 三表和已保存的日统计，核对原 1,370 笔评分 ID 与金额；不训练、不读取七月。每组按金额合计除以 BTC 数量合计，提供 Call/Put 总览和固定标称期限的分面图；小于 5 笔用空心点、缺失日期留空。PNG/SVG、分组关键统计、来源清单分别位于 `outputs/diagnostics/official_base_payout_stratified_20261008/{figures,metrics,provenance}/`。

## 当前滚动验证入口

```sh
.venv/bin/python -m pp_models.rolling_study plan
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pp_models.rolling_study run --run-id ID
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.verify_rolling_study --run-dir outputs/model_studies/ID
```

`plan` 只核对范围；`run` 执行七个固定方法，无参数搜索。4/5/6 月分别评分 854/726/1,373 笔，每日训练仅用 `[D−30d,D)` 创建且 D 前已结算记录，7 月不读取、不评分。新入口将 v1 的 train/validation 联集赋予 v2 滚动角色，沿用原有有效样本名单，前月已结算订单可以进入之后日期的训练。三个月为历史回放，非全新独立留出数据；[完整规则与限制](docs/rolling_validation.md)。

## v1 固定存储分割（历史复现）

配置文件：[config/preprocessing_v1.json](config/preprocessing_v1.json)。区间均左闭右开，同时要求订单结算时间严格早于所属组结束时间。

| 分割 | 创建时间范围（UTC） | 风险研究样本 | 官方基础价共同样本 |
| --- | --- | ---: | ---: |
| train | 2026-02-01 ≤ t < 2026-06-01 | 2,253 | 2,020 |
| validation | 2026-06-01 ≤ t < 2026-07-01 | 2,261 | 1,373 |
| test | 2026-07-01 ≤ t < 2026-07-31 17:11:48 | 2,541 | 2,541 |

- 最后时刻取官方留出数据开始时间；若输入元数据与该固定时刻冲突，程序报错，不自动挪动边界。当前不读取或合并官方验证订单文件。
- 原始 7,204 条中，125 条未通过共享技术条件，24 条结算时间跨越或恰好落在所属组结束点，最终风险样本 7,055 条，共同样本 5,934 条。各排除原因按顺序互斥计数。
- 正常提前结算保留；跨边界记录不移入另一组。赔付缺失不填零，基础价缺失不插补。
- 共同样本是风险样本中的固定子集，保持相同 ID、所属组和次序；缺失基础价的有效订单仅用于不含基础价的风险研究。共同样本的实际验证覆盖从 6 月 12 日开始，训练覆盖至 5 月 28 日，不能理解为完整月份均匀覆盖，详见汇总。
- 7 月已查看过描述统计，称为**内部后期测试集**，不能称为完全未接触的官方盲测。用户后来确认采用按天更新：train 提供启动历史，validation 选择算法及窗口 / 平滑等超参数，test 按冻结规则逐日评价。测试期已结算的订单可用于之后日期的模型更新，但不能影响已经产生的预测，也不能依据测试结果重新选择模型、窗口或余量策略。
- 当前产物保留固定日期分区和样本名单。每天的训练池需在这些数据中按创建时间窗口与 `closedAt < 预测日UTC零点` 再筛选；预处理与加载接口本身不执行滚动训练。v1 排除的 24 条跨边界记录不自动补回。不要将完整验证 / 测试数据一次性合并后训练再回测。

## 底层数据读取与定价时特征

```python
from pp_data import load_split

train = load_split("train")                 # 共同样本，含官方基础价
validation = load_split("validation")
# 以上为日期分区，不是每天的最终训练池。
# 逐日基准还须按预测日筛选历史窗口及已经结算的标签。
# 这是 v1 存储接口；当前 4–6 月评价由 rolling_protocol 统一生成。
# 当前仍保留 7 月，不执行测试期评价。

risk_train = load_split("train", cohort="risk")  # 更大样本，特征不含基础价
ablation = load_split("train", use_base=False)   # 保持共同样本，移除基础价特征
```

`X` 仅包含白名单数值特征；`y` 是每 BTC 实际赔付；`metadata` 独立保留时间、历史成交保费和质量标记，**不能拼进模型输入**。三者以 `order_id` 对齐。ID 使用原始数据行序，来源文件校验值记录在清单中。

基础特征为方向、产品期限、合同剩余期限、数量、执行价、最近已结束行情价格、`log(行情价 / 执行价)`，以及过去 60、240、1440 分钟的对数收益和实现波动。波动定义为窗口内一分钟对数收益平方和的平方根，不年化。14 个可用特征中包含 1 个可选 API Base；风险口径使用其余 13 个。

N 分钟收益及波动要求 N+1 根完整、有效、连续 K 线，最近一根结束时间不晚于下单且距离不超过 60 秒。24 小时特征因此要求 1,441 根 K 线，略严于原审计的 1,440 根覆盖统计；当前所有技术候选均满足要求。没有全样本缩放、插补、分位数或风险阈值拟合；以后若模型需要学习数据变换，只能在该预测日合法的历史训练池上拟合，其规则在验证期确定。

重新运行入口会重建相同产物。加载时验证全部输出哈希、生成代码和配置，并解析校验过的同一份内存内容；若代码或配置变化、产物损坏或混入不同版本，必须重新生成。清单最后写入，未完整发布的一批文件不能通过校验。输出必须使用专用目录，不允许与源目录重叠或覆盖陌生文件。预处理可以输出空分组供检查，但模型加载时会用明确的 `ValueError` 拒绝没有有效样本的分割或口径。

## 结构

```text
pp_audit/                       审计程序
  common.py                     字段定义、解析、指标格式
  orders.py                     订单质量、样本数量和分组统计
  market.py                     行情质量和可用分钟
  alignment.py                  订单与已结束行情对齐、时间切分
  __main__.py                   运行入口和分类输出
pp_data/                        唯一模型数据处理与加载入口
  config.py                     读取并校验固定配置
  features.py                   定价时可见特征白名单
  pipeline.py                   共享筛选、固定分割、标签与元数据
  io.py                         生成清单、文件校验、load_split
  __main__.py                   python -m pp_data
pp_models/                      三模型与共同逐日前向框架
  config.py                     特征/超参数校验与口径解析
  estimators.py                 GAM / EBM / 浅层树统一接口
  data.py                       固定数据加载与每日可用历史
  evaluation.py                 每日拟合、预测和关键统计
  runs.py                       唯一运行目录、来源和验证引用
  __main__.py                   check（不训练）/ train（显式训练）
  scoped_data.py                只读取并校验 train/validation 六表
  baselines.py                  B0 整体均值 / B1 平滑分组均值
  study_config.py               有限搜索与验证阶段定义
  study.py                      预算、顺序调参、冻结确认、汇总
  study_storage.py              候选记录、校验续跑、进程锁
  directional_features.py       多窗口方向波动，核对原样本和时点
  directional_gam.py            四个 GAM 结构与中心化方向差异
  directional_study.py          plan / run：仅六月后续探索
  market_activity_features.py   六项价格动态＋四项平台创建频率
  market_activity_gam.py        D→E→F 顺序特征对照
  market_activity_study.py      plan / run：固定样本、仅六月顺序实验
  anchor_models.py              官方 / 方向×期限基础及统一 / 四项动态乘数
  anchor_study.py               plan / run：共同样本、仅六月基础比较
config/preprocessing_v1.json    唯一时间边界与处理版本
config/models_v1.json           初始模型配置，尚未调优
config/validation_study_v1.json  每模型 5–6 套候选与确认规则
config/gam_directional_v1.json   四种结构的有限平滑搜索与固定参数对照
config/gam_market_activity_v1.json 价格动态与平台频率的顺序实验
config/anchor_comparison_v1.json 两种基础的相同有限修正搜索
requirements-models.txt         单独的模型依赖
tests/                          人工构造的小样本验证
docs/research_decisions.md       研究问题、处理方案与决策状态
docs/benchmark_design.md        初版训练目标、简单基准和逐日前向协议
docs/model_training.md          模型运行、输出分类与不训练边界
docs/validation_training_plan.md 三模型训练/调参/评估与算力估算
docs/plans/                     实现计划与验证记录
outputs/model_runs/<run-id>/    未来训练后才创建
  metrics/{summary,daily}.csv   关键指标，不保存逐笔预测
  provenance/                  配置、来源与产物哈希
outputs/model_studies/<run-id>/ 未来验证研究运行后才创建
  trials/                      各候选关键指标与校验记录
  metrics/                     调参榜、确认期对照
  provenance/                  阶段覆盖、锁定参数与执行状态
outputs/training_plans/         只读规划产生的关键数量与资源估算
outputs/data_audit/
  orders/key_metrics.csv        订单关键指标
  orders/segment_statistics.csv 按月份、方向、期限的统计
  market/key_metrics.csv        行情关键指标
  alignment/key_metrics.csv     行情覆盖、验证资料和匹配限制
  alignment/cutoff_statistics.csv 时间切分时已知结果的样本数量
outputs/preprocessing/
  datasets/{train,validation,test}/
    features.csv                预测特征（含order_id；基础价允许为空）
    targets.csv                 总赔付、每BTC赔付、是否正赔付
    metadata.csv                时间、来源行号、共同样本标记、实际保费
  reports/split_statistics.csv  六组样本数量、覆盖和时间范围
  reports/exclusion_statistics.csv 互斥排除原因及保留数量
  manifest.json                 来源、配置、代码、环境与产物校验值
```

`key_metrics.csv` 使用统一列：`metric`（稳定标识）、`value`、`unit`、`denominator`、`rate`、`definition`。`rate` 是 0–1 比例，仅在适用时填写；空值表示不可计算或不适用，不能当成零。统计口径写在 `definition` 中。

## 样本口径

- **有结果记录**：赔付字段为有限非负数，保费和数量有效。保费与赔付总额始终使用同一批记录；缺失赔付不填零。这些描述统计仍可能包含时间异常，不能称为模型收益。
- **技术条件候选样本**：输入完整有效、数量与执行价为正、方向编码与文字一致、到期晚于创建、结算不早于创建、结果有效、单位保费计算一致；完全重复记录只保留首条。有效期限是正整数秒，不强制限定为当前七种产品。正常的提前结算不排除。
- **基础价共同候选样本**：在上一口径上，要求基础价为正、markup 有效且非负、价格差额恒等式成立。
- 以上是共享审计候选口径；预处理在此基础上增加固定分割、标签可获知与完整行情条件后导出模型数据。**输出不代表 PP 业务语义已经获得官方确认**。基础价匹配可能使用未来报价，因缺少匹配时刻无法逐笔核验。
- 合同实际剩余期限与产品标称期限的偏差单独统计，绝对偏差超过 60 秒只标记，不自动删除；数秒到几十秒的延迟不直接判为错误。
- 行情按完整一分钟使用，只有结束时间不晚于创建时间的 K 线可以匹配。重复时刻的所有行情行均视为不明确，不任意挑选其中一行。
- 计算 60、240、1440 分钟覆盖时，要求最近 K 线不超过 60 秒、整个窗口每分钟连续且均通过行情质量检查。不能用零散分钟数量冒充连续覆盖。
- 时间切分只把 `createdAt < cutoff` 且 `closedAt < cutoff` 的有结果记录视为当时可获知；测试边界处才结算的记录保守排除。
- 赔付最高 1% 集中度使用有结果记录数量乘以 1% 后向上取整，按赔付金额排序；空样本或总赔付为零时该占比留空。

## 当前资料边界

API Base 的匹配时间、完整报价日志、未购买记录、对冲与手续费成本、正式评分规则未提供。原始数据已按提供方规则过滤负 markup。审计会保留这些限制及相关数量，不推断真实需求、正式模型收益或官方 baseline 得分。

## 15 分钟快速响应实验

`.venv/bin/python -m pp_models.fast_response_study plan|run --run-id ID`：在官方基础值上增加 Put 的非负快速响应，Call 保持原值。市场状态每 15 分钟更新，停止有效下跌后额外状态每段减半；幅度仍按日用已结算历史学习。只用原 train/validation 共同样本，与原值、每日统一系数比较，不读取七月测试集。[训练入口说明](docs/model_training.md#8-官方-base--15-分钟快速响应)与[冻结实验计划](docs/plans/2026-10-08-fast-response.md)。

5 分钟加速累积另有入口：`.venv/bin/python -m pp_models.accelerated_response_study plan|run --run-id ID`。比较连续下跌压力的线性与平方响应，含无收缩对照，当前倍率上限 4；仍只使用原 train/validation。[使用说明](docs/model_training.md#9-5-分钟线性与平方累积响应)。

当前平方响应的约束试验：`.venv/bin/python -m pp_models.guarded_response_study plan|run --run-id ID`，顺序测试低风险加价惩罚、每日幅度上调限制，并报告峰值欠估与普通时段高估的取舍。[使用说明](docs/model_training.md#10-低风险加价惩罚与每日上调限制)。

终点分布试验入口：`.venv/bin/python -m pp_models.terminal_distribution_study plan|run --run-id ID`。先比较静态/动态行情分布的区间质量，再以到期赔付代理修正官方 Base；只调整 <=8h Put，其余订单保持原值。固定小网格、原共同样本，仅六月。[使用说明](docs/model_training.md#11-终点收益分布初步实验)。

高额赔付信号探索：`.venv/bin/python -m scripts.payout_signal_analysis --run-id ID`。只读既有 train/validation，比较市场与平台交易活动、价格动态和合约控制量，输出相关性、条件尾部发生率及跨日期敏感性。结果位于 `outputs/diagnostics/ID/`，仅保存分类汇总。[分析口径](docs/plans/2026-10-09-payout-signals.md)。

业务约束试验：`.venv/bin/python -m pp_models.constrained_study plan|run --run-id ID`。比较合约状态、双侧尾部修正、强收缩订单流，Put/Call 独立校准；仍只用原共同样本及六月。当前增益有限、Call 高额单低估加重，不替换已有模型。[评价报告](outputs/model_studies/constrained_pricing_20261009_verified/reports/evaluation.md)与[训练说明](docs/model_training.md#13-合约状态双侧尾部与强收缩订单流)。

三类互补信号探索：`scripts.fetch_complementary_data` 获取官方公开的六月前盘口、OI/资金费率、DVOL 输入；`scripts.complementary_signals` 比较固定背景与三个信息组的逐日尾部概率探针，并核对高额赔付关联。独立输入位于 `outputs/external_data/`，研究统计位于 `outputs/diagnostics/`，没有改动定价模型。[分析报告](outputs/diagnostics/complementary_signals_20261009/reports/analysis.md)、[固定口径](docs/plans/2026-10-09-complementary-signals.md)与[命令说明](docs/model_training.md#14-三类互补信息探索)。
