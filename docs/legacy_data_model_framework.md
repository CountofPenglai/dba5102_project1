# 2026-10-07基础框架说明（历史版本）

此文保留初期入口与存储约定。当前模型、日期角色和结果以[主说明](../README.md)及[v3协议](validation_protocol_v3.md)为准；下文“未训练”“7月内部测试”仅描述当时状态。仓库内文档链接已按当前文件位置调整，命令和正文中的程序路径仍以仓库根目录为基准。

# PP 定价研究：数据处理与模型框架

当前已实现数据审计、统一预处理，以及 GAM、EBM、浅层梯度提升树的统一训练框架。**尚未启动真实模型训练。** 审计仍只输出关键统计量；预处理生成固定分割数据、标签和来源清单；未来模型运行也默认只保存关键指标和来源，不导出逐笔预测、模型文件或模拟收益报告。原始文件保留在 `../docs/`，程序不会修改它们。

研究中各问题的证据、已确定原则与待讨论方案统一记录在 [研究问题与处理方案](research_decisions.md)。

当前模型入口见 [模型使用说明](model_training.md)，当前研究优先使用 [仅验证期的训练与调参计划](validation_training_plan.md)。三模型与 B0/B1 均值对照均已实现，增加有限搜索、阶段确认和中断续跑；**尚无真实训练成绩**。本阶段只读取训练期与验证期 CSV。

## 仓库范围

仓库包含程序、配置、人工测试与研究文档。提供方的原始资料和数据不随仓库发布；`outputs/` 下的生成数据与统计、本地虚拟环境和凭据文件均由 `.gitignore` 排除。文档中指向 `../docs/` 或 `../../docs/` 的提供方资料链接依赖原本的本地目录，不属于仓库文件。

新环境需自行准备已获授权的输入数据：可放在默认的相邻 `docs/Jasper Vault Options Pricing Optimisation(Included Base Premium)/` 目录，或在审计与预处理入口通过 `--data-dir` 指定。先生成预处理产物，再检查模型计划；仓库中的测试使用人工样本，不依赖提供方数据。

## 运行

需要 Python 3.10+，依赖见 `requirements.txt`。

```sh
python -m pip install -r requirements.txt
python -m pp_audit
python -m pp_data
```

运行位置为本目录。两个入口均接受 `--data-dir`、`--output-dir`。`pp_audit --cutoff` 只改变审计查询时间，不改变任何模型数据分割；模型预处理只有 `python -m pp_data` 一个入口，使用下方唯一配置，不接受比例、随机种子或切分日期参数。时间均按 UTC 解析。

模型依赖位于 `requirements-models.txt`，建议在项目 `.venv` 中安装。安全检查为 `.venv/bin/python -m pp_models check`；无参数只显示帮助，不训练、不写文件。未来明确执行 `.venv/bin/python -m pp_models train --split validation --run-id validation_v1` 才会启动训练。本轮未执行该训练命令，完整测试使用 `.venv/bin/python -m unittest discover -s tests -v`。

本轮新增的推荐入口为 `.venv/bin/python -m pp_models study-plan`：只列出调参/确认范围、候选与资源预算，不训练。将来显式使用 `study-run --run-id ID` 才会执行完整验证期研究，`--resume` 支持中断续跑。该流程没有测试集参数，不访问测试集 CSV；旧单配置 `check/train` 入口仍保留，但本阶段不调用其测试期路径。

## 唯一训练、验证、测试分割

配置文件：[config/preprocessing_v1.json](../config/preprocessing_v1.json)。区间均左闭右开，同时要求订单结算时间严格早于所属组结束时间。

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

## 后续模型统一读取

```python
from pp_data import load_split

train = load_split("train")                 # 共同样本，含官方基础价
validation = load_split("validation")
# 以上为日期分区，不是每天的最终训练池。
# 逐日基准还须按预测日筛选历史窗口及已经结算的标签。
# 6月选择算法与超参数，7月按冻结规则逐日前向评价。

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
config/preprocessing_v1.json    唯一时间边界与处理版本
config/models_v1.json           初始模型配置，尚未调优
config/validation_study_v1.json  每模型 5–6 套候选与确认规则
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
