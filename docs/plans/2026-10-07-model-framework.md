# 三类赔付模型与统一训练入口 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 GAM、EBM、浅层梯度提升树和统一逐日训练入口，本次不调用真实估计器 fit，不运行项目训练。

**Architecture:** 新增独立 `pp_models` 包，沿用 `pp_data.load_split` 和固定 v1 数据；模型工厂、逐日历史选择、统计、产物和命令入口各自独立。默认只显示帮助，`check` 只检查，只有 `train` 会训练；未来仅保存关键汇总与来源清单，不默认保存逐笔预测和每日模型。

**Tech Stack:** Python、pandas/numpy、pyGAM LinearGAM、InterpretML EBM、scikit-learn HistGradientBoostingRegressor；模型依赖独立列入 requirements-models.txt。

**Spec:** `../research_decisions.md` 问题 10、11及 `../benchmark_design.md`；用户最新指令明确授权记录并实现三模型，但暂不训练。

## Global Constraints

- 三模型指 GAM、EBM、浅层梯度提升树；B0/B1 的既有设计保留，本次不擅自增加两步模型、风险余量或需求模型。
- 每 BTC 实际赔付为标签；保留真实零值和大额值；输出预测非负化与原始预测分开计数。
- 默认过去 30 天创建且在预测日 UTC 零点之前结算；当天只训练一次，订单特征仍按下单时点取值。
- 固定 train / validation / test 日期与 v1 样本名单保持不变。
- 三模型共用同一候选特征、训练历史与评分订单，默认 risk、不含基础价；共同样本可显式启用基础价。
- 内置随机早停关闭，固定迭代数；配置是起始值，不是调优结论。
- 7 月运行必须引用同规则、同代码与同数据的已完成 6 月运行，防止无记录更换配置。
- 本次所有测试只检查构造、边界与调度，用显式人工替身模拟估计器；不调用任何真实模型 fit（包括合成数据）。

## Review Focus

1. 窗口起点、日零点结算、当天结算、未来结果只能影响合法后续日。
2. 预测时的列顺序、禁止字段、缺失/无穷值与输入标签错位应明确拒绝。
3. 不含基础价与含基础价的训练池一致性、共同样本指标口径。
4. 无参数 / check 不训练、不创建输出；已有 run-id 不覆盖，失败运行不能用于测试期锁定。
5. 常量特征、单一方向、全零赔付与小样本的明确行为；本次不声称真实拟合已验证。

## Task 1: 配置、模型接口与只构造检查

**Files:** config/models_v1.json; pp_models/config.py; pp_models/estimators.py; requirements-models.txt; tests/test_models.py.

**Interfaces:** `load_model_config(path=None) -> dict`; `resolve_config(config, cohort, use_base, models) -> dict`; `build_model(name, resolved) -> PayoutRegressor`，提供 `fit(X,y)`、`predict_raw(X)`、`predict(X)`。固定列校验；未拟合拒绝预测。

- [x] 写缺失接口、禁止字段、顺序错位、非有限值、负预测修正的失败测试。
- [x] 运行并观察失败；实现延迟导入和三个真实模型的构造。GAM 连续变量为少量惩罚样条，方向用线性 0/1 项以支持单一方向窗口；EBM 显式两对交互；树深最多 3。
- [x] 运行构造与行为测试，始终禁止真实 fit；记录依赖版本。

## Task 2: 逐日调度与关键统计

**Files:** pp_models/data.py; pp_models/evaluation.py; tests/test_model_evaluation.py.

**Interfaces:** `load_pool(split, resolved, data_dir=None) -> ModelSplit`; `daily_batches(pool, split, window_days)` 只提供合法 train IDs / prediction IDs；`evaluate(pool, split, resolved, model_factory=build_model) -> (summary, daily)`。

- [x] 人工边界数据证明窗口左闭右开、结算严格早于预测日；未来标签改变不改变过去预测；日内不更新。
- [x] 实现逐日重新构造模型、训练集转换与预测；最少 30 条历史，不足时报错，不暗中扩大窗口或换模型。
- [x] 手算核对 RMSE、MAE、按数量计算的总赔付偏差、零分母；共同子集路径经代码审查，仅在内存持有逐笔预测。

## Task 3: 运行入口、来源与文档

**Files:** pp_models/__init__.py; pp_models/__main__.py; pp_models/runs.py; tests/test_model_cli.py; README.md; docs/model_training.md; docs/research_decisions.md.

**Interfaces:** `main(argv=None) -> int`；`check` 校验配置、模型构造、数据与窗口覆盖，输出 JSON；`train --split validation|test --run-id ID` 才执行训练；test 必须 `--validation-run PATH`。

- [x] CLI 无参数 / check 禁止真实 fit、不写训练目录；运行重复 ID、无验证来源、来源篡改应拒绝。
- [x] 实现 `outputs/model_runs/<run-id>/{metrics/{summary,daily}.csv,provenance/{config,manifest}.json}`，产物带配置/代码/数据/依赖哈希来源，最后发布清单。
- [x] 更新说明和研究记录，区分“已实现、未训练”与“已验证效果”。
- [x] 全套 unittest、三模型只构造检查、真实数据 check；核实既有预处理哈希未变，未生成模型训练产物。

## 执行记录与裁定

- 用户已明确要求实施，直接在本会话执行，不重复申请方案批准。
- 当前目录不是 Git 仓库，无法使用依赖 Git 的工作树、提交和技能记账脚本；在本计划记录执行结果，不初始化 Git。
- 本次不进行真实 fit。依赖加载与构造可以检查，但数值拟合、收敛和预测效果留待用户授权训练后验证；测试替身只证明调度协议。
- 初始特征选方向、剩余期限、moneyness、市场价格、60m/1440m 波动、60m 收益。先避免同时输入所有高度相关窗口，未来修改需配置版本和验证记录。
- 接口依赖：Task 2 使用 Task 1 的 resolved 配置与预测适配器；Task 3 使用 Task 2 的加载、覆盖与统计接口；无共享状态修改。

## 验证记录

- Task 1：四项测试先因缺失模型包失败，实现后四项通过；真实估计器的 fit 在构造测试中被设为一旦调用即报错。
- Task 2：四项测试先因缺失评价模块失败，实现后八项模型相关测试通过；用人工均值探针检查未来标签隔离，未拟合真实模型。
- Task 3：四项入口/存储测试先失败，实现后全套 43 项测试通过（31 项原有、12 项新增）。
- `check` 在真实数据上确认：risk 验证 2,261 笔 / 30 日，历史池最少 935；benchmark 含 Base 验证 1,373 笔 / 19 日，历史池最少 382。均返回 `training_performed: false`；没有计算模型成绩。
- 模型依赖安装在项目独立 `.venv`：pyGAM 0.12.0、interpret-core 0.7.8、scikit-learn 1.7.2，使用 scipy 1.16.3。没有更改已有数据处理依赖或预处理源代码。
- 按 executing-plans / requesting-code-review 要求由独立审查代理复核；审查者再次运行 43 项测试，未发现需修复的问题，也未调用真实拟合。数值收敛及已拟合模型行为仍按用户要求留待训练阶段验证。
- 未采用的额外功能：B0/B1 估计器、风险余量、需求模型、模型持久化和解释导出均不属于本次三模型实现；没有遗留的审查缺陷。
