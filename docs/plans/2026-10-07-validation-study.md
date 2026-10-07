# Validation-only model study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 计划并实现三模型有限调参和确认流程，估算资源；本轮不训练、不读取真实测试集 CSV。

**Architecture:** 在 pp_models 内扩展 scoped_data、study_config、study、study_storage。复用模型适配器与评价器；评价器增加验证期子区间、可获知标签筛选和诊断统计。既有 pp_data 代码/哈希保持不动。

**Tech Stack:** 现有 Python/pandas/numpy/pyGAM/InterpretML/sklearn；threadpoolctl 已安装，用于限制拟合线程。

**Spec:** `../validation_training_plan.md` 与 `../../config/validation_study_v1.json`。

## Global Constraints

- 不调用真实模型 fit，包括合成数据；调度只用人工探针。
- 不打开 datasets/test 的任何真实文件；只验证 manifest 和 train/validation 六个表。
- 固定 UTC v1 全局分割、30 天已结算历史、7 个共同输入、风险口径无 Base。
- 6 月前 20 日调参、后 10 日确认，调参评分标签必须在确认开始前可知。
- 最多 5/6/6 候选，确认各一个，不根据确认成绩重新调参。
- 紧凑指标与可验证来源，既有目录拒绝覆盖；明确 resume 可复用完整候选。

## Review Focus

1. 旧 load_split 会读取所有分组做哈希；新流程不能因此碰到测试 CSV。
2. 调参日创建但跨阶段结算的标签不能影响选参，合法后续日仍可用于拟合。
3. 参数搜索非笛卡尔积，重复配置复用；确认数据不能进入选择规则。
4. 恢复时参数、程序、数据、依赖漂移及产物篡改应失败；失败候选不能算成功。
5. 结果以共同订单与同一历史评估，负预测与零分母明确；算力区间是估算而非实测。

## Task 1: 仅读取开发数据与评价子区间

**Files:** pp_models/scoped_data.py, data.py, evaluation.py, baselines.py; tests/test_validation_scope.py.

**Interfaces:** `load_validation_pool(resolved, dataset_dir)` 返回 ModelSplit；`evaluate(..., period=(start,end))` 仅评分该阶段且 closed<end，未来历史仍严格按天。

- [x] 先写禁止测试文件读取、阶段标签隔离、波动阈值、B0/B1 手算失败测试。
- [x] 实现六表内存快照、同一 manifest 校验、期内订单评分筛选及紧凑诊断、均值基准。
- [x] 运行相关测试与既有套件，未拟合真实估计器。

## Task 2: 有限搜索、资源预算与可恢复记录

**Files:** pp_models/study_config.py, study.py, study_storage.py; tests/test_model_study.py.

**Interfaces:** `plan_study(pool, base, study)` 返回不训练的预算；`run_study(..., evaluator=evaluate)` 实施依次候选、按调参分数选择、锁定确认；缓存由配置/阶段哈希识别。

- [x] 先写 17 候选上限、1% 简单优先、阶段继承、确认无选参作用、恢复/篡改/失败测试。
- [x] 实现搜索与候选原子记录，显式 resume 和来源锁定；失败即停，不继续挑幸存者。
- [x] 用人工探针验证完整调度，真实 fit 始终禁止。

## Task 3: 命令、计划产物、文档与复核

**Files:** pp_models/__main__.py; README.md; docs/model_training.md; docs/research_decisions.md; tests/test_study_cli.py.

- [x] 写 study-plan 不拟合/不读取测试、study-run 显式执行与参数拒绝测试。
- [x] 实现命令并输出只读规划摘要到 training_plans；更新文档与估算。
- [x] 运行全套无真实拟合测试；独立代码审查；核实未生成真实训练输出。

## 执行裁定与记录

- 用户明确要求先完成计划并实现流程，直接实施，不重复申请计划批准；不启动训练。
- 项目不是 Git 仓库，继续用本计划记录，不创建工作树/提交。
- 只读硬件查询因沙箱权限受限后获自动批准完成：Apple M5 / 10 核 / 16 GiB；没有跑压力测试或训练。
- Task 2 消费 Task 1 的 period 评价与六表池；Task 3 只调用 validation-only 接口，不经过旧的 test 数据路径。
- Task 1 的 4 项、Task 2 的 4 项、Task 3 最初 2 项测试均先确认缺失行为失败，再实现通过；模型构造检查没有调用真实估计器 fit。
- 独立审查发现自定义 dataset-dir 未传递到输出目录保护：补充直接 / 子目录 × 新建 / 恢复的入口回归测试，先复现失败，再传递路径并通过；复审确认该问题已解决，无其他实质问题。
- 最终全套 54 项测试通过（2.161 秒）；该耗时只是协议 / 调度测试耗时，不能当作训练测速。
- 最终只读 study-plan 在拦截真实模型 fit 与测试表读取的保护下通过，与已保存资源摘要一致；调参 1,379 条、确认 879 条、跨阶段标签 3 条、最多 370 次拟合。
- 已确认 outputs/model_studies 和 outputs/model_runs 均不存在；本轮只生成 training_plans 中的关键资源统计，真实模型数值收敛与效果仍未验证。
