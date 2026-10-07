# 统一数据预处理 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan inline, test first, then obtain one independent review.

**Goal:** 由一个入口生成可复现、按时间固定分割、特征与标签分离的数据，后续模型统一加载。

**Architecture:** 新增 `pp_data`，复用 `pp_audit` 的订单与行情质量规则。唯一配置固定时间边界，先建立全部有效订单的分割，再以共同样本标记提供不同研究口径；不各自随机分割。输出 CSV 数据、少量汇总和带校验值的清单。

**Tech Stack:** Python 3.10+，现有 pandas 2.2 / numpy 1.26+，标准库 unittest；不新增依赖。

**Spec:** 本文件以下“固定设计”及 `../research_decisions.md` 中已确认的处理原则。用户本轮明确要求实现，直接完成可审阅结果；不增加审批往返。当前目录非 Git 仓库，在用户指定目录工作，不创建仓库或提交。

## 固定设计

- UTC 时间，区间左闭右开：训练 `[2026-02-01, 2026-06-01)`；验证 `[2026-06-01, 2026-07-01)`；内部测试 `[2026-07-01, 2026-07-31 17:11:48)`。末端取已知官方留出开始时间，若元数据声明不一致则报错，不能自动漂移。
- 每组订单要求 `closedAt < 本组结束时刻`。恰好在边界结算或跨边界的记录剔除并汇总，不转移到下一组。保留正常提前结算。7 月已查看描述统计，只称内部后期测试。
- `config/preprocessing_v1.json` 是唯一分割定义；CLI 只提供输入与输出目录，不提供随机种子、比例或其他分割参数。
- 候选样本完全复用审计的 `technical` 与 `common` 规则。风险研究保留缺失基础价的有效订单；官方共同样本标记为 `benchmark_eligible`，两口径使用同一 order_id 和同一时间分割。
- 特征白名单：方向、产品秒数、合同剩余秒数、数量、执行价、已结束行情价格、`log(S/K)`、过去 60/240/1440 分钟的对数收益与实现波动 `sqrt(sum(log_return**2))`，以及单独可选的官方基础价。所有行情特征以最近已结束且距下单不超过 60 秒的 K 线为锚点；N 分钟收益和波动要求 N+1 根有效连续 K 线。无插补、无缩放、无全样本分位数或波动状态阈值。
- 标签：实际总赔付、每 BTC 赔付、是否正赔付。结算时间、历史成交保费、markup、时间偏差等只进 metadata；绝不进入特征。
- 每笔保留源文件行序对应的稳定 ID，数据按创建时间及 ID 稳定排序。源文件哈希与代码、配置哈希记录在 manifest，输出文件记录哈希，重复生成内容一致。
- 分类目录：`outputs/preprocessing/datasets/{train,validation,test}/{features,targets,metadata}.csv`，`reports/{split_statistics,exclusion_statistics}.csv`，`manifest.json`。新任务明确需要可训练数据，因此在保留关键汇总的同时生成必要的数据集；不输出图表、模型或预测收益。
- 标准加载接口 `load_split(split, cohort='benchmark', use_base=None, output_dir=None)` 返回以 ID 为索引的 `X`、每 BTC 标签 `y`、独立 metadata。benchmark 默认带基础价，risk 默认不带；risk 请求基础价时报错，不隐式插补或缩小样本。
- 所有输入、数据一致性先检查；输出不得与源目录重叠，不覆盖陌生文件。清单最后写入；加载校验输出哈希、配置及 ID 对齐，拒绝混杂或损坏的产物。

## Global Constraints

- 原始数据不变；复用既有审计测试并运行完整测试集。
- 没有真实需求估计、价格加成、训练模型或根据测试集选择参数。
- 缺失标签不补零；基础价缺失不补值；派生非有限值不能进入模型。
- 切分统计只报告样本数、覆盖和时间范围，不新增加测试集赔付或收益表现分析。

## Review Focus

1. 边界上创建 / 结算、跨边界订单；UTC 与相同时刻的稳定顺序。
2. 修改未来行情、最终赔付或 closedAt 不得改变任何当时可观测特征。
3. 行情缺口、重复分钟和窗口首个收益所需的额外一分钟，不得被 rolling 或匹配掩盖。
4. 基础价缺失订单仍在 risk 口径，benchmark 真正是相同分割的子集；元数据不被当特征。
5. 重跑、输出篡改、源目录重叠或陌生已有文件，应保持原始数据安全并避免加载混合版本。

## Tasks

### Task 1: 统一分割和事前特征

**Files:** `config/preprocessing_v1.json`, `pp_data/config.py`, `pp_data/features.py`, `pp_data/pipeline.py`, `tests/test_preprocessing.py`。

**Interfaces:** `load_config() -> dict`; `build_features(orders, usable_market) -> (features, last_candle_close)`；`prepare(orders_raw, market_raw) -> PreparedData`（三个 split 的表及两份汇总）。

- [x] 编写边界结算、缺失基础价、事后字段隔离、未来行情不影响、连续 N+1 分钟、正常提前结束保留的测试，观察缺失功能失败。
- [x] 实现固定分割、白名单特征、分组表和排除汇总；运行全部测试（25 个通过）。

### Task 2: 统一生成与加载入口

**Files:** `pp_data/__init__.py`, `pp_data/__main__.py`, `pp_data/io.py`, `tests/test_preprocessing_io.py`。

**Interfaces:** `python -m pp_data [--data-dir PATH] [--output-dir PATH]`; `load_split(...) -> ModelSplit(X, y, metadata)`。

- [x] 先测可重现输出、源文件不变、两口径加载、损坏文件拒绝、缺列无产物、输出目录冲突，观察失败。
- [x] 实现来源清单、持久化与加载校验；运行完整测试（29 个通过）。

### Task 3: 真实数据验证与文档

**Files:** `README.md`, `docs/research_decisions.md`, 上述分类产物。

- [x] 真实数据生成，核对样本总数、分割互斥、时间边界、白名单、源文件哈希和重复生成哈希。
- [x] 更新唯一入口、时间分割、字段说明和研究决策实施状态；独立复核代码后修正发现并运行完整测试（31 个通过）。

## 执行记录

- 设计决定：选 2–5 月训练、6 月验证、7 月内部测试，使固定训练期比 2–4 月方案有更多样本，且保留两个连续后期窗口。此前滚动窗口建议由本次唯一分割取代；更改需要新版本。
- 设计决定：测试结束取官方留出开始时刻，未结算记录保守剔除；可能少用少量订单，但避免跨入官方留出期。
- 开始状态：现有审计程序已读；当前无 Git 仓库，无分割产物；在既有目录实现。
- Task 1: complete — 新模块缺失时测试失败，实现后完整 25/25 通过；无修改既有审计规则。
- Task 2: complete — 加载入口缺失时测试失败，实现后完整 29/29 通过；两次 CLI 输出字节一致，原始文件未变。
- Task 3: 真实数据首次生成：7,204 → 技术条件 7,079 → 固定边界保留 7,055；共同样本 5,934。5 个原始资料文件哈希保持不变；文档已更新，独立复核待完成。
- Task 3: complete — 用独立标准库 CSV / 日期逻辑核对全部样本 ID、标签、时间归属和共同样本子集；修正后再次生成真实数据，12 个文件逐字节一致，6 个 split/cohort 加载组合均通过，5 个源文件未变。
- Final review: 独立只读复核已完成，发现读取与校验之间发生重建可能混用版本；确定性交错测试复现旧特征 + 新标签。改为一次读取、校验并解析同一字节快照，回归测试 RED → GREEN。
- Final review: 空样本组原来可能返回空训练数据或触发 NumPy TypeError。按统一加载接口的输入有效性要求处理为明确 ValueError；不让模型训练阶段收到无法解释的类型错误。回归测试 RED → GREEN，完整测试 31/31 通过。
- Final review: 提供方基础价真实历史可用性和赔付精确定义无法由代码复核确认，继续保留既有资料限制；固定分割是本次研究设计，不构成未来表现保证。若这些假设变化，需要更新口径并重建数据。
- 最终状态：所有计划任务完成，无未处理代码复核问题；没有生成或训练模型，没有拟合需求或波动阈值。
- 后续研究协议补充：用户已选择按天更新的逐日前向评估，详见 `../benchmark_design.md`。本计划对应的固定日期分区、数据文件及其排除规则保持不变；每日滚动历史筛选 / 训练是后续 benchmark 的职责，不由已完成的预处理自动执行。
