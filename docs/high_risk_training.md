# 高风险GBDT训练入口

已完成2026-10-09首轮q97.5与q95标签对照，结果见[评价报告](../outputs/model_studies/high_risk_gbdt_q975_q950_20261009_verified/reports/evaluation.md)。当前不升级自动抬价：q97.5没有20%概率报警；q95报警78笔、命中8笔，共同样本上两种anchor的整体误差略升。7月继续保留。

## 运行

在项目目录下使用现有`.venv`和`requirements-models.txt`，无需新增依赖：

```sh
# 只显示协议、特征、模型设置；不读取数据，不训练，不写结果
.venv/bin/python -m pp_models.high_risk_study plan

# 同时运行两个固定标签、两个方向及两种anchor的固定倍率对照
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pp_models.high_risk_study run --run-id NEW_ID

# 独立重建每日阈值与拟合，核对关键汇总，补充概率范围与五个最大赔付日
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.verify_high_risk_study --run-dir outputs/model_studies/NEW_ID
```

唯一日期来源为`config/validation_protocol_v2.json`，本入口没有测试期参数。每次使用新run-id，禁止覆盖已完成结果。无参数不会启动训练。单线程正式运行约11秒、0.48 GiB量级内存，独立核验另需相近时间。

## 文件与参数

- [固定训练配置](../config/high_risk_gbdt_v1.json)：两个单位赔付标签、浅层树参数、样本支持要求、报警概率及倍率。
- [分类器](../pp_models/high_risk_classifier.py)：输入白名单检查、Call/Put隔离、原生缺失值处理、历史率回退及报警门控。
- [研究入口](../pp_models/high_risk_study.py)：严格逐日前向训练、共同样本anchor、分类与金额评价、来源快照。
- [特征与标签说明](high_risk_features.md)：每日q97.5/q95标签、32项特征、未知标签与缺失规则。

默认80轮、学习率0.05、深度2、最多4叶、每叶至少30笔、L2=10；不加类别权重、随机早停或参数搜索。每方向训练至少100笔、10个正例、20个负例、3个正例创建日期，否则使用平滑历史发生率并禁止抬价。

**标签阈值不同于报警概率：**q95/q97.5定义结果是否高风险；分类器预测概率达到20%才触发1.5倍。两种标签使用同一报警概率与倍率。额外的10%/30%/50%只输出识别诊断，不参与挑选最优策略。

分类使用全部risk订单（4,074笔评分），Base与两步金额比较仅用相同benchmark订单（2,953笔）。两步anchor使用原benchmark历史，不因分类器获得额外risk数据而变化。最终赔付只作为监督目标/评价，实际结束时间仅用于训练标签可用性和事后诊断，均不得进入分类器X。

## 输出

`outputs/model_studies/<run-id>/{metrics,provenance,reports}`仅保存汇总，不导出逐笔预测、标签或模型。`classification.csv`、`threshold_diagnostic.csv`报告识别效果，`macro_monthly.csv`和`overlay.csv`报告金额误差，`training_support.csv`报告训练与回退覆盖。独立核验补充`probability_ranges.csv`、`largest_payout_days.csv`与`verification.json`。

AP须与相同标签的历史率对照和正例比例比较，精确率与正常订单误报率的分母不同；报告不足以证明独立事件层面的统计显著改善。固定倍率只检验报警位置，未估计需求、成交变化或实际利润。
