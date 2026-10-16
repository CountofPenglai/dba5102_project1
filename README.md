# PP定价研究 H0与官方Base的有限混合

更新：2026-10-16。研究目标是估计BTC保险订单的每BTC预期赔付。当前项目只维护报告需要的模型、统一数据处理、验证与公告日历；原始订单和行情不随仓库发布。

**当前结论：** Call采用40% H0、Put采用15% H0的混合，在已见验证集的月等权误差上略优于Base，但全期逐单RMSE和完整风险约束仍未超过Base。它是开发候选，尚不是已验证的部署方案。

| 同一5–7月队列 | 月等权RMSE | 月等权MAE | 全期逐单RMSE |
|---|---:|---:|---:|
| Base | 293.564338 | 184.439853 | **295.150078** |
| H0 | 298.010152 | 186.263090 | 303.531640 |
| Call50% / Put25% H0 | 293.114750 | 183.652670 | 296.046896 |
| Call40% / Put15% H0 | **293.048025** | **183.558057** | 295.614043 |

单位USDC/BTC。月等权RMSE是三个单月MSE等权平均后开方；逐单RMSE按订单数加权。候选改善的95%区间仍跨0，Call净金额偏差扩大10.81%。完整解释见[统一研究结果](docs/reports/2026-10-15-report-unified-rerun.md)。

## 项目结构

```text
pp_audit/       原始订单、行情与关键质量统计
pp_data/        特征、标签、元数据统一池及校验加载
pp_models/      H0、报告对照配方、统一评估与运行归档
pp_calendar/    宏观日历的时区、历史可知性与版本维护
config/         一份验证协议、数据规则和保留模型设置
scripts/        数据迁移、日历维护、独立结果核验
external_data/  官方公告日历、冻结快照与公开来源
tests/          合成数据测试
docs/           汇报Word、数据/模型说明和当前关键结果
outputs/        本地生成的数据与完整运行，不提交Git
```

数据、模型、统计和归档职责已分离。H0、评分、尾部上下文、重采样等公共函数不再藏在旧GAM、静态分布或合同残差实验中。当前报告比较97个配置：96个原有对照和1个新公告规则；没有新增选参网格。模型与保留理由见[模型说明](docs/model_design.md)。

## 唯一训练验证口径

[验证协议v3](config/validation_protocol_v3.json)是日期角色的唯一来源：4月训练，5–7月滚动验证，从5月1日起连续每7日更新，使用此前30日创建且更新前已结算的历史订单。右端为2026-07-31 17:11:48 UTC，不含该时点。同队列4,640单，5/6/7月分别726/1,373/2,541单。

预处理现在输出一套平面数据，不再创建train、validation、test子目录。迁移保留原报告样本，不补回历史排除的跨月结算订单；这一样本局限仍需披露。7月属于已见验证，当前没有独立内部测试集。新官方验证文件禁止读取。详见[数据流程](docs/data_pipeline.md)和[验证口径](docs/validation_protocol_v3.md)。

H0输出“正赔付概率 × Base × 正赔付金额系数”，Call与Put分别拟合。定价输入只能来自合同与已完成行情；最终赔付和实际终止时间只用于合法历史训练筛选及事后评价。当前周更依赖已结算标签反馈，尚未证明适用于官方无标签部署。

## 环境与统一入口

Python 3.10+；结果环境为Python 3.12。测试用合成数据，不需要私有订单。

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-models.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m pp_audit --help
.venv/bin/python -m pp_data --help
.venv/bin/python -m pp_models --help
```

首次从旧报告数据迁移，使用显式路径并写入新目录：

```sh
.venv/bin/python scripts/migrate_report_data.py --source-dir /absolute/path/to/old/preprocessing --output-dir outputs/preprocessing
```

迁移工具校验旧文件哈希，保留所有行与数值；它是读取旧容器的唯一兼容入口。模型加载器仅接受新格式。若从原始资料重建，使用[数据流程](docs/data_pipeline.md)中的队列参照选项，否则可能扩大样本，不能直接替代当前报告。

运行全部报告配方并独立核验：

```sh
.venv/bin/python -m pp_models --run-id report_local --data-dir outputs/preprocessing --market-path /absolute/path/to/btc_market_history.csv
.venv/bin/python scripts/verify_report_unified.py outputs/model_studies/report_local
```

run-id必须未被占用。入口拟合H0相关配方，按报价时刻应用固定公告规则，统一评分并核对96个未改模型的参考统计；只读取截至验证终点的旧行情前缀。输入、源码、配置、参数和环境留在本地完整运行，Git只保留[当前关键统计](docs/results/README.md)。旧单项训练入口、旧模型框架和旧分割配置已移除。

## 官方公告维护

[公告日历](external_data/macro_events/README.md)独立于订单维护，最后核对时间为2026-10-14。82条日程含65个主要事件，CPI/非农覆盖2026–2027，FOMC延至2028年1月。历史可知性、改期证据和时区规则均保留。

本分支在报告中主要展示外部数据搜集与维度拓展：把官方来源整理为可追溯、按报价时点可用的事件数据。小额修正规则是模型接入检验，具体效果放在附录，不作为主要优化成果。

```sh
.venv/bin/python scripts/update_macro_calendar.py --source external_data/macro_events/sources/2026-10-14/verified_schedule_history.json --check-only
```

后续更新另建来源包与快照，不能覆盖旧版本；没有自动抓取或定时任务。当前公告流程已替换为[逐事件小额规则](config/announcement_overlay_v1.json)：仅使用当时有预告证据的日程，单场最高+1%，发布前24小时半衰期增强，发布后60分钟归零，多场相加封顶+5%；Call/Put一致，同场FOMC去重。无需训练，也不使用赔付反馈。

## 汇报与交接

[商业汇报大纲 Word](docs/deliverables/研究报告大纲_H0与Base混合模型.docx) · [交接索引](docs/research_document_index.md) · [模型说明](docs/model_design.md) · [清理记录](docs/plans/2026-10-15-report-architecture-cleanup.md)。

研究报告于2026-10-16扩充为正文与PPT制作附录，沿用六人、12页幻灯片、9.5分钟的章节映射。正文解释数据预处理、H0核心逻辑、关键探索、分方向混合和外部公告；附录提供完整分月与分组成绩、模型参数、训练批次、公告支持、风险统计及逐页图表取数指南。

当前统计来源是`report_announcement_v1_20261016`。公告分支月等权RMSE为293.557，MAE为184.489；相对Base，RMSE微降而MAE略升，增益区间跨0。H0与混合数字保持一致。上一轮结果只保留作数值回归参照。清理前版本保留在Git提交`7d66a5f`，无关旧实验不再复制到当前目录。原始资料和其他研究分支不受本次清理影响。尚未建立需求模型，赔付预测改善不能直接等同于利润提升。
