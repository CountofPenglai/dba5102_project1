# 统一数据流程

当前数据存储版本为`preprocessing_v2`，模型日期角色来自唯一的[验证协议v3](../config/validation_protocol_v3.json)。数据处理只负责质量、定价可用特征、目标和时间元数据，不给订单写入永久训练或测试身份。

## 文件与接口

```text
outputs/preprocessing/
  datasets/features.csv
  datasets/targets.csv
  datasets/metadata.csv
  reports/dataset_statistics.csv
  reports/exclusion_statistics.csv
  manifest.json
```

三张表以`order_id`严格对齐，按创建时间和原始行号排序。清单记录来源、列结构、类型、哈希和生产时的代码指纹。加载时验证全套产物与当前数据契约；代码后续重构不会使相同数据自动失效。写入采用原子发布，拒绝覆盖不同或损坏的已存在数据。

`load_dataset(path, cohort="benchmark", use_base=True)`返回`ModelData(X, y, metadata)`。benchmark保留有完整价格字段的可比订单；risk保留所有合格风险订单，但不暴露缺失的Base，不作补值。`canonical_identity(path)`验证数据并返回完整身份。

| 内容 | 保留范围及用途 |
|---|---|
| 特征 | Call/Put、名义期限、剩余期限、BTC数量、执行价、市场价、对数价格位置、60/240/1440分钟收益与波动、官方Base |
| 目标 | 最终赔付USDC、每BTC赔付、是否正赔付；仅供已结算训练和评价 |
| 元数据 | 原始行号、创建/结束/到期时间、最后完整K线时间、队列标记、原成交价与markup等审计字段；不自动进入模型 |

`option_type=0`为Call，`1`为Put。`realized_vol_Nm`为完整分钟对数收益平方和开方，不年化；N分钟至少需要N+1根连续K线。匹配的最后K线不得晚于报价，年龄不超过60秒。实际结束时间不进入特征。

## 复现已有报告

旧报告池只通过显式迁移工具读取，旧目录名不再决定模型角色：

```sh
python scripts/migrate_report_data.py --source-dir /absolute/path/to/old/preprocessing --output-dir outputs/preprocessing
```

迁移校验旧清单的全部11个产物，合并三份物理容器并保留全部数值与原始行号。现有池含7,055笔risk、5,934笔benchmark；其中有4月之前的订单，但模型不会使用它们训练。4,640笔属于本报告5–7月验证队列。

**迁移不是扩大样本。** 旧预处理曾排除跨分区边界尚未结算的订单，本次保留这一历史样本选择，避免架构清理改变成绩。旧排除详情可从原清单指纹和清理前Git版本追溯；当前统计不应解释为所有潜在订单上的表现。

## 从原始数据构建

```sh
python -m pp_data --data-dir /absolute/path/to/authorized/raw --output-dir outputs/rebuilt --cohort-reference outputs/preprocessing
```

原始目录需包含`pp_transactions_training.csv`及`btc_market_history.csv`。原始行号用于稳定身份；不得先重新排序源订单。提供`--cohort-reference`会限制到已验证参照池的全部订单，并核对特征、目标与元数据内容（只容许极小浮点舍入差异），失败时拒绝发布；输出目录必须与参照及输入分开。

不提供该选项时，构建器使用研究起点到最终验证截止的连续区间，只排除最终截止时未知的标签，不在6月或7月边界另作剔除。因此原生重建可能比迁移池多行，需要另行授权新的队列比较；不能冒称与当前报告等价。统一报告入口会拒绝改变4,640单的比较队列。

新官方验证文件不参与该流程。首次构建必须显式指定授权资料目录；默认相邻目录仅为便利值，不适用于所有工作树。
