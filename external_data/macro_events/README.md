# 美国宏观公告日历

独立公共数据集，仅维护CPI、Employment Situation（简称NFP/非农）和FOMC的**计划发布时间**，不依赖任何保险订单文件，不包含公告数值、市场预期、统计数值修订或未来行情。当前核对时点：**2026-10-14 12:30:02 UTC**。新版补齐1–3月历史预告及2月日程改期证据，未来覆盖不变。

## 当前覆盖与来源

| 家族 | 当前覆盖 | 美国东部发布时间 | 行数 |
|---|---|---|---:|
| CPI | 2026、2027全年公布日程 | 08:30 | 24 |
| NFP | 2026、2027全年公布日程 | 08:30 | 24 |
| FOMC声明 | 2026、2027全年及2028-01-26 | 14:00 | 17 |
| FOMC记者会 | 同上，与声明分开 | 14:30 | 17 |

合计82行，其中65个主要公告。FOMC记者会与同日声明不构成两次独立宏观事件。NFP代表整份就业报告发布，不能把该时点的市场影响全部归于非农人数。

主要来源：[BLS 2026年历](https://www.bls.gov/schedule/2026/home.htm)、[BLS 2027年历](https://www.bls.gov/schedule/2027/home.htm)、[CPI发布页](https://www.bls.gov/schedule/news_release/cpi.htm)、[就业报告发布页](https://www.bls.gov/schedule/news_release/empsit.htm)、[美联储会议日历](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)。FOMC时间依据[美联储2024年日程公告](https://www.federalreserve.gov/newsevents/pressreleases/monetary20240809a.htm)给出的14:00/14:30规则；未来各场时间属于官方通用规则推定，尚非逐场实际发生确认。

所有当地时间使用`America/New_York`转换，包含夏令时。08:30对应冬季13:30 UTC、夏季12:30 UTC；禁止固定减4小时。未来日期可能调整，`coverage`仅表示该版本已核对日程的日期范围，不保证将来不变。

## 文件组织

- `calendar.json`：当前可读取版本，与最新快照逐字节一致。
- `snapshots/20261014T123002Z.json`：当前不可覆盖版本；`snapshots/20261014T115935Z.json`为保留的上一版本。历史回放只选择当时已发布的快照。
- `sources/2026-10-14/verified_schedule_history.json`：当前来源包，保存官方URL、必要摘录、82条输入记录及1–3月补证。
- `sources/2026-10-14/verified_schedule.json`：上一来源包，继续保留，不能误用它重建最新补证版本。
- `sources/2026-10-14/fomc.html`：当次取得的美联储公开页面。BLS直接下载受限，使用官方页面检索文本存证；没有伪造完整HTML快照。

## 字段字典

| 字段 | 含义 |
|---|---|
| `event_id` | 稳定事件标识；CPI/NFP按**统计参考月**命名，不是发布日期月份 |
| `family` / `kind` | CPI/NFP/FOMC；release/statement/press_conference |
| `reference_period` | 统计参考期或FOMC会议标识 |
| `local_date` / `local_time` | 纽约计划日期和时刻，供来源核对 |
| `event_time_local` / `event_time_utc` | 带时区的派生时间，是合同跨公告判定的时间戳 |
| `status` | `historical_scheduled`为本快照时点前的计划时间，**不是实际已发布证明**；未来BLS为`scheduled`，未来FOMC为`tentative` |
| `time_basis` | 官方逐项时间或`official_recurring_time_rule`（官方重复时间规则） |
| `source_id` | 对应`sources`内官方日程来源 |
| `knowledge_basis` / `known_by_utc` | 最迟可确认已公开时间及依据；`current_snapshot`只证明本次核对时已知 |
| `announcement_source_id` | 历史预告来源：如上期CPI新闻稿说明下次发布日期 |
| `knowledge_time_precision` | 日级更新不能伪造具体发布时间；2月改期采用次日纽约00:00作为保守界限 |
| `revision_source_id` / `superseded_schedule` | 改期来源及旧日期、原来源；仅供版本审计，不把旧日期计作实际发生的第二次公告 |
| `observed_at_utc` / `as_of_utc` | 本次核对、发布版本时点，不冒充历史数据获取时间 |
| `coverage` | 各家族纽约日期的起点及排他终点；边界外必须报未知 |
| `input_sha256` | 规范化来源包哈希；运行产物另保存整个文件字节哈希 |

1–7月CPI/NFP已补齐上期新闻稿或官方改期日历证据，2026年FOMC取得2024年官方预告。因此旧市场期间的19个主要公告均有发布前公开证据；具体订单仍须检查其创建时刻不早于`known_by`。其余日期如无历史预告证据，只按当前版本时点已知处理。**该数据集不是所有历史日程修订的完整档案，known_by也不是首次通知时刻**。

2月是明确的改期例外：[1月就业稿](https://www.bls.gov/news.release/archives/empsit_01092026.htm)预告2月6日，[1月CPI稿](https://www.bls.gov/news.release/archives/cpi_01132026.htm)预告2月11日；[2月日历](https://www.bls.gov/schedule/2026/02_sched.htm)及[官方改期表](https://www.bls.gov/bls/2025-lapse-revised-release-dates.htm)分别改为11日和13日。日历标注2月4日更新，没有日内时间，故修订日期的`known_by`取2月5日00:00纽约（05:00 UTC）。原日期保留在`superseded_schedule`，不把1月预告证据错误地挂到新日期上。查询修订前历史仍返回未知，不自动恢复当时的完整旧日程。

当前订单支持和回退结论见[统一报告](../../docs/reports/2026-10-15-report-unified-rerun.md)。历史预告证据保留在上述来源包中。1月有市场但没有有效订单；历史证据齐全不等于足够训练样本。

## 查询与未知处理

`pp_calendar.calendar`独立于订单数据池维护；模型运行固定日历快照并记录哈希。

```python
from pathlib import Path
from pp_calendar.calendar import select_snapshot, query_events

calendar = select_snapshot(Path('external_data/macro_events'), as_of=prediction_time)
events = query_events(calendar, start=prediction_time, end=planned_expiry,
                      as_of=prediction_time)
```

`query_events`采用`start < event <= end`，默认排除记者会。超出覆盖、窗口内事件当时未知或没有历史快照支持的空窗口均抛出`ValueError`，不能编码成“无事件=0”。`select_snapshot`只选择发布时间不晚于查询时点的版本；2026-10-14以前尚无本项目实际归档快照，会拒绝选择。

旧数据分析属于依据官方历史预告作的回溯研究，直接核验相关订单的`known_by`；不声称使用了当年由本项目实时归档的版本。统计入口固定具体快照，日历后续更新不改变既有结果。低层`events_asof`只是证据过滤器，不用于单独生成“是否公告”特征。

## 更新流程

1. 核对上述官方年历、近期发布页和FOMC日历；保留官方变更和历史预告的来源。不要按“每月第一个星期五”自行生成非农日期。
2. 新建`sources/YYYY-MM-DD/verified_schedule.json`，复制并增补完整来源包，填写真实核对时间。保留旧事件ID，不删除历史记录；日程改变时保留旧快照，更新新版本的时间和当前获知证据。暂不支持静默删除/取消，遇正式取消须先显式扩展状态处理。
3. 检查后发布：

```sh
.venv/bin/python scripts/update_macro_calendar.py --source external_data/macro_events/sources/YYYY-MM-DD/verified_schedule.json --check-only
.venv/bin/python scripts/update_macro_calendar.py --source external_data/macro_events/sources/YYYY-MM-DD/verified_schedule.json
```

4. 检查事件数量、各家族月份覆盖、时区和新版本差异。发布器拒绝同名快照覆盖、时间倒退或旧事件无声消失；当前入口原子替换。每次官方后续测试前重新核对未来日程，不能因为覆盖到2027年就停止维护。

这是可重复的来源包校验和发布流程，**不是自动抓取服务**，没有设置定时任务。新官方验证/未取得测试数据无需打开即可更新日历；日历更新也不会自动启动训练、评分或改价。
