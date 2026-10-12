"""Compare official Base and realized payout per BTC, by side and nominal tenor.

Run in an environment with requirements-plots.txt:
    python scripts/plot_official_base_payout_stratified.py

Reads only the three canonical validation tables and existing experiment statistics.
Reconstructs the frozen scoring population and checks its ID hash; never trains.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta
import io
import json
import math
from pathlib import Path

from plot_official_base_payout import PROJECT, RUN, chinese_font, digest
import matplotlib
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

OUTPUT = PROJECT/'outputs/diagnostics/official_base_payout_stratified_20261008'
DATA = PROJECT/'outputs/preprocessing'
BLUE, ORANGE = '#2563A6', '#D65A31'


def fingerprint(value):
    # Same serialization as pp_models.runs, without importing training dependencies.
    return digest((json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())


def read_csv_checked(path, expected, sources):
    payload = path.read_bytes()
    if digest(payload) != expected:
        raise ValueError(f'Source checksum mismatch: {path}')
    sources[str(path.resolve())] = expected
    return list(csv.DictReader(io.StringIO(payload.decode('utf-8-sig'))))


def load_orders(run, data):
    manifest = json.loads((run/'provenance/manifest.json').read_text())
    canonical = json.loads((data/'manifest.json').read_text())
    if manifest['status'] != 'complete' or canonical['producer'] != 'pp_data':
        raise ValueError('Expected completed study and canonical preprocessing data')
    names = [f'datasets/{split}/{kind}.csv' for split in ('train', 'validation')
             for kind in ('features', 'targets', 'metadata')]
    identity = dict(scope=['train', 'validation'], config_sha256=canonical['config_sha256'],
        preprocessing_code=canonical['code_sha256'], artifacts={n:canonical['artifacts'][n] for n in names})
    if fingerprint(identity) != manifest['identity']['source']['development']['dataset_sha256']:
        raise ValueError('Canonical data identity differs from the source experiment')
    sources, tables = {}, {}
    for kind in ('features', 'targets', 'metadata'):
        name = f'datasets/validation/{kind}.csv'
        rows = read_csv_checked(data/name, canonical['artifacts'][name]['sha256'], sources)
        tables[kind] = {r['order_id']:r for r in rows}
        if len(tables[kind]) != len(rows):
            raise ValueError('Duplicate order IDs')
    if not (list(tables['features']) == list(tables['targets']) == list(tables['metadata'])):
        raise ValueError('Validation table IDs are not aligned')
    periods = manifest['identity']['study']['periods']
    records, checks = [], {}
    for phase in ('tuning', 'followup'):
        start, end = map(datetime.fromisoformat, periods[phase])
        validation = canonical['config']['splits']['validation']
        if not datetime.fromisoformat(validation['start']) <= start < end <= datetime.fromisoformat(validation['end']):
            raise ValueError('Period must remain inside canonical validation')
        own = []
        for order_id, meta in tables['metadata'].items():
            created, closed = map(datetime.fromisoformat, (meta['created_at_utc'], meta['closed_at_utc']))
            if meta['benchmark_eligible'] != 'True' or not (start <= created < end and closed < end):
                continue
            x, y = tables['features'][order_id], tables['targets'][order_id]
            quantity, base, payout = float(x['quantity_btc']), float(x['api_base_price_per_btc_usdc']), float(y['payout_usdc'])
            side, tenor = int(x['option_type']), int(x['product_duration_seconds'])
            if side not in (0, 1) or tenor <= 0 or quantity <= 0 or base <= 0 or payout < 0:
                raise ValueError('Invalid canonical order')
            if not all(math.isfinite(v) for v in (quantity, base, payout)):
                raise ValueError('Nonfinite order value')
            if not math.isclose(payout, float(y['payout_per_btc_usdc'])*quantity, rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError('Payout units do not reconcile')
            own.append(dict(order_id=order_id, created=created, day=created.replace(hour=0, minute=0, second=0, microsecond=0),
                phase=phase, side=side, tenor=tenor, quantity=quantity, official=base*quantity, actual=payout))
        own.sort(key=lambda r:r['created'])
        expected = manifest['identity']['plan']['periods'][phase]
        if len(own) != expected['prediction_rows'] or fingerprint([r['order_id'] for r in own]) != expected['prediction_ids_sha256']:
            raise ValueError('Scored order IDs differ from the frozen experiment')
        name = f'metrics/{phase}_daily.csv'
        original = read_csv_checked(run/name, manifest['files'][name], sources)
        comparisons = 0
        for row in original:
            if row['model'] != 'official_raw' or row['cohort'] != 'benchmark':
                continue
            day = datetime.fromisoformat(row['day_utc'])
            selected = [r for r in own if r['day'] == day and
                        (row['group'] == 'overall' or r['side'] == int(row['value']))]
            if len(selected) != int(row['orders']):
                raise ValueError('Daily or directional order count mismatch')
            for key, field in (('official','predicted_total_usdc'),('actual','actual_total_usdc')):
                if not math.isclose(sum(r[key] for r in selected), float(row[field]), rel_tol=1e-12, abs_tol=1e-8):
                    raise ValueError('Daily or directional amounts differ from the source experiment')
            comparisons += 1
        checks[phase] = dict(orders=len(own), prediction_ids_sha256=expected['prediction_ids_sha256'],
            daily_directional_rows_reconciled=comparisons)
        records.extend(own)
    return records, sources, checks


def aggregate(records, by_tenor=False):
    groups = defaultdict(list)
    for row in records:
        groups[(row['day'], row['side'], row['tenor'] if by_tenor else None)].append(row)
    result = []
    for (day, side, tenor), rows in sorted(groups.items()):
        quantity = sum(r['quantity'] for r in rows)
        official, actual = (sum(r[k] for r in rows) for k in ('official', 'actual'))
        result.append(dict(day_utc=day.isoformat(), direction='Call' if side == 0 else 'Put',
            product_duration_seconds=tenor, orders=len(rows), quantity_btc=quantity,
            official_total_usdc=official, actual_total_usdc=actual,
            official_per_btc=official/quantity, actual_per_btc=actual/quantity))
    return result


def plot_panel(ax, rows, dates, upper):
    lookup = {datetime.fromisoformat(r['day_utc']):r for r in rows}
    for key, color in [('official_per_btc',BLUE), ('actual_per_btc',ORANGE)]:
        ax.plot(dates, [lookup[d][key] if d in lookup else float('nan') for d in dates],
            color=color, linewidth=1.9, zorder=2)
        for sparse in (False, True):
            points = [d for d in dates if d in lookup and (lookup[d]['orders'] < 5) == sparse]
            ax.scatter(points, [lookup[d][key] for d in points], s=24, edgecolors=color,
                facecolors='white' if sparse else color, linewidths=1.1, zorder=3)
    ax.set_ylim(0, upper)
    ax.set_xlim(dates[0]-timedelta(hours=9), dates[-1]+timedelta(hours=9))
    ax.grid(axis='y', color='#E2E8F0', linewidth=.8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _:f'{x:,.0f}'))
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[12,15,18,21,24,27,30]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.tick_params(axis='both', labelsize=10, length=0, pad=6, colors='#526176')
    for side in ('left','bottom'):
        ax.spines[side].set_color('#CBD5E1')
    ax.axvspan(datetime(2026,6,24,tzinfo=dates[0].tzinfo)-timedelta(hours=12),
        datetime(2026,6,25,tzinfo=dates[0].tzinfo)+timedelta(hours=12), color='#FFF1E8', zorder=0)


def legend(fig, y):
    handles = [Line2D([],[],color=BLUE,linewidth=2,marker='o',label='官方基础值'),
        Line2D([],[],color=ORANGE,linewidth=2,marker='o',label='实际赔付'),
        Line2D([],[],color='#64748B',linestyle='none',marker='o',markerfacecolor='white',label='空心点：当日该组少于 5 笔')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.075,y), ncol=3, frameon=False, fontsize=11)


def save(fig, directory, name):
    for extension in ('png','svg'):
        fig.savefig(directory/f'{name}.{extension}', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--data-dir', type=Path, default=DATA)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    records, sources, checks = load_orders(args.run_dir, args.data_dir)
    overview, stratified = aggregate(records), aggregate(records, True)
    first, last = min(r['day'] for r in records), max(r['day'] for r in records)
    dates = [first+timedelta(days=i) for i in range((last-first).days+1)]
    for category in ('figures','metrics','provenance'):
        (args.output_dir/category).mkdir(parents=True, exist_ok=True)
    for name, rows in [('daily_direction',overview), ('daily_direction_tenor',stratified)]:
        with (args.output_dir/f'metrics/{name}.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    plt.rcParams.update({'font.family':chinese_font(), 'font.size':12, 'axes.unicode_minus':False,
        'svg.fonttype':'path', 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 1, figsize=(13.6,9), dpi=170, sharex=True, sharey=True)
    fig.subplots_adjust(left=.085,right=.97,bottom=.19,top=.79,hspace=.35)
    fig.suptitle('每 BTC 官方基础值与实际赔付：按方向', x=.085,y=.965,ha='left',fontsize=20,color='#17263C')
    fig.text(.085,.914,f'{first:%Y-%m-%d} — {last:%m-%d}  |  {len(records):,} 笔共同评分订单  |  此总览仍包含期限构成变化',
        fontsize=11,color='#526176')
    legend(fig,.875)
    upper = max(max(r['official_per_btc'],r['actual_per_btc']) for r in overview)*1.12
    for ax, direction in zip(axes, ('Call','Put')):
        rows = [r for r in overview if r['direction']==direction]
        plot_panel(ax,rows,dates,upper)
        ax.set_title(f"{direction}  ·  {sum(r['orders'] for r in rows):,} 笔",loc='left',fontsize=13,pad=10)
        ax.set_ylabel('USDC / BTC',fontsize=11)
    axes[-1].set_xlabel('订单创建日期（UTC）',labelpad=10)
    fig.text(.085,.10,'每 BTC 数值 = 当日该方向的金额合计 ÷ BTC 数量合计；两条线使用同一批订单，未乘校准系数。',fontsize=10,color='#64748B')
    fig.text(.085,.055,'浅橙背景标出 6/24–25；缺少订单的日期留空，不补零。期限构成控制见另一张分面图。',fontsize=10,color='#64748B')
    save(fig,args.output_dir/'figures','official_vs_actual_per_btc_by_direction')
    tenors = sorted({r['tenor'] for r in records})
    fig, axes = plt.subplots(len(tenors),2,figsize=(14.4,3.05*len(tenors)+2.3),dpi=170,sharex=True,sharey='row',squeeze=False)
    fig.subplots_adjust(left=.075,right=.975,bottom=.092,top=.897,hspace=.40,wspace=.12)
    fig.suptitle('控制方向与期限后的每 BTC 赔付',x=.075,y=.974,ha='left',fontsize=22,color='#17263C')
    fig.text(.075,.946,f'{first:%Y-%m-%d} — {last:%m-%d}  |  每个面板固定方向和标称期限  |  同一行纵轴尺度相同',fontsize=12,color='#526176')
    legend(fig,.93)
    for i, tenor in enumerate(tenors):
        label = '30 分钟' if tenor==1800 else f'{tenor/3600:g} 小时'
        own = [r for r in stratified if r['product_duration_seconds']==tenor]
        upper = max(max(r['official_per_btc'],r['actual_per_btc']) for r in own)*1.13
        for j, direction in enumerate(('Call','Put')):
            rows = [r for r in own if r['direction']==direction]
            ax = axes[i,j]
            plot_panel(ax,rows,dates,upper)
            ax.set_title(f"{direction} · {label}   |   {sum(r['orders'] for r in rows):,} 笔 / {len(rows)} 日",loc='left',fontsize=12,pad=9)
            if j==0:
                ax.set_ylabel('USDC / BTC',fontsize=11)
            if not rows:
                ax.text(.5,.5,'无评分订单',transform=ax.transAxes,ha='center',color='#64748B')
            ax.tick_params(labelbottom=True)
    for ax in axes[-1]:
        ax.set_xlabel('订单创建日期（UTC）',labelpad=10)
    fig.text(.075,.047,'每 BTC 数值 = 当日同方向、同期限的金额合计 ÷ BTC 数量合计。保留全部期限；无订单时留空、不连线。',fontsize=11,color='#64748B')
    fig.text(.075,.022,'空心点为少于 5 笔的小样本；浅橙背景为 6/24–25。该图控制了标称期限构成，未控制 moneyness 等其他差异。',fontsize=11,color='#64748B')
    save(fig,args.output_dir/'figures','official_vs_actual_per_btc_by_direction_tenor')
    files = sorted(p for category in ('figures','metrics') for p in (args.output_dir/category).iterdir() if p.is_file())
    info = dict(source_sha256=sources,scoring_population=checks,orders=len(records),test_used=False,
        training_performed=False,matplotlib_version=matplotlib.__version__,
        script_sha256={str(p.resolve()):digest(p.read_bytes()) for p in
            [Path(__file__),Path(__file__).with_name('plot_official_base_payout.py')]},
        per_btc_definition='sum(amount_USDC) / sum(quantity_BTC) within each day/side/nominal-tenor cell',
        sparse_threshold_orders=5, missing_cells='gaps, never zero or interpolation',
        artifacts={str(p.relative_to(args.output_dir)):digest(p.read_bytes()) for p in files})
    (args.output_dir/'provenance/plot.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(args.output_dir.resolve()),orders=len(records),directions=len(overview),
        direction_tenor_cells=len(stratified),checks=checks),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
