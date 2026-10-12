"""Plot existing daily aggregates; no training or access to order/test datasets.

Install requirements-plots.txt in a plotting environment, then run from the project:
    python scripts/plot_official_base_payout.py
"""
import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

# Keep Matplotlib's font/cache writes out of the user's global configuration.
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'pp_plot_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT/'outputs/model_studies/anchor_comparison_20261008'
OUTPUT = PROJECT/'outputs/diagnostics/official_base_payout_20261008'


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def load_daily(run):
    manifest = json.loads((run/'provenance/manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('The source experiment must be complete')
    records, sources = [], {}
    for phase in ('tuning', 'followup'):
        tables = {}
        for suffix in ('', '_daily'):
            name = f'metrics/{phase}{suffix}.csv'
            payload = (run/name).read_bytes()
            sha = digest(payload)
            if sha != manifest['files'][name]:
                raise ValueError(f'Source checksum mismatch: {name}')
            sources[name] = sha
            rows = csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))
            tables[suffix] = [r for r in rows if r['model'] == 'official_raw'
                and r['cohort'] == 'benchmark' and r['group'] == 'overall' and r['value'] == 'all']
        if len(tables['']) != 1 or not tables['_daily']:
            raise ValueError('Expected one official raw summary and nonempty daily records')
        phase_rows = []
        for row in tables['_daily']:
            item = dict(day=datetime.fromisoformat(row['day_utc']), phase=phase, orders=int(row['orders']),
                official=float(row['predicted_total_usdc']), actual=float(row['actual_total_usdc']))
            if item['orders'] <= 0 or any(not math.isfinite(item[k]) or item[k] < 0 for k in ('official', 'actual')):
                raise ValueError('Invalid daily amounts or order count')
            phase_rows.append(item)
        summary = tables[''][0]
        if sum(r['orders'] for r in phase_rows) != int(summary['orders']):
            raise ValueError('Daily order counts do not reconcile')
        for key, field in (('official', 'predicted_total_usdc'), ('actual', 'actual_total_usdc')):
            if not math.isclose(sum(r[key] for r in phase_rows), float(summary[field]), rel_tol=1e-12, abs_tol=1e-8):
                raise ValueError('Daily amounts do not reconcile with the summary')
        records.extend(phase_rows)
    records.sort(key=lambda r: r['day'])
    if len({r['day'] for r in records}) != len(records):
        raise ValueError('Duplicate dates across phases')
    return records, sources


def chinese_font():
    candidates = [Path('/System/Library/Fonts/STHeiti Medium.ttc'),
        Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('C:/Windows/Fonts/msyh.ttc')]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            return font_manager.FontProperties(fname=str(path)).get_name()
    raise RuntimeError('A Chinese font is required; add a font path in chinese_font()')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--output-dir', type=Path, default=OUTPUT)
    args = parser.parse_args()
    records, sources = load_daily(args.run_dir)
    dates = [r['day'] for r in records]
    count = sum(r['orders'] for r in records)
    plt.rcParams.update({'font.family': chinese_font(), 'font.size': 12, 'axes.unicode_minus': False,
        'svg.fonttype': 'path', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(13.6, 7.4), dpi=180)
    fig.subplots_adjust(left=.085, right=.97, bottom=.20, top=.79)
    fig.suptitle('官方基础值与实际赔付', x=.085, y=.955, ha='left', fontsize=21, fontweight='bold', color='#17263C')
    fig.text(.085, .895, f'{dates[0]:%Y-%m-%d} — {dates[-1]:%m-%d}  |  验证期共同样本 {count:,} 笔  |  每日总额',
        fontsize=12, color='#526176')
    ax.plot(dates, [r['official'] for r in records], color='#2563A6', linewidth=2.4,
        marker='o', markersize=4.5, label='官方基础值折算金额', zorder=3)
    ax.plot(dates, [r['actual'] for r in records], color='#D65A31', linewidth=2.4,
        marker='o', markersize=4.5, label='实际赔付金额', zorder=4)
    ax.set_ylabel('金额（USDC）', labelpad=12, color='#334155')
    ax.set_xlabel('订单创建日期（UTC）', labelpad=12, color='#334155')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.tick_params(axis='both', labelsize=10, length=0, pad=8, colors='#526176')
    ax.grid(axis='y', color='#E2E8F0', linewidth=.8)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#CBD5E1')
    ax.set_ylim(0, max(max(r['actual'], r['official']) for r in records)*1.20)
    ax.set_xlim(dates[0]-timedelta(hours=9), dates[-1]+timedelta(hours=9))
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.025), ncol=2, frameon=False, borderaxespad=0, fontsize=12)
    boundary = min(r['day'] for r in records if r['phase'] == 'followup')-timedelta(hours=12)
    ax.axvline(boundary, color='#A6B3C3', linestyle=(0, (4, 4)), linewidth=1)
    ax.text(boundary+timedelta(hours=5), .95, '后段评估', transform=ax.get_xaxis_transform(),
        color='#7A8799', fontsize=10, va='top')
    peak = max(records, key=lambda r: r['actual'])
    ax.annotate(f"{peak['day']:%m/%d}\n实际 {peak['actual']:,.0f}\n基础 {peak['official']:,.0f}",
        xy=(peak['day'], peak['actual']), xytext=(peak['day']+timedelta(days=1.15), peak['actual']*1.04),
        fontsize=11, color='#334155', va='center', linespacing=1.5,
        arrowprops=dict(arrowstyle='-', color='#94A3B8', linewidth=1),
        bbox=dict(boxstyle='round,pad=.45', fc='white', ec='#E2E8F0'))
    fig.text(.085, .085, '官方金额 = Σ（API Base × BTC 数量），未乘校准系数；实际金额为相同订单的最终赔付。',
        fontsize=10, color='#64748B')
    fig.text(.085, .045, '按创建日归组。沿用既有评分样本及阶段标签边界；无可比样本的日期未补零，未使用七月测试集。',
        fontsize=10, color='#64748B')
    figures = args.output_dir/'figures'
    figures.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(figures/f'official_base_vs_realized_payout_daily.{extension}', facecolor='white')
    plt.close(fig)
    provenance = args.output_dir/'provenance'
    provenance.mkdir(exist_ok=True)
    info = {'source_run': str(args.run_dir.resolve()), 'source_sha256': sources,
        'script': str(Path(__file__).resolve()), 'script_sha256': digest(Path(__file__).read_bytes()),
        'matplotlib_version': matplotlib.__version__, 'prediction_days': len(records), 'orders': count,
        'official_total_usdc': sum(r['official'] for r in records),
        'actual_total_usdc': sum(r['actual'] for r in records), 'test_used': False,
        'grouping': 'order creation day UTC; same scored orders; raw Base times BTC quantity',
        'artifacts': {p.name: digest(p.read_bytes()) for p in sorted(figures.glob('official_base_vs_realized_payout_daily.*'))}}
    (provenance/'plot.json').write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'figures': str(figures.resolve()), 'days': len(records), 'orders': count}, ensure_ascii=False))


if __name__ == '__main__':
    main()
