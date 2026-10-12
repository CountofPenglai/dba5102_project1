"""Plot selected fast response against official Base using saved aggregates only."""
import argparse
import csv
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
from plot_official_base_payout import chinese_font, digest, plt, mdates, FuncFormatter, PROJECT

RUN = PROJECT/'outputs/model_studies/fast_response_20261008'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=RUN)
    parser.add_argument('--fast-model', choices=['official_fast','official_power5','official_linear5','official_calm','official_calm_rate'], default='official_fast')
    args = parser.parse_args()
    manifest = json.loads((args.run_dir/'provenance/manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Experiment must be complete')
    records, sources = [], {}
    for phase in ('tuning', 'followup'):
        name = f'metrics/{phase}_daily.csv'
        payload = (args.run_dir/name).read_bytes()
        sources[name] = digest(payload)
        if sources[name] != manifest['files'][name]:
            raise ValueError('Daily source checksum mismatch')
        for row in csv.DictReader(io.StringIO(payload.decode())):
            if row['group'] != 'option_type' or row['value'] != '1' or row['cohort'] != 'benchmark':
                continue
            if row['model'] == 'official_raw' or (row['model'] == args.fast_model and row['selected'] == 'True'):
                records.append(row)
    by_model = {model: {datetime.fromisoformat(r['day_utc']): r for r in records if r['model'] == model}
                for model in ('official_raw', args.fast_model)}
    raw, fast = by_model.values()
    if not raw or set(raw) != set(fast) or len(records) != 2*len(raw):
        raise ValueError('Expected one comparable Put row per method and day')
    dates = sorted(raw)
    for day in dates:
        for key in ('orders', 'actual_total_usdc'):
            if raw[day][key] != fast[day][key]:
                raise ValueError('Comparison order counts/targets differ')
    plt.rcParams.update({'font.family': chinese_font(), 'font.size': 11, 'axes.unicode_minus': False,
        'svg.fonttype': 'path', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), dpi=170, sharex=True,
                             gridspec_kw={'height_ratios': [2.2, 1]})
    fig.subplots_adjust(left=.09, right=.97, top=.84, bottom=.14, hspace=.25)
    method = {'official_fast':'15 分钟快速响应','official_power5':'5 分钟平方累积','official_linear5':'5 分钟线性累积','official_calm':'5 分钟响应＋低风险惩罚','official_calm_rate':'5 分钟响应＋双约束'}[args.fast_model]
    fig.suptitle('Put：官方基础值 × '+method, x=.09, ha='left', y=.96, fontsize=20, color='#17263C')
    fig.text(.09, .903, '同一批验证订单的每日金额；'+method+'，响应幅度按日学习', fontsize=11, color='#526176')
    ax = axes[0]
    for table, field, label, color in [(raw, 'actual_total_usdc', '实际赔付', '#D65A31'),
            (raw, 'predicted_total_usdc', '官方原值', '#2563A6'),
            (fast, 'predicted_total_usdc', '官方 × '+method, '#208368')]:
        ax.plot(dates, [float(table[d][field]) for d in dates], marker='o', markersize=3.5,
                linewidth=2, label=label, color=color)
    ax.set_ylabel('Put 每日金额（USDC）')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.legend(loc='upper left', frameon=False, ncol=3)
    axes[1].plot(dates, [float(fast[d]['multiplier_mean']) for d in dates], color='#208368', label='当日 Put 平均乘数')
    axes[1].plot(dates, [float(fast[d]['multiplier_max']) for d in dates], color='#7C64A8', linestyle='--', label='当日 Put 最大乘数')
    axes[1].axhline(1., color='#94A3B8', linewidth=.8)
    axes[1].set_ylabel('价格乘数')
    axes[1].legend(loc='upper left', ncol=2, frameon=False)
    for ax in axes:
        ax.grid(axis='y', color='#E2E8F0', linewidth=.7)
        ax.axvline(datetime.fromisoformat('2026-06-20T12:00:00+00:00'), color='#94A3B8', linestyle=':', linewidth=1)
        ax.spines['left'].set_color('#CBD5E1'); ax.spines['bottom'].set_color('#CBD5E1')
        ax.set_xlim(dates[0]-timedelta(hours=9), dates[-1]+timedelta(hours=9))
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axes[1].set_xlabel('订单创建日期（UTC）')
    fig.text(.09, .055, '虚线右侧为 6/21–6/30 后段评估。图示为实际运行得到的快速响应；六月结果为探索性验证，未用七月测试集。',
             fontsize=9, color='#64748B')
    figures = args.run_dir/'figures'; figures.mkdir(exist_ok=True)
    paths = []
    for ext in ('png', 'svg'):
        path = figures/f'put_fast_response_daily.{ext}'
        fig.savefig(path, facecolor='white'); paths.append(path)
    plt.close(fig)
    info = {'source_sha256': sources, 'fast_model': args.fast_model, 'script_sha256': digest(Path(__file__).read_bytes()),
            'helper_sha256': digest(Path(__file__).with_name('plot_official_base_payout.py').read_bytes()),
            'orders': sum(int(raw[d]['orders']) for d in dates), 'days': len(dates), 'test_used': False,
            'files': {str(p.relative_to(args.run_dir)): digest(p.read_bytes()) for p in paths}}
    (args.run_dir/'provenance/plot.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
