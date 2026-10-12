"""Plot the terminal-distribution probe from checked aggregate CSVs only."""
import argparse
import csv
from datetime import datetime, timedelta
import io
import json
from pathlib import Path
from plot_official_base_payout import chinese_font, digest, plt, mdates, FuncFormatter, PROJECT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=PROJECT/'outputs/model_studies/terminal_distribution_20261009')
    args=parser.parse_args();run=args.run_dir
    manifest=json.loads((run/'provenance/manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('Study must be complete')
    sources={}
    def read(name):
        payload=(run/name).read_bytes();sources[name]=digest(payload)
        if sources[name]!=manifest['files'][name]:raise ValueError('Source checksum mismatch')
        return list(csv.DictReader(io.StringIO(payload.decode())))
    models={'raw':('官方原值','#2563A6'), 'scaled':('Put 统一倍率','#94A3B8'),
            'static':('静态分布修正','#9B66BA'), 'dynamic':('动态分布修正','#208368')}
    tables={k:{} for k in models}
    for phase in ('tuning','followup'):
        for r in read(f'metrics/{phase}_daily.csv'):
            if r['group']!='option_type' or r['value']!='1' or r['cohort']!='benchmark':continue
            mode=r['distribution_mode']
            if mode in ('static','dynamic') and r['selected']!='True':continue
            day=datetime.fromisoformat(r['day_utc'])
            if day in tables[mode]:raise ValueError('Duplicate method/day')
            tables[mode][day]=r
    raw=tables['raw'];dates=sorted(raw)
    for table in tables.values():
        if set(table)!=set(raw) or not raw:raise ValueError('Mismatched scoring dates')
        for d in dates:
            if any(table[d][k]!=raw[d][k] for k in ('orders','actual_total_usdc')):
                raise ValueError('Mismatched scoring orders or outcomes')
    plt.rcParams.update({'font.family':chinese_font(),'font.size':11,'axes.unicode_minus':False,
        'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(13,8),dpi=170,sharex=True,gridspec_kw={'height_ratios':[2.2,1]})
    fig.subplots_adjust(left=.09,right=.97,top=.83,bottom=.15,hspace=.23)
    fig.suptitle('终点价格分布能否改善赔付预测？',x=.09,ha='left',y=.96,fontsize=20,color='#17263C')
    fig.text(.09,.906,'相同 Put 订单的每日总额；到期赔付代理经历史实际赔付校准；不含额外风险余量',color='#526176')
    axes[0].plot(dates,[float(raw[d]['actual_total_usdc']) for d in dates],label='实际赔付',color='#D65A31',lw=2.4,marker='o',ms=3)
    for mode,(label,color) in models.items():
        axes[0].plot(dates,[float(tables[mode][d]['predicted_total_usdc']) for d in dates],label=label,color=color,
                     lw=1.7,ls='--' if mode=='scaled' else '-')
        if mode!='raw':
            axes[1].plot(dates,[float(tables[mode][d]['terminal_put_multiplier_mean']) for d in dates],
                         label=label,color=color,lw=1.7,ls='--' if mode=='scaled' else '-')
    axes[0].set_ylabel('Put 每日金额（USDC）')
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:,.0f}'))
    axes[0].legend(loc='upper left',ncol=3,frameon=False)
    axes[1].set_ylabel('Put 平均乘数')
    axes[1].axhline(1,color='#94A3B8',lw=.7)
    axes[1].legend(loc='upper left',ncol=3,frameon=False)
    for ax in axes:
        ax.grid(axis='y',color='#E2E8F0',lw=.7)
        ax.axvline(datetime.fromisoformat('2026-06-20T12:00:00+00:00'),color='#94A3B8',ls=':',lw=1)
        ax.set_xlim(dates[0]-timedelta(hours=9),dates[-1]+timedelta(hours=9))
    axes[1].xaxis.set_major_locator(mdates.DayLocator(interval=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axes[1].set_xlabel('订单创建日期（UTC）')
    fig.text(.09,.065,'虚线右侧为 6/21–6/30 后段评估。分布参数固定；赔付修正按日更新，仅用此前已结算订单。',fontsize=9,color='#64748B')
    fig.text(.09,.035,'超过 8 小时的 Put 保持原值并保留评分。六月已反复观察，属于探索性验证；未使用七月测试集。',fontsize=9,color='#64748B')
    figures=run/'figures';figures.mkdir(exist_ok=True);paths=[]
    def save(fig,name):
        for ext in ('png','svg'):
            path=figures/f'{name}.{ext}';fig.savefig(path,facecolor='white');paths.append(path)
        plt.close(fig)
    save(fig,'terminal_put_daily')
    rows=[r for r in read('metrics/market_summary.csv') if r['phase']=='followup' and r['volatility_group']=='all']
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),dpi=170)
    fig.subplots_adjust(left=.075,right=.975,top=.75,bottom=.24,wspace=.36)
    fig.suptitle('后段行情预测：覆盖率与预测精度',x=.075,ha='left',y=.95,fontsize=19,color='#17263C')
    fig.text(.075,.86,'每 30 分钟一个起点；两个方法使用相同历史和评分起点；分布中心固定为零对数收益',color='#526176')
    for method,label,color in [('static','静态历史分布','#9B66BA'),('dynamic','EWMA 动态分布','#208368')]:
        data=sorted([r for r in rows if r['method']==method],key=lambda r:int(r['horizon_minutes']))
        horizons=[int(r['horizon_minutes']) for r in data]
        if horizons!=[30,60,120,240,480]:raise ValueError('Missing market horizon')
        for ax,key,scale in zip(axes,['covered_90','width_90','crps'],[100,1,1]):
            ax.plot(range(5),[float(r[key])*scale for r in data],marker='o',label=label,color=color,lw=2)
            ax.set_xticks(range(5),['0.5','1','2','4','8']);ax.grid(axis='y',color='#E2E8F0',lw=.7)
            ax.set_xlabel('预测期限（小时）')
    axes[0].axhline(90,color='#94A3B8',ls='--',lw=1)
    axes[0].set_ylabel('90% 区间实际覆盖率（%）');axes[0].legend(frameon=False,fontsize=9)
    axes[1].set_ylabel('90% 区间平均宽度（对数收益 bp）')
    axes[2].set_ylabel('CRPS（对数收益 bp，越低越好）')
    fig.text(.075,.085,'覆盖率接近目标不等于整体更准：必须结合区间宽度、区间评分和 CRPS。重叠窗口不能视作独立事件。',fontsize=9,color='#64748B')
    save(fig,'terminal_market_diagnostics')
    info={'source_sha256':sources,'script_sha256':digest(Path(__file__).read_bytes()),
          'helper_sha256':digest(Path(__file__).with_name('plot_official_base_payout.py').read_bytes()),
          'test_used':False,'files':{str(p.relative_to(run)):digest(p.read_bytes()) for p in paths}}
    (run/'provenance/plot.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
