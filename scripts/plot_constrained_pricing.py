"""Render checked aggregate model comparisons; no raw data or model fitting."""
import argparse
import csv
from datetime import datetime
import io
import json
from pathlib import Path
from plot_official_base_payout import PROJECT, plt, mdates, FuncFormatter, chinese_font, digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=PROJECT/'outputs/model_studies/constrained_pricing_20261009_verified')
    run=parser.parse_args().run_dir;manifest=json.loads((run/'provenance/manifest.json').read_text());sources={}
    assert manifest['status']=='complete'
    def read(name):
        payload=(run/name).read_bytes();sources[name]=digest(payload)
        assert sources[name]==manifest['files'][name]
        return list(csv.DictReader(io.StringIO(payload.decode())))
    plt.rcParams.update({'font.family':chinese_font(),'font.size':11,'axes.unicode_minus':False,
        'svg.fonttype':'path','axes.spines.top':False,'axes.spines.right':False})
    models={'raw':('官方原值','#2563A6'),'scaled':('分方向统一倍率','#94A3B8'),
            'static_reference':('上一轮静态分布','#9B66BA'),'flow':('合约＋双尾＋订单流','#208368')}
    rows=read('metrics/followup_daily.csv');fig,axes=plt.subplots(2,1,figsize=(12.5,8.5),dpi=160,sharex=True)
    fig.subplots_adjust(left=.1,right=.97,top=.78,bottom=.15,hspace=.26)
    fig.suptitle('业务约束实验：峰值低估与事后高估仍然存在',x=.1,y=.96,ha='left',fontsize=20,color='#17263C')
    fig.text(.1,.902,'后段 6/21–6/30，同一批订单的日度赔付总额；动态响应每 5 分钟更新，系数按日学习',color='#526176')
    handles=[]
    for ax,side,title in zip(axes,['1','0'],['Put · 300 笔','Call · 579 笔']):
        tables={m:sorted([r for r in rows if r['group']=='option_type' and r['value']==side and r['variant']==m],key=lambda r:r['day_utc']) for m in models}
        raw=tables['raw'];dates=[datetime.fromisoformat(r['day_utc']) for r in raw]
        line,=ax.plot(dates,[float(r['actual_total_usdc']) for r in raw],label='实际赔付',color='#D65A31',lw=2.3,marker='o',ms=3)
        local=[line]
        for mode,(label,color) in models.items():
            table=tables[mode]
            assert [(r['day_utc'],r['orders'],r['actual_total_usdc']) for r in table]==[(r['day_utc'],r['orders'],r['actual_total_usdc']) for r in raw]
            line,=ax.plot(dates,[float(r['predicted_total_usdc']) for r in table],label=label,color=color,lw=1.8,ls='--' if mode=='scaled' else '-')
            local.append(line)
        if not handles:handles=local
        ax.set_title(title,loc='left',fontsize=13);ax.set_ylabel('每日金额（USDC）')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:,.0f}'));ax.grid(axis='y',color='#E2E8F0')
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.095,.885),ncol=3,frameon=False,fontsize=10)
    axes[-1].xaxis.set_major_locator(mdates.DayLocator());axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    axes[-1].set_xlabel('订单创建日期（UTC）')
    fig.text(.1,.065,'新模型的多数改善来自分方向整体校准；订单流额外作用很小，Call 高额赔付的低估反而加重。',fontsize=10,color='#526176')
    fig.text(.1,.032,'六月已反复用于研究，仅作探索。未使用七月测试集；预测金额不代表实际保费或利润。',fontsize=9,color='#64748B')
    figures=run/'figures';figures.mkdir(exist_ok=True);paths=[]
    def save(fig,name):
        for ext in ('png','svg'):
            path=figures/f'{name}.{ext}';fig.savefig(path,facecolor='white');paths.append(path)
        plt.close(fig)
    save(fig,'constrained_daily_payout')
    market=[r for r in read('metrics/market_summary.csv') if r['phase']=='followup' and r['volatility_group']=='all']
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.9),dpi=160)
    fig.subplots_adjust(left=.075,right=.975,top=.74,bottom=.24,wspace=.34)
    fig.suptitle('双侧调整：下尾有所改善，长周期仍不平衡',x=.075,y=.95,ha='left',fontsize=19,color='#17263C')
    fig.text(.075,.857,'90% 中央区间理想左右越界率各 5%；须同时检查宽度和整体预测评分',color='#526176')
    for method,label,color in [('dynamic','原动态分布','#94A3B8'),('calibrated','双侧调整分布','#208368'),('static','静态分布','#9B66BA')]:
        data=sorted([r for r in market if r['method']==method],key=lambda r:int(r['horizon_minutes']))
        assert [int(r['horizon_minutes']) for r in data]==[30,60,120,240,480]
        for ax,key,scale in zip(axes,['lower_miss_90','upper_miss_90','crps'],[100,100,1]):
            ax.plot(range(5),[float(r[key])*scale for r in data],marker='o',label=label,color=color,lw=1.9)
            ax.set_xticks(range(5),['0.5','1','2','4','8']);ax.set_xlabel('预测期限（小时）');ax.grid(axis='y',color='#E2E8F0')
    for ax in axes[:2]:ax.axhline(5,color='#475569',ls=':',lw=1)
    axes[0].set_ylabel('下方越界率（%）');axes[1].set_ylabel('上方越界率（%）')
    axes[2].set_ylabel('CRPS（对数收益 bp，越低越好）')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.52,.09),ncol=3,frameon=False)
    fig.text(.075,.035,'8h 调整后下方越界仍为 8.84%，上方仅 0.43%；高波动长周期区间变宽，评分恶化。重叠起点非独立事件。',fontsize=9,color='#64748B')
    save(fig,'constrained_tail_diagnostics')
    info={'source_sha256':sources,'script_sha256':digest(Path(__file__).read_bytes()),
          'helper_sha256':digest(Path(__file__).with_name('plot_official_base_payout.py').read_bytes()),
          'files':{str(p.relative_to(run)):digest(p.read_bytes()) for p in paths}}
    provenance=run/'provenance';(provenance/'plot_source.py').write_bytes(Path(__file__).read_bytes())
    (provenance/'plot_helper.py').write_bytes(Path(__file__).with_name('plot_official_base_payout.py').read_bytes())
    (provenance/'plot.json').write_text(json.dumps(info,indent=2)+'\n')
    paths += [provenance/n for n in ('plot.json','plot_source.py','plot_helper.py')]
    manifest['files'].update({str(p.relative_to(run)):digest(p.read_bytes()) for p in paths})
    (provenance/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
