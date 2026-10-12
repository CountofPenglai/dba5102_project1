"""Plot fixed signal families from checked aggregate tail-lift statistics."""
import argparse
import csv
import io
import json
from pathlib import Path
import numpy as np
from plot_official_base_payout import PROJECT,plt,chinese_font,digest

FEATURES=[('market_trades_per_min_15m','市场成交频率 · 15m'),
    ('market_trades_per_min_60m','市场成交频率 · 60m'),
    ('platform_put_count_15m','产品 Put 购买数 · 15m'),
    ('platform_put_count_60m','产品 Put 购买数 · 60m'),
    ('platform_call_count_60m','产品 Call 购买数 · 60m'),
    ('platform_total_count_60m','产品总购买数 · 60m'),
    ('market_vol_change_60m','波动增量 · 60m'),
    ('market_down_rms_60m','下行波动 · 60m'),
    ('market_mean_trade_quote_60m','市场平均单笔成交额 · 60m')]
PERIODS=['all_june','early','late','without_june24_25']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=PROJECT/'outputs/diagnostics/payout_signals_20261009')
    run=parser.parse_args().run_dir
    manifest=json.loads((run/'provenance/manifest.json').read_text());name='metrics/tail_alerts.csv'
    payload=(run/name).read_bytes()
    if manifest['status']!='complete' or digest(payload)!=manifest['files'][name]:raise ValueError('Unverified signal study')
    rows=list(csv.DictReader(io.StringIO(payload.decode())))
    lookup={(r['direction'],r['feature'],r['period']):r for r in rows if r['signal_group']=='high'}
    plt.rcParams.update({'font.family':chinese_font(),'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path'})
    fig,axes=plt.subplots(1,2,figsize=(13.5,7.8),dpi=170)
    fig.subplots_adjust(left=.245,right=.88,top=.79,bottom=.22,wspace=.13)
    fig.suptitle('交易活跃时，高额赔付是否更常见？',x=.055,y=.96,ha='left',fontsize=21,color='#17263C')
    fig.text(.055,.902,'数值为高信号组的高额赔付发生率 ÷ 同方向总体发生率；1 表示没有提升，2 表示发生率为两倍',color='#526176')
    fig.text(.055,.858,'高信号组：超过预测日前历史 q80；高额赔付：超过同方向、期限历史每 BTC 赔付 q90',color='#526176',fontsize=10)
    for ax,side,label in zip(axes,['1','0'],['Put · 915 笔','Call · 1,343 笔']):
        data=np.array([[float(lookup[(side,feature,period)]['lift']) for period in PERIODS] for feature,_ in FEATURES])
        with np.errstate(divide='ignore'):colors=np.log2(np.maximum(data,1e-6))
        im=ax.imshow(colors,cmap='RdBu_r',vmin=-2,vmax=2,aspect='auto')
        for i in range(len(FEATURES)):
            for j in range(4):
                value=data[i,j];text='—' if not np.isfinite(value) else f'{value:.2f}'
                ax.text(j,i,text,ha='center',va='center',fontsize=11,color='white' if abs(colors[i,j])>1.3 else '#17263C')
        ax.set_title(label,pad=17,fontsize=14)
        ax.set_xticks(range(4),['六月全部','前 20 日','后 10 日','去掉 24–25 日'])
        ax.tick_params(axis='both',length=0,pad=10)
        ax.set_yticks(range(len(FEATURES)),[label for _,label in FEATURES] if side=='1' else ['']*len(FEATURES))
        for spine in ax.spines.values():spine.set_visible(False)
    cax=fig.add_axes([.907,.22,.018,.57]);bar=fig.colorbar(im,cax=cax,ticks=[-2,-1,0,1,2])
    bar.ax.set_yticklabels(['0.25×','0.5×','1×','2×','4×']);bar.set_label('尾部发生率倍数')
    fig.text(.055,.115,'Put 信号的前后段表现差异较大；Call 的交易活动和波动增量关联更稳定。倍数不代表赔付金额或定价倍率。',fontsize=10,color='#526176')
    fig.text(.055,.068,'这是 41 项预设信号中的代表性汇总；完整统计保留全部候选。重叠窗口、日期聚集和样本筛选限制推断，未做因果解释。',fontsize=9,color='#64748B')
    fig.text(.055,.035,'仅六月探索，未使用七月；阶段和剔除峰值日的比较为敏感性诊断，不是独立确认。',fontsize=9,color='#64748B')
    paths=[]
    for ext in ['png','svg']:
        p=run/f'figures/signal_tail_lift.{ext}';fig.savefig(p,facecolor='white');paths.append(p)
    plt.close(fig)
    info={'source':name,'source_sha256':digest(payload),'script_sha256':digest(Path(__file__).read_bytes()),
          'helper_sha256':digest(Path(__file__).with_name('plot_official_base_payout.py').read_bytes()),
          'files':{str(p.relative_to(run)):digest(p.read_bytes()) for p in paths}}
    (run/'provenance/plot.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
