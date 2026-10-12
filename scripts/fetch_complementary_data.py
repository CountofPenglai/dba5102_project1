"""Fetch only public pre-July research inputs; never contacts private APIs.

python -m scripts.fetch_complementary_data --destination outputs/external_data/complementary_20261009
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import pandas as pd

START=pd.Timestamp('2026-04-29T00:00:00Z')
END=pd.Timestamp('2026-07-01T00:00:00Z')


def fetch_one(job):
    url,path,checksum=job
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=path.read_bytes() if path.exists() else urlopen(url,timeout=40).read()
    digest=hashlib.sha256(payload).hexdigest()
    if checksum:
        check_path=Path(str(path)+'.CHECKSUM')
        check=check_path.read_bytes() if check_path.exists() else urlopen(url+'.CHECKSUM',timeout=40).read()
        if digest!=check.decode().split()[0]:raise ValueError(f'Checksum mismatch: {url}')
        check_path.write_bytes(check)
    else:
        body=json.loads(payload)
        if 'error' in body or not body.get('result',{}).get('data'):
            raise ValueError(f'Empty/error DVOL response: {url}')
        if body['result'].get('continuation') is not None:
            raise ValueError('DVOL week requires pagination; do not silently truncate')
    path.write_bytes(payload)
    return {'url':url,'path':str(path),'sha256':digest,'bytes':len(payload),'official_checksum_verified':checksum}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args();root=args.destination.resolve();jobs=[]
    for day in pd.date_range(START,END,inclusive='left',freq='D'):
        for kind in ['metrics','bookDepth']:
            name=f'BTCUSDC-{kind}-{day:%Y-%m-%d}.zip'
            jobs.append((f'https://data.binance.vision/data/futures/um/daily/{kind}/BTCUSDC/{name}',root/kind/name,True))
    for month in ['2026-04','2026-05','2026-06']:
        name=f'BTCUSDC-fundingRate-{month}.zip'
        jobs.append((f'https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDC/{name}',root/'fundingRate'/name,True))
    for day in pd.date_range(START,END,inclusive='left',freq='7D'):
        last=min(day+pd.Timedelta(days=7),END)-pd.Timedelta(milliseconds=1)
        url=('https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC&resolution=3600'
             f'&start_timestamp={int(day.timestamp()*1000)}&end_timestamp={int(last.timestamp()*1000)}')
        jobs.append((url,root/'dvol'/f'BTC-DVOL-{day:%Y-%m-%d}.json',False))
    with ThreadPoolExecutor(max_workers=4) as executor:
        records=[]
        for record in executor.map(fetch_one,jobs):
            records.append(record)
            if len(records)%20==0:print(f'Checked {len(records)}/{len(jobs)} public input files',flush=True)
    for r in records:r['path']=str(Path(r['path']).relative_to(root))
    manifest={'downloaded_at_utc':datetime.now(timezone.utc).isoformat(),'start':START.isoformat(),
       'end_exclusive':END.isoformat(),'funding_warmup':'April monthly file; all events before July',
       'availability_caveat':'Historical archive; event timestamps do not certify real-time publication or receipt',
       'files':records}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    alignment=[]
    for date in ['2026-06-01','2026-06-24','2026-06-25']:
        name=f'BTCUSDC-1m-{date}.zip'
        url=f'https://data.binance.vision/data/futures/um/daily/klines/BTCUSDC/1m/{name}'
        record=fetch_one((url,root/'alignment_check'/name,True))
        record['path']=name;record['date']=date;alignment.append(record)
    (root/'alignment_check/manifest.json').write_text(json.dumps(alignment,indent=2)+'\n')
    print(json.dumps({'files':len(records),'bytes':sum(r['bytes'] for r in records),'destination':str(root)}))


if __name__=='__main__':main()
