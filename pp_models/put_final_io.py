"""Separate authorized post-development access; existing development readers unchanged."""
import csv
from datetime import datetime
import hashlib
from io import BytesIO
from pathlib import Path

import pandas as pd

from pp_audit.market import audit_market
from pp_data.config import load_config


def read_test_market_prefix(path):
    end = pd.Timestamp(load_config()['splits']['test']['end'])
    # At the partial final minute, only the preceding completed candle is usable.
    candle_boundary = end.floor('min')
    payload, digest, count, previous, boundary = bytearray(), hashlib.sha256(), 0, None, None
    with Path(path).open('rb') as stream:
        header = stream.readline()
        columns = next(csv.reader([header.decode('utf-8-sig')]))
        time_index = columns.index('open_time_utc')
        payload.extend(header); digest.update(header)
        for line in stream:
            fields = next(csv.reader([line.decode('utf-8')]))
            stamp = datetime.fromisoformat(fields[time_index].replace('Z', '+00:00'))
            if stamp.tzinfo is None or (previous is not None and stamp < previous):
                raise ValueError('Test market prefix is not chronological/timezone-aware')
            if stamp >= candle_boundary:
                boundary = stamp.isoformat(); break
            previous = stamp; payload.extend(line); digest.update(line); count += 1
    market = audit_market(pd.read_csv(BytesIO(payload))).usable
    if market.empty or not market.close_time_utc.lt(end).all():
        raise ValueError('Invalid completed test market prefix')
    return market, {'source_path': str(Path(path).resolve()), 'cutoff_exclusive': end.isoformat(),
        'completed_candle_open_cutoff': candle_boundary.isoformat(), 'prefix_sha256': digest.hexdigest(),
        'prefix_rows': count, 'usable_rows': len(market), 'boundary_timestamp_only': boundary,
        'last_usable_close_utc': market.close_time_utc.max().isoformat()}
