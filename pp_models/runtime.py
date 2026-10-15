"""Report run storage, input guard and historical market reader."""
import csv
from datetime import datetime
import hashlib
from io import BytesIO
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile
import pandas as pd
from pp_audit.market import audit_market
from pp_data.config import PROJECT, DEFAULT_DATA, DEFAULT_OUTPUT, load_protocol
from pp_data.io import canonical_identity

def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def run_destination(root, run_id, dataset_dir=DEFAULT_OUTPUT):
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", run_id):
        raise ValueError("run-id must start with a lowercase letter and contain only letters, digits, _ or -")
    root = Path(root).resolve()
    protected = [DEFAULT_DATA, Path(dataset_dir), PROJECT / "pp_data", PROJECT / "pp_audit",
                 PROJECT / "pp_models", PROJECT / "config", PROJECT / "docs"]
    for protected_dir in protected:
        protected_dir = protected_dir.resolve()
        if root == protected_dir or protected_dir in root.parents or root in protected_dir.parents:
            raise ValueError("Run output overlaps source, configuration or canonical data")
    destination = root / run_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Run already exists: {destination}; use a new run-id")
    return destination


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(canonical_json(value))
            stream.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_tables(root,phase,tables):
    dest=root/'metrics'/phase;dest.mkdir(parents=True,exist_ok=True)
    for name,table in tables.items():table.to_csv(dest/(name+'.csv'),index=False,float_format='%.17g')


def read_market_prefix(path):
    end = pd.Timestamp(load_protocol()['validation_end'])
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
                raise ValueError('Report market prefix is not chronological/timezone-aware')
            if stamp >= candle_boundary:
                boundary = stamp.isoformat(); break
            previous = stamp; payload.extend(line); digest.update(line); count += 1
    market = audit_market(pd.read_csv(BytesIO(payload))).usable
    if market.empty or not market.close_time_utc.lt(end).all():
        raise ValueError('Invalid completed report market prefix')
    return market, {'source_path': str(Path(path).resolve()), 'cutoff_exclusive': end.isoformat(),
        'completed_candle_open_cutoff': candle_boundary.isoformat(), 'prefix_sha256': digest.hexdigest(),
        'prefix_rows': count, 'usable_rows': len(market), 'boundary_timestamp_only': boundary,
        'last_usable_close_utc': market.close_time_utc.max().isoformat()}



class AccessGuard:
    """The official unlabeled holdout is outside this report's data scope."""
    def __init__(self):
        self.official_attempts = 0

    def check(self, event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes)): return
        path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0])
        if path.name == 'pp_transactions_validation.csv':
            self.official_attempts += 1
            raise PermissionError('New official validation is forbidden')


def environment_snapshot():
    result = {'python': platform.python_version()}
    for package in ('numpy', 'pandas', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl'):
        result[package] = importlib.metadata.version(package)
    return result
