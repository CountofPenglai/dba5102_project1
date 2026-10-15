"""Versioned public schedules, isolated from the frozen preprocessing package."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

NY = ZoneInfo('America/New_York')
FAMILIES = ('CPI', 'NFP', 'FOMC')


def timestamp(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('An aware timestamp is required')
    return dt


def utc(value):
    return value.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def build_calendar(bundle):
    if bundle['timezone'] != 'America/New_York' or bundle['schema_version'] != 1:
        raise ValueError('Unsupported calendar schema')
    asof = timestamp(bundle['as_of_utc'])
    source_ids = [s['source_id'] for s in bundle['sources']]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError('Duplicate sources')
    rows, seen = [], set()
    for raw in bundle['records']:
        row = raw.copy()
        if row['event_id'] in seen or row['family'] not in FAMILIES:
            raise ValueError('Duplicate event or invalid family')
        seen.add(row['event_id'])
        for key in ['source_id', 'announcement_source_id']:
            if row.get(key) is not None and row[key] not in source_ids:
                raise ValueError('Unknown source')
        local = datetime.fromisoformat(row['local_date']+'T'+row['local_time']).replace(tzinfo=NY)
        row['event_time_local'] = local.isoformat()
        row['event_time_utc'] = utc(local)
        if row['knowledge_basis'] == 'current_snapshot':
            known = asof
        else:
            if not row.get('known_by_local') or not row.get('announcement_source_id'):
                raise ValueError('Historical knowledge needs dated source evidence')
            known = datetime.fromisoformat(row['known_by_local']).replace(tzinfo=NY)
        if known > asof:
            raise ValueError('Evidence cannot postdate this snapshot')
        row['known_by_utc'] = utc(known)
        row['observed_at_utc'] = utc(asof)
        # "Historical" means a past scheduled timestamp, not proof of actual delivery.
        row['status'] = ('historical_scheduled' if local < asof else
                         'tentative' if row['family'] == 'FOMC' else 'scheduled')
        rows.append(row)
    rows.sort(key=lambda r: (r['event_time_utc'], r['event_id']))
    return {'schema_version': 1, 'as_of_utc': utc(asof), 'timezone': 'America/New_York',
            'coverage': bundle['coverage'], 'sources': bundle['sources'], 'events': rows,
            'input_sha256': hashlib.sha256(json.dumps(bundle,sort_keys=True).encode()).hexdigest()}


def events_asof(calendar, at, include_press=False):
    """Evidence filter only; use query_events for a coverage-checked feature query."""
    at = timestamp(at)
    return [r for r in calendar['events'] if timestamp(r['known_by_utc']) <= at
            and (include_press or r['kind'] != 'press_conference')]


def assert_coverage(calendar, start, end, families=FAMILIES):
    start, end = timestamp(start), timestamp(end)
    if end <= start:
        raise ValueError('Invalid interval')
    for family in families:
        coverage = calendar['coverage'][family]
        lo = datetime.fromisoformat(coverage['start']).replace(tzinfo=NY)
        hi = datetime.fromisoformat(coverage['end_exclusive']).replace(tzinfo=NY)
        if start < lo or end > hi:
            raise ValueError(f'Unknown calendar coverage for {family}; refresh required')


def query_events(calendar, start, end, as_of, families=FAMILIES, include_press=False):
    """Return known events in (start, end], rejecting unknown rather than encoding zero.

    Dated prior-announcement evidence can support a retrospective event window.
    It cannot certify absence in an empty historical window or reconstruct revisions.
    """
    assert_coverage(calendar, start, end, families)
    lo, hi, at = timestamp(start), timestamp(end), timestamp(as_of)
    rows = [r for r in calendar['events'] if r['family'] in families
            and (include_press or r['kind'] != 'press_conference')
            and lo < timestamp(r['event_time_utc']) <= hi]
    unknown = [r['event_id'] for r in rows if timestamp(r['known_by_utc']) > at]
    if unknown or (not rows and at < timestamp(calendar['as_of_utc'])):
        raise ValueError('Unknown historical calendar information: '+', '.join(unknown))
    return rows


def select_snapshot(destination, as_of):
    """Latest immutable publication available at as_of; never fall forward."""
    at = timestamp(as_of); candidates = []
    for path in (Path(destination)/'snapshots').glob('*.json'):
        calendar = json.loads(path.read_text())
        published = timestamp(calendar['as_of_utc'])
        if published <= at:
            candidates.append((published, calendar))
    if not candidates:
        raise ValueError('No snapshot existed at the requested time')
    return max(candidates, key=lambda item: item[0])[1]


def publish_calendar(bundle, destination):
    calendar = build_calendar(bundle)
    destination = Path(destination)
    name = timestamp(calendar['as_of_utc']).strftime('%Y%m%dT%H%M%SZ')+'.json'
    payload = json.dumps(calendar, ensure_ascii=False, indent=2)+'\n'
    path = destination/'snapshots'/name
    if path.exists():
        raise FileExistsError('Immutable snapshot already exists: '+str(path))
    current = destination/'calendar.json'
    if current.exists():
        previous = json.loads(current.read_text())
        if timestamp(previous['as_of_utc']) >= timestamp(calendar['as_of_utc']):
            raise ValueError('New publication must advance the snapshot time')
        if {r['event_id'] for r in previous['events']} - {r['event_id'] for r in calendar['events']}:
            raise ValueError('Cannot silently delete previously published events')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(payload)
    temporary = destination/'calendar.json.tmp'
    temporary.write_text(payload)
    temporary.replace(current)
    return path
