"""Calendar-only exposure remaining after quote time; Base and outcomes are forbidden."""
import numpy as np
import pandas as pd
from pp_calendar.calendar import assert_coverage

BASIS = ['pre_exposure', 'cross_release', 'post_exposure']
FEATURE_COLUMNS = ['option_type', *BASIS, 'event_id', 'event_time', 'event_month', 'calendar_status']


def kernel_integral(start, end, anchors, cutoff):
    """Integral of the upper envelope; a later release resets the decaying kernel."""
    value = 0.
    rate = np.log(2.)/60.
    for i, anchor in enumerate(anchors):
        stop = min(cutoff, anchors[i+1] if i+1 < len(anchors) else cutoff)
        lo, hi = max(start, anchor), min(end, stop)
        if hi > lo:
            a, b = (lo-anchor).total_seconds()/60, (hi-anchor).total_seconds()/60
            value += (np.exp(-rate*a)-np.exp(-rate*b))/rate
    return value


def build_features(quotes, calendar):
    if (set(quotes) != {'created', 'remaining_seconds', 'option_type'} or not quotes.index.is_unique
            or quotes.created.dt.tz is None or quotes.created.isna().any()
            or not quotes.option_type.isin([0, 1]).all()):
        raise ValueError('Calendar quote whitelist or time/direction alignment violated')
    events = calendar['events']; groups = []
    for row in events:
        if row['kind'] == 'press_conference': continue
        time = pd.Timestamp(row['event_time_utc']); parts = [row]
        if row['family'] == 'FOMC':
            parts += [p for p in events if p['family'] == 'FOMC' and p['kind'] == 'press_conference'
                      and pd.Timestamp(p['event_time_utc']).date() == time.date()]
        anchors = sorted(pd.Timestamp(p['event_time_utc']) for p in parts)
        known = max(pd.Timestamp(p['known_by_utc']) for p in parts)
        groups.append((row['event_id'], time, anchors, known))
    rows = []
    for q in quotes.itertuples():
        out = {'option_type': int(q.option_type), **dict.fromkeys(BASIS, 0.), 'event_id': '',
               'event_time': pd.NaT, 'event_month': '', 'calendar_status': 'no_identified_event'}
        if not np.isfinite(q.remaining_seconds) or q.remaining_seconds <= 0:
            out['calendar_status'] = 'invalid_contract'; rows.append(out); continue
        start = q.created; end = start+pd.Timedelta(seconds=float(q.remaining_seconds))
        try:
            assert_coverage(calendar, (start-pd.Timedelta(minutes=240)).isoformat(), end.isoformat())
        except ValueError:
            out['calendar_status'] = 'outside_coverage'; rows.append(out); continue
        candidates = [g for g in groups if start < g[1]+pd.Timedelta(minutes=240)
                      and end > g[1]-pd.Timedelta(minutes=60)]
        if len(candidates) != 1:
            if candidates: out['calendar_status'] = 'multiple_events'
            rows.append(out); continue
        event_id, time, anchors, known = candidates[0]
        out.update(event_id=event_id, event_time=time, event_month=time.strftime('%Y-%m'))
        if known > start:
            out['calendar_status'] = 'unknown_evidence'; rows.append(out); continue
        cutoff = time+pd.Timedelta(minutes=240)
        pre = max((min(end, time)-max(start, time-pd.Timedelta(minutes=60))).total_seconds()/3600, 0.)
        cross = float(any(start < t <= end for t in anchors))
        total = kernel_integral(time, cutoff, anchors, cutoff)
        post = kernel_integral(start, end, anchors, cutoff)/total
        out.update(pre_exposure=pre, cross_release=cross, post_exposure=post, calendar_status='known_event')
        rows.append(out)
    result = pd.DataFrame(rows, index=quotes.index, columns=FEATURE_COLUMNS)
    result['event_time'] = pd.to_datetime(result.event_time, utc=True)
    return result


def pool_features(pool, calendar):
    return build_features(pd.DataFrame({'created':pool.metadata.created_at_utc,
        'remaining_seconds':pool.X.remaining_seconds,'option_type':pool.X.option_type}),calendar)
