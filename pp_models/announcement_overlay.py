"""Small additive calendar adjustments, with no fitting or outcome inputs.

Evidence controls availability. Time to release controls magnitude. Unknown
events never affect predictions, event grouping, or the known-event ledger.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pp_calendar.calendar import assert_coverage

DIAGNOSTIC_COLUMNS = ['multiplier', 'raw_adjustment', 'adjustment', 'active_events',
                      'material_events', 'capped', 'status']
LEDGER_COLUMNS = ['order_id', 'quote_time', 'event_id', 'family', 'anchor_time',
                  'known_by_utc', 'phase', 'weight', 'raw_adjustment', 'applied_adjustment']


def aware_timestamp(value):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('Calendar evidence needs an aware timestamp')
    return stamp.tz_convert('UTC')


@dataclass(frozen=True)
class AnnouncementOverlay:
    # A separate report experiment, not an automatic layer in the delivered blend.
    per_event_cap: float = .01
    total_cap: float = .05
    pre_half_life_hours: float = 24.
    post_minutes: float = 60.

    def __post_init__(self):
        values = [self.per_event_cap, self.total_cap, self.pre_half_life_hours, self.post_minutes]
        if not np.isfinite(values).all() or not all(v > 0 for v in values) or not self.per_event_cap <= self.total_cap <= 1:
            raise ValueError('Invalid bounded announcement rule')

    def predict_diagnostics(self, quotes, calendar):
        """Return per-quote multipliers and a ledger containing known contributions.

        Input is quote time and direction only. This is a quote-time calendar
        overlay, not a measure of whether the remaining contract crosses release.
        The two directions deliberately have identical rules in this version.
        """
        if (not isinstance(quotes, pd.DataFrame) or set(quotes) != {'created', 'option_type'}
                or not quotes.index.is_unique or not quotes.option_type.isin([0, 1]).all()
                or not isinstance(quotes.created.dtype, pd.DatetimeTZDtype) or quotes.created.isna().any()):
            raise ValueError('Announcement quote whitelist/time/direction violated')
        published = aware_timestamp(calendar['as_of_utc'])
        events, seen = [], set()
        for row in calendar['events']:
            if row['event_id'] in seen:
                raise ValueError('Duplicate calendar event')
            seen.add(row['event_id'])
            family, kind = row['family'], row['kind']
            if (family not in ('CPI', 'NFP', 'FOMC') or
                    kind not in (('statement', 'press_conference') if family == 'FOMC' else ('release',))):
                raise ValueError('Unsupported calendar event')
            # Missing or retrospective-only evidence is unknown, even after release.
            if row.get('known_by_utc') is None:
                continue
            time = aware_timestamp(row['event_time_utc'])
            known = aware_timestamp(row['known_by_utc'])
            if known > published:
                raise ValueError('Evidence postdates calendar snapshot')
            if known >= time:
                continue
            key = ('US_FOMC_'+time.tz_convert('America/New_York').strftime('%Y-%m-%d')
                   if family == 'FOMC' else row['event_id'])
            events.append((key, family, time, known))

        rows, ledger = [], []
        for q in quotes.itertuples():
            at = q.created
            out = dict(multiplier=1., raw_adjustment=0., adjustment=0., active_events=0,
                       material_events=0, capped=False,
                       status='unknown_no_active_evidence' if at < published else 'no_active_known_event')
            try:
                assert_coverage(calendar, (at-pd.Timedelta(minutes=self.post_minutes)).isoformat(),
                                (at+pd.Timedelta(microseconds=1)).isoformat())
            except ValueError:
                out['status'] = 'outside_coverage'; rows.append(out); continue
            contributions = {}
            for key, family, time, known in events:
                # Availability follows the evidence timestamp, not today's knowledge
                # that the event eventually occurred.
                if known > at:
                    continue
                until = (time-at).total_seconds()
                # Approach the per-event cap exponentially before release, then
                # decay linearly to zero over the fixed post-release interval.
                weight = (float(np.exp2(-until/(3600*self.pre_half_life_hours))) if until > 0
                          else max(0., 1+until/(60*self.post_minutes)))
                if weight <= 0:
                    continue
                # FOMC anchors are alternatives within one event, never additive.
                item = dict(order_id=q.Index, quote_time=at.isoformat(), event_id=key, family=family,
                            anchor_time=time.isoformat(), known_by_utc=known.isoformat(),
                            phase='pre' if until > 0 else 'post', weight=weight,
                            raw_adjustment=self.per_event_cap*weight)
                previous = contributions.get(key)
                if previous is None or item['weight'] > previous['weight']:
                    contributions[key] = item
            raw = sum(c['raw_adjustment'] for c in contributions.values())
            if contributions:
                # Independent events add percentage points, never compound multipliers.
                adjustment = min(self.total_cap, raw)
                for item in contributions.values():
                    # Preserve an exact additive attribution when the total cap binds.
                    ledger.append(dict(item, applied_adjustment=item['raw_adjustment']*adjustment/raw))
                out.update(multiplier=1+adjustment, raw_adjustment=raw, adjustment=adjustment,
                           active_events=len(contributions),
                           material_events=sum(c['raw_adjustment'] >= .001 for c in contributions.values()),
                           capped=raw > self.total_cap, status='active_known_events')
            rows.append(out)
        return (pd.DataFrame(rows, index=quotes.index, columns=DIAGNOSTIC_COLUMNS),
                pd.DataFrame(ledger, columns=LEDGER_COLUMNS))
