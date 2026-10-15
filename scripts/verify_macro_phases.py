"""Direct-slice verification of saved macro phases; no fitting or validation reads."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pp_data.config import PROJECT
from scripts.analyze_base_regimes import digest, install_guards
from scripts.analyze_macro_phases import RESULT, load_inputs


def verify(root):
    guards = install_guards()
    manifest = json.loads((root/'provenance/manifest.json').read_text())
    run = json.loads((root/'provenance/run.json').read_text())
    for name, sha in manifest['outputs'].items():
        assert digest(root/name) == sha, name
    for name, sha in manifest['code'].items():
        assert digest(PROJECT/name) == sha, name
    for name, sha in run['protected_hashes'].items():
        assert digest(PROJECT/name) == sha, name
    calendar_path = PROJECT/run['calendar_path']
    assert digest(calendar_path) == run['calendar_sha256']
    calendar = json.loads(calendar_path.read_text())
    for e in calendar['events']:
        local = pd.Timestamp(e['local_date']+'T'+e['local_time'], tz='America/New_York')
        assert local.tz_convert('UTC') == pd.Timestamp(e['event_time_utc'])
    frame, market, _, events, info, _ = load_inputs()
    assert info['prefix_sha256'] == run['old_market']['prefix_sha256']
    phases = {'pre60': (-60, 0), 'post0_15': (0, 15),
              'post15_60': (15, 60), 'post60_240': (60, 240)}
    prices = market.set_index('open_time_utc').close.sort_index()
    prices = prices.reindex(pd.date_range(prices.index.min(), prices.index.max(), freq='min'))
    returns = np.log(prices/prices.shift(1))

    def close(actual, expected):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-7, equal_nan=True)

    def direct_window(start, end):
        idx = pd.date_range(start.ceil('min'), end.floor('min'), freq='min', inclusive='left')
        r = returns.reindex(idx).to_numpy()
        if not len(r) or not np.isfinite(r).all():
            return np.full(4, np.nan)
        return np.array([np.sqrt(np.mean(r*r)), np.sqrt(np.mean(np.maximum(r, 0)**2)),
                         np.sqrt(np.mean(np.minimum(r, 0)**2)), r.sum()])*10000

    def in_period(table, period):
        if period == 'warmup': return table.loc[table.month.lt('2026-04')]
        if period == 'development': return table.loc[table.month.between('2026-04', '2026-06')]
        if period == 'study': return table.loc[table.month.ge('2026-04')]
        return table.loc[table.month.eq(period)]

    def read(name): return pd.read_csv(root/'metrics'/f'{name}.csv')

    mr = read('market_phases'); control_count = 0
    for row in mr.itertuples():
        t = pd.Timestamp(row.time); lo, hi = phases[row.phase]
        start = t+pd.Timedelta(minutes=lo); end = t+pd.Timedelta(minutes=hi)
        observed = direct_window(start, end)
        close(observed, [getattr(row, m+'_observed') for m in ['total', 'call', 'put', 'return']])
        assert row.complete == bool(np.isfinite(observed).all())
        controls = [] if pd.isna(row.control_times) else [pd.Timestamp(v) for v in row.control_times.split('|')]
        assert len(controls) == row.controls
        for c in controls:
            local, anchor = c.tz_convert('America/New_York'), t.tz_convert('America/New_York')
            assert local.time() == anchor.time() and local.weekday() == anchor.weekday()
            assert abs((local.date()-anchor.date()).days) in (7, 14)
            a, b = c+pd.Timedelta(minutes=lo), c+pd.Timedelta(minutes=hi)
            assert not ((events.time-pd.Timedelta(minutes=60) < b) &
                        (events.time+pd.Timedelta(minutes=240) > a)).any()
        means = np.mean([direct_window(c+pd.Timedelta(minutes=lo), c+pd.Timedelta(minutes=hi))
                         for c in controls], axis=0) if len(controls) >= 2 else np.full(4, np.nan)
        close(means, [getattr(row, m+'_control') for m in ['total', 'call', 'put', 'return']])
        for j, m in enumerate(['total', 'call', 'put', 'return']):
            close(observed[j]-means[j], getattr(row, m+'_difference'))
            if m != 'return': close(observed[j]/means[j], getattr(row, m+'_ratio'))
        control_count += len(controls)
    ms = read('market_summary')
    for row in ms.itertuples():
        g = in_period(mr, row.period)
        g = g.loc[g.phase.eq(row.phase)]
        if row.family != 'all': g = g.loc[g.family.eq(row.family)]
        v = g[row.metric+'_'+row.statistic].dropna()
        assert row.calendar_events == len(g) and row.events == len(v)
        close([v.mean(), v.median()], [row.mean, row.median])
        close((v > (1 if row.statistic == 'ratio' else 0)).mean(), row.positive_fraction)

    planned = frame.created+pd.to_timedelta(frame.remaining, unit='s')
    event_map = {e.event_id: e for e in events.itertuples()}
    masks = {}
    for e in events.itertuples():
        delta = (frame.created-e.time).dt.total_seconds()/60
        for phase, (lo, hi) in phases.items():
            masks[(e.event_id, phase)] = (delta >= lo) & (delta < hi)
        masks[(e.event_id, 'cross_primary')] = (frame.created < e.time) & (planned >= e.time)
    orders, crossing = read('order_phases'), read('cross_event_support')
    for row in pd.concat([orders, crossing]).itertuples():
        e = event_map[row.event_id]
        g = frame.loc[masks[(row.event_id, row.phase)] & frame.side.eq(row.side)]
        b = g.loc[g.base.notna()]; err = b.base-b.actual
        close([len(g), len(b), err.mean(), (err**2).mean(), abs(err).mean()],
              [row.risk_orders, row.base_orders, row.bias, row.mse, row.mae])
        assert row.known_orders == g.created.ge(e.known_time).sum()
        assert row.cross_primary_orders == ((g.created < e.time) & (planned.loc[g.index] >= e.time)).sum()
        cash = [(b.quantity*(-err).clip(lower=0)).sum(), (b.quantity*err.clip(lower=0)).sum(),
                (b.quantity*err).sum()] if len(b) else [np.nan]*3
        close(cash, [row.cash_under, row.cash_over, row.cash_bias])
        future = np.maximum((planned.loc[g.index]-g.created.clip(lower=e.time)).dt.total_seconds()/60, 0)
        cross_post = ((planned.loc[g.index]-e.time).dt.total_seconds()/60).where(
            (g.created < e.time) & (planned.loc[g.index] >= e.time))
        close([g.remaining.mean()/3600, future.mean(), cross_post.mean()],
              [row.remaining_hours_mean, row.future_post_release_minutes_mean, row.cross_post_release_minutes_mean])
    exposure = read('exposure_support')
    for row in exposure.itertuples():
        t = event_map[row.event_id].time; lo, hi = phases[row.phase]
        overlap = (planned.clip(upper=t+pd.Timedelta(minutes=hi))-
                   frame.created.clip(lower=t+pd.Timedelta(minutes=lo))).dt.total_seconds().clip(lower=0)/60
        g = frame.loc[overlap.gt(0) & frame.side.eq(row.side)]
        close([len(g), g.base.notna().sum(), overlap.loc[g.index].sum()],
              [row.risk_orders, row.base_orders, row.overlap_minutes_sum])
    os = read('order_summary')
    for row in os.itertuples():
        g = in_period(orders, row.period)
        g = g.loc[g.side.eq(row.side) & g.phase.eq(row.phase)]
        if row.family != 'all': g = g.loc[g.family.eq(row.family)]
        b = g.loc[g.base_orders.gt(0)]; n = b.base_orders.sum()
        close([len(g), g.risk_orders.gt(0).sum(), len(b), g.base_orders.ge(5).sum(),
               g.risk_orders.sum(), n], [row.calendar_events, row.risk_events, row.base_events,
                                       row.events_ge5_base, row.risk_orders, row.base_orders])
        for m in ['bias', 'mse', 'mae']:
            close([b[m].mean(), (b[m]*b.base_orders).sum()/n if n else np.nan],
                  [getattr(row, m+'_event_equal'), getattr(row, m+'_order_equal')])
    history = read('historical_support')
    for row in history.itertuples():
        origin = pd.Timestamp(row.origin); nr = nb = er = eb = 0
        for e in events.itertuples():
            if e.time >= origin: continue
            mask = masks[(e.event_id, row.phase)] & frame.created.lt(origin) & frame.closed.lt(origin)
            mask &= frame.created.ge(e.known_time)
            if row.side != 'all': mask &= frame.side.eq(row.side)
            r, b = int(mask.sum()), int((mask & frame.base.notna()).sum())
            nr += r; nb += b; er += r > 0; eb += b > 0
        close([nr, nb, er, eb], [row.risk_orders, row.base_orders, row.risk_events, row.base_events])
    matched, match_summary = read('matched_phases'), read('matched_summary')
    for row in matched.itertuples():
        expected = (masks[(row.event_id, row.phase)] & frame.side.eq(row.side) & frame.base.notna()).sum()
        assert row.treated_orders == expected
        assert row.treated_matched <= row.treated_orders
    for row in match_summary.itertuples():
        g = in_period(matched, row.period)
        g = g.loc[g.side.eq(row.side) & g.phase.eq(row.phase) & g.control.eq(row.control)]
        n = g.treated_orders.sum(); v = g[row.metric+'_difference'].dropna()
        close([len(g), g.treated_orders.gt(0).sum(), n, g.treated_matched.sum(),
               g.treated_matched.sum()/n if n else np.nan, len(v), v.mean()],
              [row.calendar_events, row.eligible_events, row.treated_orders, row.treated_matched,
               row.coverage, row.events, row.mean])
    for name, sha in run['protected_hashes'].items(): assert digest(PROJECT/name) == sha, name
    assert guards == {'official_validation_open_attempts': 0, 'model_fit_predict_calls': 0}
    result = {'calendar_timestamps': len(calendar['events']), 'market_phase_windows': len(mr),
              'market_control_windows': control_count, 'market_summary_rows': len(ms),
              'quote_phase_cells': len(orders), 'cross_event_cells': len(crossing),
              'exposure_cells': len(exposure), 'order_summary_rows': len(os),
              'historical_support_rows': len(history), 'matched_cells': len(matched),
              'matched_summary_rows': len(match_summary), 'protected_files': len(run['protected_hashes']),
              'code_calendar_output_checksums': True, 'guards': guards,
              'verifier_sha256': digest(Path(__file__))}
    path = root/'provenance/verification.json'; path.write_text(json.dumps(result, indent=2)+'\n')
    manifest['outputs']['provenance/verification.json'] = digest(path)
    (root/'provenance/manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, default=RESULT)
    with threadpool_limits(limits=1): verify(parser.parse_args().output)
