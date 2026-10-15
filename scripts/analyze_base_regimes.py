"""Read-only Base diagnostics on frozen April-July orders; no model fitting.

Only aggregated statistics are persisted. Official validation access and model
fit/predict calls are blocked. Market reads stop at the old July boundary.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pp_data.config import DEFAULT_DATA, DEFAULT_OUTPUT, PROJECT
from pp_data.features import BASE_FEATURE
from pp_models.data import load_pool
from pp_models.directional_features import augment_pool
from pp_models.put_final_io import read_test_market_prefix

START = pd.Timestamp('2026-04-01T00:00:00Z')
END = pd.Timestamp('2026-07-31T17:11:48Z')
MONTHS = ['2026-04', '2026-05', '2026-06', '2026-07']
REPEATS, SEED = 2000, 20261014
SINGLES = ['tenor_hours', 'position_bin', 'volatility_bin', 'trend_bin', 'relative_bin']
PAIRS = [('tenor_hours', 'position_bin'), ('tenor_hours', 'volatility_bin'),
         ('position_bin', 'volatility_bin'), ('volatility_bin', 'trend_bin')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_guards():
    counts = {'official_validation_open_attempts': 0, 'model_fit_predict_calls': 0}

    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            if Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).name == 'pp_transactions_validation.csv':
                counts['official_validation_open_attempts'] += 1
                raise PermissionError('Official validation access is forbidden in this audit')

    def profile(frame, event, arg):
        if event == 'call' and frame.f_code.co_name in {'fit', 'fit_transform', 'partial_fit', 'predict', 'predict_raw', 'predict_components'}:
            module = frame.f_globals.get('__name__', '')
            if module.startswith(('pp_models', 'sklearn')):
                counts['model_fit_predict_calls'] += 1
                raise PermissionError('Model fitting/prediction is forbidden in this audit')

    sys.addaudithook(audit)
    sys.setprofile(profile)
    return counts


def block_weights(month):
    start = pd.Timestamp(month+'-01', tz='UTC')
    days = pd.date_range(start, start+pd.offsets.MonthBegin(1), inclusive='left')
    rng = np.random.default_rng(SEED+int(month[-2:]))
    starts = rng.integers(0, len(days), size=(REPEATS, int(np.ceil(len(days)/7))))
    indices = ((starts[:, :, None]+np.arange(7)) % len(days)).reshape(REPEATS, -1)[:, :len(days)]
    weights = np.zeros((REPEATS, len(days)), dtype=int)
    np.add.at(weights, (np.repeat(np.arange(REPEATS), len(days)), indices.ravel()), 1)
    assert (weights.sum(axis=1) == len(days)).all()
    return days, weights


def metrics(frame, calendar=None):
    e = frame[BASE_FEATURE]-frame.actual
    q, b = frame.quantity_btc, frame[BASE_FEATURE]
    result = dict(orders=len(frame), days=frame.day.nunique(),
        supported=len(frame) >= 30 and frame.day.nunique() >= 5,
        actual_mean=frame.actual.mean(), base_mean=b.mean(), mean_error=e.mean(),
        rmse=float(np.sqrt(np.mean(e**2))), mae=abs(e).mean(),
        zero_payout_rate=frame.actual.eq(0).mean(), quantity_btc=q.sum(),
        actual_cash=(q*frame.actual).sum(), base_cash=(q*b).sum(),
        cash_bias=(q*e).sum(), cash_over=(q*e.clip(lower=0)).sum(),
        cash_under=(q*(-e).clip(lower=0)).sum(),
        mse_multiplier_slope=2*np.mean(b*e), mae_multiplier_slope=np.mean(b*np.sign(e)))
    np.testing.assert_allclose(result['cash_bias'], result['cash_over']-result['cash_under'], atol=1e-8)
    if calendar is not None:
        days, weights = calendar
        daily = pd.DataFrame({'day': frame.day, 'error': e}).groupby('day').error.agg(['sum', 'size']).reindex(days, fill_value=0)
        den = weights@daily['size'].to_numpy()
        samples = (weights@daily['sum'].to_numpy())[den > 0]/den[den > 0]
        result['bootstrap_valid'] = len(samples)
        result['mean_error_ci_low'], result['mean_error_ci_high'] = np.quantile(samples, [.025, .975]) if len(samples) >= 1900 else (np.nan, np.nan)
        sums = daily['sum'].to_numpy(); counts = daily['size'].to_numpy()
        leave = [(sums.sum()-sums[j:j+7].sum())/(counts.sum()-counts[j:j+7].sum())
                 for j in range(len(days)-6) if counts.sum()-counts[j:j+7].sum() >= 10]
        result.update(delete_7d_valid=len(leave), delete_7d_bias_min=min(leave) if leave else np.nan,
                      delete_7d_bias_max=max(leave) if leave else np.nan)
    return result


def add_groups(pool):
    frame = pool.X.copy()
    frame['actual'] = pool.y
    frame['created'] = pool.metadata.created_at_utc
    frame['closed'] = pool.metadata.closed_at_utc
    frame['day'] = frame.created.dt.floor('D')
    frame['month'] = frame.created.dt.strftime('%Y-%m')
    frame['side'] = frame.option_type.map({0: 'call', 1: 'put'})
    frame['tenor_hours'] = (frame.product_duration_seconds/3600).map(lambda x: f'{x:g}')
    direction = 1-2*frame.option_type
    frame['position'] = direction*frame.log_moneyness/np.maximum(frame.total_vol_240m*np.sqrt(frame.remaining_seconds/60), 1e-6)
    frame['direction_volatility'] = np.where(frame.option_type.eq(1), frame.down_vol_240m, frame.up_vol_240m)
    frame['relative_volatility'] = np.where(frame.option_type.eq(1), frame.down_vol_60m, frame.up_vol_60m)/np.maximum(frame.direction_volatility, 1e-12)
    frame['trend'] = direction*frame.log_return_60m/np.maximum(frame.total_vol_240m*np.sqrt(60), 1e-6)
    frame['trend_bin'] = pd.cut(frame.trend, [-np.inf, -.5, .5, np.inf], labels=['favorable', 'neutral', 'adverse']).astype(str)
    thresholds = []
    for side in ['call', 'put']:
        ref = frame.loc[frame.side.eq(side) & frame.created.lt(START) & frame.closed.lt(START)]
        if len(ref) < 30:
            raise ValueError('Insufficient pre-April reference support')
        mask = frame.side.eq(side)
        for feature, column in [('position', 'position_bin'), ('direction_volatility', 'volatility_bin'), ('relative_volatility', 'relative_bin')]:
            cut = ref[feature].quantile([1/3, 2/3]).to_numpy()
            if not np.isfinite(cut).all() or cut[0] >= cut[1]:
                raise ValueError('Degenerate reference cuts')
            frame.loc[mask, column] = pd.cut(frame.loc[mask, feature], [-np.inf, *cut, np.inf], labels=['low', 'middle', 'high']).astype(str)
            thresholds.append(dict(side=side, feature=feature, reference_orders=len(ref), reference_days=ref.day.nunique(),
                                   reference_latest_close=ref.closed.max().isoformat(), q33=cut[0], q67=cut[1]))
    return frame, pd.DataFrame(thresholds)


def stability(groups):
    rows = []
    for (side, family, group), part in groups.groupby(['side', 'family', 'group']):
        adequate = part.loc[part.supported]
        four = len(adequate) == 4
        positive, negative = int(adequate.mean_error.gt(0).sum()), int(adequate.mean_error.lt(0).sum())
        same = four and (positive == 4 or negative == 4)
        ci_pos = int((adequate.mean_error_ci_low > 0).sum()); ci_neg = int((adequate.mean_error_ci_high < 0).sum())
        deletion_same = bool(same and (adequate.delete_7d_valid > 0).all() and (
            (adequate.delete_7d_bias_min > 0).all() or (adequate.delete_7d_bias_max < 0).all()))
        slopes = adequate.mse_multiplier_slope
        rows.append(dict(side=side, family=family, group=group, months_present=len(part), supported_months=len(adequate),
            min_month_orders=part.orders.min(), min_month_days=part.days.min(),
            over_months=positive, under_months=negative, same_bias_sign_four_months=same,
            ci_over_months=ci_pos, ci_under_months=ci_neg,
            deletion_same_sign_four_months=deletion_same,
            same_mse_slope_sign_four_months=bool(four and ((slopes > 0).all() or (slopes < 0).all())),
            status='insufficient_support' if not four else ('mixed_bias_sign' if not same else
                ('same_sign_with_four_intervals' if ci_pos == 4 or ci_neg == 4 else 'same_sign_uncertain'))))
    return pd.DataFrame(rows)


def tenor_standardization(frame, warmup):
    """Same broad tenor weights across all three states and all four months."""
    frame = frame.loc[frame.side.eq('put')].copy(); warmup = warmup.loc[warmup.side.eq('put')].copy()
    for part in (frame, warmup):
        part['band'] = pd.cut(part.product_duration_seconds/3600, [0, 1, 4, np.inf], labels=['up_to_1h', '2_to_4h', '8h_plus']).astype(str)
    weights = warmup.band.value_counts(normalize=True)
    rows = []
    for family in ['position_bin', 'volatility_bin', 'trend_bin']:
        levels = sorted(frame[family].unique())
        cells = frame.groupby(['month', family, 'band']).agg(orders=('actual', 'size'), days=('day', 'nunique'))
        shared = []
        for band in sorted(weights.index):
            if all((month, level, band) in cells.index and cells.loc[(month, level, band), 'orders'] >= 20
                   and cells.loc[(month, level, band), 'days'] >= 3 for month in MONTHS for level in levels):
                shared.append(band)
        for (month, level), part in frame.groupby(['month', family]):
            supported = part.loc[part.band.isin(shared)]
            adjusted = np.nan
            if shared:
                residual = supported.assign(error=supported[BASE_FEATURE]-supported.actual).groupby('band').error.mean()
                adjusted = sum(float(weights[band]/weights[shared].sum())*residual[band] for band in shared)
            rows.append(dict(family=family, group=level, month=month, shared_tenor_bands='|'.join(shared),
                             original_orders=len(part), covered_orders=len(supported), coverage=len(supported)/len(part),
                             original_mean_error=(part[BASE_FEATURE]-part.actual).mean(),
                             fixed_tenor_mean_error=adjusted))
    return pd.DataFrame(rows)


def check_previous(frame, reference):
    count = 0
    for phase, months in [('development_daily', MONTHS[:3]), ('test_daily', MONTHS[3:])]:
        previous = pd.read_csv(reference/'metrics'/phase/'summary.csv', dtype={'month': str, 'value': str})
        previous = previous.loc[previous.model.eq('official_raw') & previous.group.isin(['overall', 'direction'])]
        for _, row in previous.iterrows():
            part = frame.loc[frame.month.isin(months)]
            if row.month != 'all': part = part.loc[part.month.eq(row.month)]
            if row.group == 'direction': part = part.loc[part.option_type.eq(int(row.value))]
            got = metrics(part)
            for old, new in [('orders', 'orders'), ('days', 'days'), ('rmse_per_btc', 'rmse'), ('mae_per_btc', 'mae'),
                             ('actual_total_usdc', 'actual_cash'), ('predicted_total_usdc', 'base_cash'),
                             ('total_bias_usdc', 'cash_bias'), ('gross_underprediction_usdc', 'cash_under'),
                             ('gross_overprediction_usdc', 'cash_over')]:
                np.testing.assert_allclose(got[new], row[old], rtol=1e-11, atol=1e-8)
            count += 1
    return count


def analyze(output):
    if output.exists(): raise ValueError('Choose a new dedicated output directory')
    guards = install_guards()
    reference = PROJECT/'outputs/model_studies/put_final_july_20261014'
    old_files = json.loads((reference/'provenance/manifest.json').read_text())['files']
    for name, expected in old_files.items():
        if digest(reference/name) != expected: raise ValueError('Prior result changed: '+name)
    canonical_digest = digest(DEFAULT_OUTPUT/'manifest.json')
    market, market_info = read_test_market_prefix(DEFAULT_DATA/'btc_market_history.csv')
    expected_market = json.loads((reference/'provenance/test_execution.json').read_text())['market_prefix']
    for key in ['prefix_rows', 'prefix_sha256', 'last_usable_close_utc']:
        if market_info[key] != expected_market[key]: raise ValueError('Old market prefix changed: '+key)
    pool, feature_info = augment_pool(load_pool('test', {'cohort': 'benchmark', 'use_base': True}, DEFAULT_OUTPUT), market)
    all_frame, thresholds = add_groups(pool)
    warmup = all_frame.loc[all_frame.created.lt(START) & all_frame.closed.lt(START)]
    frame = all_frame.loc[all_frame.created.ge(START) & all_frame.created.lt(END)].copy()
    assert len(frame.loc[frame.month.lt('2026-07')]) == 2953 and len(frame.loc[frame.month.eq('2026-07')]) == 2541
    verified = check_previous(frame, reference)
    calendars = {month: block_weights(month) for month in MONTHS}
    rows = []; overall = []
    for (month, side), part in frame.groupby(['month', 'side']):
        rows.append(dict(month=month, side=side, family='overall', group='all', **metrics(part, calendars[month])))
        for columns in [[x] for x in SINGLES]+([list(x) for x in PAIRS] if side == 'put' else []):
            for key, group in part.groupby(columns, sort=True):
                label = '|'.join(map(str, key if isinstance(key, tuple) else (key,)))
                rows.append(dict(month=month, side=side, family=' x '.join(columns), group=label, **metrics(group, calendars[month])))
        overall.append(dict(month=month, side=side, **metrics(part, calendars[month])))
    for month, part in frame.groupby('month'):
        overall.append(dict(month=month, side='all', **metrics(part, calendars[month])))
    groups = pd.DataFrame(rows)
    # Every partition reproduces the direction/month totals; pairs are not added together.
    for (month, side, family), part in groups.groupby(['month', 'side', 'family']):
        parent = groups.loc[groups.month.eq(month)&groups.side.eq(side)&groups.family.eq('overall')].iloc[0]
        assert part.orders.sum() == parent.orders
        for column in ['mean_error', 'mae']:
            np.testing.assert_allclose(np.average(part[column], weights=part.orders), parent[column], atol=1e-10)
        np.testing.assert_allclose(np.average(part.rmse**2, weights=part.orders), parent.rmse**2, rtol=1e-12)
        np.testing.assert_allclose(part.cash_bias.sum(), parent.cash_bias, atol=1e-8)
    risk = load_pool('test', {'cohort': 'risk', 'use_base': False}, DEFAULT_OUTPUT)
    coverage = []
    for month in MONTHS:
        for side, name in [(0, 'call'), (1, 'put')]:
            ids = risk.X.index[risk.metadata.created_at_utc.dt.strftime('%Y-%m').eq(month)&risk.X.option_type.eq(side)]
            present = ids.isin(frame.index)
            coverage.append(dict(month=month, side=name, risk_orders=len(ids), base_orders=int(present.sum()),
                base_coverage=float(present.mean()), actual_mean_with_base=risk.y.loc[ids[present]].mean(),
                actual_mean_without_base=risk.y.loc[ids[~present]].mean()))
    tables = {'overall_by_month': pd.DataFrame(overall), 'base_groups': groups,
              'cross_month_stability': stability(groups), 'coverage_by_month': pd.DataFrame(coverage),
              'fixed_thresholds': thresholds, 'tenor_standardization': tenor_standardization(frame, warmup)}
    for directory in ['metrics', 'provenance', 'reports']: (output/directory).mkdir(parents=True, exist_ok=False)
    for name, table in tables.items(): table.to_csv(output/'metrics'/f'{name}.csv', index=False, float_format='%.17g')
    sources = {str(p.relative_to(PROJECT)): digest(p) for p in [
        Path(__file__), PROJECT/'pp_models/data.py', PROJECT/'pp_models/directional_features.py',
        PROJECT/'pp_models/put_final_io.py', PROJECT/'pp_data/io.py', PROJECT/'pp_audit/market.py',
        PROJECT/'pp_audit/common.py', PROJECT/'docs/plans/2026-10-14-base-multiplier-research.md']}
    info = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), analysis='Base only; no models fitted or evaluated',
        audit_scope='April-July old canonical orders; new official validation forbidden', guards=guards,
        scoring_orders=len(frame), scoring_put_orders=int(frame.side.eq('put').sum()),
        reference_orders=len(warmup), reference= str(reference),
        canonical_manifest_sha256=canonical_digest, market_prefix=market_info, directional_features=feature_info,
        source_sha256=sources, verified_previous_base_groups=verified,
        bootstrap=dict(repeats=REPEATS, seed=SEED, block_calendar_days=7, minimum_valid=1900,
                       intervals='exploratory pointwise; no multiplicity correction'),
        sufficient_support=dict(month_orders=30, month_days=5),
        local_slopes='Analytical derivative at multiplier 1 only; not fitted coefficients or candidate scores',
        no_per_order_outputs=True)
    assert guards == {'official_validation_open_attempts': 0, 'model_fit_predict_calls': 0}
    assert digest(DEFAULT_OUTPUT/'manifest.json') == canonical_digest
    for name, expected in old_files.items(): assert digest(reference/name) == expected
    (output/'provenance/analysis.json').write_text(json.dumps(info, ensure_ascii=False, indent=2)+'\n')
    (output/'reports/README.md').write_text('# Base分组审计\n\n仅使用旧4–7月数据，无训练、新官方验证集访问或逐笔输出。\n\n详细解释见项目 `docs/reports/2026-10-14-base-regimes.md`。\n')
    manifest = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'provenance/manifest.json').write_text(json.dumps({'files': manifest}, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'orders': len(frame), 'group_rows': len(groups),
                      'previous_base_groups_verified': verified, **guards}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT/'outputs/diagnostics/base_regimes_20261014')
    args = parser.parse_args()
    analyze(args.output.resolve())
