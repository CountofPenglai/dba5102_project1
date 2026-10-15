import unittest
import numpy as np
import pandas as pd
from pp_data.features import BASE_FEATURE
from tests.test_h0_time_decay import fixture
from pp_models.h0_hourly_amount import HourlyAmount, market_features, amount_features
from pp_models.h0_hourly_amount_study import predict_period, score
from pp_models.h0_time_decay_study import PROTOCOL, predict_period as old_predict


def inputs():
    p = fixture(); x = p.X.copy(); k = np.arange(len(x))
    x['market_price_usdc_per_btc'] = 10000.
    f = pd.DataFrame({'up_vol_60m': .001*(1+k%19)/10, 'down_vol_60m': .001*(1+k%13)/10,
        'up_vol_240m': .0006, 'down_vol_240m': .0005, 'available': True}, index=x.index)
    return p, x, f


class HourlyModelTests(unittest.TestCase):
    def test_direction_selection_and_volatility_scale(self):
        p, x, f = inputs(); anchor = pd.Series(200., index=x.index)
        a = amount_features(x, f, anchor)
        x.option_type = 0; b = amount_features(x, f, anchor)
        call = p.X.option_type.eq(0)
        np.testing.assert_array_equal(a.loc[call, 'level'], b.loc[call, 'level'])
        np.testing.assert_allclose(a.loc[~call, 'speed'], np.log((f.loc[~call, 'down_vol_60m']+1e-8)/(.0005+1e-8)))
        np.testing.assert_allclose(a.loc[call, 'level'], np.log1p(10000*f.loc[call, 'up_vol_60m']*np.sqrt(x.loc[call, 'remaining_seconds']/60)/200))

    def test_fixed_parameters_respond_to_market_and_bounds_and_missing_fallback(self):
        p, x, f = inputs(); anchor = pd.Series(200., index=x.index)
        a = amount_features(x, f, anchor); target = 200*(1+.6*a.level)
        m = HourlyAmount('amount_joint').fit(x.option_type, a, target, p.metadata.created_at_utc, anchor)
        before = m.state(); ordinary, _ = m.predict(x.option_type, a, anchor)
        shock = a.copy(); shock['level'] += 3
        shifted, _ = m.predict(x.option_type, shock, anchor)
        self.assertFalse(np.allclose(ordinary, shifted)); self.assertEqual(before, m.state())
        self.assertTrue(((shifted/anchor >= .9) & (shifted/anchor <= 1.1)).all())
        shock.loc[shock.index[:5], ['level', 'speed']] = np.nan; shock.loc[shock.index[:5], 'available'] = False
        pred, support = m.predict(x.option_type, shock, anchor)
        np.testing.assert_array_equal(pred[:5], anchor.iloc[:5]); self.assertFalse(support[:5].any())

    def test_gate_preserves_ordinary_and_never_lowers_and_sparse_fallback(self):
        p, x, f = inputs(); anchor = pd.Series(200., index=x.index); a = amount_features(x, f, anchor)
        a['level'] = np.tile(np.linspace(0, 1, 50), int(np.ceil(len(a)/50)))[:len(a)]
        a['speed'] = np.log(2.); y = pd.Series(400., index=x.index)
        model = HourlyAmount('amount_gate').fit(x.option_type, a, y, p.metadata.created_at_utc, anchor)
        pred, _ = model.predict(x.option_type, a, anchor)
        np.testing.assert_array_equal(pred[a.level.le(.5)], anchor.loc[a.level.le(.5)])
        self.assertGreater(pred.max(), 200); self.assertLessEqual(pred.max(), 220.00000001)
        self.assertTrue((pred >= 200).all())
        sparse = HourlyAmount('amount_gate').fit(x.option_type.iloc[:30], a.iloc[:30], y.iloc[:30], p.metadata.created_at_utc.iloc[:30], anchor.iloc[:30])
        np.testing.assert_array_equal(sparse.predict(x.option_type, a, anchor)[0], anchor)

    def test_sides_are_fitted_independently_and_zero_anchor_safe(self):
        p, x, f = inputs(); anchor = pd.Series(200., index=x.index); a = amount_features(x, f, anchor)
        y = p.y.copy(); m = HourlyAmount('amount_level').fit(x.option_type, a, y, p.metadata.created_at_utc, anchor)
        y.loc[x.option_type.eq(1)] *= 100
        n = HourlyAmount('amount_level').fit(x.option_type, a, y, p.metadata.created_at_utc, anchor)
        self.assertEqual(m.state()['0'], n.state()['0'])
        anchor[:] = 0.; z = HourlyAmount('amount_level').fit(x.option_type, a, y, p.metadata.created_at_utc, anchor)
        np.testing.assert_array_equal(z.predict(x.option_type, a, anchor)[0], 0.)


class HourlyFeaturesTests(unittest.TestCase):
    def test_future_candles_irrelevant_and_gaps_explicit(self):
        t = pd.date_range('2026-04-01', periods=310, freq='min', tz='UTC')
        market = pd.DataFrame({'open_time_utc': t, 'close_time_utc': t+pd.Timedelta(milliseconds=59999),
            'close': 100*np.exp(np.arange(310)*.001)})
        dates = pd.Series([t[280]+pd.Timedelta(seconds=30)], index=['q'])
        a = market_features(dates, market)
        changed = market.copy(); changed.loc[280:, 'close'] *= 1000
        pd.testing.assert_frame_equal(a, market_features(dates, changed))
        self.assertAlmostEqual(a.up_vol_60m.iloc[0], .001)
        self.assertAlmostEqual(a.down_vol_60m.iloc[0], 0.)
        missing = market_features(dates, market.drop(index=200))
        self.assertFalse(missing.available.iloc[0])
        self.assertTrue(missing[['up_vol_240m', 'down_vol_240m']].isna().all().all())


class HourlyProtocolTests(unittest.TestCase):
    def test_frozen_probability_and_reference_and_weekly_feedback(self):
        p, x, f = inputs(); p.X['market_price_usdc_per_btc'] = x.market_price_usdc_per_btc
        protocol = {**PROTOCOL, 'validation_end': '2026-05-15T00:00:00Z'}
        a, sa = predict_period(p, f, protocol); old, _, _ = old_predict(p, protocol)
        h0 = a.loc[a.model.eq('h0')]
        np.testing.assert_array_equal(h0.prediction, old.loc[old.model.eq('h0'), 'prediction'])
        for name in ('amount_level', 'amount_joint', 'amount_gate'):
            np.testing.assert_array_equal(a.loc[a.model.eq(name), 'probability'], h0.probability)
        mask = p.metadata.created_at_utc.ge('2026-05-01') & p.metadata.created_at_utc.lt('2026-05-08')
        p.y.loc[mask] = 1000.
        b, sb = predict_period(p, f, protocol)
        first = '2026-05-01T00:00:00+00:00'; second = '2026-05-08T00:00:00+00:00'
        pd.testing.assert_frame_equal(a.loc[a.origin.eq(first)], b.loc[b.origin.eq(first)])
        self.assertEqual(sa[first], sb[first]); self.assertNotEqual(sa[second], sb[second])
        self.assertTrue(a.month.eq('2026-05').all())

    def test_unsettled_labels_and_pre_april_labels_never_enter(self):
        p, x, f = inputs(); p.X['market_price_usdc_per_btc'] = x.market_price_usdc_per_btc
        protocol = {**PROTOCOL, 'validation_end': '2026-05-15T00:00:00Z'}
        mask = p.metadata.created_at_utc.ge('2026-04-23')
        p.metadata.loc[mask, 'closed_at_utc'] = pd.Timestamp('2026-08-01', tz='UTC')
        a, sa = predict_period(p, f, protocol)
        p.y.loc[mask | p.metadata.created_at_utc.lt('2026-04-01')] = 1e9
        b, sb = predict_period(p, f, protocol)
        pd.testing.assert_frame_equal(a, b); self.assertEqual(sa, sb)

    def test_statistics_do_not_hide_unsupported_monthly_tails(self):
        from pp_models.h0_hourly_amount_statistics import tables
        p, x, f = inputs(); p.X['market_price_usdc_per_btc'] = x.market_price_usdc_per_btc
        protocol = {**PROTOCOL, 'validation_end': '2026-06-15T00:00:00Z'}
        pred, _ = predict_period(p, f, protocol); scored, _ = score(p, pred, protocol['validation_end'])
        out = tables(scored, protocol['validation_start'], protocol['validation_end'], uncertainty=False)
        monthly = out['monthly_risk_constraints']
        self.assertEqual(set(monthly.month), {'2026-05', '2026-06'})
        thin = monthly.tail_orders.lt(30) | monthly.tail_days.lt(5)
        self.assertTrue(thin.any()); self.assertFalse(monthly.loc[thin, 'direction_risk_pass'].any())
        self.assertTrue(out['response_diagnostics'].groupby(['month', 'side']).orders.nunique().eq(1).all())


if __name__ == '__main__': unittest.main()
