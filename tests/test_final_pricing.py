"""Delivery contract: label-free inference must preserve the reported model."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from fixtures import candles, order
from pp_models.h0 import H0, MODEL_INPUTS, h0_state
from pp_pricing import FinalPricingModel


def features():
    return pd.DataFrame({
        'option_type': [0, 0, 1, 1, 0, 1],
        'api_base_price_per_btc_usdc': [80., 150., 120., 200., 180., 90.],
        'remaining_seconds': [3600., 7200., 3600., 7200., 14400., 14400.],
        'product_duration_seconds': [3600., 7200., 3600., 7200., 14400., 14400.],
        'realized_vol_60m': [.01, .02, .015, .02, 0., .03],
        'log_moneyness': [0., .01, -.01, .02, .03, -.02],
    }, index=['a', 'b', 'c', 'd', 'e', 'f'])[MODEL_INPUTS]


def artifact(parameters):
    return dict(schema_version='h0_base_blend_v1', model='blend_c040_p015',
                fitted_at='2026-07-31T00:00:00+00:00',
                input_columns=MODEL_INPUTS, alpha={'0': .4, '1': .15},
                parameters=parameters, provenance={'source': 'synthetic test'})


class FinalPricingTests(unittest.TestCase):
    def setUp(self):
        self.x = features()
        self.h0 = H0().fit(self.x, pd.Series([0., 180., 250., 0., 100., 50.], index=self.x.index))
        self.artifact = artifact(h0_state(self.h0))
        self.model = FinalPricingModel(self.artifact)
        self.created = pd.Series(pd.Timestamp('2026-08-01', tz='UTC'), index=self.x.index)

    def test_matches_h0_and_directional_formula(self):
        result = self.model.predict(self.x, created_at=self.created)
        p, s = self.h0.predict_components(self.x)
        b = self.x.api_base_price_per_btc_usdc.to_numpy()
        a = np.where(self.x.option_type.eq(0), .4, .15)
        np.testing.assert_allclose(result.h0_per_btc_usdc, p*s, rtol=1e-13)
        np.testing.assert_allclose(result.prediction_per_btc_usdc, a*p*s+(1-a)*b, rtol=1e-13)
        np.testing.assert_allclose(result.pricing_multiplier, result.prediction_per_btc_usdc/b)
        self.assertTrue(result.index.equals(self.x.index))

    def test_single_class_and_missing_direction_match_h0(self):
        x = self.x.loc[self.x.option_type.eq(0)]
        h0 = H0().fit(x, pd.Series(0., index=x.index))
        model = FinalPricingModel(artifact(h0_state(h0)))
        result = model.predict(self.x, created_at=self.created)
        np.testing.assert_allclose(result.h0_per_btc_usdc, h0.predict_raw(self.x))

    def test_rejects_future_parameters_bad_features_and_misaligned_times(self):
        for times in [self.created - pd.Timedelta(days=2), self.created.iloc[::-1],
                      pd.Series(pd.NaT, index=self.x.index)]:
            with self.assertRaises(ValueError): self.model.predict(self.x, created_at=times)
        for x in [self.x.assign(realizedProfit_USDC=0), self.x.assign(realized_vol_60m=np.nan),
                  self.x.assign(api_base_price_per_btc_usdc=0), self.x.assign(option_type=2)]:
            with self.assertRaises(ValueError): self.model.predict(x, created_at=self.created)

    def test_rejects_corrupt_artifact(self):
        for change in [('alpha', {'0': 1.1, '1': .15}), ('schema_version', 'unknown')]:
            bad = copy.deepcopy(self.artifact); bad[change[0]] = change[1]
            with self.assertRaises(ValueError): FinalPricingModel(bad)
        bad = copy.deepcopy(self.artifact); bad['parameters']['0']['scale'][0] = 0.
        with self.assertRaises(ValueError): FinalPricingModel(bad)

    def test_official_adapter_ignores_outcomes_and_future_candles(self):
        orders = pd.DataFrame([order(createdAt='2026-08-01 01:01:00',
            expirationDate='2026-08-01 02:01:00'), order(optionType=1,
            createdAt='2026-08-01 01:01:00', expirationDate='2026-08-01 02:01:00')])
        market = candles(start='2026-08-01', periods=100)
        got = self.model.predict_orders(orders, market)
        changed = orders.assign(realizedProfit_USDC=999999, closedAt='2040-01-01',
                                premiumAmount_USDC=999999, markup_per_BTC_USDC=999999)
        market.loc[market.open_time_utc.ge(pd.Timestamp('2026-08-01 01:01', tz='UTC')),
                   ['open', 'high', 'low', 'close']] *= 2
        pd.testing.assert_frame_equal(got, self.model.predict_orders(changed, market))
        with self.assertRaises(ValueError): self.model.predict_orders(orders, market.iloc[:30])

    def test_cli_works_without_training_or_private_data_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); model = root/'model.json'; src = root/'quotes.csv'; out = root/'out.csv'
            model.write_text(json.dumps(self.artifact))
            self.x.assign(created_at_utc=self.created).rename_axis('order_id').to_csv(src)
            args = [sys.executable, '-m', 'pp_pricing', '--features', str(src),
                    '--model', str(model), '--output', str(out)]
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = pd.read_csv(out)
            self.assertEqual(saved.order_id.tolist(), self.x.index.tolist())
            before = out.read_bytes()
            again = subprocess.run(args, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(before, out.read_bytes())

    def test_export_checks_source_hash_and_label_cutoff(self):
        from scripts.export_final_model import export_model
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); provenance = root/'run'/'provenance'; provenance.mkdir(parents=True)
            stamp = '2026-07-31T00:00:00+00:00'
            state = dict(origin=stamp, earliest_training_create='2026-07-01T00:00:00+00:00',
                         latest_training_close='2026-07-30T23:59:59+00:00', history_orders=100,
                         models={'h0': {'parameters': h0_state(self.h0)}})
            parameters = provenance/'parameters.json'
            parameters.write_text(json.dumps({'time_decay': {stamp: state}}))
            manifest = dict(status='complete', producer='report_unified_v3',
                official_validation_open_attempts=0, artifacts={
                    'provenance/parameters.json': hashlib.sha256(parameters.read_bytes()).hexdigest()})
            (provenance/'manifest.json').write_text(json.dumps(manifest))
            out = root/'model.json'
            export_model(root/'run', out)
            self.assertEqual(FinalPricingModel.load(out).artifact['parameters'], self.artifact['parameters'])
            with self.assertRaises(FileExistsError): export_model(root/'run', out)
            parameters.write_text(parameters.read_text()+' ')
            with self.assertRaises(ValueError): export_model(root/'run', root/'bad.json')
            state['latest_training_close'] = stamp
            parameters.write_text(json.dumps({'time_decay': {stamp: state}}))
            manifest['artifacts']['provenance/parameters.json'] = hashlib.sha256(parameters.read_bytes()).hexdigest()
            (provenance/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): export_model(root/'run', root/'bad.json')

    def test_cli_preserves_literal_na_order_ids(self):
        ids = ['NA', 'NULL', 'N/A', '0001', 'call', 'put']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); src = root/'quotes.csv'; out = root/'out.csv'
            self.x.assign(created_at_utc=self.created).set_axis(ids).rename_axis('order_id').to_csv(src)
            result = subprocess.run([sys.executable, '-m', 'pp_pricing', '--features', str(src),
                '--output', str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = pd.read_csv(out, converters={'order_id': str})
            self.assertEqual(saved.order_id.tolist(), ids)

    def test_import_does_not_load_training_libraries(self):
        result = subprocess.run([sys.executable, '-c',
            "import sys; import pp_pricing; assert 'sklearn' not in sys.modules; "
            "assert 'pp_models.report_unified_study' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_delivery_evidence_verifier_runs_without_private_data(self):
        result = subprocess.run([sys.executable, 'scripts/verify_delivery.py'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['status'], 'passed')

    def test_public_verifier_rejects_changed_coefficients(self):
        from scripts import verify_delivery as delivery
        project = delivery.PROJECT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(project/'docs/results', root/'docs/results')
            shutil.copytree(project/'docs/deliverables', root/'docs/deliverables')
            shutil.copy2(project/'README.md', root/'README.md')
            (root/'pp_pricing').mkdir()
            path = root/'pp_pricing/model.json'
            data = json.loads((project/'pp_pricing/model.json').read_text())
            data['parameters']['0']['severity'] *= 10
            path.write_text(json.dumps(data))
            with patch.object(delivery, 'PROJECT', root):
                with self.assertRaises(ValueError): delivery.verify()

    def test_raw_adapter_accepts_mixed_timestamps_and_preserves_ids(self):
        market = candles(start='2026-08-01', periods=100)
        orders = pd.DataFrame([
            order(order_id='0001', createdAt='2026-08-01 01:01:00', expirationDate='2026-08-01 02:01:00'),
            order(order_id='b', optionType=1, createdAt='2026-08-01T01:02:00Z',
                  expirationDate='2026-08-01T02:02:00.000Z')])
        result = self.model.predict_orders(orders, market)
        self.assertEqual(result.index.tolist(), ['0001', 'b'])
        uniform = orders.copy()
        for name in ['createdAt', 'expirationDate']:
            uniform[name] = pd.to_datetime(uniform[name], format='mixed', utc=True)
        pd.testing.assert_frame_equal(result, self.model.predict_orders(uniform, market))
        for ids in [['a', 'a'], [' ', 'b'], [None, 'b']]:
            with self.assertRaises(ValueError):
                self.model.predict_orders(orders.assign(order_id=ids), market)


if __name__ == '__main__':
    unittest.main()
