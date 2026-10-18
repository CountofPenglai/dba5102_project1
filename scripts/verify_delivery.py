"""Verify delivery evidence; optionally replay all reported weeks without fitting."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd
from pp_data.config import load_protocol
from pp_data.io import load_dataset
from pp_models.pricing_inputs import MODEL_INPUTS
from pp_models.runtime import AccessGuard
from pp_models.verification import h0_values
from pp_pricing import FinalPricingModel


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(run_dir=None, data_dir=None):
    if (run_dir is None) != (data_dir is None):
        raise ValueError('Supply --run-dir and --data-dir together')
    guard = AccessGuard()
    sys.addaudithook(guard.check)
    result_root = PROJECT/'docs/results'
    sources = json.loads((result_root/'source_manifest.json').read_text())
    checked = 0
    for item in sources['files']:
        if digest(result_root/item['saved']) != item['saved_sha256']:
            raise ValueError('Published result hash changed: '+item['saved'])
        checked += 1
    audit = json.loads((result_root/'report_data_audit_20261016.json').read_text())
    if digest(PROJECT/audit['report']) != audit['report_sha256']:
        raise ValueError('Word report changed after its data audit')
    if digest(result_root/'source_manifest.json') != audit['result_source_manifest_sha256']:
        raise ValueError('Published result manifest changed after the report audit')

    published = result_root/sources['current_run']
    manifest = json.loads((published/'manifest.json').read_text())
    model_path = PROJECT/'pp_pricing/model.json'
    # A claimed source hash alone cannot detect edited coefficients in the JSON.
    # Check the exact artifact against the recorded, independently replayed export.
    delivery = json.loads((result_root/'delivery_verification_20261018.json').read_text())
    if digest(model_path) != delivery['model_sha256']:
        raise ValueError('Delivered model artifact changed after its verified export')
    model = FinalPricingModel.load(model_path)
    if model.artifact['provenance']['parameters_sha256'] != manifest['artifacts']['provenance/parameters.json']:
        raise ValueError('Delivered model does not reference the reported parameter source')
    comparison = pd.read_csv(published/'validation/model_comparison.csv').set_index('model')
    expected = {
        'official_raw': [293.564338, 184.439853, 295.150078],
        'h0': [298.010152, 186.263090, 303.531640],
        'blend_c050_p025': [293.114750, 183.652670, 296.046896],
        'blend_c040_p015': [293.048025, 183.558057, 295.614043],
    }
    columns = ['macro_rmse_per_btc', 'macro_mae_per_btc', 'pooled_rmse_per_btc']
    # Validate the published numerical evidence; the concise README links to it
    # without duplicating the report's metric table.
    for name, values in expected.items():
        np.testing.assert_allclose(comparison.loc[name, columns].to_numpy(float), values, atol=5.1e-7, rtol=0)
    summary = pd.read_csv(published/'validation/summary.csv')
    for month in ['2026-05', '2026-06']:
        group = summary.loc[summary.month.eq(month) & summary.group.eq('overall')].set_index('model')
        if not (group.loc['h0', ['rmse_per_btc', 'mae_per_btc']] <
                group.loc['official_raw', ['rmse_per_btc', 'mae_per_btc']]).all():
            raise ValueError('H0 May/June improvement claim does not match the evidence')

    replay = None
    if run_dir is not None:
        run_dir = Path(run_dir)
        parameter_path = run_dir/'provenance/parameters.json'
        if digest(parameter_path) != model.artifact['provenance']['parameters_sha256']:
            raise ValueError('Replay requires the delivered model source run')
        states = json.loads(parameter_path.read_text())['time_decay']
        latest = max(states, key=pd.Timestamp)
        if model.artifact['fitted_at'] != latest or model.parameters != states[latest]['models']['h0']['parameters']:
            raise ValueError('Delivered coefficients differ from the original last weekly state')
        pool = load_dataset(data_dir, cohort='benchmark', use_base=True)
        protocol = load_protocol()
        created = pool.metadata.created_at_utc
        frames = []; max_difference = 0.
        for origin, state in states.items():
            start, stop = pd.Timestamp(origin), pd.Timestamp(state['stop'])
            q = pool.X.index[created.ge(start) & created.lt(stop)]
            x = pool.X.loc[q, MODEL_INPUTS]
            artifact = copy.deepcopy(model.artifact)
            artifact.update(fitted_at=origin, parameters=state['models']['h0']['parameters'])
            output = FinalPricingModel(artifact).predict(x, created_at=created.loc[q])
            p, s = h0_values(x, artifact['parameters'])
            np.testing.assert_allclose(output.h0_per_btc_usdc, p*s, rtol=1e-12, atol=1e-9)
            max_difference = max(max_difference, float(np.max(np.abs(output.h0_per_btc_usdc-p*s))))
            frames.append(pd.DataFrame(dict(actual=pool.y.loc[q],
                prediction=output.prediction_per_btc_usdc, month=created.loc[q].dt.strftime('%Y-%m'))))
        frame = pd.concat(frames)
        expected_ids = pool.X.index[created.ge(pd.Timestamp(protocol['validation_start'])) &
                                   created.lt(pd.Timestamp(protocol['validation_end']))]
        if not frame.index.is_unique or set(frame.index) != set(expected_ids) or len(frame) != 4640:
            raise ValueError('Replay validation cohort changed')
        error = frame.prediction-frame.actual
        values = dict(macro_rmse_per_btc=float(np.sqrt(error.pow(2).groupby(frame.month).mean().mean())),
            macro_mae_per_btc=float(error.abs().groupby(frame.month).mean().mean()),
            pooled_rmse_per_btc=float(np.sqrt(error.pow(2).mean())),
            pooled_mae_per_btc=float(error.abs().mean()))
        for key, value in values.items():
            np.testing.assert_allclose(value, comparison.loc['blend_c040_p015', key], rtol=1e-12, atol=1e-9)
        replay = dict(weekly_states=len(states), orders=len(frame), maximum_h0_absolute_difference=max_difference,
                      final_blend_metrics=values)
    return dict(status='passed', published_files_checked=checked, report_hash_unchanged=True,
        headline_metrics_checked=12, h0_may_june_rmse_and_mae_verified=True,
        model_fitted_at=model.artifact['fitted_at'], model_sha256=digest(PROJECT/'pp_pricing/model.json'),
        weekly_replay=replay, model_fit_calls=0, official_validation_open_attempts=guard.official_attempts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output', type=Path, help='Optional new compact JSON evidence file')
    args = parser.parse_args()
    if args.output and (args.output.exists() or args.output.is_symlink()):
        parser.error('Output already exists')
    result = verify(args.run_dir, args.data_dir)
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(payload)
    print(payload, end='')


if __name__ == '__main__':
    main()
