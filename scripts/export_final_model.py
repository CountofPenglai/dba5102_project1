"""Export the last verified H0 weekly state; never fit or read order datasets."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from pp_models.pricing_inputs import MODEL_INPUTS
from pp_pricing import FinalPricingModel


def export_model(run_dir, output):
    """Publish a portable JSON model from a completed report run, without overwrite."""
    run_dir, output = Path(run_dir), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output already exists: {output}')
    source = run_dir/'provenance/parameters.json'
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = json.loads((run_dir/'provenance/manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('producer') != 'report_unified_v3'
            or manifest.get('official_validation_open_attempts') != 0
            or manifest.get('artifacts', {}).get('provenance/parameters.json') != digest):
        raise ValueError('Parameter hash or completed report provenance does not match')
    states = json.loads(payload)['time_decay']
    origin = max(states, key=pd.Timestamp)
    state = states[origin]
    stamp = pd.Timestamp(origin)
    earliest = pd.Timestamp(state['earliest_training_create'])
    latest_close = pd.Timestamp(state['latest_training_close'])
    if (any(pd.isna(t) or t.tzinfo is None for t in (stamp, earliest, latest_close))
            or state['origin'] != origin or not stamp-pd.Timedelta(days=30) <= earliest < stamp
            or not latest_close < stamp or state['history_orders'] < 100):
        raise ValueError('Invalid weekly history or label-availability cutoff')
    artifact = dict(
        schema_version='h0_base_blend_v1', model='blend_c040_p015', fitted_at=origin,
        input_columns=MODEL_INPUTS, alpha={'0': .4, '1': .15},
        parameters=state['models']['h0']['parameters'],
        provenance=dict(source_run=run_dir.name, parameters_sha256=digest,
            history_orders=state['history_orders'], earliest_training_create=state['earliest_training_create'],
            latest_training_close=state['latest_training_close'], history_days=30,
            original_update_days=7, export_refit=False, label_feedback_at_inference=False,
            note='Last existing weekly state; not an all-data refit. Frozen inference is not the rolling backtest.'),
    )
    FinalPricingModel(artifact)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)+'\n')
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        artifact = export_model(args.run_dir, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(f"Exported {artifact['model']} fitted at {artifact['fitted_at']} to {args.output}")


if __name__ == '__main__':
    main()
