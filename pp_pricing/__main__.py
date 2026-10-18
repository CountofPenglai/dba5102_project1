"""CLI for official-format quotes or prepared features; never starts training."""
import argparse
from pathlib import Path
import sys

import pandas as pd

from pp_models.pricing_inputs import MODEL_INPUTS
from .model import DEFAULT_MODEL, ORDER_INPUTS, FinalPricingModel, with_order_index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--orders', type=Path, help='Official-format orders; labels may be absent')
    inputs.add_argument('--features', type=Path, help='Prepared MODEL_INPUTS plus created_at_utc')
    parser.add_argument('--market', type=Path, help='BTCUSDC minute candles; required with --orders')
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL, help='Frozen JSON parameter artifact')
    parser.add_argument('--output', type=Path, required=True, help='New prediction CSV; existing files refused')
    args = parser.parse_args()
    if bool(args.orders) != bool(args.market):
        parser.error('--market is required only with --orders')
    if args.output.exists() or args.output.is_symlink():
        parser.error('Output already exists; choose a new path')
    try:
        model = FinalPricingModel.load(args.model)
        allowed = ORDER_INPUTS if args.orders else MODEL_INPUTS + ['created_at_utc']
        # Column selection keeps outcomes out of the inference adapter entirely.
        frame = pd.read_csv(args.orders or args.features,
                            usecols=lambda name: name in allowed or name == 'order_id',
                            # Preserve literal IDs such as NA/NULL as well as zeros;
                            # missing-value parsing still applies to numeric features.
                            converters={'order_id': str})
        frame = with_order_index(frame)
        if frame.index.name is None:
            frame.index.name = 'source_row'
        if args.orders:
            result = model.predict_orders(frame, pd.read_csv(args.market))
        else:
            result = model.predict(frame[MODEL_INPUTS], created_at=frame.created_at_utc)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also prevents a race from overwriting another result.
        with args.output.open('x', encoding='utf-8', newline='') as stream:
            result.to_csv(stream, float_format='%.17g')
    except (ValueError, KeyError, AttributeError, OSError) as error:
        parser.error(str(error))
    print(f'Wrote {len(result)} predictions to {args.output}; parameters frozen at {model.fitted_at}',
          file=sys.stderr)


if __name__ == '__main__':
    main()
