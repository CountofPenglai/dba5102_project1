"""Run with `python -m pp_audit`; write aggregate CSVs only."""

import argparse
import json
from pathlib import Path
import pandas as pd
from .orders import audit_orders
from .market import audit_market
from .alignment import audit_alignment


def main(argv=None):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Read-only PP research data audit")
    parser.add_argument("--data-dir", type=Path, default=project.parent / "docs" / "Jasper Vault Options Pricing Optimisation(Included Base Premium)")
    parser.add_argument("--output-dir", type=Path, default=project / "outputs" / "data_audit")
    parser.add_argument("--cutoff", action="append", help="UTC cutoff; repeat for multiple internal research cutoffs")
    args = parser.parse_args(argv)
    cutoffs = args.cutoff if args.cutoff is not None else ["2026-05-01", "2026-06-01", "2026-07-01"]
    try:
        metadata_path = args.data_dir / "pp_research_datasets_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object")
        orders = audit_orders(pd.read_csv(args.data_dir / "pp_transactions_training.csv"))
        market = audit_market(pd.read_csv(args.data_dir / "btc_market_history.csv"))
        alignment, cutoff_table = audit_alignment(
            orders, market, metadata, cutoffs,
            (args.data_dir / "pp_transactions_validation.csv").is_file(),
        )
        tables = {
            "orders/key_metrics.csv": orders.metrics,
            "orders/segment_statistics.csv": orders.segments,
            "market/key_metrics.csv": market.metrics,
            "alignment/key_metrics.csv": alignment,
            "alignment/cutoff_statistics.csv": cutoff_table,
        }
        # Validate and compute everything before creating any output.
        for name, table in tables.items():
            destination = args.output_dir / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(destination, index=False, encoding="utf-8-sig", float_format="%.10g")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Audit failed: {error}\n")
    print(f"Audited {len(orders.rows):,} orders and {len(market.rows):,} candles.")
    print(f"Saved {len(tables)} aggregate CSVs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
