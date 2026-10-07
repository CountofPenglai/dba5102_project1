"""Canonical preprocessing entry point: python -m pp_data."""

import argparse
from pathlib import Path
from .config import DEFAULT_DATA, DEFAULT_OUTPUT
from .io import build_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the canonical chronological PP research datasets")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = build_dataset(args.data_dir, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Preprocessing failed: {error}\n")
    print(result.statistics[["split", "cohort", "rows"]].to_string(index=False))
    print(f"Saved canonical datasets and manifest to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
