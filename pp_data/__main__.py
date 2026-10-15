"""Canonical preprocessing entry point: python -m pp_data."""

import argparse
from pathlib import Path
from .config import DEFAULT_DATA, DEFAULT_OUTPUT
from .io import build_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build one chronological PP research pool")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohort-reference", type=Path,
                        help="Retain exactly the source IDs from a verified flat dataset")
    args = parser.parse_args(argv)
    try:
        result = build_dataset(args.data_dir, args.output_dir, cohort_reference=args.cohort_reference)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Preprocessing failed: {error}\n")
    print(result.statistics[["cohort", "rows"]].to_string(index=False))
    print(f"Saved chronological dataset and manifest to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
