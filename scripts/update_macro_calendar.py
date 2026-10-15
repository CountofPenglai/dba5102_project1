"""Publish an independently verified public-source bundle; never fetch order data."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pp_calendar.calendar import build_calendar, publish_calendar
from pp_data.config import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, default=PROJECT/'external_data/macro_events')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    bundle = json.loads(args.source.read_text())
    calendar = build_calendar(bundle)
    print(json.dumps({'as_of': calendar['as_of_utc'], 'events': len(calendar['events']),
                      'coverage': calendar['coverage']}, ensure_ascii=False))
    if not args.check_only:
        print(publish_calendar(bundle, args.destination))


if __name__ == '__main__':
    main()
