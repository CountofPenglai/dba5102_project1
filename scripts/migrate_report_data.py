"""One-time, lossless migration of verified historical report bundles to one pool.

Only the known old artifact paths are read. Their directory names identify
physical containers and never determine chronology or model evaluation roles.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from pp_data.config import load_config
from pp_data.io import (
    TABLE_KINDS, _artifact_bytes, _check_nonoverlap, _guard_output,
    _new_manifest, _parse_artifact, _publish_dataset, _validate_tables,
)
from pp_data.pipeline import PreparedData, pool_statistics

LEGACY_BUNDLES = ("train", "validation", "test")
LEGACY_ARTIFACTS = {
    f"datasets/{bundle}/{kind}.csv" for bundle in LEGACY_BUNDLES for kind in TABLE_KINDS
} | {"reports/split_statistics.csv", "reports/exclusion_statistics.csv"}


def migrate_report_data(source_dir, output_dir):
    """Verify all legacy artifacts, preserve all table values, publish a flat pool."""
    if Path(source_dir).is_symlink() or Path(output_dir).is_symlink():
        raise ValueError("Migration directories must not be symbolic links")
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    _check_nonoverlap(source, output)
    _guard_output(source, output)
    manifest_bytes = _artifact_bytes(source, "manifest.json")
    try:
        old = json.loads(manifest_bytes)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("Invalid legacy manifest") from error
    if not isinstance(old, dict) or old.get("producer") != "pp_data" or old.get("version") != "preprocessing_v1":
        raise ValueError("Expected a verified legacy pp_data manifest")
    records = old.get("artifacts")
    # Exact allowlist rejects foreign/traversal/absolute paths before any reads.
    if not isinstance(records, dict) or set(records) != LEGACY_ARTIFACTS:
        raise ValueError("Legacy artifact list is incomplete or contains foreign paths")
    snapshots = {name: _artifact_bytes(source, name) for name in sorted(LEGACY_ARTIFACTS)}
    loaded = {name: _parse_artifact(payload, records[name], name)
              for name, payload in snapshots.items()}
    tables = {}
    for bundle in LEGACY_BUNDLES:
        ids = [loaded[f"datasets/{bundle}/{kind}.csv"].get("order_id") for kind in TABLE_KINDS]
        if any(values is None for values in ids) or not ids[0].equals(ids[1]) or not ids[0].equals(ids[2]):
            raise ValueError("Legacy bundle table IDs are not aligned")
    for kind in TABLE_KINDS:
        pieces = [loaded[f"datasets/{bundle}/{kind}.csv"] for bundle in LEGACY_BUNDLES]
        if any(list(piece) != list(pieces[0]) for piece in pieces):
            raise ValueError(f"Legacy {kind} table schemas differ")
        tables[kind] = pd.concat(pieces, ignore_index=True)
        if tables[kind].order_id.isna().any() or tables[kind].order_id.duplicated().any():
            raise ValueError("Legacy order IDs must be globally unique")
    times = pd.to_datetime(tables["metadata"].created_at_utc, utc=True, errors="raise", format="mixed")
    positions = tables["metadata"].assign(_created=times).sort_values(
        ["_created", "source_row_number"], kind="stable").index
    tables = {kind: table.loc[positions].reset_index(drop=True) for kind, table in tables.items()}
    manifest = _new_manifest("migrated_report_pool")
    _validate_tables(tables, manifest["config"], manifest["research_end"])
    old_exclusions = loaded["reports/exclusion_statistics.csv"]
    if not {"reason", "rows", "definition"}.issubset(old_exclusions):
        raise ValueError("Legacy exclusion report schema mismatch")
    excluded = pd.to_numeric(old_exclusions.loc[old_exclusions.reason.ne("retained"), "rows"], errors="raise")
    if excluded.isna().any() or (excluded < 0).any() or (excluded % 1 != 0).any():
        raise ValueError("Legacy exclusion counts are invalid")
    exclusions = pd.DataFrame([
        {"reason": "historical_report_pool_exclusions", "rows": int(excluded.sum()),
         "definition": "Historical exclusions retained, including omitted month-boundary labels; no rows were restored"},
        {"reason": "retained", "rows": len(tables["metadata"]),
         "definition": "Every row from all verified physical historical data bundles"},
    ])
    result = PreparedData(**tables, statistics=pool_statistics(tables["metadata"]),
                          exclusions=exclusions, config=load_config())
    manifest["migration_source"] = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "artifacts": {name: records[name]["sha256"] for name in sorted(records)},
        "version": old["version"],
    }
    manifest["input_sha256"] = old.get("input_sha256", {})
    manifest["limitations"].append(
        "The migrated report pool retains historical exclusions, including month-boundary unsettled orders; it is not an expanded raw rebuild")
    _publish_dataset(source, output, result, manifest)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = migrate_report_data(args.source_dir, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Migration failed: {error}\n")
    print(result.statistics[["cohort", "rows"]].to_string(index=False))
    print(f"Saved verified flat report pool to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
