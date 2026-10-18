# Obsolete configuration and results cleanup

**Goal:** Remove superseded result snapshots while keeping the final model, current report evidence and numerical regression inputs usable.

**Scope:** This worktree only. Preserve all eight live JSON configurations, the current published report/source manifest, private preprocessing data, frozen pricing artifact, current report run and shared-training verification run. No model fitting, official validation access, commit or push.

## Tasks

- [x] Confirm configuration consumers and result references before removal.
- [x] Keep only `summary.csv` and `model_comparison.csv` in the old numerical reference; remove retired announcement rows without changing retained cell values.
- [x] Remove superseded configuration/protocol/result snapshots and the two completed local runs `report_unified_v3_20261015_r2` and `report_cleanup_v3_20261015`.
- [x] Update configuration/result indexes and historical retention statements; keep a compact cleanup record.
- [x] Verify identical regression checks before/after, unchanged protected hashes, all tests, final delivery replay, local links and clean diff whitespace.

Historical tracked files remain recoverable from Git. The old reference directory is a numerical fixture, not an alternative active configuration or report source. Existing source manifests are immutable historical evidence and will not be rewritten to hide prior paths.
