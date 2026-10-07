# Data Audit Implementation Plan

> Execution: superpowers:executing-plans, inline. The user explicitly requested implementation in this directory. No additional approval stage is necessary.

**Goal:** Reproducible read-only data audit with only decision-relevant aggregate statistics.

**Architecture:** Separate order, market and alignment audits. One CLI writes five CSVs, grouped by subject. Shared parsers distinguish missing, malformed and non-finite values.

**Tech Stack:** Python, pandas, numpy, unittest.

**Spec:** `../../README.md`, especially the output and sample definitions.

## Global Constraints

- Work only in the requested `project1_program` directory; preserve source files.
- No charts, row-level exports, cleaned datasets, trained models or speculative demand estimates.
- Use UTC and completed candles; count missing labels separately from zero payouts.
- Output metrics with explicit denominators and definitions; unavailable metrics stay blank.
- Directory is not a Git repository. No worktree, initialization, branch or commit is needed.

## Review Focus

- Missing/non-finite outcomes must not become zero payouts or contribute premiums to unmatched totals.
- Duplicate, missing or malformed candles must not pass continuous history checks.
- Outcomes settled at/after a cutoff cannot become training labels at that cutoff.
- Future-candle and invalid-time information must not enter candidate coverage counts.
- Empty/all-missing groups and missing optional metadata cannot produce misleading healthy zeroes.

## Tasks

- [x] 1. Write fixture-based tests and public audit stubs; observe failures. Tests cover overlapping exclusions, 0 vs missing payout, price units and normal settlement latency.
- [x] 2. Implement `audit_orders(raw) -> OrderAudit` and `audit_market(raw) -> MarketAudit`. Return aggregate tables plus internal parsed rows/masks for alignment. Run tests.
- [x] 3. Write alignment and CLI tests first. Implement `audit_alignment(orders, market, metadata, cutoffs, validation_present) -> (metrics, cutoffs)` and the CLI. Verify full windows, exact boundaries and input preservation.
- [x] 4. Run the full audit, independently reconcile headline numbers, review code, and report research implications with links to output statistics.

## Execution Record

- Initial inspection: no existing code or AGENTS.md; source contains two principal CSVs, dictionary, metadata, guide and project brief.
- Scope decision: candidate sample counts are diagnostic only; no automatic data cleaning is exported.
- Task 1–2: nine fixture tests failed on unimplemented audits, then all nine passed after implementation.
- Task 3: seven new alignment/CLI tests failed before implementation, then the full 16-test suite passed.
- Full run: 7,204 orders, 338,917 candles; five aggregate CSVs. SHA-256 of all five source-directory files unchanged before/after execution. Verified using Python bundle with pandas 2.2.3.
- Independent standard-library reconciliation matched labeled/technical/common counts, anomaly unions, common-sample monetary totals and the full minute grid.
- Independent read-only review confirmed core masks, continuity and cutoff logic. Two optional-metadata issues were reproduced and fixed with regression tests: missing filter counts no longer raise or imply zero; invalid validation-end timestamps no longer imply full market coverage.
- Review follow-up: premium omission was assessed as a diagnostic completeness issue because otherwise an excluded financial input could not be traced directly. Added a dedicated invalid-premium count and separated it from arithmetic identity errors, verified with a failing-then-passing fixture.
- Final verification: all 19 tests pass; full audit rerun successfully. Outputs remain exactly five categorized aggregate CSVs. No deferred code findings.
- Business assumptions remain provisional: nonnegative outcome-as-payout interpretation, possible API quote look-ahead, and the official pricing benchmark's transformations are not established by this audit. Candidate counts must not be treated as a final cleaning decision.
