# Shared rolling splits and training implementation plan

**Goal:** Centralize reusable temporal splitting, H0 fitting, quote context and settled forecast history while preserving all reported predictions.

**Scope:** User-requested architecture refactor. Preserve the existing delivery changes, report artifacts and frozen model. No tuning, new official validation access, expanded cohort, commit or push.

## Interfaces and responsibilities

- `pp_data.splitting`: validate temporal protocol; yield `QuoteWindow` / `RollingWindow` objects with origin, exclusive stop, ordered query/history IDs and validation role; provide the sole causal `history_ids` selector.
- `pp_models.validation`: historical tail thresholds and common prediction metadata, using the shared splitter. Re-export `history_ids` for current callers; independent verification keeps its independent arithmetic.
- `pp_models.training`: one `TrainingSession` per run. Reuse each `(origin, half_life)` H0 fit across branches, enforce pool/protocol identity and unchanged source content, expose actual fit/cache counters. `TimeDecayH0(None)` is exactly the original H0 plus diagnostics.
- `pp_models.forecast_history`: cache prior forward predictions; expose labels only after strict settlement cutoff. No contemporaneous or future labels in dynamic blend selection.
- Existing study modules call these services and retain only branch-specific corrections. `report_unified_study.predict_all` owns and shares the session. Public CLI and study signatures remain compatible via optional keyword-only session arguments.

## Tasks

- [x] Write temporal boundary, cache reuse/invalidation and delayed-label tests, and confirm failure before implementation.
- [x] Implement splitting/context/training/history modules with English contracts.
- [x] Migrate all study branches and announcement query windows; record actual fit reuse in new run provenance.
- [x] Run full tests and all 97 recipes on the existing 4,640-order report cohort; compare current frozen metrics/parameters and run independent verification.
- [x] Update bilingual README, model/data architecture and concise verification evidence.

Completed evidence: [refactor verification](../reports/2026-10-18-shared-training.md). All 105 tests passed; all 97 recipes retained their previous statistics and parameter bytes; H0-family fits fell from 78 to 39. Independent report and delivery verification both passed.

## Acceptance

- Same continuous seven-day clock, 30-day creation history, strict `closed < origin`, monthly expanding q90 and warm-up behavior.
- No leakage from query labels or actual holding times; announcement scheduling must remain independent of settlement and labels.
- No cache sharing between different datasets, protocols or changed input content.
- H0 fitted once per supported week, with weighted variants once per supported validation week; reported recipe identities and outcomes unchanged.
- Final `pp_pricing` import remains free of training libraries, and the shipped JSON remains byte-identical.
