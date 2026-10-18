# Delivery readiness implementation plan

**Goal:** Deliver the reported H0/Base directional blend through a clear inference entry point, preserve reproducible evidence, and document it in English followed by Chinese.

**Scope:** The user's five delivery requests on 2026-10-18 authorize documentation, necessary cleanup, comments, and an official-test interface. Keep the selected formula and weights unchanged. No tuning, new official validation access, full-data refit, commit, or push in this task.

**Architecture:** `pp_pricing` owns frozen inference; `pp_models` retains report experiments. Both share the exact H0 feature transform. Export only the last verified weekly H0 parameter state, with provenance. The announcement branch remains a separate research comparison, not part of the final blend.

## Tasks

- [x] Add inference tests first: parity with trained H0, directional weights, label isolation, UTC/as-of checks, invalid/missing inputs, completed candles, output order and safe CLI output.
- [x] Extract shared pricing-input checks/transform; add necessary English comments to H0 and time-boundary logic without changing arithmetic.
- [x] Add `pp_pricing` Python/CLI interfaces and an explicit export tool. Publish a small JSON parameter snapshot and synthetic examples; no private rows.
- [x] Audit remaining architecture. Retain report-relevant negative results and the sole legacy migration bridge; remove only genuinely unused or misleading entry points/references.
- [x] Update README (English first), inference contract, model/data index references, and a report-to-code evidence checklist.
- [x] Run the full suite, clean-checkout-style inference smoke tests, independent report verification, artifact/source hash checks, and all-week inference parity. Record compact results.

## Acceptance and review focus

1. Official-format input does not require or use outcomes, actual closure, paid premium, or markup.
2. Missing Base or unusable market history fails clearly; no rows are silently dropped or fabricated.
3. Frozen July parameters cannot score earlier timestamps; label-free future inference does not imply weekly retraining or proven out-of-sample superiority.
4. Final prediction equals `alpha * H0 + (1-alpha) * Base`, with Call 0.40 / Put 0.15. Predictions and multipliers retain input order.
5. All 4,640 reported validation predictions can be replayed with their original weekly states; headline metrics and published evidence remain unchanged.

Implementation uses test-first checks for the new interface. Existing reports and audited Word content are preserved. Prior audit edits in this worktree are retained.

## Completion evidence

- 97 tests passed; a clean directory ran both CLI input modes without private data or training libraries.
- 12 weekly states / 4,640 orders reproduced the reported final blend; 19 published file hashes and the audited Word hash matched.
- Independent report verification passed for 97 model summaries and 2,024 risk rows; no new official validation access or real-data fitting.
- Remaining report kernels, migration bridge, and immutable reference statistics are necessary, not obsolete. The obsolete announcement result index was corrected.
- Compact evidence: `docs/results/delivery_verification_20261018.json`; delivery mapping: `docs/reports/2026-10-18-delivery-readiness.md`.

- Independent review completed; artifact-hash verification, mixed timestamp parsing, and Python order-ID consistency were fixed with regression tests. No review findings remain open within this delivery scope.
