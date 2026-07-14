# MANCE Reference Status

Date: July 13, 2026

## What Is Done

- Official upstream repository cloned at `/Volumes/Backups/FARO/external/mance`.
- FARO adapter added at `research/scripts/run_mance_reference_numpy_store.py`.
- Full Waterbirds official-code MANCE++ reference run completed on all official
  train/validation/external embeddings for seeds 0-4.
- Waterbirds statistical report written to
  `research/artifacts/mance_reference_statistical_report.json`.
- Camelyon17 official-code MANCE++ diagnostic completed first on a stratified
  20,000 train / 8,000 validation / 8,000 external subset, then expanded to
  40,000 train / 16,000 validation / 16,000 external and 80,000 train /
  24,000 validation / 24,000 external.
- Full no-cap Camelyon17 official-code MANCE++ run completed on all 302,436
  training, 68,464 validation, and 85,054 external frozen-representation
  examples.
- Receipts written to
  `research/artifacts/camelyon17_mancepp_reference_diagnostic_receipt.json`,
  `research/artifacts/camelyon17_mancepp_reference_40k_diagnostic_receipt.json`,
  and
  `research/artifacts/camelyon17_mancepp_reference_80k_diagnostic_receipt.json`.
- Full no-cap receipt written to
  `research/artifacts/camelyon17_mancepp_reference_full_nocap_receipt.json`
  and mirrored to
  `/Volumes/Backups/FARO/artifacts/mance_reference/camelyon17_mancepp_reference_full_nocap_receipt.json`.

## Waterbirds Claim-Grade Result

| Metric | Before mean | After mean | Delta mean | Delta 95% CI |
| --- | ---: | ---: | ---: | ---: |
| External source leakage balanced accuracy | 0.905937 | 0.446842 | -0.459096 | 0.006679 |
| External target balanced accuracy | 0.739030 | 0.691529 | -0.047502 | 0.002494 |
| External worst target-source accuracy | 0.490654 | 0.528349 | 0.037695 | 0.005410 |
| Validation source leakage balanced accuracy | 0.906550 | 0.452296 | -0.454254 | 0.011561 |
| Validation target balanced accuracy | 0.745691 | 0.670909 | -0.074781 | 0.007879 |

This is a full-split, five-seed, official-code reference baseline for
Waterbirds.

## Camelyon17 Claim-Grade Result

The strongest Camelyon17 run now uses the full no-cap frozen-representation
store: 302,436 training, 68,464 validation, and 85,054 external examples. It is
marked claim-grade by the reference adapter.

| Metric | Before | After | Delta |
| --- | ---: | ---: | ---: |
| Validation source leakage balanced accuracy | 0.849188 | 0.563460 | -0.285728 |
| Validation target balanced accuracy | 0.897353 | 0.887627 | -0.009726 |
| External target balanced accuracy | 0.890505 | 0.874104 | -0.016401 |
| External worst target-source accuracy | 0.846545 | 0.813695 | -0.032850 |

The earlier 20,000 / 8,000 / 8,000, 40,000 / 16,000 / 16,000, and
80,000 / 24,000 / 24,000 diagnostics showed the same qualitative pattern:
strong validation source-leakage reduction with target-utility loss. External
source leakage remains blank for Camelyon17 because the binary source encoding
has a single source class in the external split.

## Claim Boundary

MANCE++ is now claim-grade for Waterbirds and Camelyon17. Manuscript claims
must still state the method boundary: this is official-code MANCE++ evidence
under FARO's frozen-representation protocol, not proof that FARO is a universal
state-of-the-art erasure method. Exact upstream R-LACE, TaCo, LEACE, and
SPLINCE/SPLICE receipts remain optional strengthening beyond the scoped
protocol claim.
