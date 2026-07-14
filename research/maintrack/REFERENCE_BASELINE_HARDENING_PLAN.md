# VERA Reference Baseline Hardening Plan

## Current Status

The local VERA packet has claim-ready official rows for Waterbirds and
Camelyon17-WILDS, but the adversarial review is intentionally not submission
ready after the July 2026 novelty update. The remaining issue is not the VERA
method specification or the Camelyon evidence. The issue is baseline
defensibility: current SPLINCE/SPLICE-style, R-LACE-style, and TaCo-style rows
are proxy stress tests unless the corresponding reference implementations are
run or the claims are scoped so reference parity is not required.

As of July 13, 2026, the official upstream MANCE repository has been cloned and
wired into VERA through `research/scripts/run_mance_reference_numpy_store.py`.
Waterbirds now has a full-split, five-seed, official-code MANCE++ reference
package with a claim-grade statistical report. The mean external source
leakage balanced accuracy drops from 0.905937 to 0.446842, while mean external
target balanced accuracy drops from 0.739030 to 0.691529 and worst-group
target accuracy improves from 0.490654 to 0.528349.

Camelyon17 now has a full no-cap official-code MANCE++ reference row on all
302,436 training, 68,464 validation, and 85,054 external frozen-representation
examples. Validation source leakage drops from 0.849188 to 0.563460, while
validation target balanced accuracy drops from 0.897353 to 0.887627. External
target balanced accuracy drops from 0.890505 to 0.874104, and external worst
target-source accuracy drops from 0.846545 to 0.813695. This clears the prior
Camelyon-specific MANCE++ reference blocker under VERA's frozen-representation
protocol.

The scaling audit at
`research/artifacts/camelyon17_mance_scaling_feasibility.json` estimated the
full no-cap run at 4.83 hours under a linear lower-bound model and 19.78 hours
under the most recent observed superlinear scaling. The actual full no-cap run
completed in 11,760.208 seconds, below the linear lower-bound estimate.

The upstream inventory at
`research/artifacts/upstream_baseline_reference_inventory.json` now pins
official repositories and commits for MANCE++, R-LACE, TaCo, and LEACE. This
improves reproducibility but does not convert the current R-LACE or TaCo rows
into reference-parity rows. Matched receipts under VERA's locked splits are
still required before the manuscript can claim exact upstream comparisons for
those methods.

INLP and LEACE have now been hardened separately through
`research/scripts/run_linear_eraser_reference_rows.py`. The resulting
`research/artifacts/linear_eraser_reference_report.json` contains real
frozen-feature INLP rows and official LEACE rows using the pinned
`concept-erasure` implementation on Waterbirds, Camelyon17, and GaitPDB. This
clears the cheap-linear-eraser reviewer objection without changing the broader
claim boundary: the paper may report these rows, but it still must not claim
universal erasure state of the art.

## Remaining Reference-Parity Work

The strongest MANCE/MANCE++ blocker is now cleared for Waterbirds and
Camelyon17, and the cheap INLP/LEACE blocker is cleared for frozen-feature
reference rows. Remaining reference-parity work is optional strengthening:
exact matched receipts for R-LACE, TaCo, and an identified official
SPLINCE/SPLICE implementation. Until those receipts exist, those rows remain
proxy stress tests, and VERA should continue to be framed as a certified
selection and abstention layer over candidate edits rather than a universal
state-of-the-art erasure method.

## Required Artifacts

A reference MANCE/MANCE++ run should write these artifacts.

| Artifact | Purpose |
| --- | --- |
| `research/artifacts/waterbirds_mancepp_reference_seed*_receipt.json` | Waterbirds reference receipts with claim boundary |
| `research/artifacts/mance_reference_camelyon17_result_receipt.json` | Camelyon17 reference receipt or documented infeasibility |
| `research/artifacts/mance_reference_statistical_report.json` | Paired comparison against VERA and GroupDRO-style baselines |
| `research/artifacts/mance_reference_environment.json` | Commit, dependency, and runtime provenance |
| `research/artifacts/faro_baseline_fairness_report.json` | Materialized baseline audit report with proxy/reference labels |

Current diagnostic artifacts:

| Artifact | Status |
| --- | --- |
| `research/artifacts/mance_reference_statistical_report.json` | Claim-grade five-seed Waterbirds MANCE++ reference statistics |
| `research/artifacts/camelyon17_mancepp_reference_full_nocap_receipt.json` | Claim-grade full no-cap Camelyon17 MANCE++ reference receipt |
| `research/artifacts/camelyon17_mancepp_reference_diagnostic_receipt.json` | Official upstream MANCE++ code, stratified Camelyon17 subset, not claim-grade |
| `research/artifacts/camelyon17_mancepp_reference_40k_diagnostic_receipt.json` | Expanded official upstream MANCE++ Camelyon17 subset diagnostic, not claim-grade |
| `research/artifacts/camelyon17_mancepp_reference_80k_diagnostic_receipt.json` | Largest subset diagnostic before the full no-cap claim-grade receipt |
| `research/artifacts/camelyon17_mance_scaling_feasibility.json` | Local full no-cap runtime feasibility estimate for Camelyon17 MANCE++ |
| `research/artifacts/upstream_baseline_reference_inventory.json` | Pinned official repository inventory for MANCE++, R-LACE, TaCo, and LEACE |
| `research/artifacts/linear_eraser_reference_report.json` | Real frozen-feature INLP rows and official LEACE rows on Waterbirds, Camelyon17, and GaitPDB |
| `research/artifacts/waterbirds_mancepp_reference_seed*_receipt.json` | Official upstream MANCE++ code, full Waterbirds splits, claim-grade |
| `/Volumes/Backups/FARO/artifacts/mance_reference/camelyon17_mancepp_reference_diagnostic_receipt.json` | External-drive copy of the same diagnostic receipt |

## Pass Criteria

The baseline blocker is cleared when the adversarial review can verify all of
the following conditions.

The baseline fairness report is materialized locally and reports zero failing
checks. Every table and caption distinguishes reference implementations from
proxy stress tests. The MANCE++ receipts use the locked splits, source labels,
target labels, metrics, seeds, and confidence-interval policy expected by VERA.

For AAAI/NeurIPS-level claims, a pinned upstream repository alone is not enough.
Each reference-parity row must have a receipt, environment provenance, split
hashes, and metric definitions matching VERA's reported rows.

## Claim Boundary

The defensible contribution sentence remains: VERA is a certified
edit-or-abstain protocol over a candidate eraser family, not a replacement for
every modern eraser. This framing survives INLP, LEACE, R-LACE, TaCo, SPLINCE,
and MANCE because those methods propose edits, while VERA decides whether an
edit is certifiably safe.
