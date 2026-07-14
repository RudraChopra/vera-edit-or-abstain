# Reference Baseline Matrix

Date: July 13, 2026

## Purpose

This matrix prevents VERA from overstating baseline coverage. A baseline can be
used in one of three ways: as a local reference implementation, as an official
upstream reference implementation, or as a scoped proxy stress test. Only the
second category supports reference-parity claims.

## Current Status

| Baseline | Status | Evidence | Allowed claim |
| --- | --- | --- | --- |
| ERM probe | Local reference | Shared frozen representations and locked splits | Standard linear-probe baseline under matched conditions |
| Source-balanced ERM | Local reference | Shared frozen representations and locked splits | Source-reweighted linear baseline |
| Group-reweighted ERM | Local reference | Shared target-source group weights | Robust linear-probe baseline |
| GroupDRO-style probe | Local reference | Shared group-weighted solver | Strong robust probe baseline, not full deep GroupDRO training |
| INLP projection | Local frozen-feature reference row | `research/artifacts/linear_eraser_reference_report.json` | Real INLP-style iterative nullspace row under locked frozen-feature splits |
| LEACE erasure | Official upstream reference row | `research/artifacts/linear_eraser_reference_report.json` plus pinned `concept-erasure` upstream code | Official LEACE frozen-feature reference row under locked splits |
| SPLINCE/SPLICE-style erasure | Scoped proxy | Local task-preserving erasure stress row | Proxy only; no reference-parity claim |
| R-LACE/RLACE-style erasure | Official upstream pinned; current row scoped proxy | `research/artifacts/upstream_baseline_reference_inventory.json` plus local adversarial linear erasure stress row | Proxy only until an exact matched upstream receipt exists |
| TaCo-style erasure | Official upstream pinned; current row scoped proxy | `research/artifacts/upstream_baseline_reference_inventory.json` plus local target-conditioned erasure stress row | Proxy only until an exact matched upstream receipt exists |
| MANCE++ | Official upstream reference on Waterbirds | `research/artifacts/mance_reference_statistical_report.json` | Full official-code MANCE++ comparison on Waterbirds |
| MANCE++ | Official upstream reference on Camelyon17 | `research/artifacts/camelyon17_mancepp_reference_full_nocap_receipt.json` | Full no-cap official-code MANCE++ comparison on Camelyon17 frozen representations |
| MANCE++ | Large official-code diagnostics on Camelyon17 | `research/artifacts/camelyon17_mancepp_reference_80k_diagnostic_receipt.json` | Historical subset diagnostics only; superseded by full no-cap receipt |

## Reviewer-Facing Boundary

The paper may claim that VERA is compared against matched robust probes, real
frozen-feature INLP rows, official LEACE frozen-feature rows, and full
official-code MANCE++ Waterbirds/Camelyon17 baselines. It may also say that
official upstream repositories for MANCE++, R-LACE, TaCo, and LEACE are pinned
locally. It must not claim universal erasure state of the art or reference
parity for SPLINCE, R-LACE, or TaCo until those exact reference
implementations are run under the same splits and audited.

## Next Baseline Work

The highest-value next reference implementations are exact matched receipts for
R-LACE and TaCo using the pinned upstream repositories, or an identified
official SPLINCE/SPLICE implementation. VERA's main claim should remain the
certified edit-or-abstain protocol rather than state-of-the-art erasure.
