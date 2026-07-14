# VERA Reproducibility Checklist

## Scope

This checklist defines what must be reproducible before Paper A is submitted to
ICLR, NeurIPS, AAAI, ICML, ICDM, or a journal extension. It covers the VERA
method paper, not the old science-fair materials. The authoritative
machine-readable manifest is
`research/configs/faro_paper_a_reproducibility.json`.

## Fixed Claims

The paper's reproducible claim is that VERA estimates a leakage-utility
frontier over candidate representation edits, applies the smallest edit whose
simultaneous intervals certify source reduction and target-risk preservation,
and abstains when the certified safe set is empty. The reproducibility packet
does not support clinical deployment claims, universal concept-erasure
optimality, or state-of-the-art claims against reference implementations that
have not been run.

## Required Seeds and Statistics

Official benchmark rows must use seeds `0, 1, 2, 3, 4` unless the benchmark is
explicitly deterministic and the manuscript says why. Every claim-ready row
must report seed-level metrics, 95 percent confidence intervals, paired
comparisons against the predeclared strongest relevant baseline, target utility,
worst-group or worst-domain utility when applicable, residual source leakage,
and the selected edit or abstention decision.

## Reproduction Commands

The current checked reproduction path is:

```bash
python3 research/scripts/audit_benchmark_claims.py
/tmp/faro-torch-venv/bin/python research/scripts/audit_reference_baseline_scope.py --no-fail
/usr/bin/python3 research/scripts/audit_upstream_baseline_references.py --no-fail
/usr/bin/python3 research/scripts/audit_aaai2027_source_readiness.py --no-fail
/usr/bin/python3 research/scripts/run_camelyon17_numpy_store_benchmark.py --store-dir /Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_full_numpy_store --results research/artifacts/camelyon17_wilds_official_multiseed_results.csv --receipt research/artifacts/camelyon17_wilds_official_result_receipt.json --statistics research/artifacts/camelyon17_wilds_official_statistical_report.json --seeds 0,1,2,3,4
/tmp/faro-torch-venv/bin/python research/scripts/run_camelyon17_projection_frontier_certificate.py
/usr/bin/python3 research/scripts/audit_mance_camelyon_scaling.py --no-fail
/usr/bin/python3 research/scripts/build_maintrack_figures.py
python3 research/scripts/audit_maintrack_readiness.py --no-fail
python3 research/scripts/run_faro_adversarial_review.py --no-fail
tectonic -X compile research/maintrack/faro_main.tex
```

The full historical artifact runner remains:

```bash
python3 research/scripts/reproduce_paper_artifacts.py
```

That runner is useful for broad regression checks, but the main-track readiness
audit is the authoritative submission gate for Paper A.

## Current Claim-Ready Rows

Waterbirds and Camelyon17-WILDS are the current durable official claim-ready
families in the local packet. Waterbirds has full local image metadata, frozen
ResNet-18 embeddings, five seeds, paired statistics, and a passing receipt.
Waterbirds is a negative or abstention row: VERA abstains under the locked rule
while group-reweighted ERM is stronger on worst-group accuracy.
Camelyon17-WILDS has a complete 455,954-example frozen ResNet-18 embedding
store, full NumPy-store conversion, five locked protocol rows, paired
statistics, a passing high-stakes official receipt, and a full
projection-frontier abstention certificate. It supports
representation-reliability claims only, not clinical deployment claims.
CivilComments-WILDS has prior full-store stress artifacts, but it is not
counted as claim-ready while its receipt and paired statistical report are
dataless local placeholders.

## Current Required Blocker

The high-stakes benchmark blocker is cleared by Camelyon17-WILDS. The remaining
submission blocker is that the adversarial internal review and readiness audit
must clear critical and major findings. CivilComments-WILDS remains a
non-durable optional/third-family row until its local receipt and statistics are
rehydrated or regenerated.

The upstream baseline inventory now pins official local repositories for
MANCE++, R-LACE, TaCo, and LEACE. These pins improve reproducibility, but
R-LACE, TaCo, and LEACE remain proxy rows until exact matched receipts are
generated. The AAAI-27 source readiness audit currently verifies source
integrity and reports a single expected warning: local final AAAI PDF
compilation still requires PDFLaTeX.

## Artifact Policy

Small scripts, configuration files, figure builders, receipts, statistical
reports, and manuscript sources should be released. Third-party datasets and
large embedding stores should not be committed to the public repository. The
release should instead provide download instructions, manifest paths, checkpoint
hashes, validation reports, and regeneration commands.

## Pre-Submission Checklist

- The novelty lock, algorithm specification, theory target, baseline protocol,
  statistical integrity plan, and this reproducibility checklist are present.
- The machine-readable reproducibility manifest validates with
  `audit_reproducibility_packet.py`.
- The benchmark claim audit reports the current official claim-ready rows.
- The reference baseline scope audit verifies that proxy rows and official
  reference rows are labeled separately.
- The upstream baseline inventory verifies the pinned official-code repositories
  used to define future reference-parity work.
- The AAAI-27 source readiness audit has no source failures; any PDFLaTeX
  warning must be resolved before actual AAAI upload.
- The main-track figure builder regenerates all manuscript figures from
  existing CSV/JSON artifacts.
- The manuscript compiles locally from `research/maintrack/faro_main.tex`.
- The readiness audit has no failing checks.
- The adversarial internal review has zero critical and zero major findings.
- Camelyon17 or an equivalent high-stakes benchmark has a passing official
  receipt and paired statistics.
