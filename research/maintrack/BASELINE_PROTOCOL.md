# VERA Baseline Protocol

## Purpose

Every baseline row must answer the same question as VERA: how much source
leakage is reduced, how much target utility is preserved, and whether the edit
is safe under the declared epsilon and delta thresholds.

## Required Families

- ERM probe and source-balanced ERM.
- GroupDRO-style, group-balanced, IRM-style, and VREx-style probes when the
  benchmark supplies group or environment labels.
- INLP and LEACE for linear concept removal.
- R-LACE/RLACE, TaCo, SPLINCE or SPLICE, and MANCE where faithful implementations are
  available.
- Source-probe projection and VERA frontier candidates as internal stress
  baselines.

## Proxy Labels

Current SPLINCE/SPLICE-style, R-LACE/RLACE-style, TaCo-style, and MANCE-style rows are
proxy stress tests unless the official reference implementation is actually run
and audited. Tables and captions must say proxy when proxy code is used.

## Metrics

Each official row reports target balanced accuracy, worst-group or worst-domain
accuracy, source leakage, selected edit or abstention decision, seed-level
variation, paired confidence interval evidence, and a receipt. Camelyon17-WILDS
is bounded to hospital-shift representation reliability; it is not a clinical
deployment baseline.
