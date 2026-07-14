# FARO Statistical Integrity

## Claim-Ready Unit

A claim-ready benchmark row requires a materialized receipt, a statistical
report, locked splits, no data leakage, source leakage metrics, target utility
metrics, worst-group or worst-domain metrics when applicable, and an explicit
claim boundary.

## Seeds

Official stochastic rows use seed list `0, 1, 2, 3, 4`. Deterministic rows may
repeat the locked protocol rows only when the receipt states that the solver is
deterministic and the manuscript does not treat repeated rows as independent
randomness.

## Confidence Interval Policy

Reports must include a 95 percent confidence interval or a clearly scoped
deterministic equivalent. Paired comparisons should use the predeclared
strongest relevant baseline, usually GroupDRO-style or group-reweighted ERM for
the current official rows.

## Current Official Rows

Waterbirds and Camelyon17-WILDS are the durable claim-ready rows in the local
packet. CivilComments-WILDS remains prior non-durable stress evidence until its
receipt and statistical report are materialized locally. Camelyon17-WILDS is a
high-stakes representation-reliability benchmark, not evidence of clinical
safety.
