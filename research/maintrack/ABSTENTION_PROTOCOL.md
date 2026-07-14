# VERA Abstention Protocol

## Decision Labels

VERA has two terminal decisions: `EDIT` and `ABSTAIN`.

## Calibration

For each candidate edit, VERA computes calibrated uncertainty intervals for
target balanced accuracy, worst target-source accuracy, and source leakage
balanced accuracy. Calibration must be fixed before external metrics are used.

## ABSTAIN Condition

VERA returns ABSTAIN when every candidate edit fails at least one certified
constraint:

- target balanced accuracy falls below the allowed tolerance,
- worst target-source accuracy falls below the allowed tolerance,
- source leakage reduction is too small to justify the edit,
- or confidence intervals are too wide to certify the edit.

## Required Evidence

The paper must show both synthetic and real benchmark abstention cases. The
synthetic case demonstrates geometry; the real benchmark case demonstrates that
ABSTAIN is not just a toy behavior.
