# Alzheimer Environment-Shift Scope

## Boundary

Alzheimer or dementia datasets are not part of the current claim-grade packet.
They can be added only if the dataset exposes a meaningful source or environment
label such as site, scanner, cohort, protocol, acquisition year, or demographic
collection stratum.

## Why This Matters

VERA studies target-preserving source removal. A medical dataset without an
explicit source label does not test VERA's core question. It would create a
weak medical story and a reviewer could correctly call it opportunistic.

## Current Medical Row

Camelyon17-WILDS is the current high-stakes benchmark row because it has a
hospital-shift structure and official splits. The claim is representation
reliability under shift, not clinical safety or deployment readiness.
