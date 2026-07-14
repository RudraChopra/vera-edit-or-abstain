# VERA Main-Track Project Spec

## Project

VERA (Verified Erasure under Reweighting Ambiguity) is a certified
edit-or-abstain protocol for representation interventions under distribution
shift. The project is Paper A: a method paper aimed at ICLR, ICML, NeurIPS,
AAAI, and related journal expansions.

## Core Claim

VERA does not claim to be a universal replacement for erasure methods such as
INLP, LEACE, RLACE, TaCo, SPLINCE, or MANCE++. Given identity and edited
representations for the same examples, it certifies paired target harm and
post-edit leakage for every deployment distribution in a declared bounded
reweighting class. It abstains when the evidence cannot support that external
contract. A finite-family testing layer is inherited from Learn Then Test style
risk control and is not itself a novelty claim.

## Benchmark Evidence

The benchmark package currently includes claim-grade official Waterbirds and
Camelyon17 evidence for VERA, a five-seed claim-grade official-code MANCE++
reference package on Waterbirds, and a large official-code MANCE++ diagnostic
on Camelyon17. Camelyon17 MANCE++ remains diagnostic because full reference
nearest-neighbor manifold estimation is not locally practical at 455,954
examples on the available Mac.

## Submission Boundary

The paper may claim shift-robust paired edit certification only after the new
theorems, simulations, and false-acceptance studies pass
`VERA_AIRTIGHT_SPEC.md`. It may not claim state-of-the-art nonlinear erasure on
every benchmark unless all closest reference baselines are run claim-grade on
those same benchmarks. The earlier finite-candidate validation certificate is
an implementation baseline, not the headline contribution.
