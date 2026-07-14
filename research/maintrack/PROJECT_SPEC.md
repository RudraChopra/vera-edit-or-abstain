# FARO Main-Track Project Spec

## Project

FARO is a certified edit-or-abstain protocol for representation interventions
under distribution shift. The project is Paper A: a method paper aimed at ICLR,
ICML, NeurIPS, AAAI, and related journal expansions.

## Core Claim

FARO does not claim to be a universal replacement for erasure methods such as
INLP, LEACE, RLACE, TaCo, SPLINCE, or MANCE++. The claim is narrower and more
defensible: given a family of candidate representation edits, FARO selects only
frontier edits that satisfy target-preservation and source-leakage constraints,
and it abstains when the evidence does not certify a safe edit.

## Benchmark Evidence

The benchmark package currently includes claim-grade official Waterbirds and
Camelyon17 evidence for FARO, a five-seed claim-grade official-code MANCE++
reference package on Waterbirds, and a large official-code MANCE++ diagnostic
on Camelyon17. Camelyon17 MANCE++ remains diagnostic because full reference
nearest-neighbor manifold estimation is not locally practical at 455,954
examples on the available Mac.

## Submission Boundary

The paper may claim that FARO is a protocol for safe edit selection and
abstention. It may not claim state-of-the-art nonlinear erasure on every
benchmark unless all closest reference baselines are run claim-grade on those
same benchmarks.
