# FARO Novelty Sweep

## Scope

This sweep compares FARO against INLP, LEACE, RLACE, R-LACE, TaCo, SPLINCE,
SPLICE, MANCE, MANCE++, and domain generalization methods.

## Current Defensible Contribution Sentence

FARO introduces a certified frontier-aware edit-or-abstain decision layer for
source removal: it evaluates candidate representation edits under simultaneous
target-preservation and source-leakage uncertainty, selects the smallest
certified safe edit, or returns an auditable abstention certificate.

## Distinction From Erasure

INLP, LEACE, RLACE, R-LACE, TaCo, SPLINCE, SPLICE, MANCE, and MANCE++ define
ways to remove or reduce encoded concept information. FARO defines when an edit
should be accepted at all. It can wrap erasers, compare erasers, or abstain when
no candidate in the audited family satisfies the preservation contract.

## Distinction From Domain Generalization

Domain-generalization methods optimize predictive robustness under environment
shift. FARO audits representation edits after source labels are known. The
object of study is the certified leakage-utility frontier, not training a new
robust predictor from scratch.

## 2026 Update

MANCE++ is the closest recent nonlinear erasure baseline. The local packet now
contains full official-code Waterbirds MANCE++ evidence and a large Camelyon17
diagnostic. Manuscript claims must preserve that boundary.
