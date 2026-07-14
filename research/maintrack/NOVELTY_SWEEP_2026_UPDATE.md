# FARO Novelty Sweep Update, July 2026

## Current Defensible Contribution Sentence

FARO reframes source or concept removal as a certified edit-or-abstain
selection problem: given a fixed representation and a candidate family of
erasers, it estimates the leakage-utility frontier, selects the smallest edit
whose simultaneous intervals certify source reduction and target-risk
preservation, and returns an abstention certificate when no candidate is
certifiably safe.

## Why This Survives Close Prior Work

INLP removes linearly decodable attributes by repeatedly training classifiers
and projecting onto null spaces. R-LACE frames linear erasure as a minimax game.
LEACE gives a closed-form linear eraser. TaCo targets nonlinear recovery of
protected attributes. SPLINCE directly protects target covariance while erasing
linear concept predictability. MANCE and MANCE++ add a recent manifold-aware
nonlinear erasure family with broad text and vision evaluation.

Those methods make FARO's scope narrower but sharper. FARO should not claim to
be the strongest eraser. It should claim to be the decision protocol that asks
whether any candidate eraser, including methods such as LEACE, R-LACE, TaCo,
SPLINCE, or MANCE, is safe enough to deploy under predeclared target-risk and
source-leakage thresholds. The novelty is the certified frontier and abstention
contract, not another unconditional projection.

## Reviewer Attack And Defense

A reviewer can say: "SPLINCE already preserves task information, and MANCE is a
state-of-the-art nonlinear eraser, so why is FARO needed?" The answer is that
task-preserving erasure and manifold-aware erasure still output an edit. FARO
adds a statistical decision boundary around any such edit. If the confidence
intervals cannot certify both target preservation and source reduction, FARO
does not edit and writes a certificate explaining why.

## Required Claim Boundary

The manuscript may say that FARO is compatible with modern erasers as candidate
edits and that it provides a calibrated selection/abstention layer. It must not
say that FARO dominates MANCE, SPLINCE, TaCo, R-LACE, LEACE, or INLP unless the
corresponding reference implementations are actually run under matched splits,
representations, metrics, and confidence intervals.

## Next Hardening Step

Before external submission, either run a reference MANCE/MANCE++ baseline or
state explicitly that the current MANCE-style row is a proxy stress test. The
paper remains strongest if FARO is evaluated as a protocol that can abstain
around strong erasers rather than as a replacement eraser.
