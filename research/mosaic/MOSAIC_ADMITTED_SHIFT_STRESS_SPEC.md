# MOSAIC admitted-shift stress specification

## Status

This is a post-review deterministic stress analysis. Its design was finalized
after a small development pilot, so it is not described as preregistered or as
fresh confirmation. Every eligible job from the existing locked 100-job bridge
study is included.

## Question

For a release accepted by the direct target-table comparator, does there exist
an explicit target law inside the corresponding MOSAIC bridge class on which
that direct release violates the registered contract?

## Construction

At each utility threshold, retain every job deployed by the direct target-table
rule. For its selected candidate:

1. Use the empirical reference table as a fixed population center. This center
   belongs to its own simultaneous confidence region.
2. Use the strict-v2 learned source-blind transform and retained mass for each
   task label.
3. Enumerate the direct channel's finite Bayes-attacker assignments. For the
   maximizing assignment, put each source-specific residual distribution on
   the fine-token row that maximizes that source's correct released-token mass.
4. Form the alternative population
   `q* = t p T + (1 - t) r*` exactly. This is an admitted member of the learned
   bridge class by construction.
5. Evaluate the direct release and the candidate-matched strict MOSAIC release
   on the same `q*`. MOSAIC deploys only when its strict outward privacy and
   utility bounds pass the registered threshold; otherwise it abstains.

The primary threshold is worst-stratum error 0.40 with normalized within-label
source advantage 0.35. Secondary rows report the complete registered threshold
frontier: 0.30, 0.35, 0.40, 0.45, and 0.49.

## Required checks

- Reconstructed laws are valid probability tensors.
- Every reconstructed law satisfies the displayed bridge decomposition to
  numerical tolerance.
- The reconstructed direct risk matches the independently enumerated exact
  worst-class risk.
- Direct target-table selection bounds pass before stress evaluation.
- Every deployed strict MOSAIC release passes its outward bounds and its exact
  population evaluation on the reconstructed law.
- Every job and abstention is retained; no dataset or seed is filtered after
  computing the stress outcome.

## Claim boundary

The result is an exact, real-table-anchored counterfactual stress test over the
class MOSAIC certifies. It does not claim that the constructed residual shift
was observed in the untouched diagnostic fold or that it predicts the frequency
of future real-world drift.
