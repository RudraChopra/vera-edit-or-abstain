# Novelty Decision

## Verdict

The original VERA paper is beyond repair as a top-main-track submission if its
only new claim is finite-candidate confidence-bound selection with abstention.
That mechanism is an application of Learn Then Test style risk control. It can
remain an implementation component, but it cannot remain the contribution.

## Replacement Research Question

What is the largest deployment reweighting budget under which a representation
edit can be simultaneously certified to preserve target utility and resist
registered sensitive-attribute recovery attacks, and when must that radius
collapse to zero?

The paired quantity matters. For target loss, VERA evaluates the incremental
harm of an edit relative to the identity intervention on the same example. For
leakage, VERA audits fresh heterogeneous attackers and certifies the worst
preregistered attacker, not the eraser's native probe. This removes the need to
name or discover environments in the primary method.

## New Name

VERA now expands to **Verified Erasure under Reweighting Ambiguity**.

## Required Delta Over Prior Work

The final paper must prove and evaluate all of the following together:

1. a simultaneous lower confidence bound on each edit's maximum common shift
   radius, obtained by inverting robust paired-harm and attacker-leakage bands
   over a continuum of density-ratio budgets;
2. a worst-group mixture corollary for the annotated-group setting;
3. a support-mismatch impossibility theorem showing that a nontrivial
   distribution-free certificate is impossible when deployment can put mass
   where validation has none;
4. a comparison showing why point selection, an IID LTT certificate, and a
   shift-robust paired certificate make measurably different deployment
   decisions; and
5. an honest boundary that no finite attacker portfolio certifies erasure
   against all measurable recovery algorithms.

## Kill Criteria

The pivot fails and must be redesigned again if the literature audit finds the
same erasure-shift-radius object and guarantee, the robust bounds are vacuous on
all five real datasets, or naive selection does not produce honest external
contract violations. Internal audit scripts cannot waive these criteria.
