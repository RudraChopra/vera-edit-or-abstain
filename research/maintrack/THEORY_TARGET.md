# FARO Theory Target

## Safe-Acceptance Theorem

The current manuscript states and proves a uniform safe-acceptance and
abstention theorem. Let `U_ext(a)` be external target utility,
`U_val_hat(a)` be validation target utility, `tau` be the minimum acceptable
target utility, and `epsilon` be a uniform validation-to-external uncertainty
radius. On the event that every candidate edit is covered by this radius, any
edit accepted by FARO has external target utility at least `tau`. If no
candidate satisfies the lower-confidence-bound condition, FARO returns
`ABSTAIN`.

## Simultaneous Frontier Theorem

The manuscript now also states and proves a simultaneous frontier
certification theorem. For a finite candidate frontier, FARO accepts only edits
whose lower confidence bound on external target utility exceeds the target
threshold and whose upper confidence bound on external source leakage remains
below the leakage threshold. On the event that the validation intervals
simultaneously cover all frontier target and leakage quantities, any accepted
edit satisfies both external constraints. If the certified feasible set is
empty, FARO returns `ABSTAIN`.

## Why This Is Nontrivial

The theorem is not the obvious claim that erasure can hurt accuracy. The useful
statement is about decision safety: FARO separates edit construction from edit
deployment and proves that the abstention rule controls false acceptance of
unsafe edits under explicit uncertainty assumptions.

## False-Acceptance Control

The manuscript now includes an explicit corollary: if FARO's simultaneous
validation intervals cover every target and leakage quantity on the audited
frontier with probability at least `1-delta`, then the probability that FARO
accepts an edit violating either external constraint is at most `delta`. This
turns the coverage event in the frontier theorem into the reviewer-facing risk
statement: false acceptance is controlled by the failure probability of the
simultaneous interval system.

## Proof Obligations

The proof defines the candidate frontier, the target-preservation constraint,
the leakage constraint, the confidence radii, and the event on which validation
estimates uniformly cover external metrics. The cost of conservatism remains:
FARO may abstain even when an edit would have worked, but it should not accept
edits outside the certified region on the coverage event.

## Finite-Sample Validation Certificate

The manuscript now includes a Hoeffding plus union-bound theorem for finite
candidate frontiers. For each edit and each target class or target-source
group, the validation recall/accuracy estimate receives a simultaneous
high-probability confidence radius. Balanced accuracy inherits an averaged
radius over class recalls, and worst-group accuracy inherits a conservative
maximum group radius. These radii instantiate the abstract `epsilon_U` and
`epsilon_L` terms in the simultaneous frontier theorem.

The next theory upgrade should tighten these conservative radii, for example
with empirical Bernstein bounds, bootstrap simultaneous intervals, or a
stability analysis for learned probes.
