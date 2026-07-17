# Response to the July 17 Adversarial Review

This note records the factual disposition of the attached AAAI review before
any new P0 outcome is generated. It is an internal revision ledger, not
submission-facing evidence and not a substitute for independent review.

## Factual Triage

The review is correct that the present controlled-shift table does not establish
an empirical deployment advantage over IID LTT. At the independent follow-up's
primary setting, IID LTT deployed 109 edits with zero observed shifted-contract
violations and retained 109 of 183 shifted-law safe opportunities. VERA's vector
envelope deployed 59 edits with zero observed violations and retained 59 of 183.
The current manuscript must not imply that this table alone demonstrates VERA's
practical superiority.

The review is also correct that the boosted-tree stress result is serious. The
tree was outside the registered portfolio and found five violations among the
59 VERA vector deployments in the independent follow-up. This does not
mathematically falsify a certificate explicitly scoped to the four registered
attacker families. It does falsify any broader informal interpretation of that
certificate as robust removal of source information.

The review is correct that the exact finite-reference reweighting study is not
a natural deployment study. It establishes known ambiguity-set membership and
tests the finite-sample mathematics, but it cannot by itself establish that a
future public-benchmark split or a real deployment belongs to the declared
ambiguity set. The final study therefore specifies a distinct natural
group-mixture diagnostic with an explicit within-environment conditional-stability
assumption and separate reporting of within-environment discrepancies.

The review is correct that the successful 8,000-observation follow-up was
chosen after a failed 4,000-observation primary. It remains valid only as a
non-pooled, post-failure independent follow-up. It is not a final prospective
confirmation of the revised P0 hypothesis.

Two logistical claims in the review are stale. The repository has a 12-page
named technical supplement, and a private anonymous supplement build exists.
Neither fact makes the package submission-ready: proof review, anonymous archive
construction, source inspection, and venue-policy verification remain open.

The concern about a coherent deployment law is partly addressed in the current
main-paper text: VERA's target and leakage claims are explicitly separate
conditional stress contracts rather than an unstated factorized joint law. The
theorem statement, proof, and implementation still require independent expert
review before that clarification can be treated as verified.

## P0 Repair Plan

The original unused protocol at `research/prereg_vera_p0_confirmation.json`
is transparently superseded before any seed 173--236 outcome is created because
its receipts did not retain construction-fold outcomes needed to replay its
stress-design decision. Version 2, at
`research/prereg_vera_p0_confirmation_v2.json`, is generated only after the
receipt code is committed and before any fresh outcome is created. It retains
the original file and its hash as an immutable pre-outcome record rather than
rewriting it in place.

Version 2 selects the fixed design edit by construction target balanced
accuracy, then selects the supported stress cell from construction-only target
harm and five registered attacker surplus values. The required construction
arrays are recorded in every receipt, so an independent analyzer can replay
that choice without looking at certification or external outcomes.

The final protocol uses the prior completed blocks only as development evidence
and never pools them with the final block.
The protocol registers all four requested budgets
Gamma = 1, 1.1, 1.25, and 1.5, fixes Gamma = 1.25 as the primary profile, and
selects its supported stress cell on construction-only data. It requires the
same candidate edits, construction data, contracts, utility tie-break, and
certification evidence for IID LTT and VERA.

The registered attacker portfolio is expanded from linear logistic regression,
RBF--Nystroem logistic regression, random forest, and a two-layer MLP to add a
gradient-boosted tree. A distance-weighted KNN attacker becomes the new
out-of-portfolio stress family. This change will be evaluated only on fresh
seeds. The prior boosted-tree result remains reported as a negative limitation;
it is not retroactively converted into in-portfolio evidence.

The final protocol has deliberately demanding success conditions: a measurable
IID-LTT shifted-contract exposure, VERA safety at the primary profile, useful
VERA retention at Gamma = 1.1 and 1.25, a paired seed-cluster comparison, and
fully reported KNN stress. Failure of any condition is a result, not a cue to
quietly select a different profile or rerun the protocol under a new name.

## Remaining Human Gates

No automated run can complete the two independent proof checks, cold novelty
and methods reviews, authorship verification, license approval, anonymous
archive inspection, or the AAAI upload. Those remain explicit gates even if the
new experiments are favorable.
