# VERA Main-Track Research Program

## Identity

**VERA: Support-Aware Certification of Representation Edits Under Deployment Shift**

VERA is a general machine-learning method for deciding whether a proposed
representation edit remains acceptable under a declared deployment shift. It is
software-only and domain-general; the evidence spans vision, medical vision,
text, and time series. Camelyon17 is a high-stakes reliability benchmark, not a
clinical-safety claim.

## Core Thesis

Concept erasers construct interventions, but a deployment system still needs to
know how much shift an intervention can survive. VERA evaluates paired target
harm against the identity edit and leakage from fresh, heterogeneous attackers.
For each candidate, it lower-certifies a support-aware vector of groupwise
density-ratio budgets under which every registered contract holds. The output
is a certified shift envelope, its common-radius summary, limiting contracts,
or `ABSTAIN`.

The finite-candidate decision layer is an application of Learn Then Test and
related distribution-free risk control. The novelty claim is deliberately
narrower: the paired multi-contract representation-edit shift envelope, its simultaneous
lower certificate over a continuum of deployment budgets, its common-radius
geometry, and its support-mismatch boundary.
Its evidence-efficiency extension allocates a fixed certification budget to
minimize worst normalized additive DKW slack for one candidate selected on an
independent design fold. This is a certificate-specific surrogate, not a claim
of generic active-design novelty or frontier-wide power optimality.

## Evidence Contract

The locked evidence package has four parts. The exact balanced study contains
54 cells: six validation sizes, three error levels, three shift budgets, and
2,000 replicates per cell. An independent implementation replays every seeded
cell and checks both false acceptance and predicted abstention. A separate
216-cell exact grid varies validation size, candidate-family size,
validated-environment count, and error level, again with 2,000 replicates per
cell; each family includes the identity control action in its multiplicity
count.

The original real study is a preregistered 200-run matrix:

- five official-code erasers: INLP, R-LACE, LEACE, TaCo, and MANCE++;
- five datasets: Waterbirds, Camelyon17-WILDS, CivilComments-WILDS, Bios, and
  GaitPDB; and
- eight untouched seeds (5--12) with one shared split, preprocessing,
  target-probe, and attacker protocol per dataset. Pilot seeds 0--4 are
  excluded from confirmation.

Every run must carry the locked preregistration hash, the same runner commit, a
pinned and clean upstream checkout, split hashes, and hashes for its per-example
audit arrays. Missing or proxy rows fail closed.

A disjoint 800-run replication on seeds 13--44 establishes the prior IID
uncertainty-control result. The central shift experiment is separately locked
at `Gamma=1.1`: four supported datasets, five official eraser families, and 64
fresh seeds (45--108), for 1,280 runs. Its deployment laws are controlled
reweightings with machine-verified ambiguity membership. A separate design fold
chooses the shifted supported cell and the locked square-score evidence
allocation; final shifted-law outcomes are untouched by selection. A
pre-outcome additive-allocation extension uses disjoint certification streams
and cannot replace the primary decision. Camelyon17 is retained outside this
matrix as the unsupported-hospital boundary.

That first 64-seed controlled-shift primary passed safety, paired reduction,
and vector/common advantage but failed usefulness. A non-pooling follow-up was
therefore locked before any seeds 109--172 outcomes existed, using the same
supported datasets and official eraser frontier but promoting the previously
registered 8,000-observation targeted-allocation sensitivity setting to a new
follow-up primary. The follow-up completed another 1,280 official-method runs,
passed the strict receipt audit and sealed two-way replay comparison, and
passed all registered gates: validation selection violated 24/186 deployed
decisions, whereas VERA deployed 59 edits with zero shifted-contract
violations and retained 59/183 oracle-safe opportunities. The first primary
remains reported as failed and is not pooled with the follow-up.

## Scientific Boundaries

VERA does not certify clinical deployment, universal concept removal, or
security against every measurable attacker. Its shift guarantee is restricted
to deployment distributions absolutely continuous with respect to the
certification distribution and bounded by the declared density ratio. A
deployment environment absent from certification is not covered; the
unsupported-cell theorem shows why no certification-data-only protocol can repair that
without an additional structural assumption.

Configuration-level tests that reuse the same seeds, samples, thresholds, or
nested fractions are not treated as independent evidence. The analysis retains
the preregistered diagnostics but uses seed-blocked sensitivity analyses for
inferential claims. Corrections are recorded in
[`ANALYSIS_CORRECTION_LEDGER.md`](ANALYSIS_CORRECTION_LEDGER.md).

## Completion Rule

The scientific execution contract is
[`VERA_AIRTIGHT_SPEC.md`](VERA_AIRTIGHT_SPEC.md). The authoritative current
submission tracker is [`GOAL_1_58_STATUS.md`](GOAL_1_58_STATUS.md); older
generated readiness artifacts describe earlier source states and cannot
override it. Submission readiness requires all
scientific gates, a complete anonymous and named paper package, four
role-specific cold reviews, and one fresh post-revision review from researchers
who publish in machine learning. Internal scripts
cannot substitute for those reviews or guarantee acceptance. OpenAI Codex's
extensive assistance is disclosed in the manuscript; every listed human author
must complete [`HUMAN_AUTHOR_VERIFICATION_GATE.md`](HUMAN_AUTHOR_VERIFICATION_GATE.md)
and personally verify the entire submission.

The shorter seven-goal map is [`SEVEN_GATE_STATUS.md`](SEVEN_GATE_STATUS.md).
Both live trackers intentionally remain red until every empirical,
presentation, and human-only condition has direct evidence; a green
file-presence audit cannot override them.

Current paper source remains authoritative. The tracked named AAAI PDFs were
rebuilt with local TeX Live/pdfTeX on July 17, 2026 and their rendered text
includes the 24/186 versus 0/59 follow-up result. Fresh anonymous PDFs were also
built and copied to a private review location outside the named checkout, but
are intentionally not tracked in this named branch per the double-blind release
boundary. Any submission still requires a final human page-by-page PDF
inspection and source/archive check.

## Three-Venue Scientific Record

The AAAI-27, ICLR 2027, and NeurIPS 2027 variants are three explanations of one
scientific record, not independent experiments. Their shared requirements are
frozen in [`THREE_VENUE_CONTENT_VARIANT_SPEC.md`](THREE_VENUE_CONTENT_VARIANT_SPEC.md),
and every final number, failed gate, interval, and title branch must come from
the single interface in
[`SHARED_RESULT_MANIFEST_SPEC.md`](SHARED_RESULT_MANIFEST_SPEC.md) and its
machine-readable
[`SHARED_RESULT_MANIFEST_SCHEMA.json`](SHARED_RESULT_MANIFEST_SCHEMA.json).
Outcome-blind
ICLR and NeurIPS content sources are under
[`venue_variants/`](venue_variants/). Their section-level equivalence is bound
by [`VENUE_VARIANT_CLAIM_MAP.md`](VENUE_VARIANT_CLAIM_MAP.md). They remain
format-pending until official 2027 materials are available and verified.
