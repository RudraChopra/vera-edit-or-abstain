# Cold AAAI Main-Track Review: MOSAIC

Review date: 2026-07-18
Evidence freeze: corrected bridge v2 and final anonymous package

## Recommendation

**Accept / weak-accept leaning (approximately 6.5 on a 10-point scale).** This
is a serious AAAI main-track submission with a novel technical combination, a
theorem that addresses deployment shift directly, and unusually strong
reproducibility evidence. I would submit it to the main track rather than a
workshop.

I would not honestly assign strong accept. The external guarantee remains
conditional on the target population represented by a labeled bridge sample,
the primary real-data deployments come from one dataset, and the claim-grade
real result required a transparently disclosed post-audit software correction.
Independent expert proof review is also still missing.

## Summary

MOSAIC releases one persistent task-specific stochastic token, or abstains. A
single confidence region on the pre-release token table covers every channel
and decoder optimized on that table. A new finite-sample bridge program uses
labeled target data to learn one source-blind Markov transform and the largest
uniformly supportable retained mass; unexplained source-specific contamination
is charged adversarially. The resulting program controls the exact finite-token
Bayes source attacker and worst-stratum task error under the certified shift
class.

## Strengths

1. The paper squarely addresses Learn Then Test, conformal risk control, FARE,
   robust validation, and Blackwell/Le Cam comparison. It claims none of those
   ingredients as new. Table 1 makes the narrow compositional delta explicit.
2. The main guarantee is about an external population admitted by a labeled
   target bridge. It no longer assumes that a validation interval somehow
   covers a shifted external metric.
3. The synthetic reason-for-existence experiment is decisive. Naive continuum
   selection falsely accepts 42.1% of 1,000 hard-boundary tables, while MOSAIC
   records none. In the retention cell, MOSAIC safely deploys 56.3% versus 2.5%
   for held-out certification and 3.3% for finite LTT.
4. The matched 500-candidate study isolates the continuum advantage: 57.9%
   certified deployment for MOSAIC versus 28.3% for Holm LTT, with no observed
   false acceptances for either rule.
5. The real bridge study supplies the previously missing decision ladder. At
   the primary 0.40 utility contract, unconditional deployment, validation-only
   selection, and a bridge plug-in violate 38/80, 18/60, and 7/47 estimable
   diagnostics. Strict MOSAIC deploys 20/100 with 0/20 observed violations and
   abstains on every missing-support Camelyon job.
6. The threat model now matches the theorem. The public object is explicitly a
   task-specific finite token, sampled once and persisted. Bounded fresh access
   uses the product channel, and unbounded fresh access has an impossibility
   result.
7. Protocol discipline is exceptional: locked seeds, complete receipts, exact
   population evaluators, outward-rounded decisions, deterministic replay,
   exact-rational replay, correction-scope audit, and preserved failed records.

## Remaining Risks

1. The bridge certifies the sampled target stratum laws, not arbitrary later
   drift. A new population or tokenizer requires a new bridge.
2. At the primary real contract, all 20 strict deployments are BiasBios jobs.
   Three Waterbirds jobs deploy only at 0.49; CivilComments and GaitPDB abstain.
   Thus cross-domain execution is broad, but positive real-domain retention is
   not. Zero violations among 20 has a one-sided 95% upper bound of 13.9%.
3. The v2 result is corrective evidence, not untouched confirmation. The
   correction is narrow and auditable: all 918 changed label certificates, and
   no others, contain an exactly zero transform-output column; all exact
   rational slacks remain nonnegative. Transparency limits the damage, but some
   reviewers will still discount this experiment.
4. The method deliberately trades a reusable embedding for a compact
   task-specific interface. Decoder enumeration also scales exponentially with
   the released alphabet, so larger interfaces require structured solvers.
5. The proof and novelty audits are internal. A probability/statistics expert
   and a fair-representation expert should still review the paper cold.

## Earlier Critique Resolution

- **Narrow two-token interface:** owned explicitly; L=4 sensitivity is reported
  and the paper does not claim a general-purpose edited embedding.
- **Repeated-query leakage:** resolved by persistent release semantics, a
  product-channel certificate for bounded fresh draws, and an asymptotic
  impossibility result for unbounded draws.
- **Weak 0.49 utility contract:** real outcomes are reported across 0.30, 0.35,
  0.40, 0.45, and 0.49; 0.40 is the primary contract.
- **Unjustified shift set:** the bridge LP learns the common transform and
  contamination budget from labeled target data with simultaneous coverage.
- **Unclear novelty boundary:** the main paper includes a component-level prior
  work table and 40 cited references.
- **Thin real decision evidence:** expanded from 25 to 100 jobs with a
  deployment-rule ladder and explicit missing-support abstention.

## Verification Performed

- The full repository suite passes 217 tests plus 14 subtests.
- A fresh extraction of the 740-member anonymous archive passes every checksum,
  all 109 packaged tests, and its identity scan.
- Clean-extraction deterministic replay matches all 100 v2 receipts, 1,300
  candidate bridges, and 1,400 global optimizations.
- Clean-extraction exact-rational replay verifies 1,300 bridge certificates and
  1,400 outward release bounds with minimum exact slack zero.
- The correction-scope audit compares 2,600 label certificates and reports no
  out-of-scope change.
- The anonymous main paper has seven content pages plus two reference pages;
  the supplement has nine pages. All fonts are embedded. There are no overfull
  boxes, unresolved references, missing citations, anonymous identity strings,
  or visible layout defects.

## Final Judgment

MOSAIC now clears the bar for a credible AAAI main-track paper. Its strongest
case is the complete theorem-to-decision chain: a same-table continuum event, a
data-certified external bridge, sharp abstention boundaries, and measured
failure of simpler deployment rules. Its largest remaining weakness is not
wording but independent validation. A strong-accept judgment would require a
human proof audit and a fresh bridge confirmation that retains safe releases on
more than one real dataset. On the evidence currently available, accept-leaning
is the highest defensible rating.
