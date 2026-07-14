# VERA Top-Conference Execution Specification

This file is the authoritative completion standard. A passing internal script
is evidence only for the exact condition it checks. It is never evidence of
novelty, scientific importance, or acceptance probability.

## Scientific Contribution

VERA means Verified Erasure under Reweighting Ambiguity. Given identity and
edited representations for the same examples, VERA certifies the incremental
target harm and post-edit sensitive leakage over a declared deployment-shift
class. Its primary class is all distributions with a bounded density ratio
relative to validation. Annotated worst-group mixtures are a corollary. VERA
returns the largest jointly certifiable reweighting radius for each edit and
`ABSTAIN` when even the IID contract cannot be established.

The finite-candidate testing layer is explicitly attributed to Learn Then Test
and related risk-control work. It is not claimed as novel.

## Gate 1: Theory

- Prove bounded-reweighting certification for paired edit harm.
- Control target harm and every preregistered attacker simultaneously.
- Invert the risk bands into a simultaneous lower confidence bound on each
  edit's maximum common deployment-shift radius.
- Prove that the radius remains valid for a deployment budget chosen after the
  certificate is observed.
- Derive worst-group mixture certification as a corollary.
- Prove support-mismatch impossibility with a two-world or Le Cam argument.
- Derive false-acceptance control at level `delta`.
- Validate coverage for every tested `(n, m, attackers, shift cap, delta)` cell.
- Hash and commit `prereg.json` before claim-grade certification runs.

## Gate 2: Theory Matched By Data

- Run 1,000 synthetic replicates at six validation sizes and three delta levels.
- Report Clopper-Pearson intervals for empirical false acceptance.
- Overlay predicted and observed abstention curves.
- Subsample each real dataset at 5%, 10%, 25%, 50%, and 100%.
- Require observed transitions to lie in preregistered predicted bands on at
  least four of five datasets.

## Gate 3: Killer Experiment

- Compare always deploy, best validation point estimate, IID LTT, VERA, and
  oracle.
- Use five datasets, at least four erasers, nine contract pairs, five seeds,
  and four validation sizes.
- Find honest, preregistered regimes where naive selection violates the locked
  external contract at least 20% of the time on each dataset.
- Require VERA false acceptance at or below `delta` in every claim-grade cell.
- Report exact paired McNemar tests with Holm correction and discordant counts.
- Quantify deployment retention relative to the oracle with intervals.

## Gate 4: Zero-Proxy Baselines

- Pin official upstream commits for INLP, RLACE, LEACE, MANCE++, and TaCo or
  SPLINCE.
- Run Waterbirds, Camelyon17 WILDS, Bios, CivilComments WILDS, and GaitPDB.
- Use the same frozen representations, splits, probe family, and seeds 0-4
  within each dataset.
- Emit one JSON receipt per run.
- Fail table generation if any cell lacks a receipt or uses proxy code.

## Gate 5: Abstract Number

- Verify the sentence reporting naive false acceptance, VERA false acceptance,
  and deployment retention directly from regenerated receipts.
- Require at least a 15 percentage-point false-acceptance reduction in an
  honest regime, or lead with the theory and forced-abstention result instead.

## Gate 6: Presentation

- Figure 1 must teach the method in three panels: paired edit harm under
  reweighting, the certified shift radius versus abstention, and a real receipt.
- Use vector, colorblind-safe figures readable at 50% zoom.
- Fill the venue's permitted content pages without formatting hacks.
- Include at least 40 verified references spanning erasure, risk control,
  selective prediction, distribution shift, probing, fairness certificates,
  and deployment auditing.
- Purge stale project names from source, figures, artifacts, and PDF metadata.
- Produce anonymous and named source/PDF variants with clean metadata.

## Gate 7: External Adversarial Review

- Obtain two completed cold reviews from researchers who publish in machine
  learning.
- Record every critical and major finding in `review_response_ledger.md`.
- Fix or rebut each finding in the paper itself.
- Require both reviewers to agree that LTT overlap is explicitly addressed.

## Submission Machinery

- Register OpenReview and use one identity consistently.
- Verify current official deadlines and style files for each target venue.
- Compile at the exact page limit with no margin or spacing manipulation.
- Anonymize paper, supplement, code archive, links, and PDF metadata.
- Provide a seeded anonymous archive that reproduces the main table in one
  command.
- Complete the venue reproducibility checklist.
- Upload proofs, full tables, and per-dataset details by the supplement deadline.
- Register the abstract before the venue's abstract deadline.
- Use the primary area closest to trustworthy ML, distribution shift, or
  uncertainty quantification.

## Completion Rule

Every gate must have authoritative evidence. Missing human reviews, missing
official baselines, unrun experiment cells, unverified references, or an
uncompiled final PDF keep the project incomplete. No conference acceptance can
be guaranteed; completion means submission-ready evidence meeting this spec.
