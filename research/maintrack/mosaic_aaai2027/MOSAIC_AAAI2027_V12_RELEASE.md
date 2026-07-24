# MOSAIC AAAI 2027 V12 Release

This is the frozen path-to-9 submission package. It keeps the V12 paper name
and title while adding the theory, breadth, comparison, and adoption work from
`mosaic_path_to_9.md`.

## Submission files

- `MOSAIC_AAAI2027_V12_ANONYMOUS.pdf`
- `MOSAIC_AAAI2027_V12_SUPPLEMENT_ANONYMOUS.pdf`
- `MOSAIC_AAAI2027_V12_REPRODUCIBILITY_CHECKLIST.pdf`
- `MOSAIC_AAAI2027_V12_CODE_DATA_SUPPLEMENT.zip`

The named paper and named supplement are included for arXiv and direct review.

## Added evidence

- Three natural-shift domains now yield 75 releases in 120 jobs with zero
  held-out violations: BiasBios 20/20, geographic ACS 15/60, and natural
  CINIC-10 origin shift 40/40.
- A locked powered Qwen2.5-1.5B hidden-state study releases 5/5 temporal
  moderation interfaces with zero held-out violations and zero violations in
  500 operational replays.
- A separately locked supported multi-hospital Camelyon17 study releases 5/5
  pathology interfaces even at the .35 utility contract, with zero held-out
  violations and zero violations in 500 operational replays.
- A prospectively frozen 2023 ACS panel confirms one natural direct-rule
  contract violation that MOSAIC had rejected, with empirical error .4245 and
  a familywise lower bound of .4054 against the .40 contract.
- A real proxy-label path releases at the two largest calibration sizes,
  including the full 78,267-row fold, after correctly abstaining on two smaller
  folds.
- The sharp residual-support theorem characterizes the infinite-sample
  conservatism floor; its independent census finds residual shift larger than
  sampling uncertainty in all 35 primary BiasBios and ACS releases.
- A matched local-DP randomized-response baseline releases 0/35 interfaces at
  the .40 utility contract, while MOSAIC releases 35/35 and has lower certified
  error in every job.
- The 40-seed CINIC extension uses one stricter familywise allocation and
  passes its locked gate. The minimum leave-one-domain-out release rate is
  43.8%.
- An official FARE comparison uses the same ACS raw inputs, partition,
  proxy-calibrated source contract, and utility contract. All eight FARE leaf
  candidates abstain, with zero false acceptances.
- A real ACS source imputer reaches .840 balanced accuracy. Token-dependent
  calibration correctly forces abstention when the latent-law uncertainty is
  too large.
- The supplement proves a binary certification lower bound and the matching
  universal-region rate, bounded correlated-session certification, a
  multiplicative transcript-capacity bound, anytime-valid recertification,
  finite-cover and all-threshold continuous extensions, token-dependent proxy
  calibration, and a sharp privacy-utility conflict bound.
- A 120-cell designer chart maps contract, alphabet size, and shift budget to
  sufficient labels per source-label stratum.
- The installable package includes monitoring, report generation, and the
  `mosaic-audit` command. The Lean 4 core checks adaptive selection and bridge
  residual reconstruction.

## Verification

- Main paper: 9 pages total; references begin on page 8.
- Supplement: 16 pages before final Camelyon integration.
- Code/data archive: 15.9 MB, below the 50 MB limit.
- Full test suite: 313 tests and 14 subtests pass before the final rebuild.
- Submission audit: pass.
- Anonymous PDF and archive identity scans: pass.
- Visual PDF inspection: pass.

The real production deployment was excluded from this cycle by request.
External replication is prepared as a stand-alone packet but is not claimed
until an unaffiliated evaluator completes it.
