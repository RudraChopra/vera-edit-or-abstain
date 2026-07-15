# VERA Reproducibility Checklist

## Scope

This checklist covers the claim-grade VERA paper, its exact synthetic studies,
and the untouched-seed official-code experiment. The locked scientific object
is the support-aware certificate for paired target harm and balanced attacker
leakage. It does not certify perfect erasure, clinical safety, or deployment
outside the declared support and bounded-density-ratio model.

## Locked Protocols

- Primary real-study preregistration:
  `research/prereg_confirmatory_balanced.json` with SHA-256
  `2767b64f1fa844f512026c5cc4d5e81bca4ed92bef9c8dfd5287dec2918395aa`.
- Real learning-curve diagnostic:
  `research/prereg_real_learning_curve_diagnostic.json` and its sidecar.
- Secondary ablations:
  `research/prereg_confirmatory_secondary_ablations.json` and its sidecar.
- Candidate/group-count exact extension:
  `research/prereg_exact_family_grid.json` and its sidecar.
- Independent stress replication:
  `research/prereg_independent_stress_replication.json` with SHA-256
  `348c2784ec8d6bc9cef3c8449dccd4a07c394f1e3b54f8231b7c9c88b52caad3`.
- Seeds `0` through `4` are exploratory and excluded from confirmation.
- Seeds `5` through `12` are the eight untouched confirmatory seeds.
- Seeds `13` through `44` are disjoint independent stress-replication seeds.
- The primary analysis uses all nine registered threshold pairs at full
  certification size and `Gamma=1.0`; `Gamma=1.01` is the fixed shift
  sensitivity.

## Data And Methods

The real matrix contains five public datasets: Waterbirds,
Camelyon17-WILDS, CivilComments-WILDS, Bios, and GaitPDB. It crosses five
pinned official eraser implementations: INLP, R-LACE, LEACE, TaCo, and
MANCE++. One receipt is required for every dataset, eraser, and untouched seed,
for exactly 200 design-stage official-code receipts and zero proxy rows. The
independent stress replication repeats the official-code matrix on 32 disjoint
seeds, for 800 additional receipts under a separate preregistration. Each
receipt records the upstream remote and commit, runner commit, split hashes,
preregistration hash, candidate settings, and per-example NPZ hashes.

The external split is never used to construct an edit, train a target model or
attacker, certify a candidate, or select a deployment rule. It is opened only
for locked evaluation after candidate construction. Camelyon17 center 2 is
outside certification support and therefore forces VERA to abstain; its
single-class external source slice makes balanced leakage non-estimable.

## Statistical Checks

- The candidate-wise fixed-profile intersection-union test spends
  `delta / 12` per candidate and requires every component contract to pass.
- The post-selection envelope uses simultaneous coverage over all registered
  candidate-contract pairs.
- The exact balanced study contains 54 cells and 2,000 repetitions per cell.
- The family/grid extension contains 216 cells and 2,000 repetitions per cell.
- Primary real-data inference averages the nine correlated threshold outcomes
  within each seed and applies an exact one-sided sign-flip test, followed by
  Holm correction over the four externally estimable datasets.
- The independent stress replication uses one locked contract per supported
  dataset and requires, without post-hoc replacement, point-selection external
  violations at or above 20%, VERA violations at or below `delta`, and
  Holm-corrected one-sided paired McNemar `p <= .05` on all four externally
  estimable datasets. Camelyon17 remains the forced-abstention support-boundary
  case.
- A sign-only test is a secondary robustness check. Configuration-level
  McNemar and Clopper-Pearson calculations are reported as dependent-cell
  diagnostics, not independent-trial inference.
- Safe-retention uncertainty is reported with a seed-cluster bootstrap; the
  configuration-level binomial interval remains a labeled diagnostic.

## Reproduction

From the anonymous archive, compact reproduction verifies frozen rows and
rebuilds the main table without private datasets or upstream repositories:

```bash
python research/scripts/reproduce_vera_submission.py
```

Full reproduction requires the public datasets, pinned upstream repositories,
and immutable NPZ paths recorded by the preregistration and receipts:

```bash
python research/scripts/reproduce_vera_submission.py --full
```

The claim-grade audit sequence is:

```bash
python research/scripts/audit_exact_balanced_simulation.py
python research/scripts/audit_exact_family_grid_simulation.py
python research/scripts/audit_official_eraser_receipts.py \
  --prereg research/prereg_confirmatory_balanced.json \
  --hash-file research/prereg_confirmatory_balanced.sha256 \
  --receipt-dir research/artifacts/confirmatory_balanced_receipts \
  --output research/artifacts/confirmatory_balanced_receipt_audit.json
python research/scripts/analyze_vera_confirmatory_balanced.py
python research/scripts/audit_vera_confirmatory_analysis.py
python research/scripts/audit_vera_confirmatory_compact.py
python research/scripts/analyze_vera_learning_curve_diagnostic.py
python research/scripts/analyze_vera_confirmatory_ablations.py
python research/scripts/build_vera_confirmatory_results.py
python research/scripts/audit_official_eraser_receipts.py \
  --prereg research/prereg_independent_stress_replication.json \
  --hash-file research/prereg_independent_stress_replication.sha256 \
  --receipt-dir research/artifacts/independent_stress_replication_receipts \
  --output research/artifacts/independent_stress_replication_receipt_audit.json
python research/scripts/analyze_vera_independent_stress_replication.py
python research/scripts/audit_vera_independent_stress_replication.py
python research/scripts/audit_vera_independent_stress_compact.py
python research/scripts/build_vera_independent_stress_package.py
python research/scripts/audit_presentation_readiness.py
python research/scripts/audit_goal_completion.py --no-fail
```

The design-stage confirmatory audit rehashes and directly loads all 480 raw
candidate NPZs, recomputes 25,920 candidate-configuration rows with a separate
certificate implementation, replays every decision rule, and regenerates the
headline and inferential diagnostics. The independent stress audit repeats the
same raw-array replay on the 800-receipt disjoint-seed matrix and the anonymous
archive additionally runs a compact frozen-row replay of those independent
selection decisions.

## Release Policy

Release source code, frozen preregistrations, compact aggregate rows, receipt
JSON files, audit reports, figures, paper sources, and the anonymous archive.
Do not commit third-party datasets, derived embedding stores, or private human
review/account records. Public-data derivatives remain on external storage and
are represented by manifests and content hashes.

## Final Human Gates

Technical audits do not establish novelty, proof correctness, ethical
eligibility, or likely acceptance. Before submission, the human author must
verify every official checklist response, confirm the venue deadline and
OpenReview account, inspect the anonymous archive, and close two genuine cold
reviews from researchers who publish in machine learning. The manuscript's
generative-AI disclosure must remain present and must accurately describe the
full scope of assistance.
