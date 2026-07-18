# VERA: Evidence-Efficient Certification of Representation Edits Under Deployment Shift

VERA is a shift-aware decision layer for representation editing. Given a
registered frontier of edits, it certifies a supported vector of bounded
density-ratio shifts under which one edit simultaneously satisfies paired
target-harm and post-edit leakage contracts. It also reports a conservative
common shift radius and prospectively allocates certification evidence toward
limiting supported cells. VERA returns `ABSTAIN` when even the IID contract
cannot be certified or when deployment requires support absent from
certification.

The finite-candidate selection mechanism follows Learn Then Test and related
risk-control work; it is not claimed as new. The proposed object is the paired,
multi-attacker **support-aware edit shift envelope**, its common-radius summary,
its prospective additive multi-cell evidence-allocation rule, and its
support-mismatch boundary. The locked controlled primary uses the earlier
square-score allocator; a separately frozen prospective extension evaluates the
full additive allocator without replacing a primary gate.
The formal leakage guarantee covers only the registered attacker portfolio;
the acronym is intentionally left unexpanded, and the method does not claim
universal concept removal.
Some immutable early preregistrations retain the retired record label
`Verified Erasure under Reweighting Ambiguity`. That historical label is not
the current title, acronym expansion, or scientific claim.

OpenAI Codex assisted extensively with research ideation, literature discovery,
theorem and proof drafting, implementation, experiment orchestration,
statistical analysis, figures, and manuscript drafting. It is not an author or
a citable source. Any human submission requires independent verification of
the complete work and retains full responsibility for every claim and policy
obligation.

Included:

- The original pilot lock in `research/prereg_real.json` and the untouched-seed
  confirmatory lock in `research/prereg_confirmatory_balanced.json`, each with
  a SHA-256 sidecar
- The locked 216-cell theorem coverage grid over validation size, candidate
  count, validated-group count, and delta in
  `research/prereg_exact_family_grid.json`
- The locked disjoint-seed independent stress replication in
  `research/prereg_independent_stress_replication.json`
- The prospective 64-seed controlled supported-shift lock in
  `research/prereg_controlled_shift.json`, with its separately committed
  SHA-256 sidecar, preregistered cap-8 analysis contract, independent cap-8
  replay requirement, and byte-unchanged cap-4 implementation sensitivity
- The shift-aware theory and manuscript material under `research/maintrack/`
- One shared cross-venue scientific record, result-manifest schema, and
  section-level claim map under `research/maintrack/`; venue-specific formats
  remain conditional on their verified official materials and policies
- Official-code adapters, analysis, and fail-closed audits under
  `research/scripts/`
- Per-run JSON receipts and small verification artifacts under
  `research/artifacts/`
- The complete final P0 matrix (seeds 173--236) and lossless compressed primary
  artifact. Its independently audited result is a negative confirmation for
  VERA's planned superiority-over-IID-LTT headline; see
  `research/maintrack/P0_RESULT_SUMMARY_2026-07-18.md` before relying on any
  earlier controlled-shift result.

Excluded:

- Raw third-party datasets
- Large frozen embedding stores
- External-drive-only generated arrays
- Local virtual environments and `.git` metadata from the working directory

Core verification commands:

```bash
python research/scripts/reproduce_vera_submission.py

# Full replay when the external per-example arrays are mounted:
python research/scripts/reproduce_vera_submission.py --full

# Individual gates:
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
python research/scripts/verify_reference_manifest.py
python -m unittest research.tests.test_vera_robust_certificate \
  research.tests.test_vera_analysis -v
python research/scripts/audit_goal_completion.py --no-fail

# Final P0 integrity and independent-reader gates. The full per-example arrays
# remain on the mounted external drive.
python research/scripts/audit_vera_p0_receipts.py
python research/scripts/analyze_vera_p0_confirmation.py
python research/scripts/replay_vera_p0_exact_risks.py
python research/scripts/audit_vera_p0_reader_agreement.py
```

The five claim-grade eraser families are INLP, R-LACE, LEACE, TaCo, and MANCE++.
The real study spans Waterbirds, Camelyon17-WILDS, CivilComments-WILDS, Bios,
and GaitPDB. Seeds 0--4 are disclosed pilot evidence; the locked 200-run
confirmation uses untouched seeds 5--12, the independent stress replication
uses disjoint seeds 13--44, and the prospective controlled-shift study uses
fresh seeds 45--108. Large third-party datasets, frozen
embedding stores, generated audit arrays, and local environments are
intentionally kept off GitHub; their hashes and compact receipts remain
auditable here.

This repository does not claim that conference acceptance is guaranteed. The
completion audit remains fail-closed until every empirical, presentation, and
human-review gate has evidence. Every listed human author must also complete
`research/maintrack/HUMAN_AUTHOR_VERIFICATION_GATE.md` before submission.

The authoritative current status is
`research/maintrack/GOAL_1_58_STATUS.md`. Older generated readiness artifacts
describe earlier source states and cannot override that tracker.
