# VERA: Verified Erasure under Reweighting Ambiguity

VERA is a shift-aware decision layer for representation editing. Given a
registered frontier of edits, it lower-certifies the largest bounded
density-ratio shift under which one edit simultaneously satisfies paired target
harm and post-edit leakage contracts. VERA returns `ABSTAIN` when even the IID
contract cannot be certified or when deployment includes an environment absent
from certification.

The finite-candidate selection mechanism follows Learn Then Test and related
risk-control work; it is not claimed as new. The proposed object is the paired,
multi-attacker **support-aware erasure shift envelope**, its common-radius
summary, and its support-mismatch boundary.

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
- The shift-aware theory and manuscript material under `research/maintrack/`
- Official-code adapters, analysis, and fail-closed audits under
  `research/scripts/`
- Per-run JSON receipts and small verification artifacts under
  `research/artifacts/`

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
```

The five claim-grade eraser families are INLP, R-LACE, LEACE, TaCo, and MANCE++.
The real study spans Waterbirds, Camelyon17-WILDS, CivilComments-WILDS, Bios,
and GaitPDB. Seeds 0--4 are disclosed pilot evidence; the locked 200-run
confirmation uses untouched seeds 5--12, and the independent stress replication
uses disjoint seeds 13--44. Large third-party datasets, frozen
embedding stores, generated audit arrays, and local environments are
intentionally kept off GitHub; their hashes and compact receipts remain
auditable here.

This repository does not claim that conference acceptance is guaranteed. The
completion audit remains fail-closed until every empirical, presentation, and
human-review gate has evidence. Every listed human author must also complete
`research/maintrack/HUMAN_AUTHOR_VERIFICATION_GATE.md` before submission.
