# FARO Reference Baseline Scope Audit

Generated at UTC: `2026-07-14T00:52:52.628492+00:00`
Reference scope ready: `True`
Universal erasure SOTA claim allowed: `False`

| Status | Check | Evidence |
| --- | --- | --- |
| pass | `baseline_fairness_ready` | baseline_ready=True; fail_count=0 |
| pass | `proxy_rows_labeled` | proxy_rows=6; proxy_statuses=['pass', 'pass', 'pass', 'pass', 'pass', 'pass'] |
| pass | `mance_waterbirds_reference_ready` | claim_grade_reference_row=True; claim_grade_statistics=True |
| pass | `mance_camelyon_reference_ready` | claim_grade_reference_row=True; full_nocap_receipt_materialized=True; claim_boundary=Claim-grade for the full no-cap Camelyon17 frozen-representation store, with external source leakage intentionally null because the external split has one binary source class. |
| pass | `mance_camelyon_scaling_documented` | full_no_cap_completed=True; compute_blocker=False; storage_blocker=False; full_no_cap_observed_run=True; linear_lower_bound_seconds=17390.03850483477; recent_superlinear_estimate_seconds=71202.37299226955 |
| pass | `upstream_baseline_inventory_ready` | inventory_ready=True; fail_count=0; expected=MANCE++/R-LACE/TaCo/LEACE pinned plus SPLINCE boundary |
| pass | `claim_ledger_ready` | claim_ledger_ready=True; fail_count=0 |
| pass | `no_universal_erasure_sota_claim` | manuscripts and claim ledger explicitly deny universal/SOTA erasure claims |
| pass | `reference_parity_boundary_written` | close erasure baselines and reference-parity boundary are named in paper-facing docs |
