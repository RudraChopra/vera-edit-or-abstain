# VERA Exact Goal Completion Audit

Generated at UTC: `2026-07-15T02:14:19.989117+00:00`
Goal complete: `False`
Literal requested bar complete: `False`
Registered protocol complete: `False`

> This audit is fail-closed. A stronger replacement is recorded separately; it does not silently check a literal requested box. This audit does not predict acceptance or substitute for peer review.

## Literal Requested Bar

| Status | Gate | Evidence | Required next |
| --- | --- | --- | --- |
| pass | `requested_goal_1_theory`: Requested theory grid and proofs | registered_theory_pass=True; candidate_counts_tested=[5, 9, 13, 17]; group_counts_tested=[1, 3, 5]; explicit_m_and_group_variation=True; run_prereg_bytes_committed=True; family_grid_independently_verified=True; identity_candidate_included=True; configured_m=12; configured_groups=3 | None. |
| pass | `requested_goal_2_theory_data`: Requested synthetic and real curve overlay | registered_theory_data_pass=True; report_present=True; passed=None; dataset_count=5; tracking=5; exact_replay=True; confirmatory_vera_control=True; strict_seed_control=True; raw_rows=25920; raw_mismatches=0 | None. |
| fail | `requested_goal_3_killer_experiment`: Requested strict false-acceptance study | independent_stress_report_present=True; receipt_audit_pass=True; analysis_audit_pass=True; grid_valid=True; threshold_grid_valid=True; supported_datasets_passing_all_three=3/4; global_vera_control=True; camelyon_forced_abstention=True | Complete the locked independent stress replication: all four supported datasets must hit point-selection >=20%, VERA <= delta, and Holm-corrected McNemar <= .05, with Camelyon17 forced abstention. |
| pass | `requested_goal_4_zero_proxy_baselines`: Requested official baselines on untouched seeds | independent_receipt_count=800/800; seeds=[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]; proxies=0; invalid=0; pinned=True | None. |
| fail | `requested_goal_5_memorable_number`: Requested receipted headline or theory lead | independent_stress_present=True; registered_memorable_number_pass=False; independent_report_passed=False; verified=True; X=0.2734375; Y=0.0078125; Z=0.5098039215686274; X_minus_Y=0.265625; package_passed=True; registered_pass_conditions_met=False | Meet the independent stress X/Y/Z headline exactly; the older theory-only fallback is disabled once the independent strict replication exists. |
| pass | `requested_goal_6_presentation`: Requested seven-page presentation package | registered_presentation_pass=True; figure_1_panel_count=3; repository_forbidden_hits=0; oversized_unscanned=0; report_present=True; pages=7; verified_references=53; forbidden_hits=0; anonymous_clean=True; named_clean=True | None. |
| fail | `requested_goal_7_external_review`: Requested two human cold reviews | registered_external_review_pass=False; report_present=True; completed=0; ML_publishers=0; unresolved_critical=0; unresolved_major=0; unaddressed_LTT=0; unaddressed_PRC=0 | Obtain two real cold reviews from ML publishers and close every critical/major item in a response ledger. |

## Registered Scientific Protocol

| Status | Gate | Evidence | Required next |
| --- | --- | --- | --- |
| pass | `goal_1_shift_aware_theory`: Shift-aware certification and impossibility | prereg_hash_valid=True; proof_blocks=9; required_labels_present=True; synthetic_cells=54; synthetic_grid_valid=True; independent_verification=True; theory_implementation_consistency=True | None. |
| pass | `goal_2_theory_matched_by_data`: Theory matched by synthetic and real data | report_present=True; passed=None; dataset_count=5; tracking=5; exact_replay=True; confirmatory_vera_control=True; strict_seed_control=True; raw_rows=25920; raw_mismatches=0 | None. |
| fail | `goal_3_killer_experiment`: Deployment rules head to head | independent_stress_report_present=True; receipt_audit_pass=True; analysis_audit_pass=True; grid_valid=True; threshold_grid_valid=True; supported_datasets_passing_all_three=3/4; global_vera_control=True; camelyon_forced_abstention=True | Complete the locked independent stress replication with all four supported datasets passing the naive-failure, VERA-control, and Holm-corrected paired-test endpoints. |
| pass | `goal_4_zero_proxy_baselines`: Official baselines on five datasets | independent_receipt_count=800/800; seeds=[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44]; missing=0; proxies=0; invalid=0; pinned=True | None. |
| fail | `goal_5_memorable_number`: Receipted abstract result | independent_report_passed=False; verified=True; X=0.2734375; Y=0.0078125; Z=0.5098039215686274; X_minus_Y=0.265625; package_passed=True; registered_pass_conditions_met=False | Derive X/Y/Z from the independent stress receipts and package the audited abstract sentence. |
| pass | `goal_6_presentation`: Top-conference presentation | report_present=True; pages=7; verified_references=53; forbidden_hits=0; anonymous_clean=True; named_clean=True | None. |
| fail | `goal_7_external_adversarial_review`: Two external cold reviews | report_present=True; completed=0; ML_publishers=0; unresolved_critical=0; unresolved_major=0; unaddressed_LTT=0; unaddressed_PRC=0 | Obtain two real cold reviews from ML publishers and close every critical/major item in a response ledger. |

## Submission Machinery

- **fail** `submission_machinery`: report_present=True; missing_or_unconfirmed=['openreview_account_human_confirmed', 'single_email_human_confirmed', 'deadlines_human_confirmed', 'scientific_content_human_verified', 'authorship_criteria_human_confirmed', 'ai_assistance_disclosure_human_confirmed']

## Declared Replacements

- `at least five claim-grade seeds` -> `seeds 5-12 as untouched confirmatory runs`: Eight untouched seeds exceed the requested minimum; seeds 0-4 informed protocol design and are exploratory.
