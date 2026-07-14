# VERA Exact Goal Completion Audit

Generated at UTC: `2026-07-14T18:36:30.656386+00:00`
Goal complete: `False`
Literal requested bar complete: `False`
Registered protocol complete: `False`

> This audit is fail-closed. A stronger replacement is recorded separately; it does not silently check a literal requested box. This audit does not predict acceptance or substitute for peer review.

## Literal Requested Bar

| Status | Gate | Evidence | Required next |
| --- | --- | --- | --- |
| pass | `requested_goal_1_theory`: Requested theory grid and proofs | registered_theory_pass=True; candidate_counts_tested=[5, 9, 13, 17]; group_counts_tested=[1, 3, 5]; explicit_m_and_group_variation=True; run_prereg_bytes_committed=True; family_grid_independently_verified=True; identity_candidate_included=True; configured_m=12; configured_groups=3 | None. |
| pass | `requested_goal_2_theory_data`: Requested synthetic and real curve overlay | registered_theory_data_pass=True; report_present=True; passed=None; dataset_count=5; tracking=5; exact_replay=True; confirmatory_vera_control=True; strict_seed_control=True; raw_rows=25920; raw_mismatches=0 | None. |
| fail | `requested_goal_3_killer_experiment`: Requested strict false-acceptance study | grid_valid=True; naive_failure_datasets=2 (required >=1); vera_control=True; seed_blocked_significant=0; discordant_counts=True; retention_intervals=True | Meet the prespecified >=20% naive-failure, strict VERA-control, seed-blocked significance, and retention conditions without post-hoc tuning. |
| pass | `requested_goal_4_zero_proxy_baselines`: Requested official baselines on untouched seeds | receipt_count=200/200; seeds=[5, 6, 7, 8, 9, 10, 11, 12]; requested_seeds=[5, 6, 7, 8, 9, 10, 11, 12]; proxies=0; invalid=0 | None. |
| pass | `requested_goal_5_memorable_number`: Requested receipted headline or theory lead | X_Y_Z_pass=True; theory_forced_abstention_alternative=True; report_present=True; X=0.21875; Y=0.0; Z=0.3082191780821918; X_minus_Y=0.21875; verified=True; descriptive_caveat=True | None. |
| fail | `requested_goal_6_presentation`: Requested seven-page presentation package | registered_presentation_pass=True; figure_1_panel_count=3; repository_forbidden_hits=460; oversized_unscanned=1; report_present=True; pages=7; verified_references=53; forbidden_hits=0; anonymous_clean=True; named_clean=True | Pass the full presentation audit, including an explicit three-panel Figure 1 check and zero forbidden-name hits. |
| fail | `requested_goal_7_external_review`: Requested two human cold reviews | registered_external_review_pass=False; report_present=True; completed=0; ML_publishers=0; unresolved_critical=0; unresolved_major=0; unaddressed_LTT=0; unaddressed_PRC=0 | Obtain two real cold reviews from ML publishers and close every critical/major item in a response ledger. |

## Registered Scientific Protocol

| Status | Gate | Evidence | Required next |
| --- | --- | --- | --- |
| pass | `goal_1_shift_aware_theory`: Shift-aware certification and impossibility | prereg_hash_valid=True; proof_blocks=9; required_labels_present=True; synthetic_cells=54; synthetic_grid_valid=True; independent_verification=True; theory_implementation_consistency=True | None. |
| pass | `goal_2_theory_matched_by_data`: Theory matched by synthetic and real data | report_present=True; passed=None; dataset_count=5; tracking=5; exact_replay=True; confirmatory_vera_control=True; strict_seed_control=True; raw_rows=25920; raw_mismatches=0 | None. |
| fail | `goal_3_killer_experiment`: Deployment rules head to head | report_present=True; rules=['always_deploy_balanced', 'external_balanced_oracle', 'point_selection_balanced', 'vera_balanced_envelope', 'vera_balanced_iut']; naive_failure_datasets=2; vera_observed_rate=0.0; delta=0.05; seed_blocked_significant_datasets=0 | Complete the preregistered rule grid, observed false-acceptance analysis, seed-blocked paired tests, and retention intervals. |
| pass | `goal_4_zero_proxy_baselines`: Official baselines on five datasets | report_present=True; receipts=200/200; missing=0; proxies=0; invalid=0; pinned=True | None. |
| pass | `goal_5_memorable_number`: Receipted abstract result | report_present=True; X=0.21875; Y=0.0; Z=0.3082191780821918; X_minus_Y=0.21875; verified=True; descriptive_caveat=True | None. |
| pass | `goal_6_presentation`: Top-conference presentation | report_present=True; pages=7; verified_references=53; forbidden_hits=0; anonymous_clean=True; named_clean=True | None. |
| fail | `goal_7_external_adversarial_review`: Two external cold reviews | report_present=True; completed=0; ML_publishers=0; unresolved_critical=0; unresolved_major=0; unaddressed_LTT=0; unaddressed_PRC=0 | Obtain two real cold reviews from ML publishers and close every critical/major item in a response ledger. |

## Submission Machinery

- **fail** `submission_machinery`: report_present=True; missing_or_unconfirmed=['openreview_account_human_confirmed', 'single_email_human_confirmed', 'anonymous_archive_reproduces_main_table', 'deadlines_human_confirmed', 'scientific_content_human_verified', 'authorship_criteria_human_confirmed', 'ai_assistance_disclosure_human_confirmed']

## Declared Replacements

- `at least five claim-grade seeds` -> `seeds 5-12 as untouched confirmatory runs`: Eight untouched seeds exceed the requested minimum; seeds 0-4 informed protocol design and are exploratory.
