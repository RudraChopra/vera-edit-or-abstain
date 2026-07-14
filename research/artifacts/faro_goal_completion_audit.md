# VERA Exact Goal Completion Audit

Generated at UTC: `2026-07-14T06:01:30.742071+00:00`
Goal complete: `False`

> This audit is fail-closed. It does not predict acceptance or substitute for peer review.

| Status | Gate | Evidence | Required next |
| --- | --- | --- | --- |
| pass | `goal_1_shift_aware_theory`: Shift-aware certification and impossibility | prereg_hash_valid=True; proof_blocks=6; required_labels_present=True; synthetic_cells=18; synthetic_grid_valid=True; independent_verification=True | None. |
| fail | `goal_2_theory_matched_by_data`: Theory matched by synthetic and real data | report_present=False; passed=None; dataset_count=None; tracking=None; all_false_acceptance_controlled=None | Run the locked real-data subsampling study and verify predicted/observed overlays on at least four datasets. |
| fail | `goal_3_killer_experiment`: Deployment rules head to head | report_present=False; rules=[]; naive_failure_datasets=None; vera_upper=None; delta=None; significant_datasets=None | Complete the preregistered four-rule grid, global false-acceptance analysis, McNemar tests, and retention intervals. |
| fail | `goal_4_zero_proxy_baselines`: Official baselines on five datasets | report_present=True; receipts=0/125; missing=125; proxies=0; invalid=0; pinned=True | Produce and validate all 125 official method/dataset/seed run receipts with zero proxy or missing rows. |
| fail | `goal_5_memorable_number`: Receipted abstract result | report_present=False; X=None; Y=None; Z=None; X_minus_Y=None; verified=None | Derive X/Y/Z from locked receipts and verify the exact abstract and introduction sentence with a gap of at least 15 points. |
| fail | `goal_6_presentation`: Top-conference presentation | report_present=False; pages=None; verified_references=None; forbidden_hits=None; anonymous_clean=None; named_clean=None | Finish and independently audit the seven-page paper, Figure 1, 40+ references, naming purge, both PDFs, and metadata. |
| fail | `goal_7_external_adversarial_review`: Two external cold reviews | report_present=False; completed=None; ML_publishers=None; unresolved_critical=None; unresolved_major=None; unaddressed_LTT=None | Obtain two real cold reviews from ML publishers and close every critical/major item in a response ledger. |
| fail | `submission_machinery`: Venue submission machinery | report_present=False; missing_or_unconfirmed=['openreview_account_human_confirmed', 'single_email_human_confirmed', 'target_style_compiles', 'exact_page_limit', 'zero_formatting_hacks', 'anonymization_complete', 'anonymous_archive_reproduces_main_table', 'reproducibility_checklist_complete', 'supplement_ready', 'deadlines_human_confirmed', 'areas_and_keywords_selected'] | Complete the technical packaging and obtain human confirmation for account, email, and venue-deadline items. |
