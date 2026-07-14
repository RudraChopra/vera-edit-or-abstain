# VERA Reproducibility Packet Audit

Generated at UTC: `2026-07-14T01:06:25.522344+00:00`

- `packet_ready`: yes
- `pass_count`: 9
- `fail_count`: 0

| Check | Status | Evidence |
| --- | --- | --- |
| `manifest_present` | pass | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/configs/faro_paper_a_reproducibility.json; exists=yes |
| `checklist_present` | pass | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/maintrack/REPRODUCIBILITY_CHECKLIST.md; exists=yes |
| `seed_policy_locked` | pass | official_seed_list=[0, 1, 2, 3, 4]; minimum_official_seeds=5; confidence_level=0.95; paired_statistics_required=True |
| `reproduction_commands_complete` | pass | command_keys=['aaai2027_source_readiness', 'adversarial_internal_review', 'benchmark_claim_audit', 'camelyon17_mance_scaling_feasibility', 'camelyon17_mancepp_80k_diagnostic', 'camelyon17_mancepp_full_nocap_reference', 'camelyon17_numpy_store_conversion', 'camelyon17_official_benchmark', 'camelyon17_projection_frontier_certificate', 'camelyon17_tar_stream_dryrun', 'claim_ledger_audit', 'gaitpdb_numpy_store', 'gaitpdb_public_locked_split_benchmark', 'iclr2026_style_draft_compile', 'maintrack_figures', 'maintrack_readiness', 'manuscript_compile', 'reference_baseline_scope_audit', 'upstream_baseline_reference_inventory']; missing=[]; script_refs_ok=yes; missing_script_refs={} |
| `reproduction_outputs_present` | pass | all_expected_outputs_present=yes; missing_outputs={} |
| `claim_rows_have_receipts` | pass | claim_ready_rows=2; receipts_present=yes; missing_claim_receipts=[] |
| `blocked_rows_scoped` | pass | blocked_rows=1; scoped=yes |
| `core_artifacts_present` | pass | core_artifact_count=58; present=yes |
| `release_policy_scoped` | pass | release_code_configs_and_small_artifacts=True; release_large_raw_data=False; anonymization_required_for_review=True |
