# VERA Reproducibility Packet Audit

Generated at UTC: `2026-07-14T19:08:51.880445+00:00`

- `packet_ready`: yes
- `pass_count`: 9
- `fail_count`: 0

| Check | Status | Evidence |
| --- | --- | --- |
| `manifest_present` | pass | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/configs/vera_reproducibility.json; exists=yes |
| `checklist_present` | pass | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/maintrack/REPRODUCIBILITY_CHECKLIST.md; exists=yes |
| `seed_policy_locked` | pass | pilot_seed_list=[0, 1, 2, 3, 4]; confirmatory_seed_list=[5, 6, 7, 8, 9, 10, 11, 12]; minimum_confirmatory_seeds=8; confidence_level=0.95; paired_statistics_required=True |
| `reproduction_commands_complete` | pass | command_keys=['claim_ledger_audit', 'compact_reproduction', 'confirmatory_raw_audit', 'exact_balanced_audit', 'exact_family_grid_audit', 'official_receipt_audit', 'presentation_audit']; missing=[]; script_refs_ok=yes; missing_script_refs={} |
| `reproduction_outputs_present` | pass | all_expected_outputs_present=yes; missing_outputs={} |
| `claim_rows_have_receipts` | pass | claim_ready_rows=5; receipts_present=yes; missing_claim_receipts=[] |
| `blocked_rows_scoped` | pass | blocked_rows=1; scoped=yes |
| `core_artifacts_present` | pass | core_artifact_count=18; present=yes |
| `release_policy_scoped` | pass | release_code_configs_and_small_artifacts=True; release_large_raw_data=False; anonymization_required_for_review=True |
