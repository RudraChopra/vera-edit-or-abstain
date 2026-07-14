# VERA Claim Ledger Audit

Generated at UTC: `2026-07-14T19:08:52.037909+00:00`

- `claim_ledger_ready`: yes
- `pass_count`: 6
- `fail_count`: 0

| Check | Status | Evidence |
| --- | --- | --- |
| `claim_ledger_present` | pass | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/configs/vera_claim_ledger.json; exists=yes |
| `claim_ledger_markdown_present` | pass | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/maintrack/CLAIM_LEDGER.md; exists=yes |
| `allowed_claims_have_evidence` | pass | allowed_claims=12; with_existing_evidence=12 |
| `forbidden_claims_registered` | pass | forbidden_claims=8; forbidden_patterns=25 |
| `audited_manuscript_paths_exist` | pass | audited_paths=['research/maintrack/aaai2027_template/AuthorKit27/vera_paper_body.tex', 'research/maintrack/aaai2027_template/AuthorKit27/vera_supplement_body.tex', 'research/maintrack/aaai2027_template/AuthorKit27/vera_aaai2027_anonymous.tex', 'research/maintrack/aaai2027_template/AuthorKit27/vera_aaai2027_named.tex', 'research/maintrack/CLAIM_LEDGER.md']; exist=yes |
| `forbidden_claims_absent_from_manuscript` | pass | forbidden_hits=[] |
