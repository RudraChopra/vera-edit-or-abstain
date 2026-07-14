# VERA Claim Ledger Audit

Generated at UTC: `2026-07-14T00:52:48.802612+00:00`

- `claim_ledger_ready`: yes
- `pass_count`: 6
- `fail_count`: 0

| Check | Status | Evidence |
| --- | --- | --- |
| `claim_ledger_present` | pass | path=/Users/rudrachopra/Documents/Science Fair/research/configs/faro_claim_ledger.json; exists=yes |
| `claim_ledger_markdown_present` | pass | path=/Users/rudrachopra/Documents/Science Fair/research/maintrack/CLAIM_LEDGER.md; exists=yes |
| `allowed_claims_have_evidence` | pass | allowed_claims=13; with_existing_evidence=13 |
| `forbidden_claims_registered` | pass | forbidden_claims=6; forbidden_patterns=19 |
| `audited_manuscript_paths_exist` | pass | audited_paths=['research/maintrack/faro_main.tex', 'research/maintrack/README.md', 'research/maintrack/PAPER_A_LOCK.md', 'research/maintrack/READINESS_GATES.md']; exist=yes |
| `forbidden_claims_absent_from_manuscript` | pass | forbidden_hits=[] |
