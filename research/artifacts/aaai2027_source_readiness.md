# AAAI-27 Source Readiness Audit

Generated at UTC: `2026-07-14T01:06:25.182585+00:00`
Source ready: `True`
Warning count: `1`

| Status | Check | Evidence | Requirement |
| --- | --- | --- | --- |
| pass | `source_materialized` | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/maintrack/aaai2027_template/AuthorKit27/faro_aaai2027_draft.tex; materialized=True | AAAI source must exist locally and not be a dataless placeholder. |
| pass | `official_style_materialized` | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/maintrack/aaai2027_template/AuthorKit27/aaai2027.sty; materialized=True | Official AAAI-27 style file must be present. |
| pass | `uses_official_submission_style` | source uses \usepackage[submission]{aaai2027} | Submission source must use the official AAAI submission style. |
| pass | `anonymous_author_block` | author block is anonymous and local identity strings are absent | AAAI double-anonymous source must not expose author identity. |
| pass | `anonymous_source_materialized` | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/maintrack/aaai2027_template/AuthorKit27/faro_aaai2027_anonymous.tex; materialized=True | Dedicated anonymous AAAI source must exist locally. |
| pass | `anonymous_source_identity_free` | anonymous_leaks=[] | Anonymous AAAI source must include code availability without named identity or GitHub URL. |
| pass | `named_source_release_metadata` | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/maintrack/aaai2027_template/AuthorKit27/faro_aaai2027_named.tex; materialized=True; has_author=True; has_repo=True | Named AAAI source must contain author metadata and the public release URL. |
| pass | `required_sections_present` | missing_sections=[] | AAAI source must include the core method-paper sections. |
| pass | `current_camelyon_mance_reference` | AAAI source contains the full no-cap Camelyon MANCE++ reference numbers | AAAI source must cite the full no-cap Camelyon MANCE++ receipt rather than superseded diagnostics. |
| pass | `no_stale_camelyon_mance_terms` | stale_terms_present=[] | Superseded 40k Camelyon MANCE++ text must be removed. |
| pass | `false_acceptance_corollary_present` | source contains the false-acceptance corollary | Theory section should include explicit false-acceptance control. |
| pass | `reviewer_attack_preempted` | source explicitly preempts the eraser-versus-decision-layer objection | AAAI source should preempt the strongest baseline-framing attack. |
| pass | `reference_boundary_present` | source separates pinned/proxy baselines and denies universal erasure SOTA | AAAI source must not overclaim reference parity. |
| pass | `clinical_boundary_present` | source states Camelyon17/GaitPDB are not clinical deployment evidence | Medical benchmark language must not imply clinical deployment readiness. |
| pass | `estimated_length_reasonable` | estimated_main_words=1221 | AAAI source should remain plausibly within the 7-page technical limit. |
| warn | `pdflatex_available` | pdflatex=<missing> | Local final AAAI PDF compilation requires PDFLaTeX. |

## Compile Blocker

PDFLaTeX is not installed locally
