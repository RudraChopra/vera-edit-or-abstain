# AAAI-27 Source Readiness Audit

Generated at UTC: `2026-07-14T19:08:09.936882+00:00`
Source ready: `True`
Warning count: `0`

| Status | Check | Evidence | Requirement |
| --- | --- | --- | --- |
| pass | `source_materialized` | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/maintrack/aaai2027_template/AuthorKit27/vera_paper_body.tex; materialized=True | AAAI source must exist locally and not be a dataless placeholder. |
| pass | `official_style_materialized` | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/maintrack/aaai2027_template/AuthorKit27/aaai2027.sty; materialized=True | Official AAAI-27 style file must be present. |
| pass | `uses_official_submission_style` | source uses \usepackage[submission]{aaai2027} | Submission source must use the official AAAI submission style. |
| pass | `anonymous_author_block` | author block is anonymous and local identity strings are absent | AAAI double-anonymous source must not expose author identity. |
| pass | `anonymous_source_materialized` | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/maintrack/aaai2027_template/AuthorKit27/vera_aaai2027_anonymous.tex; materialized=True | Dedicated anonymous AAAI source must exist locally. |
| pass | `anonymous_source_identity_free` | anonymous_leaks=[] | Anonymous AAAI source must include code availability without named identity or GitHub URL. |
| pass | `named_source_release_metadata` | path=/Volumes/Backups/FARO/github_export/vera-edit-or-abstain/research/maintrack/aaai2027_template/AuthorKit27/vera_aaai2027_named.tex; materialized=True; has_author=True; has_repo=True | Named AAAI source must contain author metadata and the public release URL. |
| pass | `required_sections_present` | missing_sections=[] | AAAI source must include the core method-paper sections. |
| pass | `official_baseline_matrix_present` | source identifies all five official erasers, 200 runs, and the zero-proxy boundary | AAAI source must describe the current official baseline matrix without proxy rows. |
| pass | `false_acceptance_corollary_present` | source contains the false-acceptance corollary | Theory section should include explicit false-acceptance control. |
| pass | `reviewer_attack_preempted` | source squarely attributes finite-family testing and distinguishes VERA from LTT and Prompt Risk Control | AAAI source should preempt the closest-prior-work objection. |
| pass | `reference_boundary_present` | source identifies pinned official baselines, excludes proxy rows, and states explicit non-novelty boundaries | AAAI source must not overclaim baseline or method novelty. |
| pass | `clinical_boundary_present` | source states Camelyon17/GaitPDB are not clinical deployment evidence | Medical benchmark language must not imply clinical deployment readiness. |
| pass | `estimated_length_reasonable` | estimated_main_words=4035 | AAAI source should remain plausibly within the 7-page technical limit. |
| pass | `latex_engine_available` | latex_engine=/opt/homebrew/bin/tectonic | Local final AAAI PDF compilation requires PDFLaTeX or Tectonic. |
