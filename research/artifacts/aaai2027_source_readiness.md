# AAAI-27 Source Readiness Audit

Generated at UTC: `2026-07-14T01:39:04.887256+00:00`
Source ready: `True`
Warning count: `1`

| Status | Check | Evidence | Requirement |
| --- | --- | --- | --- |
| pass | `source_materialized` | path=/Users/rudrachopra/Documents/Science Fair/research/maintrack/aaai2027_template/AuthorKit27/faro_aaai2027_draft.tex; materialized=True | AAAI source must exist locally and not be a dataless placeholder. |
| pass | `official_style_materialized` | path=/Users/rudrachopra/Documents/Science Fair/research/maintrack/aaai2027_template/AuthorKit27/aaai2027.sty; materialized=True | Official AAAI-27 style file must be present. |
| pass | `uses_official_submission_style` | source uses \usepackage[submission]{aaai2027} | Submission source must use the official AAAI submission style. |
| pass | `anonymous_author_block` | author block is anonymous and local identity strings are absent | AAAI double-anonymous source must not expose author identity. |
| pass | `anonymous_source_materialized` | path=/Users/rudrachopra/Documents/Science Fair/research/maintrack/aaai2027_template/AuthorKit27/faro_aaai2027_anonymous.tex; materialized=True | Dedicated anonymous AAAI source must exist locally. |
| pass | `anonymous_source_identity_free` | anonymous_leaks=[] | Anonymous AAAI source must include code availability without named identity or GitHub URL. |
| pass | `named_source_release_metadata` | path=/Users/rudrachopra/Documents/Science Fair/research/maintrack/aaai2027_template/AuthorKit27/faro_aaai2027_named.tex; materialized=True; has_author=True; has_repo=True | Named AAAI source must contain author metadata and the public release URL. |
| pass | `required_sections_present` | missing_sections=[] | AAAI source must include the core method-paper sections. |
| pass | `aaai_full_draft_length` | estimated_main_words=2822 | AAAI source should be a real full paper draft, not a short extended abstract. |
| pass | `risk_control_related_work_present` | source cites LTT, RCPS, Pareto Testing, and selective classification | AAAI source must position VERA against close risk-control and abstention literature. |
| pass | `current_camelyon_mance_reference` | AAAI source contains the full no-cap Camelyon MANCE++ reference numbers | AAAI source must cite the full no-cap Camelyon MANCE++ receipt rather than superseded diagnostics. |
| pass | `no_stale_camelyon_mance_terms` | stale_terms_present=[] | Superseded 40k Camelyon MANCE++ text must be removed. |
| pass | `validation_only_theory_boundary` | theory states validation guarantee and separates external transfer assumptions | Theory must not claim distribution-free external-shift control from validation intervals alone. |
| pass | `reviewer_attack_preempted` | source explicitly preempts the eraser-versus-decision-layer objection | AAAI source should preempt the strongest baseline-framing attack. |
| pass | `reference_boundary_present` | source separates completed LEACE/INLP rows from incomplete R-LACE/TaCo receipts and denies universal erasure SOTA | AAAI source must not overclaim reference parity. |
| pass | `real_linear_eraser_rows_present` | report=/Users/rudrachopra/Documents/Science Fair/research/artifacts/linear_eraser_reference_report.json; reference_rows_ready=True; leace_official_code=True | AAAI source must include real INLP and official LEACE frozen-feature rows. |
| pass | `protocol_constants_present` | source states thresholds, confidence level, frontier strengths, and Hoeffding multiplicity rule | AAAI source must state the protocol constants and CI construction. |
| pass | `waterbirds_reframed_as_abstention` | source labels Waterbirds as an abstention/failure-analysis row | Waterbirds result must not be framed as an accuracy win. |
| pass | `clinical_boundary_present` | source states Camelyon17/GaitPDB are not clinical deployment evidence | Medical benchmark language must not imply clinical deployment readiness. |
| pass | `reference_count_reasonable` | bibliography_items=10 | AAAI source should cite the close erasure, risk-control, and abstention literature. |
| warn | `pdflatex_available` | pdflatex=<missing> | Local final AAAI PDF compilation requires PDFLaTeX. |

## Compile Blocker

PDFLaTeX is not installed locally
