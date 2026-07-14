# FARO Baseline Fairness Audit

Generated at UTC: `2026-07-13T20:17:11.757840+00:00`
Baseline ready: `True`

| Status | Baseline | Implementation status | Requirement |
| --- | --- | --- | --- |
| pass | `erm_probe` | local_reference | ERM probe is available under the same frozen representation protocol. |
| pass | `source_balanced_erm` | local_reference | Source-balanced ERM is available where source labels are defined. |
| pass | `group_reweighted_erm` | local_reference | Group-reweighted ERM is available where target-source groups are defined. |
| pass | `group_dro_probe` | local_reference | GroupDRO-style probe is available as the robust linear-probe baseline. |
| pass | `source_probe_projection` | local_proxy | Source-probe projection is labeled as a representation-edit stress test. |
| pass | `inlp_style_projection` | style_proxy | INLP-style rows are labeled as style/proxy rows unless reference code is run. |
| pass | `leace_closed_form_affine_erasure` | style_proxy | LEACE-style rows are scoped unless exact reference parity is shown. |
| pass | `splince_style_task_preserving_erasure` | style_proxy | SPLINCE/SPLICE rows are proxy rows unless reference receipts are added. |
| pass | `rlace_style_linear_adversarial_erasure` | style_proxy | R-LACE rows are proxy rows unless reference receipts are added. |
| pass | `taco_style_target_conditioned_erasure` | style_proxy | TaCo rows are proxy rows unless reference receipts are added. |
| pass | `mancepp_reference_waterbirds` | official_reference | Official-code MANCE++ Waterbirds receipt(s): ['waterbirds_mancepp_reference_receipt.json', 'waterbirds_mancepp_reference_seed0_receipt.json', 'waterbirds_mancepp_reference_seed1_receipt.json', 'waterbirds_mancepp_reference_seed2_receipt.json', 'waterbirds_mancepp_reference_seed3_receipt.json', 'waterbirds_mancepp_reference_seed4_receipt.json'] |
| pass | `mancepp_camelyon17_boundary` | diagnostic_boundary | Camelyon17 MANCE++ is diagnostic only and must not be presented as a full reference row. |
| pass | `faro_selected_frontier_point` | method_under_test | FARO selected point and abstention decision are reported separately from baselines. |
| pass | `claim_boundary` | audit_boundary | Tables and prose distinguish proxy stress tests from official reference implementations. |
| pass | `no_sota_erasure_claim` | claim_boundary | Evidence supports FARO as certified selection/abstention, not universal SOTA erasure. |
