# FARO Statistical Significance Addendum

Generated at UTC: `2026-07-13T19:52:16.192893+00:00`

| Dataset | Comparison | Metric | Mean delta | Sign counts | p-value |
| --- | --- | --- | ---: | --- | ---: |
| waterbirds | FARO_selected minus group_reweighted_erm | `external_target_balanced_accuracy` | -0.067316 | +0/-5 of n=5 | 0.062500 |
| waterbirds | FARO_selected minus group_reweighted_erm | `external_worst_target_source_accuracy` | -0.195261 | +0/-5 of n=5 | 0.062500 |
| waterbirds | FARO_selected minus group_reweighted_erm | `external_source_leakage_balanced_accuracy` | 0.000000 | +3/-2 of n=5 | 1.000000 |
| camelyon17 | FARO_selected minus group_dro_probe | `external_target_balanced_accuracy` | 0.011499 | +5/-0 of n=5 | 0.062500 |
| camelyon17 | FARO_selected minus group_dro_probe | `external_worst_target_source_accuracy` | 0.028829 | +5/-0 of n=5 | 0.062500 |
| camelyon17 | FARO_selected minus group_dro_probe | `validation_source_leakage_balanced_accuracy` | 0.000000 | +0/-0 of n=0 |  |
| waterbirds | MANCE++ after minus before | `external_source_leakage_balanced_accuracy` | -0.459096 | +0/-5 of n=5 | 0.062500 |
| waterbirds | MANCE++ after minus before | `external_target_balanced_accuracy` | -0.047502 | +0/-5 of n=5 | 0.062500 |
| waterbirds | MANCE++ after minus before | `external_worst_target_source_accuracy` | 0.037695 | +5/-0 of n=5 | 0.062500 |
| waterbirds | MANCE++ after minus before | `validation_source_leakage_balanced_accuracy` | -0.454254 | +0/-5 of n=5 | 0.062500 |
| waterbirds | MANCE++ after minus before | `validation_target_balanced_accuracy` | -0.074781 | +0/-5 of n=5 | 0.062500 |

With five seeds, the smallest possible nonzero two-sided exact sign-test p-value is 0.0625.
These tests are therefore integrity checks on directionality, not claims of conventional p < 0.05 significance.
