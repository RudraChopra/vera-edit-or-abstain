# MANCE Reference Run

- Dataset: `waterbirds`
- Variant: `mance++`
- Claim-grade reference row: `True`
- Diagnostic reason: none
- Train/val/test examples: {'train': 4795, 'validation': 1199, 'external': 5794}

## Metrics

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `external_source_leakage_balanced_accuracy` | 0.905937 | 0.440801 | -0.465136 |
| `external_target_balanced_accuracy` | 0.739030 | 0.692340 | -0.046690 |
| `external_worst_target_source_accuracy` | 0.490654 | 0.526480 | 0.035826 |
| `validation_source_leakage_balanced_accuracy` | 0.906550 | 0.436956 | -0.469594 |
| `validation_target_balanced_accuracy` | 0.745691 | 0.657921 | -0.087770 |

## Interpretation

This run is labeled as a claim-grade official-code MANCE reference row.
