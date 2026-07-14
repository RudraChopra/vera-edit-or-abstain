# MANCE Reference Run

- Dataset: `waterbirds`
- Variant: `mance++`
- Claim-grade reference row: `True`
- Diagnostic reason: none
- Train/val/test examples: {'train': 4795, 'validation': 1199, 'external': 5794}

## Metrics

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `external_source_leakage_balanced_accuracy` | 0.905937 | 0.458405 | -0.447532 |
| `external_target_balanced_accuracy` | 0.739030 | 0.689114 | -0.049916 |
| `external_worst_target_source_accuracy` | 0.490654 | 0.529595 | 0.038941 |
| `validation_source_leakage_balanced_accuracy` | 0.906550 | 0.458634 | -0.447916 |
| `validation_target_balanced_accuracy` | 0.745691 | 0.668399 | -0.077291 |

## Interpretation

This run is labeled as a claim-grade official-code MANCE reference row.
