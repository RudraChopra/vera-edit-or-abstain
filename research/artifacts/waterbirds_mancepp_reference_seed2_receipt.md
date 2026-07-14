# MANCE Reference Run

- Dataset: `waterbirds`
- Variant: `mance++`
- Claim-grade reference row: `True`
- Diagnostic reason: none
- Train/val/test examples: {'train': 4795, 'validation': 1199, 'external': 5794}

## Metrics

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `external_source_leakage_balanced_accuracy` | 0.905937 | 0.441319 | -0.464619 |
| `external_target_balanced_accuracy` | 0.739030 | 0.693738 | -0.045292 |
| `external_worst_target_source_accuracy` | 0.490654 | 0.537383 | 0.046729 |
| `validation_source_leakage_balanced_accuracy` | 0.906550 | 0.460299 | -0.446251 |
| `validation_target_balanced_accuracy` | 0.745691 | 0.682341 | -0.063350 |

## Interpretation

This run is labeled as a claim-grade official-code MANCE reference row.
