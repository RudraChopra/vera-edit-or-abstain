# MANCE Reference Run

- Dataset: `waterbirds`
- Variant: `mance++`
- Claim-grade reference row: `True`
- Diagnostic reason: none
- Train/val/test examples: {'train': 4795, 'validation': 1199, 'external': 5794}

## Metrics

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `external_source_leakage_balanced_accuracy` | 0.905937 | 0.450811 | -0.455126 |
| `external_target_balanced_accuracy` | 0.739030 | 0.687999 | -0.051031 |
| `external_worst_target_source_accuracy` | 0.490654 | 0.528037 | 0.037383 |
| `validation_source_leakage_balanced_accuracy` | 0.906550 | 0.466132 | -0.440417 |
| `validation_target_balanced_accuracy` | 0.745691 | 0.670791 | -0.074900 |

## Interpretation

This run is labeled as a claim-grade official-code MANCE reference row.
