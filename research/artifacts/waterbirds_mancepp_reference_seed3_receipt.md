# MANCE Reference Run

- Dataset: `waterbirds`
- Variant: `mance++`
- Claim-grade reference row: `True`
- Diagnostic reason: none
- Train/val/test examples: {'train': 4795, 'validation': 1199, 'external': 5794}

## Metrics

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `external_source_leakage_balanced_accuracy` | 0.905937 | 0.442872 | -0.463065 |
| `external_target_balanced_accuracy` | 0.739030 | 0.694452 | -0.044578 |
| `external_worst_target_source_accuracy` | 0.490654 | 0.520249 | 0.029595 |
| `validation_source_leakage_balanced_accuracy` | 0.906550 | 0.439457 | -0.467092 |
| `validation_target_balanced_accuracy` | 0.745691 | 0.675094 | -0.070597 |

## Interpretation

This run is labeled as a claim-grade official-code MANCE reference row.
