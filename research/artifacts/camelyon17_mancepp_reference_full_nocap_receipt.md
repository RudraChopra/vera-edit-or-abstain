# MANCE Reference Run

- Dataset: `camelyon17`
- Variant: `mance++`
- Claim-grade reference row: `True`
- Diagnostic reason: none
- Train/val/test examples: {'train': 302436, 'validation': 68464, 'external': 85054}

## Metrics

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `external_source_leakage_balanced_accuracy` |  |  |  |
| `external_target_balanced_accuracy` | 0.890505 | 0.874104 | -0.016401 |
| `external_worst_target_source_accuracy` | 0.846545 | 0.813695 | -0.032850 |
| `validation_source_leakage_balanced_accuracy` | 0.849188 | 0.563460 | -0.285728 |
| `validation_target_balanced_accuracy` | 0.897353 | 0.887627 | -0.009726 |

## Interpretation

This run is labeled as a claim-grade official-code MANCE reference row.
