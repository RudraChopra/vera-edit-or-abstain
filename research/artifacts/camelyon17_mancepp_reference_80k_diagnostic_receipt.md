# MANCE Reference Run

- Dataset: `camelyon17`
- Variant: `mance++`
- Claim-grade reference row: `False`
- Diagnostic reason: stratified subset run, not a full matched benchmark row
- Train/val/test examples: {'train': 80000, 'validation': 24000, 'external': 24000}

## Metrics

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `external_source_leakage_balanced_accuracy` |  |  |  |
| `external_target_balanced_accuracy` | 0.877917 | 0.846542 | -0.031375 |
| `external_worst_target_source_accuracy` | 0.813167 | 0.743000 | -0.070167 |
| `validation_source_leakage_balanced_accuracy` | 0.847223 | 0.569314 | -0.277910 |
| `validation_target_balanced_accuracy` | 0.904422 | 0.890573 | -0.013849 |

## Interpretation

This is an official-code MANCE reference diagnostic. It proves integration against the upstream implementation, but it should not be cited as a full claim-grade baseline unless rerun without caps and with a finalized schedule.
