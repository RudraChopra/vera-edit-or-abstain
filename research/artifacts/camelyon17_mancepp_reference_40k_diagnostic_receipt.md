# MANCE Reference Run

- Dataset: `camelyon17`
- Variant: `mance++`
- Claim-grade reference row: `False`
- Diagnostic reason: stratified subset run, not a full matched benchmark row
- Train/val/test examples: {'train': 40000, 'validation': 16000, 'external': 16000}

## Metrics

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `external_source_leakage_balanced_accuracy` |  |  |  |
| `external_target_balanced_accuracy` | 0.875938 | 0.851438 | -0.024500 |
| `external_worst_target_source_accuracy` | 0.809875 | 0.755625 | -0.054250 |
| `validation_source_leakage_balanced_accuracy` | 0.847395 | 0.541005 | -0.306390 |
| `validation_target_balanced_accuracy` | 0.906498 | 0.886941 | -0.019557 |

## Interpretation

This is an official-code MANCE reference diagnostic. It proves integration against the upstream implementation, but it should not be cited as a full claim-grade baseline unless rerun without caps and with a finalized schedule.
