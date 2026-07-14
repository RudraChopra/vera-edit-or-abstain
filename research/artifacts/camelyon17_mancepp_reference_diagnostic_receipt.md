# MANCE Reference Run

- Dataset: `camelyon17`
- Variant: `mance++`
- Claim-grade reference row: `False`
- Diagnostic reason: stratified subset run, not a full matched benchmark row
- Train/val/test examples: {'train': 20000, 'validation': 8000, 'external': 8000}

## Metrics

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| `external_source_leakage_balanced_accuracy` |  |  |  |
| `external_target_balanced_accuracy` | 0.868625 | 0.834250 | -0.034375 |
| `external_worst_target_source_accuracy` | 0.791250 | 0.719500 | -0.071750 |
| `validation_source_leakage_balanced_accuracy` | 0.846750 | 0.538750 | -0.308000 |
| `validation_target_balanced_accuracy` | 0.911750 | 0.886625 | -0.025125 |

## Interpretation

This is an official-code MANCE reference diagnostic. It proves integration against the upstream implementation, but it should not be cited as a full claim-grade baseline unless rerun without caps and with a finalized schedule.
