# MOSAIC Auditor Quickstart

## Install

```bash
python -m pip install .
mosaic-audit doctor
```

The exact submission runtime is recorded in
`requirements-runtime.lock`. The package API is intentionally small:
`fit`, `certify`, `release_or_abstain`, and `write_report`.

## Minimal Audit

```python
from mosaic_certified_release import Mosaic, MosaicConfig

gate = Mosaic(MosaicConfig(
    privacy_advantage_threshold=0.35,
    utility_error_threshold=0.40,
))
gate.fit(construction_features, construction_labels)
result = gate.certify(
    reference_features,
    reference_labels,
    reference_sources,
    bridge_features,
    bridge_labels,
    bridge_sources,
)
gate.write_report("audit-report")

response = gate.release_or_abstain("persistent-item-id", one_feature_row)
```

`CERTIFIED` means every registered contract cleared its simultaneous
finite-sample bound. Any missing support, infeasible optimization, or failed
contract returns `ABSTAIN`. Do not replace that decision with an uncertified
fallback.

## Continuous Monitoring

`AnytimeAuditMonitor` maintains Dirichlet-mixture e-process confidence
regions. Its snapshots remain valid after optional stopping, so an auditor can
inspect the gate repeatedly without assigning a new alpha budget to every
calendar check.

## Verify a Report

```bash
mosaic-audit verify-report audit-report/certification_report.json
```

The report records configuration, runtime versions, bounds, release channel,
decoder, and the central decision in both JSON and self-contained HTML.
