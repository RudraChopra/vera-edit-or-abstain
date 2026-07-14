# FARO: Certified Edit-or-Abstain Selection

This repository contains the reproducibility-safe public release for FARO,
a representation-editing protocol that selects a certified source-removing edit
or returns an abstention certificate when no safe edit is validated.

Included:

- Manuscript sources and compiled draft PDFs under `research/maintrack/`
- Reproduction manifests and claim ledgers under `research/configs/`
- Audit, benchmark, and reference-baseline scripts under `research/scripts/`
- Small JSON/CSV/Markdown receipts under `research/artifacts/`

Excluded:

- Raw third-party datasets
- Large frozen embedding stores
- External-drive-only generated arrays
- Local virtual environments and `.git` metadata from the working directory

Main verification commands:

```bash
/usr/bin/python3 research/scripts/audit_claim_ledger.py --no-fail
/tmp/faro-torch-venv/bin/python research/scripts/audit_reference_baseline_scope.py --no-fail
/usr/bin/python3 research/scripts/audit_reproducibility_packet.py --no-fail
/usr/bin/python3 research/scripts/audit_maintrack_readiness.py --no-fail
/usr/bin/python3 research/scripts/audit_goal_completion.py --no-fail
```

The large Camelyon17 embedding store and third-party datasets are intentionally
not committed. The release includes receipts, manifests, scripts, and claim
boundaries so those artifacts can be regenerated or verified separately.
