# VERA: Verified Erasure under Reweighting Ambiguity

This repository contains the reproducibility-safe public release for VERA,
a shift-aware decision layer for representation editing. It certifies paired
target-harm and retrained-attacker leakage contracts under a declared
support-aware reweighting envelope, or returns abstention when the evidence does
not support deployment.

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
python research/scripts/reproduce_vera_submission.py
python research/scripts/audit_goal_completion.py --no-fail
```

The release includes the locked preregistrations, compact receipts, official
eraser adapters, paper sources, and fail-closed audits. Large third-party
datasets, frozen embedding stores, and raw per-example arrays are intentionally
not committed; the anonymous archive records manifests and content hashes for
those external artifacts.
