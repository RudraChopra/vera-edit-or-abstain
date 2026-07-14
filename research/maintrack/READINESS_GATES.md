# VERA Main-Track Readiness Gates

## Decision

Current decision: **not ready for ICLR 2027 main-track submission**.

## Passing Evidence Already Present

- Waterbirds now has a full 11,788-example frozen ResNet-18 TRACE table, a
  five-seed official-style vectorized receipt, paired statistics, and passing
  claim gates. It is an abstention/failure-analysis row: VERA abstains under
  the locked rule, while group-reweighted ERM is stronger on worst-group
  accuracy.
- CivilComments-WILDS has prior full-store stress artifacts, but it is not
  claim-ready while the receipt and paired statistical report are macOS File
  Provider dataless placeholders.
- Reproduction harness has a passing full-mode report for existing artifacts.
- Colored-MNIST and synthetic suites provide controlled mechanism evidence.
- The codebase has official benchmark receipts, provenance gates, and
  statistical analysis scripts.
- Camelyon17-WILDS has full local metadata, a 15-sample no-PIL frozen
  ResNet-18 encoder smoke table with 512 features, compact store smokes,
  source-aware runner smokes, a complete 455,954-row frozen-embedding store
  through the official archive tar-stream route, a full NumPy-store conversion,
  five locked protocol rows, paired statistics, and a passing official
  high-stakes benchmark receipt.
- A compact Camelyon17 frozen ResNet-18 embedding store smoke has passed with
  15 rows, 512 features, and exact float32 binary sizing.
- A source-aware Camelyon17 store smoke has been converted to
  `trace_embedding_store_v1` and consumed by a multi-seed benchmark runner with
  source-leakage metrics and an explicit non-claim-ready receipt.
- The VERA-labeled CivilComments baseline fairness audit now passes with
  matched SPLINCE/SPLICE-style, R-LACE-style, TaCo-style, and MANCE-style proxy
  rows. These are comparison stress tests, not reference parity claims.
- A deterministic VERA synthetic abstention certificate now passes: the overlap
  case returns `ABSTAIN` with zero safe candidates, while the non-overlap case
  returns `EDIT`. This is development evidence for abstention geometry, not an
  official benchmark result.
- A real CivilComments-WILDS abstention stress certificate now passes: the
  strict setting returns `ABSTAIN`, while a relaxed setting returns `EDIT`.

## Blocking Evidence

- Two durable official claim-ready benchmark families are present: Waterbirds
  and Camelyon17-WILDS. CivilComments has prior artifacts but is not counted
  while its receipt/statistical report are dataless placeholders.
- The medical/high-stakes official receipt requirement is now met by
  Camelyon17-WILDS, with a non-clinical-deployment claim boundary.
- Reference concept-erasure baselines need faithful implementations or explicit
  proxy labels.
- The manuscript now includes deterministic artifact-backed figures and local
  LaTeX compilation, but it still needs final camera-ready figure review and CI
  compilation before submission.
- The adversarial internal review must have no unresolved critical or major
  findings before submission.

## ICLR 2027 Go/No-Go Rule

Attempt ICLR 2027 only if all of these are true before submission:

- the second official benchmark receipt passes,
- the Waterbirds abstention result is framed honestly rather than as a win,
- a medical or equivalent high-stakes official receipt passes,
- the manuscript is rewritten around VERA,
- the novelty lock is explicitly defended against INLP, R-LACE, LEACE, TaCo,
  and domain generalization,
- the theory section proves the target-preservation interval, abstention
  criterion, and finite-candidate safety theorem,
- abstention is demonstrated with structured artifacts,
- baselines follow the baseline protocol and every proxy is labeled,
- the reproducibility checklist and code archive are ready,
- `python3 research/scripts/audit_maintrack_readiness.py` shows no blocking
  failures.

Otherwise, target the next strong venue rather than submitting a fragile paper.

## Current Runtime Status

The official `.venv/trace-official` environment has usable `torch`, but its
Pillow import path stalls while reading `PIL._version`. The system Python imports
PIL quickly but does not have `torch` or `torchvision`.

CivilComments-WILDS should not be counted as a durable local claim-ready row
while
`research/artifacts/civilcomments_wilds_full_store_balanced_trace_store_result_receipt.json`
and
`research/artifacts/civilcomments_wilds_full_store_balanced_trace_store_statistical_report.json`
are macOS File Provider dataless placeholders. Rehydrate or regenerate those
two files before citing CivilComments as a local official row.

A clean torch runtime is available at `/tmp/faro-torch-venv`; this path is now
a symlink to the external-drive runtime under `/Volumes/Backups/FARO/runtime`.
It can import `torch` and run the no-PIL ResNet-18 exporters:

```bash
/tmp/faro-torch-venv/bin/python \
  research/scripts/export_camelyon17_resnet18_torch_no_pil.py \
  --metadata research/artifacts/camelyon17_wilds_metadata.csv \
  --out research/artifacts/camelyon17_resnet18_torch_smoke_embeddings.csv \
  --report research/artifacts/camelyon17_resnet18_torch_smoke_report.json \
  --weights /Users/rudrachopra/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
  --max-examples-per-split 5 \
  --batch-size 8 \
  --device auto
```

The old extracted-PNG route is no longer the preferred full-data path. It is too
fragile on macOS File Provider and external drives because it creates 455,954
small PNG files. The active route streams directly from the official WILDS
archive on the external drive.

The tar-stream dry run should pass before any full Camelyon run:

```bash
/tmp/faro-torch-venv/bin/python \
  research/scripts/export_camelyon17_resnet18_torch_tar_store_no_pil.py \
  --archive /Volumes/Backups/FARO/data/wilds/camelyon17_v1.0/archive.tar.gz \
  --raw-metadata /Volumes/Backups/FARO/data/wilds/camelyon17_v1.0/metadata.csv \
  --store-dir /Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_full_store \
  --dry-run
```

The tar-stream dry run has passed: 455,954 rows and 933,793,792 raw float32
feature bytes in
`research/artifacts/camelyon17_resnet18_torch_full_store_dryrun_report.json`.

The active full-data recovery route uses the external drive so the project is
not bottlenecked by macOS File Provider placeholders under the repo path. The
official archive is already staged at
`/Volumes/Backups/FARO/data/wilds/camelyon17_v1.0/archive.tar.gz`.

```bash
/tmp/faro-torch-venv/bin/python - <<'PY'
from pathlib import Path
from wilds import get_dataset
root = Path("/Volumes/Backups/FARO/data/wilds")
root.mkdir(parents=True, exist_ok=True)
get_dataset(dataset="camelyon17", root_dir=str(root), download=True)
PY
```

For a smoke test that exercises the real archive reader without overclaiming:

```bash
/tmp/faro-torch-venv/bin/python \
  research/scripts/export_camelyon17_resnet18_torch_tar_store_no_pil.py \
  --store-dir /Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_tar_smoke_store \
  --report research/artifacts/camelyon17_resnet18_torch_tar_smoke_store_report.json \
  --weights /Users/rudrachopra/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
  --batch-size 8 \
  --device auto \
  --stop-after-new-rows 32 \
  --progress-every 16

python3 research/scripts/convert_camelyon17_f32_store_to_numpy_store.py \
  --source-report research/artifacts/camelyon17_resnet18_torch_tar_smoke_store_report.json \
  --out-dir /Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_tar_smoke_numpy_store \
  --report research/artifacts/camelyon17_resnet18_torch_tar_smoke_numpy_store_report.json \
  --force
```

The tar-stream smoke export has passed with 32 real archive rows, a byte-aligned
float32 embedding store, and successful NumPy-store conversion. It is not a
benchmark row. The full tar-stream store on the external drive now contains all
455,954 real archive rows with byte-aligned float32 embeddings. The exporter
repairs any complete binary tail rows left by an interrupted run before
appending; the next step is full-store NumPy conversion and benchmark receipt
generation.

For the full high-stakes export:

```bash
/tmp/faro-torch-venv/bin/python \
  research/scripts/export_camelyon17_resnet18_torch_tar_store_no_pil.py \
  --archive /Volumes/Backups/FARO/data/wilds/camelyon17_v1.0/archive.tar.gz \
  --raw-metadata /Volumes/Backups/FARO/data/wilds/camelyon17_v1.0/metadata.csv \
  --store-dir /Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_full_store \
  --report research/artifacts/camelyon17_resnet18_torch_full_store_report.json \
  --weights /Users/rudrachopra/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
  --batch-size 16 \
  --device auto \
  --resume
```

Then convert the full store and run the five-seed official receipt:

```bash
python3 research/scripts/convert_camelyon17_f32_store_to_numpy_store.py \
  --source-report research/artifacts/camelyon17_resnet18_torch_full_store_report.json \
  --out-dir /Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_full_numpy_store \
  --report research/artifacts/camelyon17_resnet18_torch_full_numpy_store_report.json \
  --force

python3 research/scripts/run_camelyon17_numpy_store_benchmark.py \
  --store-dir /Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_full_numpy_store \
  --results research/artifacts/camelyon17_wilds_official_multiseed_results.csv \
  --receipt research/artifacts/camelyon17_wilds_official_result_receipt.json \
  --statistics research/artifacts/camelyon17_wilds_official_statistical_report.json \
  --seeds 0,1,2,3,4
```

For runner integration without overclaiming, use the source-aware smoke bridge:

```bash
/tmp/faro-torch-venv/bin/python \
  research/scripts/export_camelyon17_resnet18_torch_store_no_pil.py \
  --metadata research/artifacts/camelyon17_wilds_metadata.csv \
  --store-dir research/artifacts/camelyon17_resnet18_torch_source_balanced_smoke_store \
  --report research/artifacts/camelyon17_resnet18_torch_source_balanced_smoke_store_report.json \
  --weights /Users/rudrachopra/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth \
  --max-examples-per-split 0 \
  --max-examples-per-split-class-source 2 \
  --batch-size 4 \
  --device auto

python3 research/scripts/convert_camelyon17_f32_store_to_numpy_store.py \
  --source-report research/artifacts/camelyon17_resnet18_torch_source_balanced_smoke_store_report.json \
  --out-dir research/artifacts/camelyon17_resnet18_torch_source_balanced_smoke_numpy_store \
  --report research/artifacts/camelyon17_resnet18_torch_source_balanced_smoke_numpy_store_report.json \
  --force

python3 research/scripts/run_camelyon17_numpy_store_benchmark.py \
  --store-dir research/artifacts/camelyon17_resnet18_torch_source_balanced_smoke_numpy_store \
  --results research/artifacts/camelyon17_resnet18_torch_source_balanced_smoke_numpy_store_multiseed_results.csv \
  --receipt research/artifacts/camelyon17_resnet18_torch_source_balanced_smoke_numpy_store_multiseed_receipt.json \
  --statistics research/artifacts/camelyon17_resnet18_torch_source_balanced_smoke_numpy_store_multiseed_statistics.json \
  --seeds 0,1,2
```

The smoke receipt has source-leakage metrics and paired statistics available,
but it is intentionally `claim_gate_passed=false` because it is capped at 20
examples and 3 seeds.

The synthetic abstention artifact can be regenerated with:

```bash
python3 research/scripts/run_faro_synthetic_abstention_certificate.py
python3 research/scripts/run_faro_real_abstention_stress.py
```

It writes:

- `research/artifacts/faro_synthetic_abstention_report.json`,
- `research/artifacts/faro_synthetic_abstention_frontier.csv`,
- `research/artifacts/faro_synthetic_overlap_abstention_certificate.json`,
- `research/artifacts/faro_synthetic_nonoverlap_edit_certificate.json`.

The real stress script writes:

- `research/artifacts/faro_real_abstention_stress_report.json`,
- `research/artifacts/faro_real_abstention_stress_report.md`,
- `research/artifacts/faro_real_abstention_stress_frontier.csv`.

Run the adversarial internal review after each readiness pass:

```bash
python3 research/scripts/run_faro_adversarial_review.py --no-fail
```

It writes:

- `research/artifacts/faro_adversarial_internal_review.json`,
- `research/artifacts/faro_adversarial_internal_review.md`.
