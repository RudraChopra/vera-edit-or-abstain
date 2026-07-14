# FARO Experiment Matrix

## Current Evidence

| Family | Dataset | Status | Role |
| --- | --- | --- | --- |
| Text / subpopulation shift | CivilComments-WILDS | Prior full-store result artifacts exist, but the receipt and statistical report are currently macOS File Provider dataless placeholders; do not count as durable claim-ready evidence until rehydrated or regenerated | Optional third family once durable |
| Spurious correlation / background shift | Waterbirds | Full Hugging Face mirror exported to 11,788 local images, frozen ResNet-18 TRACE table validated with 512 nonconstant features, five-seed vectorized receipt passed all official gates; FARO abstains and group-reweighted ERM is stronger | Current durable official family and honest abstention/failure-analysis row |
| Medical / hospital shift | Camelyon17-WILDS | Complete 455,954-row frozen ResNet-18 store, full NumPy-store conversion, five locked protocol rows, paired statistics, and a passing official high-stakes receipt | Current durable high-stakes official family; not clinical deployment evidence |
| Medical / neuroimaging source shift | Alzheimer/OASIS-style MRI | Planned only; source-label feasibility must be verified before use | Optional flagship medical stress test |
| Controlled vision shift | Colored MNIST | Reproducible development evidence | Ablation and frontier visualization |
| Synthetic latent shift | Synthetic TRACE/FARO suite | Reproducible mechanism evidence plus FARO abstention certificate | Theory-aligned sanity checks |

## Required Official Rows

1. **CivilComments-WILDS full store.** Regenerate or rehydrate the full-store
   receipt and paired statistical report before counting this as a claim-ready
   official row.
2. **Waterbirds official.** Keep as the current durable official benchmark family and
   frame it as an abstention stress test: under the locked selection rule FARO
   abstains, and group-reweighted ERM is stronger on worst-group accuracy.
3. **Camelyon17-WILDS medical shift.** Keep as the durable high-stakes official
   row: complete frozen embeddings, NumPy store, five locked protocol rows,
   paired statistics, and a non-clinical-deployment receipt boundary.
4. **Optional third family.** Add CelebA-style spurious correlation or another
   WILDS/DomainBed benchmark if time allows.
5. **Optional Alzheimer stress test.** Use only if explicit site/scanner/cohort
   source labels are available and the claim remains representation reliability,
   not diagnosis.

## Baselines

Required baselines:

- ERM probe,
- source-balanced ERM,
- group-balanced ERM,
- GroupDRO-style probe,
- VREx-style probe,
- IRM-style probe,
- source-probe projection,
- INLP-style projection,
- LEACE-style erasure,
- SPLINCE/SPLICE-style task-preserving linear erasure,
- R-LACE-style linear adversarial erasure,
- TaCo-style target-conditioned nonlinear erasure,
- MANCE-style manifold-aware erasure,
- FARO selected frontier point,
- FARO abstention/failure mode.

Reviewer-critical baseline caveats:

- reference LEACE implementation where feasible,
- reference parity for SPLINCE/SPLICE, R-LACE, TaCo, and MANCE where feasible,
- DANN/CORAL/MMD if the paper claims broad domain-generalization comparison.

## Metrics

Each official row should report:

- validation target balanced accuracy,
- external target balanced accuracy,
- external worst target-source group accuracy,
- source leakage after editing,
- abstention rate,
- paired seed-level delta versus GroupDRO and concept-erasure baselines,
- confidence intervals and sign-test or paired-test evidence.

## Stop Rules

If FARO only wins on synthetic or Colored-MNIST-like data, do not submit as a
main-conference method paper. Reposition as an empirical study of when source
erasure fails.

Do not use the medical benchmark as a clinical safety or deployment claim; it is
only a representation-reliability stress test under hospital shift.

## Current Camelyon Smoke Evidence

`research/artifacts/camelyon17_png_smoke_report.json` validates 15 local PNG
patches across train, validation, and external splits using only the Python
standard library. This proves the direct metadata paths resolve to readable
images without PIL or WILDS imports. It is not claim-grade evidence because it
does not use frozen deep embeddings or write an official receipt.

`research/scripts/export_camelyon17_resnet18_torch_no_pil.py` now exports a
15-sample frozen ResNet-18 TRACE table without PIL or torchvision. The current
report at `research/artifacts/camelyon17_resnet18_torch_smoke_report.json`
contains 15 rows, 512 features, all three splits, the official torchvision
ResNet-18 checkpoint SHA, and `claim_grade_embedding=true`. It deliberately
keeps `claim_grade_benchmark_row=false`; the claim-grade evidence is the full
Camelyon17 store, official receipt, and paired statistics.

`research/scripts/export_camelyon17_resnet18_torch_store_no_pil.py` is the
preferred scaling route. It writes a manifest CSV plus contiguous little-endian
float32 embeddings instead of a huge text feature table. The current smoke store
at `research/artifacts/camelyon17_resnet18_torch_smoke_store_report.json` has
15 rows, 512 features, exactly 30,720 embedding bytes, and
`claim_grade_embedding_store=true`. A full Camelyon17 embedding store is
preflighted at
`research/artifacts/camelyon17_resnet18_torch_full_store_dryrun_report.json`:
455,954 rows and 933,793,792 raw float32 feature bytes before manifests and
reports.

The store route now also has a class-balanced benchmark-consumption smoke:
`research/artifacts/camelyon17_resnet18_torch_balanced_smoke_store_report.json`
selects 2 examples per class per split, converts through
`research/scripts/convert_camelyon17_f32_store_to_numpy_store.py`, and writes a
non-claim-ready receipt at
`research/artifacts/camelyon17_resnet18_torch_balanced_smoke_numpy_store_result_receipt.json`.
That receipt has `store_format=trace_embedding_store_v1`,
`official_dataset=true`, and `claim_gate_passed=false` because it is capped at
12 examples.

The stronger runner smoke is source-aware:
`research/artifacts/camelyon17_resnet18_torch_source_balanced_smoke_store_report.json`
selects 2 examples per `(split, target class, source label)` where available,
yielding 20 examples with target and source diversity in train and validation.
After conversion, `research/scripts/run_camelyon17_numpy_store_benchmark.py`
writes 15 result rows across 5 methods and 3 seeds, with paired statistics
available and source leakage populated. Its receipt remains
`claim_gate_passed=false` because it is still capped at 20 examples and 3 seeds.

## Current Baseline Expansion

`research/scripts/run_splince_style_civilcomments_baseline.py` adds a matched
task-preserving linear concept-removal proxy to the FARO-labeled CivilComments
table. The current row is `task_preserving_linear_erasure` with
`external_target_balanced_accuracy_mean=0.7325`,
`external_worst_target_source_accuracy_mean=0.6678`, and
`validation_source_leakage_balanced_accuracy_mean=0.5859`. The receipt is
`research/artifacts/splince_style_civilcomments_baseline_receipt.json`.

`research/scripts/run_rlace_style_civilcomments_baseline.py` adds a matched
linear-adversarial erasure proxy to the same table. The current row is
`linear_adversarial_concept_erasure` with
`external_target_balanced_accuracy_mean=0.7315`,
`external_worst_target_source_accuracy_mean=0.6970`, and
`validation_source_leakage_balanced_accuracy_mean=0.7009`. The receipt is
`research/artifacts/rlace_style_civilcomments_baseline_receipt.json`.

`research/scripts/run_taco_style_civilcomments_baseline.py` adds a matched
target-conditioned nonlinear erasure proxy. The current row is
`target_concept_erasure` with
`external_target_balanced_accuracy_mean=0.7327`,
`external_worst_target_source_accuracy_mean=0.6937`, and
`validation_source_leakage_balanced_accuracy_mean=0.5689`. The receipt is
`research/artifacts/taco_style_civilcomments_baseline_receipt.json`.

`research/scripts/run_mance_style_civilcomments_baseline.py` adds a matched
PCA-tangent manifold-aware erasure proxy. The current row is
`manifold_aware_concept_erasure` with
`external_target_balanced_accuracy_mean=0.7325`,
`external_worst_target_source_accuracy_mean=0.6678`, and
`validation_source_leakage_balanced_accuracy_mean=0.5944`. The receipt is
`research/artifacts/mance_style_civilcomments_baseline_receipt.json`.

The baseline fairness audit now passes with 15/15 configured baseline checks.
These are reviewer-stress baselines, not reference implementation parity
claims.

## Current Synthetic Abstention Evidence

`research/scripts/run_faro_synthetic_abstention_certificate.py` writes a
deterministic frontier for two theory-aligned cases. In the overlap case,
`lambda_y_star=0.3` and `lambda_s_star=0.7`, so the certified safe set is empty
and FARO returns `ABSTAIN`. In the non-overlap case, FARO selects
`strength=0.4` and returns `EDIT`. The generated report is
`research/artifacts/faro_synthetic_abstention_report.json`. This supports the
abstention mechanism but remains development evidence, not an official
benchmark row.

Full-run preflight is now explicit:
`research/artifacts/camelyon17_resnet18_torch_full_store_dryrun_report.json`
passes on the official archive tar stream, and the real full-store export has
completed with 455,954 byte-aligned rows. The exporter treats the manifest as
the authoritative resume ledger and trims complete extra binary rows left by an
interrupted process before appending.

The full store has been converted to a runner-compatible NumPy store at
`/Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_full_numpy_store`.
`research/scripts/run_camelyon17_numpy_store_benchmark.py` writes the official
five-row protocol receipt and paired statistics at
`research/artifacts/camelyon17_wilds_official_result_receipt.json` and
`research/artifacts/camelyon17_wilds_official_statistical_report.json`, with
`claim_gate_passed=true`.

The old extracted-PNG route exposed macOS File Provider materialization
placeholders and is no longer the active high-stakes path. The active route
streams PNG bytes directly from
`/Volumes/Backups/FARO/data/wilds/camelyon17_v1.0/archive.tar.gz`, preserving the
official metadata and avoiding hundreds of thousands of small extracted files.
PNG at
`patient_046_node_3/patch_patient_046_node_3_x_15296_y_3872.png` that
File Provider reports as present but not downloaded. That path is retained only
as recovery context; the completed tar-stream route is the active claim-grade
route.
