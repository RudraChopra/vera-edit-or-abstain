# MOSAIC code and data supplement

This anonymous archive accompanies the double-blind submission **MOSAIC:
Data-Certified Stochastic Release under Structured Deployment Shift**. It
contains the method implementation, locked configurations, compact trial
outcomes, sanitized certificate receipts, and deterministic claim replays.

## Setup

The locked environment is Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0,
scikit-learn 1.9.0, and PyTorch 2.13.0. Linear and mixed-integer programs use
the HiGHS backend exposed by SciPy. All reported runs are CPU-only.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r environment/requirements.txt
```

The equivalent Conda lock is `environment/environment.yml`.

## One-command replay

From the extracted archive root:

```bash
python verify/check_hashes.py
python verify/replay_all.py --workers 2
```

`replay_all.py` runs every entry point below, asserts each paper value, and
then checks 1,300 bridge certificates and 1,400 release bounds using exact
rational arithmetic. Each command writes a new receipt under
`artifacts/reproduced/`. For a quick table-only check, add
`--skip-exact-rational`; that option does not validate the serialized
certificate arithmetic.

The full 1,000-replicate synthetic study can also be regenerated from its
archive-local source lock. This is substantially slower than compact replay:

```bash
python data/synthetic/regenerate.py --workers 8
```

The generator verifies every source and pilot-input hash before it starts,
writes the complete report to `artifacts/reproduced/`, and asserts the primary
421-versus-0 safety cell at completion.

## Claim map

| Paper result | Replay | Expected output |
|---|---|---|
| 1,000-trial safety cell | `scripts/run_synthetic_safety.py --seed 1260000` | naive continuum 421/1000; MOSAIC 0/1000; exact 95% Clopper-Pearson intervals |
| Matched retention baseline | `scripts/run_matched_baselines.py --seed 910000000` | MOSAIC 579/1000; Holm-LTT 283/1000; McNemar 2.28e-72 |
| Real 100-job comparison | `scripts/run_real_100jobs.py --seed 1200` | strict 20 (0); direct 36 (0); bridge plug-in 47 (7); validation 80 (18); unconditional 100 (38) |
| Admitted-shift stress | `scripts/run_admitted_shift_stress.py --seed 1200` | direct 16/36 violations; MOSAIC 20 safe releases and 16 abstentions; zero MOSAIC violations on all five thresholds |
| Released-interface utility | `scripts/run_utility_table.py --seed 1200` | BiasBios .863 [.857,.868], .873 four-bin, .901 full edited, .903 unedited; Waterbirds row |
| Scaling study | `scripts/run_scaling.py --seed 4100` | 75/75 certify at .40; K=64,G=4 median 0.359 seconds; n-for-radius 2000 to 12817 |
| ACS California to Texas | `scripts/run_acs_ca_tx.py --seed 1305` | abstain 5/5 at .40; release 5/5 at .45 and .49; two identity no-ops; best error .4023 to .4145 |
| Bridge power and validity | `scripts/run_bridge_power.py --seed 930000000` | valid acceptance 0%, 31%, 100%, 100%; both invalid mechanisms rejected |
| Exact certificate audit | `verify/audit_exact_rational.py --seed 1200 --workers 2` | 1,300 bridge certificates and 1,400 outward bounds, zero failures |

The `--seed` choices are restricted to the locked seed families. The aggregate
scripts replay all trials in their claim cell; the supplied seed identifies
the locked family and is recorded in the new audit receipt.

## Contents

- `src/`: bridge optimization, strict certification, exact-rational checks,
  and replay helpers.
- `scripts/`: one executable result replay per paper table or figure group.
- `data/synthetic/`: locked synthetic configuration, compact trials, and the
  full archive-local generator.
- `data/real/`: public benchmark acquisition and preprocessing protocol. Raw
  datasets are intentionally excluded.
- `artifacts/frozen/`: immutable compact outcomes and summary receipts.
- `artifacts/certificates/`: sanitized per-job token tables and strict release
  certificates used by the exact-rational audit.
- `verify/`: full replay and SHA-256 verification.
- `MANIFEST_SHA256.txt`: SHA-256 for every immutable archive member except the
  manifest itself. Newly generated files under `artifacts/reproduced/` are not
  part of the immutable manifest.

## Determinism and hardware

Claim-grade runs used Apple M4 hardware with 10 CPU cores and 16 GB unified
memory on macOS 26.2 arm64. GPU acceleration is neither required nor used.
Random seeds, candidate frontiers, thresholds, and split rules are locked.
Compact replay is deterministic. Wall-clock timing can vary with system load;
the script checks the recorded solve-time table rather than demanding identical
timing on a reviewer's machine.

## Double-blind compliance

The archive contains no author names, affiliations, acknowledgements, email
addresses, version-control history, absolute user paths, or external links.
Third-party provenance is represented by method and dataset names only. The
build fails if an identity marker, absolute user path, or web address appears.
