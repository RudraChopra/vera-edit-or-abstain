# Independent Replication Packet

## Scope

This packet is designed for an outside evaluator who has not contributed to
MOSAIC. It reproduces the public package checks, the theorem implementation
tests, the locked confirmation receipts, and the Lean soundness core. A run by
the authors is an internal replay; only an unaffiliated evaluator may describe
their run as an external replication.

## Clean-room procedure

Use a fresh Python 3.11 or newer environment:

```sh
python -m pip install -e ".[test]"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
mosaic-audit doctor
```

Build the mechanized core separately:

```sh
cd research/formal/MosaicFormal
lake build
```

Run the artifact integrity audit:

```sh
python research/mosaic/audit_mosaic_submission_package.py
```

The evaluator should record the repository commit, operating system, Python
version, Lean version, command transcript, and any mismatch. The report should
distinguish:

- byte-level receipt verification;
- recomputation from frozen feature stores;
- re-extraction from raw data;
- proof-kernel checking; and
- any result that could not be reproduced.

## Independence statement

The repository intentionally contains no prefilled claim of external
replication. An outside evaluator may add a signed report only after completing
the procedure. The manuscript may cite that report only after it exists.
