#!/usr/bin/env python3
"""Build and verify the double-blind AAAI code/data supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
DEFAULT_OUTPUT = HERE / "mosaic_aaai2027_code_data_anonymous.zip"
PACKAGE_ROOT = "mosaic_code_data_anonymous"
FIXED_ZIP_TIME = (2026, 7, 18, 0, 0, 0)

LOCKED_REPLAYS = (
    {
        "name": "synthetic_v1",
        "commit": "2c9eb880bec838ef0794b235d62c188a27d8267e",
        "prereg": "research/mosaic/prereg_mosaic_synthetic_v1.json",
        "sidecar": "research/mosaic/prereg_mosaic_synthetic_v1.sha256",
        "report": "research/artifacts/mosaic_synthetic_confirmation_v1.json",
    },
    {
        "name": "transform_exact_v2",
        "commit": "0ab29e375df096b44c57e176e83c8293343144cb",
        "prereg": "research/mosaic/prereg_mosaic_transform_exact_v2.json",
        "sidecar": "research/mosaic/prereg_mosaic_transform_exact_v2.sha256",
        "report": "research/artifacts/mosaic_transform_exact_confirmation_v2.json",
    },
)

SOURCE_PATTERNS = (
    "research/mosaic/*.py",
    "research/mosaic/*.md",
    "research/mosaic/*.json",
    "research/mosaic/*.sha256",
    "research/mosaic/requirements-*.txt",
    "research/tests/test_mosaic*.py",
    "research/scripts/official_eraser_adapters.py",
    "research/scripts/run_official_eraser_frontier.py",
    "research/maintrack/mosaic_aaai2027/make_mosaic_*_figure.py",
    "research/maintrack/mosaic_aaai2027/build_anonymous_code_package.py",
)

ARTIFACT_PATTERNS = (
    "research/artifacts/mosaic_*.json",
    "research/artifacts/mosaic_*.sha256",
    "research/artifacts/mosaic_bridge_confirmation_receipts_v1/*.json",
    "research/artifacts/mosaic_bridge_strict_receipts_v1/*.json",
    "research/artifacts/mosaic_bridge_strict_v2_receipts_v1/*.json",
    "research/artifacts/mosaic_bridge_comparator_receipts_v1/*.json",
    "research/artifacts/mosaic_direct_target_receipts_v1/*.json",
    "research/artifacts/mosaic_acs_bridge_raw_v3/*.json",
    "research/artifacts/mosaic_acs_bridge_strict_v3_receipts/*.json",
    "research/artifacts/mosaic_real_confirmation_v1/*.json",
    "research/artifacts/mosaic_real_exact_confirmation_v1/*.json",
)

FORBIDDEN_IDENTITY_MARKERS = (
    b"Rudra" + b" Chopra",
    b"Rudra" + b"Chopra",
    b"rudra" + b"chopra",
    b"/" + b"Users" + b"/",
)

ANONYMOUS_LICENSE = """MIT License

Copyright (c) 2026 The Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

ANONYMOUS_README = """# MOSAIC anonymous code and data supplement

This archive supports the double-blind submission "MOSAIC: Data-Certified
Stochastic Release under Structured Deployment Shift." It
contains the new method, locked protocols, focused tests, complete synthetic
receipts, official-method compact receipts, and independent replay programs.

## Environment

Python 3.12.13 was used with the versions pinned in
`research/mosaic/requirements-confirmation.txt` and
`research/mosaic/requirements-real.txt`, including PyTorch 2.13.0 for the
official eraser adapters. The claim-grade runs used an Apple M4
CPU, 16 GB unified memory, and macOS 26.2 arm64. Synthetic runs and all replay
programs are CPU-only.

## Fast verification

Run from the extracted archive root:

```bash
python -m pip install -r research/mosaic/requirements-confirmation.txt
PYTHONPATH=research/mosaic:research/scripts python -m pytest \
  research/tests/test_mosaic*.py -q
```

The claim-grade synthetic studies are replayed in their exact hash-locked code
environments, not against files improved after each preregistration:

```bash
cd locked_replay/synthetic_v1
PYTHONPATH=research/mosaic \
  python research/mosaic/audit_mosaic_synthetic_confirmation.py \
  --output /tmp/mosaic_synthetic_audit.json
cd ../../locked_replay/transform_exact_v2
PYTHONPATH=research/mosaic \
  python research/mosaic/audit_mosaic_transform_exact_confirmation.py \
  --prereg research/mosaic/prereg_mosaic_transform_exact_v2.json \
  --sidecar research/mosaic/prereg_mosaic_transform_exact_v2.sha256 \
  --report research/artifacts/mosaic_transform_exact_confirmation_v2.json \
  --output /tmp/mosaic_transform_exact_audit.json
```

The real-feature compact replay uses the separate real environment:

```bash
python -m pip install -r research/mosaic/requirements-real.txt
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/audit_mosaic_real_frontier.py \
  research/artifacts/mosaic_real_confirmation_v1/*.json \
  --output /tmp/mosaic_real_audit.json
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/audit_mosaic_real_transform_exact.py \
  --output /tmp/mosaic_real_transform_exact_audit.json
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/audit_mosaic_real_exact_frontier.py \
  research/artifacts/mosaic_real_exact_confirmation_v1/*.json \
  --output /tmp/mosaic_real_exact_confirmation_audit.json
```

The paired synthetic baselines and bridge-model stress test can be replayed
directly from their complete saved tables:

```bash
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/audit_mosaic_baseline_extension.py \
  --report research/artifacts/mosaic_baseline_extension_v1_schema_repaired.json \
  --output /tmp/mosaic_baseline_extension_audit.json
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/audit_mosaic_bridge_misspecification.py \
  --output /tmp/mosaic_bridge_misspecification_audit.json
```

The review-stage admitted-shift, ACS diagnosis, utility table, and scaling
artifacts are deterministic and can be rebuilt directly:

```bash
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/analyze_mosaic_admitted_shift_stress.py
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/analyze_mosaic_acs_infeasibility.py
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/summarize_mosaic_release_utility_table.py
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/run_mosaic_scaling_study.py
```

The external-shift evidence has four independent replay layers. The first
recomputes every raw finite-confidence bridge and global optimum. The next two
replay the corrected outward-rounded v2 receipts in floating-point and exact
rational arithmetic. The last compares v1 and v2 and verifies that every change
is exactly the disclosed structural-zero correction:

```bash
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/audit_mosaic_bridge_frontier.py \
  research/artifacts/mosaic_bridge_confirmation_receipts_v1/*.json \
  --output /tmp/mosaic_bridge_raw_audit.json
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/audit_mosaic_bridge_strict_v2.py \
  --original-dir research/artifacts/mosaic_bridge_confirmation_receipts_v1 \
  --strict-dir research/artifacts/mosaic_bridge_strict_v2_receipts_v1 \
  --prereg research/mosaic/prereg_mosaic_bridge_v1.json \
  --amendment research/mosaic/prereg_mosaic_bridge_strict_amendment_v2.json \
  --output /tmp/mosaic_bridge_strict_v2_audit.json
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/audit_mosaic_bridge_rational.py \
  --original-dir research/artifacts/mosaic_bridge_confirmation_receipts_v1 \
  --strict-dir research/artifacts/mosaic_bridge_strict_v2_receipts_v1 \
  --strict-amendment research/mosaic/prereg_mosaic_bridge_strict_amendment_v2.json \
  --rational-lock research/mosaic/prereg_mosaic_bridge_rational_audit_v2.json \
  --output /tmp/mosaic_bridge_rational_v2_audit.json
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/audit_mosaic_bridge_strict_correction_v2.py \
  --v1-dir research/artifacts/mosaic_bridge_strict_receipts_v1 \
  --v2-dir research/artifacts/mosaic_bridge_strict_v2_receipts_v1 \
  --v1-audit research/artifacts/mosaic_bridge_strict_audit_v1.json \
  --v2-audit research/artifacts/mosaic_bridge_strict_v2_audit_v1.json \
  --v2-rational-audit research/artifacts/mosaic_bridge_rational_v2_audit_v1.json \
  --v2-amendment research/mosaic/prereg_mosaic_bridge_strict_amendment_v2.json \
  --output /tmp/mosaic_bridge_strict_correction_v2_audit.json
```

The comparator extension additionally verifies that its protocol lock was
committed before outcomes were inspected. An extracted archive has no Git
history, so materialize a disposable local commit without changing any file,
then run its independent certificate audit:

```bash
git init
git add research
git -c user.name="Anonymous Authors" \
  -c user.email="anonymous@example.invalid" \
  commit -m "Materialize locked review package"
PYTHONPATH=research/mosaic:research/scripts \
  python research/mosaic/audit_mosaic_bridge_comparator_extension.py \
  --raw-dir research/artifacts/mosaic_bridge_confirmation_receipts_v1 \
  --comparator-dir research/artifacts/mosaic_bridge_comparator_receipts_v1 \
  --lock research/mosaic/prereg_mosaic_bridge_comparator_extension_v1.json \
  --output /tmp/mosaic_bridge_comparator_audit.json
```

The real-feature transformation stage additionally requires the public datasets,
frozen feature stores, and pinned official repositories recorded in
`research/mosaic/prereg_mosaic_real_v1.json`. Those third-party files are not
redistributed. Their complete token tables and replayable decisions are included.

## Certificate Coverage

The release contract is evaluated over the common-transform plus bounded-residual
class certified from the labeled bridge sample, not only at its empirical table.
Missing source-label strata lead to abstention. The archive
preserves the strict-v1 record and the disclosed structural-zero repair, along
with the deterministic and exact-rational v2 replays. The current strict-v2
receipts are the release record used by the paper.

`MANIFEST.sha256` authenticates every other file in the archive. Public authorship
and repository metadata are intentionally omitted for double-blind review.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def collect_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in (*SOURCE_PATTERNS, *ARTIFACT_PATTERNS):
        files.update(path for path in REPOSITORY.glob(pattern) if path.is_file())
    if not files:
        raise RuntimeError("anonymous package selection is empty")
    return sorted(files, key=lambda path: path.relative_to(REPOSITORY).as_posix())


def git_blob(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    )
    return completed.stdout


def write_locked_replays(root: Path) -> None:
    for replay in LOCKED_REPLAYS:
        replay_root = root / "locked_replay" / replay["name"]
        prereg_source = REPOSITORY / replay["prereg"]
        prereg = json.loads(prereg_source.read_text(encoding="utf-8"))
        for relative, expected in prereg["code_sha256"].items():
            data = git_blob(replay["commit"], relative)
            actual = sha256_bytes(data)
            if actual != expected:
                raise RuntimeError(
                    f"locked replay mismatch for {replay['name']}:{relative}: "
                    f"expected {expected}, found {actual}"
                )
            destination = replay_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        for relative, expected in prereg.get("pilot_artifact_sha256", {}).items():
            source = REPOSITORY / relative
            actual = sha256(source)
            if actual != expected:
                raise RuntimeError(
                    f"locked pilot mismatch for {replay['name']}:{relative}: "
                    f"expected {expected}, found {actual}"
                )
            destination = replay_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for key in ("prereg", "sidecar", "report"):
            source = REPOSITORY / replay[key]
            destination = replay_root / replay[key]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def write_manifest(root: Path) -> Path:
    manifest = root / "MANIFEST.sha256"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest:
            rows.append(f"{sha256(path)}  {path.relative_to(root).as_posix()}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return manifest


def scan_anonymity(root: Path) -> None:
    failures = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        for marker in FORBIDDEN_IDENTITY_MARKERS:
            if marker.lower() in data.lower():
                failures.append(f"{path.relative_to(root)} contains {marker!r}")
    if failures:
        raise RuntimeError("anonymous package failed identity scan:\n" + "\n".join(failures))


def verify_manifest(root: Path) -> None:
    manifest = root / "MANIFEST.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        actual = sha256(root / relative)
        if actual != expected:
            raise RuntimeError(f"manifest mismatch for {relative}")


def write_deterministic_zip(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = Path(PACKAGE_ROOT) / path.relative_to(root)
            info = zipfile.ZipInfo(relative.as_posix(), FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    temporary.replace(output)


def verify_zip(output: Path) -> tuple[int, int]:
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt archive member: {bad}")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("archive contains duplicate members")
        for marker in FORBIDDEN_IDENTITY_MARKERS:
            for name in names:
                if marker.lower() in archive.read(name).lower():
                    raise RuntimeError(f"archive identity scan failed for {name}")
    return len(names), output.stat().st_size


def build(output: Path) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="mosaic-anonymous-") as temporary:
        root = Path(temporary) / PACKAGE_ROOT
        for source in collect_files():
            destination = root / source.relative_to(REPOSITORY)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        write_locked_replays(root)
        (root / "README.md").write_text(ANONYMOUS_README, encoding="utf-8")
        (root / "LICENSE.txt").write_text(ANONYMOUS_LICENSE, encoding="utf-8")
        write_manifest(root)
        scan_anonymity(root)
        verify_manifest(root)
        write_deterministic_zip(root, output)
    members, size = verify_zip(output)
    return {
        "output": str(output),
        "sha256": sha256(output),
        "members": members,
        "bytes": size,
        "anonymity_scan": "pass",
        "zip_integrity": "pass",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(build(args.output.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
