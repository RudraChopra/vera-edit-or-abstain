#!/usr/bin/env python3
"""Lock the baseline replay audit before any baseline outcomes are produced."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_OUTPUT = ROOT / "prereg_mosaic_baseline_extension_audit_v1.json"
BASELINE_PREREG = ROOT / "prereg_mosaic_baseline_extension_v1.json"
BASELINE_OUTPUT = REPOSITORY / "research/artifacts/mosaic_baseline_extension_v1.json"
CODE_FILES = (
    "research/mosaic/audit_mosaic_baseline_extension.py",
    "research/mosaic/lock_mosaic_baseline_extension_audit.py",
    "research/mosaic/run_mosaic_baseline_extension.py",
    "research/mosaic/run_mosaic_synthetic_pilot.py",
    "research/mosaic/mosaic_exact.py",
    "research/mosaic/mosaic_invariant.py",
    "research/mosaic/mosaic_optimizer.py",
    "research/tests/test_mosaic_baseline_extension_audit.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite an existing audit lock")
    if BASELINE_OUTPUT.exists():
        raise FileExistsError("baseline outcomes already exist; pre-outcome lock refused")
    baseline_hash = sha256(BASELINE_PREREG)
    baseline_sidecar = BASELINE_PREREG.with_suffix(
        BASELINE_PREREG.suffix + ".sha256"
    ).read_text(encoding="utf-8").strip()
    if baseline_hash != baseline_sidecar:
        raise AssertionError("baseline preregistration sidecar mismatch")
    payload = {
        "project": "MOSAIC paired baseline deterministic replay audit",
        "status": "locked_before_baseline_outcomes",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_lock": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "baseline_preregistration_sha256": baseline_hash,
        "baseline_output_present_at_lock": False,
        "audit_scope": (
            "Regenerate every table from its locked seed; rerun every method; "
            "recompute population risks, deployment labels, diagnostics, exact "
            "aggregate counts, rates, means, and Clopper-Pearson intervals."
        ),
        "independence_boundary": (
            "This is a separately hash-locked deterministic replay, not an "
            "independent human replication. It shares the registered optimizer "
            "and method implementations but independently constructs risk labels "
            "and aggregate checks."
        ),
        "comparison_gate": "None; all method outcomes and orderings are retained.",
        "code_sha256": {
            relative: sha256(REPOSITORY / relative) for relative in CODE_FILES
        },
    }
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(data, encoding="utf-8")
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    sidecar.write_text(digest + "\n", encoding="utf-8")
    print(digest)


if __name__ == "__main__":
    main()
