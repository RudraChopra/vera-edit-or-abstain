#!/usr/bin/env python3
"""Lock the direct target-table comparator before its results are computed."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_OUTPUT = ROOT / "prereg_mosaic_direct_target_v1.json"
RAW_DIR = REPOSITORY / "research/artifacts/mosaic_bridge_confirmation_receipts_v1"
DIRECT_DIR = REPOSITORY / "research/artifacts/mosaic_direct_target_receipts_v1"
MANIFEST = REPOSITORY / "research/artifacts/mosaic_direct_target_manifest_v1.json"
AUDIT = REPOSITORY / "research/artifacts/mosaic_direct_target_audit_v1.json"
ORIGINAL_PREREG = ROOT / "prereg_mosaic_bridge_v1.json"
CODE_FILES = (
    "research/mosaic/run_mosaic_direct_target_comparator.py",
    "research/mosaic/audit_mosaic_direct_target_comparator.py",
    "research/mosaic/lock_mosaic_direct_target_comparator.py",
    "research/mosaic/run_mosaic_bridge_comparator_extension.py",
    "research/mosaic/audit_mosaic_bridge_frontier.py",
    "research/mosaic/mosaic_transform_exact.py",
    "research/mosaic/mosaic_transform_exact_optimizer.py",
    "research/mosaic/mosaic_optimizer.py",
    "research/mosaic/mosaic_real.py",
    "research/tests/test_mosaic_direct_target_comparator.py",
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
        raise FileExistsError("refusing to overwrite an existing direct-target lock")
    if DIRECT_DIR.exists() or MANIFEST.exists() or AUDIT.exists():
        raise FileExistsError("direct-target outcomes already exist; lock refused")
    original_hash = sha256(ORIGINAL_PREREG)
    original_sidecar = ORIGINAL_PREREG.with_suffix(
        ORIGINAL_PREREG.suffix + ".sha256"
    ).read_text(encoding="utf-8").strip()
    if original_hash != original_sidecar:
        raise ValueError("original bridge preregistration sidecar mismatch")
    receipts = sorted(RAW_DIR.glob("*.json"))
    payload = {
        "project": "MOSAIC direct target-table comparator",
        "status": "locked_post_outcome_comparator_before_execution",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_lock": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "original_preregistration_sha256": original_hash,
        "raw_receipts_present_at_lock": len(receipts),
        "raw_receipt_filenames": [path.name for path in receipts],
        "required_raw_receipt_count": 100,
        "expected_candidate_rows": 1300,
        "released_token_count": 2,
        "source_advantage_threshold": 0.35,
        "utility_thresholds": [0.30, 0.35, 0.40, 0.45, 0.49],
        "rule": (
            "Construct simultaneous multinomial confidence sets directly on each "
            "labeled target bridge table and optimize the finite-token release with "
            "identity transform and zero contamination."
        ),
        "selection": (
            "At each threshold, select the direct-target candidate with the minimum "
            "certified worst conditional error, breaking ties by candidate key. The "
            "untouched diagnostic fold never enters selection."
        ),
        "missing_support_policy": (
            "A missing target source-label stratum forces direct-target abstention."
        ),
        "primary_reporting_slice": (
            "Report every dataset, seed, threshold, deployment, abstention, target "
            "support status, diagnostic estimability, and false acceptance at tau_U=0.40."
        ),
        "timing_disclosure": (
            "This is an explicitly post-outcome reviewer-requested comparator. The "
            "original 100 raw receipts existed before this lock; no direct-target "
            "output was computed before the rule, thresholds, selection, audit, and "
            "reporting fields were fixed. It is not presented as an original "
            "preregistered outcome."
        ),
        "comparison_gate": "None. Every result is retained regardless of which rule wins.",
        "claim_boundary": (
            "Direct target certification covers only the target law sampled by the "
            "bridge fold. It does not certify the transform-plus-contamination class "
            "or post-bridge external drift addressed by MOSAIC."
        ),
        "stopping_rule": (
            "Process all 100 raw receipts and all available candidate rows with no "
            "seed replacement, threshold change, favorable ordering filter, or omission."
        ),
        "code_sha256": {
            relative: sha256(REPOSITORY / relative) for relative in CODE_FILES
        },
    }
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(data, encoding="utf-8")
    sidecar.write_text(hashlib.sha256(data.encode("utf-8")).hexdigest() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
