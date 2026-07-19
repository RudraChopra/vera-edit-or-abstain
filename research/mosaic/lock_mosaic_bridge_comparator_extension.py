#!/usr/bin/env python3
"""Lock the bridge comparator extension before receipt-content inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_OUTPUT = ROOT / "prereg_mosaic_bridge_comparator_extension_v1.json"
RAW_DIR = REPOSITORY / "research/artifacts/mosaic_bridge_confirmation_receipts_v1"
COMPARATOR_DIR = REPOSITORY / "research/artifacts/mosaic_bridge_comparator_receipts_v1"
COMPARATOR_MANIFEST = (
    REPOSITORY / "research/artifacts/mosaic_bridge_comparator_manifest_v1.json"
)
COMPARATOR_AUDIT = (
    REPOSITORY / "research/artifacts/mosaic_bridge_comparator_audit_v1.json"
)
ORIGINAL_PREREG = ROOT / "prereg_mosaic_bridge_v1.json"
CODE_FILES = (
    "research/mosaic/run_mosaic_bridge_comparator_extension.py",
    "research/mosaic/audit_mosaic_bridge_comparator_extension.py",
    "research/mosaic/lock_mosaic_bridge_comparator_extension.py",
    "research/mosaic/audit_mosaic_bridge_frontier.py",
    "research/mosaic/mosaic_bridge.py",
    "research/mosaic/mosaic_invariant.py",
    "research/mosaic/mosaic_optimizer.py",
    "research/mosaic/mosaic_transform_exact.py",
    "research/mosaic/mosaic_transform_exact_optimizer.py",
    "research/tests/test_mosaic_bridge_comparator_extension.py",
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
        raise FileExistsError("refusing to overwrite an existing comparator lock")
    if COMPARATOR_DIR.exists() or COMPARATOR_MANIFEST.exists() or COMPARATOR_AUDIT.exists():
        raise FileExistsError("comparator outcomes already exist; lock refused")
    original_hash = sha256(ORIGINAL_PREREG)
    original_sidecar = ORIGINAL_PREREG.with_suffix(
        ORIGINAL_PREREG.suffix + ".sha256"
    ).read_text(encoding="utf-8").strip()
    if original_hash != original_sidecar:
        raise ValueError("original bridge preregistration sidecar mismatch")
    existing_receipts = sorted(RAW_DIR.glob("*.json"))
    payload = {
        "project": "MOSAIC real bridge deployment-rule comparator extension",
        "status": "locked_during_raw_run_before_content_inspection",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_lock": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "original_preregistration_sha256": original_hash,
        "raw_receipts_present_at_lock": len(existing_receipts),
        "raw_receipt_filenames_present_at_lock": [
            path.name for path in existing_receipts
        ],
        "timing_disclosure": (
            "This post-preregistration extension was designed and hash-locked "
            "while the original 100-job run was in progress. The listed raw files "
            "already existed, but no receipt content or outcome aggregate was read "
            "when choosing these rules, thresholds, or reporting fields."
        ),
        "required_raw_receipt_count": 100,
        "expected_candidate_rows": 1300,
        "released_token_count": 2,
        "source_advantage_threshold": 0.35,
        "utility_thresholds": [0.30, 0.35, 0.40, 0.45, 0.49],
        "rules": {
            "mosaic_transform_exact": (
                "The original simultaneous-confidence bridge certificate and "
                "transform-exact optimizer; deploy only when the contract clears."
            ),
            "capacity_transfer": (
                "The same confidence tables and learned bridge, replacing the "
                "transform-exact transfer with the generic differential-capacity "
                "fallback; deploy only when the contract clears."
            ),
            "bridge_plugin": (
                "Fit the structured bridge and optimize the channel with all "
                "confidence radii set to zero; deploy only when point estimates clear."
            ),
            "validation_plugin": (
                "Ignore the target bridge, assume identity/no shift, set reference "
                "confidence radii to zero, and deploy only when validation point "
                "estimates clear the contract."
            ),
            "always_deploy_validation": (
                "Use the validation-only point-estimate solution with the lowest "
                "estimated error and deploy it at every utility threshold."
            ),
        },
        "selection": (
            "Within each rule and threshold, select the candidate with minimum "
            "rule-specific worst conditional error, breaking exact ties by candidate "
            "key. No diagnostic label or outcome enters selection."
        ),
        "missing_support_policy": (
            "The bridge plug-in abstains when any bridge source-label stratum is "
            "missing. MOSAIC and capacity transfer retain their registered robust "
            "missing-support behavior. Validation-only rules do not use bridge labels."
        ),
        "primary_reporting_slice": (
            "Report deployment, abstention, estimability, and false-acceptance rates "
            "for every rule at tau_U=0.40, together with all five registered thresholds."
        ),
        "comparison_gate": (
            "None. Retain and report every dataset, seed, threshold, abstention, "
            "missing-support row, and comparator ordering regardless of which rule wins."
        ),
        "audit_scope": (
            "For every stored channel, independently recompute certificate risks from "
            "the raw count tables, diagnostic risks from untouched counts, every "
            "threshold decision, and every selected candidate."
        ),
        "claim_boundary": (
            "These are matched deployment-rule and certificate ablations on a fixed "
            "finite-token interface. They do not make the plug-in rules certified, "
            "and the capacity row is a generic robust-transfer comparator rather than "
            "a separate end-to-end concept-erasure algorithm."
        ),
        "stopping_rule": (
            "Process all 100 original receipts and all available candidate rows. No "
            "seed replacement, threshold change, favorable ordering filter, or "
            "selective omission is permitted."
        ),
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
