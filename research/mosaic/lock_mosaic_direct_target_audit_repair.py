#!/usr/bin/env python3
"""Lock the mechanical direct-target audit import repair before replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_OUTPUT = ROOT / "prereg_mosaic_direct_target_audit_repair_v1.json"
DIRECT_LOCK = ROOT / "prereg_mosaic_direct_target_v1.json"
DIRECT_AUDIT = REPOSITORY / "research/artifacts/mosaic_direct_target_audit_v1.json"
DIRECT_DIR = REPOSITORY / "research/artifacts/mosaic_direct_target_receipts_v1"
CODE_FILES = (
    "research/mosaic/audit_mosaic_direct_target_comparator_v2.py",
    "research/mosaic/lock_mosaic_direct_target_audit_repair.py",
    "research/mosaic/audit_mosaic_bridge_comparator_extension.py",
    "research/mosaic/run_mosaic_direct_target_comparator.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite an existing audit-repair lock")
    if DIRECT_AUDIT.exists():
        raise FileExistsError("an audit outcome already exists")
    if len(list(DIRECT_DIR.glob("*.json"))) != 100:
        raise ValueError("direct-target receipt set is incomplete")
    payload = {
        "project": "MOSAIC direct target-table audit import repair",
        "status": "locked_audit_import_repair_before_replay",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "direct_target_lock_sha256": sha256(DIRECT_LOCK),
        "direct_target_receipt_count": 100,
        "repair": {
            "trigger": "The first audit invocation raised ImportError before reading any receipt values.",
            "root_cause": "v1 imported audit_release from the generator module instead of the comparator audit module.",
            "change": "v2 changes only that import and otherwise preserves the v1 replay logic.",
            "classification": "post-outcome mechanical audit-repair disclosure",
            "prediction_before_replay": "The corrected audit should process 100 receipts and 1300 candidates with no mismatch if the generator is correct.",
        },
        "code_sha256": {
            relative: sha256(REPOSITORY / relative) for relative in CODE_FILES
        },
    }
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(data, encoding="utf-8")
    sidecar.write_text(hashlib.sha256(data.encode("utf-8")).hexdigest() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
