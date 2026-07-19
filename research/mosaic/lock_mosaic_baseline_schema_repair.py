#!/usr/bin/env python3
"""Lock a schema-only repair after the baseline audit exposed a missing key."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_OUTPUT = ROOT / "prereg_mosaic_baseline_schema_repair_v1.json"
ORIGINAL_REPORT = REPOSITORY / "research/artifacts/mosaic_baseline_extension_v1.json"
REPAIRED_REPORT = (
    REPOSITORY
    / "research/artifacts/mosaic_baseline_extension_v1_schema_repaired.json"
)
ORIGINAL_PREREG = ROOT / "prereg_mosaic_baseline_extension_v1.json"
ORIGINAL_AUDIT_LOCK = ROOT / "prereg_mosaic_baseline_extension_audit_v1.json"
CODE_FILES = (
    "research/mosaic/repair_mosaic_baseline_report_schema.py",
    "research/mosaic/lock_mosaic_baseline_schema_repair.py",
    "research/tests/test_mosaic_baseline_schema_repair.py",
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
        raise FileExistsError("refusing to overwrite an existing schema-repair lock")
    if REPAIRED_REPORT.exists():
        raise FileExistsError("repaired report already exists; amendment lock refused")
    if not ORIGINAL_REPORT.exists():
        raise FileNotFoundError("original baseline report is absent")
    report = json.loads(ORIGINAL_REPORT.read_text(encoding="utf-8"))
    rows = report.get("replicate_results")
    if not isinstance(rows, list) or len(rows) != 8000:
        raise ValueError("original report does not contain the registered 8,000 rows")
    if any("scenario" in row for row in rows):
        raise ValueError("original rows already contain the field proposed for repair")
    payload = {
        "project": "MOSAIC paired baseline schema-only repair",
        "status": "post_outcome_schema_only_amendment",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_lock": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "trigger": (
            "The separately prelocked audit terminated before replay with "
            "KeyError('scenario') because the runner serialized ScenarioResult "
            "without a scenario field, while diagnostics and aggregate cells retained it."
        ),
        "original_report_sha256": sha256(ORIGINAL_REPORT),
        "original_preregistration_sha256": sha256(ORIGINAL_PREREG),
        "original_audit_lock_sha256": sha256(ORIGINAL_AUDIT_LOCK),
        "required_repaired_rows": 8000,
        "sample_size_to_scenario": {
            "125": "hard_safety_boundary",
            "250": "retention_and_stochastic_value",
        },
        "allowed_change": (
            "Add exactly one scenario key to each replicate row, using the unique "
            "sample-size mapping already registered and used by the original runner "
            "for aggregate cells. Add one top-level provenance object."
        ),
        "forbidden_changes": (
            "No seed, method, channel, decoder, estimate, decision, diagnostic, "
            "aggregate, interval, threshold, or pass condition may change."
        ),
        "audit_policy": (
            "Run the unchanged, pre-outcome-locked deterministic replay audit against "
            "the repaired copy and retain the original report byte-for-byte."
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
