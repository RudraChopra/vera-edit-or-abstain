#!/usr/bin/env python3
"""Add the omitted scenario key to baseline rows without changing outcomes."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from mosaic_real import sha256
from run_mosaic_official_frontier_exact_confirmation import atomic_json_dump


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_REPORT = REPOSITORY / "research/artifacts/mosaic_baseline_extension_v1.json"
DEFAULT_LOCK = ROOT / "prereg_mosaic_baseline_schema_repair_v1.json"
DEFAULT_OUTPUT = (
    REPOSITORY
    / "research/artifacts/mosaic_baseline_extension_v1_schema_repaired.json"
)


def validate_lock(path: Path) -> tuple[dict[str, object], str]:
    digest = sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists() or sidecar.read_text(encoding="utf-8").strip() != digest:
        raise ValueError("schema-repair lock sidecar mismatch")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != "post_outcome_schema_only_amendment":
        raise ValueError("schema-repair amendment is not locked")
    relative = path.resolve().relative_to(REPOSITORY.resolve())
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative.as_posix()}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout
    if committed != path.read_bytes():
        raise ValueError("schema-repair amendment is not committed")
    for code_path, expected in lock["code_sha256"].items():
        if sha256(REPOSITORY / code_path) != expected:
            raise ValueError(f"schema-repair code hash mismatch: {code_path}")
    return lock, digest


def repair_report(
    report: dict[str, object], *, sample_size_to_scenario: dict[int, str]
) -> tuple[dict[str, object], int]:
    repaired = json.loads(json.dumps(report))
    rows = repaired.get("replicate_results")
    if not isinstance(rows, list):
        raise ValueError("baseline report has no replicate_results list")
    changed = 0
    for row in rows:
        if "scenario" in row:
            raise ValueError("refusing to alter a row that already has a scenario")
        sample_size = int(row["sample_size_per_stratum"])
        if sample_size not in sample_size_to_scenario:
            raise ValueError(f"unregistered sample size: {sample_size}")
        row["scenario"] = sample_size_to_scenario[sample_size]
        changed += 1
    return repaired, changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock, lock_hash = validate_lock(args.lock)
    if sha256(args.report) != lock["original_report_sha256"]:
        raise ValueError("baseline report differs from the locked repair input")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    mapping = {
        int(sample_size): str(scenario)
        for sample_size, scenario in lock["sample_size_to_scenario"].items()
    }
    repaired, changed = repair_report(report, sample_size_to_scenario=mapping)
    if changed != int(lock["required_repaired_rows"]):
        raise AssertionError("schema repair changed an unexpected number of rows")
    repaired["schema_repair"] = {
        "kind": "scenario_key_only",
        "original_report": str(args.report),
        "original_report_sha256": lock["original_report_sha256"],
        "amendment_sha256": lock_hash,
        "repaired_rows": changed,
        "repaired_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json_dump(repaired, args.output)
    print(json.dumps(repaired["schema_repair"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
