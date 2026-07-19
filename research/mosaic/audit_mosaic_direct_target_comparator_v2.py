#!/usr/bin/env python3
"""Mechanically repaired independent replay for the direct-target comparator.

The v1 audit module imported ``audit_release`` from the generator rather than
the existing comparator audit module. This version changes only that import and
is released under a separate audit-repair lock.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from audit_mosaic_bridge_comparator_extension import audit_release, mappings_close
from audit_mosaic_bridge_frontier import table_from_counts
from mosaic_real import sha256
from run_mosaic_bridge_comparator_extension import select_candidate, threshold_key
from run_mosaic_direct_target_comparator import RULE, validate_lock
from run_mosaic_official_frontier_exact_confirmation import atomic_json_dump


def audit_one(
    original_path: Path,
    direct_path: Path,
    *,
    original_prereg_hash: str,
    lock_hash: str,
) -> dict[str, object]:
    failures: list[str] = []
    original = json.loads(original_path.read_text(encoding="utf-8"))
    direct = json.loads(direct_path.read_text(encoding="utf-8"))
    if direct.get("original_receipt_sha256") != sha256(original_path):
        failures.append("original receipt hash mismatch")
    if direct.get("original_preregistration_sha256") != original_prereg_hash:
        failures.append("original preregistration hash mismatch")
    if direct.get("direct_target_lock_sha256") != lock_hash:
        failures.append("direct-target lock hash mismatch")
    if direct.get("rule") != RULE:
        failures.append("registered rule mismatch")
    if (direct.get("dataset"), direct.get("seed")) != (
        original.get("dataset"),
        original.get("seed"),
    ):
        failures.append("dataset or seed mismatch")

    protocol = original["protocol"]
    token_count = int(protocol["fine_token_count"])
    table_delta = float(protocol["per_candidate_table_delta"])
    source_threshold = float(protocol["privacy_advantage_threshold"])
    utility_thresholds = [float(value) for value in protocol["utility_thresholds"]]
    original_by_candidate = {row["candidate"]: row for row in original["results"]}
    rows = direct["results"]
    if len(rows) != len(original_by_candidate):
        failures.append("candidate row count mismatch")
    identity = (np.eye(token_count, dtype=np.float64),)
    for row in rows:
        candidate = str(row.get("candidate"))
        original_row = original_by_candidate.get(candidate)
        if original_row is None:
            failures.append(f"{candidate}: absent from original receipt")
            continue
        if "upstream_error" in row:
            if row["upstream_error"] != original_row.get("optimization_error"):
                failures.append(f"{candidate}: upstream error mismatch")
            continue
        bridge, radii, totals = table_from_counts(
            original_row["bridge_token_counts"],
            token_count=token_count,
            familywise_delta=table_delta,
        )
        if row.get("bridge_stratum_counts") != totals.tolist():
            failures.append(f"{candidate}: bridge totals mismatch")
        if not np.allclose(np.asarray(row.get("bridge_l1_radii")), radii, atol=3e-7):
            failures.append(f"{candidate}: bridge radii mismatch")
        if np.any(totals == 0):
            if row.get("direct_target_error") != "missing target source-label stratum":
                failures.append(f"{candidate}: missing-support decision mismatch")
            continue
        release = row.get(RULE)
        if not isinstance(release, dict):
            failures.append(f"{candidate}: missing direct-target release")
            continue
        current = audit_release(
            release,
            reference=bridge,
            radii=radii,
            diagnostic_counts=original_row["diagnostic_token_counts"],
            libraries=(identity, identity),
            contaminations=(0.0, 0.0),
            source_threshold=source_threshold,
            utility_thresholds=utility_thresholds,
            family="exact",
        )
        failures.extend(f"{candidate}: {failure}" for failure in current)

    for threshold in utility_thresholds:
        name = threshold_key(threshold)
        expected = select_candidate(
            rows,
            rule=RULE,
            release_key=RULE,
            source_threshold=source_threshold,
            utility_threshold=threshold,
        )
        if not mappings_close(direct["selection_by_utility_threshold"][name], expected):
            failures.append(f"selection mismatch at {name}")
    return {
        "dataset": direct.get("dataset"),
        "seed": direct.get("seed"),
        "original": str(original_path),
        "direct_target": str(direct_path),
        "candidate_rows": len(rows),
        "failure_count": len(failures),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--direct-target-dir", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--repair-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    lock, lock_hash = validate_lock(args.lock)
    repair = json.loads(args.repair_lock.read_text(encoding="utf-8"))
    if repair.get("status") != "locked_audit_import_repair_before_replay":
        raise ValueError("audit repair is not locked")
    originals = sorted(args.raw_dir.glob("*.json"))
    direct_paths = {path.name: path for path in args.direct_target_dir.glob("*.json")}
    if len(originals) != int(lock["required_raw_receipt_count"]):
        raise ValueError("raw receipt count differs from the lock")
    if set(path.name for path in originals) != set(direct_paths):
        raise ValueError("raw and direct-target receipt filenames differ")
    rows: list[dict[str, object] | None] = [None] * len(originals)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                audit_one,
                original,
                direct_paths[original.name],
                original_prereg_hash=str(lock["original_preregistration_sha256"]),
                lock_hash=lock_hash,
            ): index
            for index, original in enumerate(originals)
        }
        for future in as_completed(futures):
            index = futures[future]
            rows[index] = future.result()
            print(f"audited {rows[index]['direct_target']}", flush=True)
    completed = [row for row in rows if row is not None]
    payload = {
        "name": "MOSAIC direct target-table independent certificate audit v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "direct_target_lock_sha256": lock_hash,
        "audit_repair_lock_sha256": sha256(args.repair_lock),
        "repair_disclosure": repair["repair"],
        "file_count": len(completed),
        "candidate_rows": sum(int(row["candidate_rows"]) for row in completed),
        "failure_count": sum(int(row["failure_count"]) for row in completed),
        "passed": all(int(row["failure_count"]) == 0 for row in completed),
        "files": completed,
    }
    atomic_json_dump(payload, args.output)
    if not payload["passed"]:
        raise AssertionError("direct-target audit found failures")


if __name__ == "__main__":
    main()
