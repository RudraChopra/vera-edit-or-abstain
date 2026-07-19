#!/usr/bin/env python3
"""Deterministically audit every corrected v2 MOSAIC bridge receipt."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import audit_mosaic_bridge_strict as audit_v1
from mosaic_real import sha256
from replay_mosaic_bridge_strict_v2 import replay_one
from run_mosaic_official_frontier_exact_confirmation import atomic_json_dump


def audit_pair(
    original_path: Path,
    strict_path: Path,
    *,
    prereg_hash: str,
    amendment_hash: str,
) -> tuple[list[str], dict[str, object]]:
    previous = audit_v1.replay_one
    audit_v1.replay_one = replay_one
    try:
        return audit_v1.audit_pair(
            original_path,
            strict_path,
            prereg_hash=prereg_hash,
            amendment_hash=amendment_hash,
        )
    finally:
        audit_v1.replay_one = previous


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-dir", required=True, type=Path)
    parser.add_argument("--strict-dir", required=True, type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    prereg_hash = sha256(args.prereg)
    prereg_sidecar = args.prereg.with_suffix(args.prereg.suffix + ".sha256")
    if prereg_sidecar.read_text(encoding="utf-8").strip() != prereg_hash:
        raise ValueError("preregistration sidecar mismatch")
    amendment_hash = sha256(args.amendment)
    amendment_sidecar = args.amendment.with_suffix(args.amendment.suffix + ".sha256")
    if amendment_sidecar.read_text(encoding="utf-8").strip() != amendment_hash:
        raise ValueError("strict amendment sidecar mismatch")
    originals = sorted(args.original_dir.glob("*.json"))
    strict = {path.name: path for path in args.strict_dir.glob("*.json")}
    failures: list[str] = []
    if set(strict) != {path.name for path in originals}:
        failures.append("original and strict receipt filename sets differ")
    summaries: list[dict[str, object] | None] = [None] * len(originals)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for index, original_path in enumerate(originals):
            strict_path = strict.get(original_path.name)
            if strict_path is None:
                continue
            future = executor.submit(
                audit_pair,
                original_path,
                strict_path,
                prereg_hash=prereg_hash,
                amendment_hash=amendment_hash,
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            pair_failures, summary = future.result()
            failures.extend(
                f"{originals[index].name}: {failure}" for failure in pair_failures
            )
            summaries[index] = summary
            print(originals[index].name, flush=True)
    files = [summary for summary in summaries if summary is not None]
    report: dict[str, object] = {
        "name": "MOSAIC strict numerical bridge v2 independent replay",
        "strict_repair_version": "v2_structural_zero_columns",
        "passed": not failures,
        "preregistration_sha256": prereg_hash,
        "strict_amendment_sha256": amendment_hash,
        "files_replayed": len(files),
        "candidate_rows_replayed": sum(
            int(file["candidate_rows"]) for file in files
        ),
        "global_optimization_replays": sum(
            int(file["global_optimizations"]) for file in files
        ),
        "minimum_membership_slack": min(
            (
                float(file["minimum_membership_slack"])
                for file in files
                if file["minimum_membership_slack"] is not None
            ),
            default=None,
        ),
        "failures": failures,
        "files": files,
    }
    atomic_json_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
