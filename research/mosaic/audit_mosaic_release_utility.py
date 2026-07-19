#!/usr/bin/env python3
"""Independently recompute every locked MOSAIC released-interface utility row."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mosaic_real import sha256
from mosaic_release_utility_common import evaluate_job, selected_jobs
from run_mosaic_official_frontier_exact_confirmation import atomic_json_dump
from run_mosaic_release_utility import validate_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock, lock_hash = validate_lock(args.lock)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    jobs = selected_jobs(args.strict_dir, lock["slices"])
    stored = {
        (str(row["dataset"]), int(row["seed"]), str(row["utility_threshold"])): row
        for row in report["results"]
    }
    if len(stored) != len(jobs):
        raise ValueError("utility report row count differs from lock")
    recomputed: list[dict[str, object] | None] = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_job, job): index for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            index = futures[future]
            recomputed[index] = future.result()
            print(f"replayed {recomputed[index]['dataset']} seed {recomputed[index]['seed']}", flush=True)
    failures: list[str] = []
    for row in recomputed:
        assert row is not None
        key = (str(row["dataset"]), int(row["seed"]), str(row["utility_threshold"]))
        if stored.get(key) != row:
            failures.append(f"utility replay mismatch: {key}")
    payload = {
        "name": "MOSAIC released-interface utility independent replay",
        "utility_lock_sha256": lock_hash,
        "utility_report_sha256": sha256(args.report),
        "release_count": len(jobs),
        "failure_count": len(failures),
        "passed": not failures,
        "failures": failures,
    }
    atomic_json_dump(payload, args.output)
    if failures:
        raise AssertionError("utility replay found mismatches")


if __name__ == "__main__":
    main()
