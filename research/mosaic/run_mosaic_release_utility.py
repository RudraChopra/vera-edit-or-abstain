#!/usr/bin/env python3
"""Run the locked post-outcome MOSAIC released-interface utility analysis."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from mosaic_real import sha256
from mosaic_release_utility_common import evaluate_job, selected_jobs
from run_mosaic_official_frontier_exact_confirmation import atomic_json_dump


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]


def validate_lock(path: Path) -> tuple[dict[str, object], str]:
    digest = sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").strip() != digest:
        raise ValueError("utility lock sidecar mismatch")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked_post_outcome_utility_analysis_before_execution":
        raise ValueError("utility analysis is not correctly locked")
    relative = path.resolve().relative_to(REPOSITORY.resolve())
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative.as_posix()}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout
    if committed != path.read_bytes():
        raise ValueError("utility lock is not committed")
    for relative_name, expected in lock["code_sha256"].items():
        if sha256(REPOSITORY / relative_name) != expected:
            raise ValueError(f"locked utility code mismatch: {relative_name}")
    return lock, digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-dir", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock, lock_hash = validate_lock(args.lock)
    jobs = selected_jobs(args.strict_dir, lock["slices"])
    if len(jobs) != int(lock["expected_deployed_release_count"]):
        raise ValueError("selected release count differs from locked expectation")
    results: list[dict[str, object] | None] = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_job, job): index for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            print(f"evaluated {results[index]['dataset']} seed {results[index]['seed']}", flush=True)
    completed = [result for result in results if result is not None]
    payload = {
        "name": "MOSAIC released-interface task utility analysis",
        "utility_lock_sha256": lock_hash,
        "slices": lock["slices"],
        "release_count": len(completed),
        "results": completed,
        "claim_boundary": lock["claim_boundary"],
    }
    atomic_json_dump(payload, args.output)


if __name__ == "__main__":
    main()
