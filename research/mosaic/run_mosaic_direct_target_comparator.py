#!/usr/bin/env python3
"""Certify target-only releases from the stored MOSAIC bridge tables.

This intentionally post-outcome analysis answers whether the bridge transfers
statistical strength beyond a direct certificate on the labeled target table.
It never changes the original MOSAIC decision or the registered bridge result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from audit_mosaic_bridge_frontier import table_from_counts
from mosaic_real import sha256
from mosaic_transform_exact_optimizer import optimize_transform_exact_channel
from run_mosaic_bridge_comparator_extension import (
    diagnostic_decision,
    select_candidate,
    serialize_solution,
    threshold_key,
)
from run_mosaic_official_frontier_exact_confirmation import atomic_json_dump


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
ORIGINAL_PREREG = ROOT / "prereg_mosaic_bridge_v1.json"
RULE = "direct_target"


def validate_lock(path: Path) -> tuple[dict[str, object], str]:
    lock_hash = sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists() or sidecar.read_text(encoding="utf-8").strip() != lock_hash:
        raise ValueError("direct-target lock sidecar mismatch")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked_post_outcome_comparator_before_execution":
        raise ValueError("direct-target comparator is not correctly locked")
    if sha256(ORIGINAL_PREREG) != lock.get("original_preregistration_sha256"):
        raise ValueError("original bridge preregistration hash mismatch")
    relative_lock = path.resolve().relative_to(REPOSITORY.resolve())
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative_lock.as_posix()}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout
    if committed != path.read_bytes():
        raise ValueError("direct-target lock is not committed")
    for relative, expected in lock.get("code_sha256", {}).items():
        if sha256(REPOSITORY / relative) != expected:
            raise ValueError(f"locked direct-target code mismatch: {relative}")
    return lock, lock_hash


def solve_candidate(
    original_row: dict[str, object], protocol: dict[str, object]
) -> dict[str, object]:
    metadata = {
        key: original_row[key]
        for key in ("candidate", "method", "strength", "provenance")
    }
    if "optimization_error" in original_row:
        return {**metadata, "upstream_error": original_row["optimization_error"]}

    token_count = int(protocol["fine_token_count"])
    table_delta = float(protocol["per_candidate_table_delta"])
    released_count = int(protocol["primary_released_token_count"])
    source_threshold = float(protocol["privacy_advantage_threshold"])
    utility_thresholds = [float(value) for value in protocol["utility_thresholds"]]
    bridge, bridge_radii, bridge_totals = table_from_counts(
        original_row["bridge_token_counts"],
        token_count=token_count,
        familywise_delta=table_delta,
    )
    row: dict[str, object] = {
        **metadata,
        "bridge_token_counts": original_row["bridge_token_counts"],
        "diagnostic_token_counts": original_row["diagnostic_token_counts"],
        "bridge_stratum_counts": bridge_totals.tolist(),
        "bridge_l1_radii": bridge_radii.tolist(),
    }
    if np.any(bridge_totals == 0):
        row["direct_target_error"] = "missing target source-label stratum"
        return row

    identity = (np.eye(token_count, dtype=np.float64),)
    direct = optimize_transform_exact_channel(
        bridge,
        l1_radii=bridge_radii,
        common_channels_by_label=(identity, identity),
        contaminations=(0.0, 0.0),
        privacy_advantage_thresholds=(source_threshold, source_threshold),
        released_token_count=released_count,
        solver_time_limit_seconds=300.0,
    )
    row[RULE] = serialize_solution(
        direct,
        risk_bound_kind="simultaneous_confidence_target_table_only",
        diagnostic_counts=original_row["diagnostic_token_counts"],
        source_threshold=source_threshold,
        utility_thresholds=utility_thresholds,
    )
    return row


def replay_one(
    original_path: Path,
    output_path: Path,
    *,
    original_prereg_hash: str,
    lock_hash: str,
) -> dict[str, object]:
    original = json.loads(original_path.read_text(encoding="utf-8"))
    if original.get("prereg_sha256") != original_prereg_hash:
        raise ValueError(f"{original_path} has the wrong bridge preregistration hash")
    protocol = original["protocol"]
    rows = [solve_candidate(row, protocol) for row in original["results"]]
    source_threshold = float(protocol["privacy_advantage_threshold"])
    selections = {
        threshold_key(float(threshold)): select_candidate(
            rows,
            rule=RULE,
            release_key=RULE,
            source_threshold=source_threshold,
            utility_threshold=float(threshold),
        )
        for threshold in protocol["utility_thresholds"]
    }
    payload: dict[str, object] = {
        "project": "MOSAIC direct target-table comparator",
        "dataset": original["dataset"],
        "seed": original["seed"],
        "protocol": protocol,
        "original_receipt": str(original_path),
        "original_receipt_sha256": sha256(original_path),
        "original_preregistration_sha256": original_prereg_hash,
        "direct_target_lock_sha256": lock_hash,
        "rule": RULE,
        "results": rows,
        "selection_by_utility_threshold": selections,
        "claim_boundary": (
            "The direct rule certifies only the distribution sampled by the labeled "
            "bridge table. Unlike MOSAIC, it does not certify a transform-plus-"
            "contamination class that can contain an external target law."
        ),
    }
    atomic_json_dump(payload, output_path)
    return {
        "dataset": payload["dataset"],
        "seed": payload["seed"],
        "output": str(output_path),
        "sha256": sha256(output_path),
        "candidate_rows": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    lock, lock_hash = validate_lock(args.lock)
    if len(args.inputs) != int(lock["required_raw_receipt_count"]):
        raise ValueError("input receipt count differs from the lock")
    if len({path.resolve() for path in args.inputs}) != len(args.inputs):
        raise ValueError("raw receipt inputs must be unique")
    if args.manifest.exists():
        raise FileExistsError(f"refusing to overwrite {args.manifest}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object] | None] = [None] * len(args.inputs)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for index, original in enumerate(args.inputs):
            output = args.output_dir / original.name
            if output.exists():
                raise FileExistsError(f"refusing to overwrite {output}")
            future = executor.submit(
                replay_one,
                original,
                output,
                original_prereg_hash=str(lock["original_preregistration_sha256"]),
                lock_hash=lock_hash,
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            summaries[index] = future.result()
            print(summaries[index]["output"], flush=True)
    files = [summary for summary in summaries if summary is not None]
    atomic_json_dump(
        {
            "name": "MOSAIC direct target-table comparator manifest",
            "direct_target_lock_sha256": lock_hash,
            "original_preregistration_sha256": lock["original_preregistration_sha256"],
            "rule": RULE,
            "files": files,
            "file_count": len(files),
            "candidate_rows": sum(int(file["candidate_rows"]) for file in files),
        },
        args.manifest,
    )


if __name__ == "__main__":
    main()
