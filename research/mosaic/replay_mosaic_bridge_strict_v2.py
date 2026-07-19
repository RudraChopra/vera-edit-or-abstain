#!/usr/bin/env python3
"""Replay all MOSAIC bridge receipts with the locked v2 numerical correction."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import replay_mosaic_bridge_strict as replay_v1
from mosaic_real import sha256
from mosaic_strict_certification_v2 import certify_bridge_membership_strict
from run_mosaic_official_frontier_exact_confirmation import atomic_json_dump


def replay_one(
    original_path: Path,
    output_path: Path,
    *,
    prereg_hash: str,
    amendment_hash: str,
) -> dict[str, object]:
    """Run the v1 pipeline with only its bridge repair replaced by v2."""

    previous = replay_v1.certify_bridge_membership_strict
    replay_v1.certify_bridge_membership_strict = certify_bridge_membership_strict
    try:
        summary = replay_v1.replay_one(
            original_path,
            output_path,
            prereg_hash=prereg_hash,
            amendment_hash=amendment_hash,
        )
    finally:
        replay_v1.certify_bridge_membership_strict = previous

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["project"] = "MOSAIC strict numerical bridge replay v2"
    payload["strict_repair_version"] = "v2_structural_zero_columns"
    payload["numerical_policy"]["structural_zero_output_rule"] = (
        "skip guard-ratio contraction only when the complete transform output "
        "column is exactly zero; recheck every original inequality"
    )
    atomic_json_dump(payload, output_path)
    summary["sha256"] = sha256(output_path)
    summary["strict_repair_version"] = payload["strict_repair_version"]
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--prereg", required=True, type=Path)
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, object] | None] = [None] * len(args.inputs)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for index, path in enumerate(args.inputs):
            output = args.output_dir / path.name
            if output.exists():
                raise FileExistsError(f"refusing to overwrite {output}")
            future = executor.submit(
                replay_one,
                path,
                output,
                prereg_hash=prereg_hash,
                amendment_hash=amendment_hash,
            )
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            summaries[index] = future.result()
            print(summaries[index]["output"], flush=True)
    files = [summary for summary in summaries if summary is not None]
    manifest = {
        "name": "MOSAIC strict numerical bridge replay v2 manifest",
        "strict_repair_version": "v2_structural_zero_columns",
        "preregistration_sha256": prereg_hash,
        "strict_amendment_sha256": amendment_hash,
        "files": files,
        "file_count": len(files),
        "candidate_rows": sum(int(file["candidate_rows"]) for file in files),
        "global_optimizations": sum(
            int(file["global_optimizations"]) for file in files
        ),
        "minimum_membership_slack": min(
            float(file["minimum_membership_slack"])
            for file in files
            if file["minimum_membership_slack"] is not None
        ),
    }
    atomic_json_dump(manifest, args.manifest)
    print(args.manifest)


if __name__ == "__main__":
    main()
