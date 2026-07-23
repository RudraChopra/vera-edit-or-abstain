#!/usr/bin/env python3
"""Compare MOSAIC with contract-matched binary randomized response."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from mosaic_transform_exact import (
    transform_exact_attacker_confidence_bound,
    transform_exact_utility_confidence_bound,
)


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "research/mosaic/prereg_mosaic_local_dp_baseline_v1.json"
OUTPUT = ROOT / "research/artifacts/mosaic_local_dp_baseline_v1.json"
BIASBIOS = ROOT / "research/artifacts/mosaic_bridge_strict_v2_receipts_v1"
ACS = ROOT / "research/artifacts/mosaic_acs_natural_shift_v1_receipts"
SOURCE_THRESHOLD = 0.35
UTILITY_THRESHOLD = 0.40
KEEP_PROBABILITY = (1.0 + SOURCE_THRESHOLD) / 2.0
LOCAL_DP_EPSILON = math.log(KEEP_PROBABILITY / (1.0 - KEEP_PROBABILITY))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def expected_protocol() -> dict[str, Any]:
    return {
        "domains": ["BiasBios-Clinical", "ACS-geographic"],
        "included_jobs": (
            "every frozen primary MOSAIC deployment in the 20-job BiasBios "
            "study and 60-job ACS study"
        ),
        "source_advantage_threshold": SOURCE_THRESHOLD,
        "utility_threshold": UTILITY_THRESHOLD,
        "local_dp_epsilon": LOCAL_DP_EPSILON,
        "randomized_response_keep_probability": KEEP_PROBABILITY,
        "fine_token_count": 4,
        "released_token_count": 2,
        "decoder_search": (
            "enumerate all 16 deterministic four-token-to-binary task maps, "
            "then apply the fixed randomized-response channel and select the "
            "map with minimum exact bridge-class worst-stratum error"
        ),
        "comparison": (
            "MOSAIC and randomized response use the same frozen reference "
            "multinomials, confidence radii, bridge transforms, contamination "
            "budgets, binary output alphabet, and task-error contract"
        ),
    }


def validate_lock(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(path):
        raise ValueError("local-DP lock sidecar mismatch")
    lock = load(path)
    if lock.get("status") != "locked_before_local_dp_outcomes":
        raise ValueError("local-DP lock has the wrong status")
    if lock.get("protocol") != expected_protocol():
        raise ValueError("local-DP protocol differs from its lock")
    for relative, expected in lock["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"locked code mismatch: {relative}")
    for relative, expected in lock["input_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"locked input mismatch: {relative}")
    for local in (path, sidecar):
        relative = local.resolve().relative_to(ROOT.resolve())
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if committed != local.read_bytes():
            raise ValueError(f"{relative} is not the committed lock")
    return lock


def randomized_response_channel(task_map: tuple[int, ...]) -> np.ndarray:
    channel = np.empty((len(task_map), 2), dtype=np.float64)
    for token, task_label in enumerate(task_map):
        channel[token, task_label] = KEEP_PROBABILITY
        channel[token, 1 - task_label] = 1.0 - KEEP_PROBABILITY
    return channel


def certify_fixed_channel(
    *,
    probabilities: np.ndarray,
    radii: np.ndarray,
    bridge: dict[str, Any],
    channel: np.ndarray,
) -> dict[str, Any]:
    decoder = (0, 1)
    privacy = []
    utility = []
    for label, certificate in enumerate(bridge["labels"]):
        transforms = (certificate["transform"],)
        contamination = float(certificate["contamination"])
        privacy.append(
            transform_exact_attacker_confidence_bound(
                probabilities[label],
                channel,
                l1_radii=radii[label],
                common_fine_token_channels=transforms,
                contamination=contamination,
            )
        )
        utility.append(
            [
                transform_exact_utility_confidence_bound(
                    probabilities[label, source],
                    channel,
                    decoder,
                    true_label=label,
                    l1_radius=float(radii[label, source]),
                    common_fine_token_channels=transforms,
                    contamination=contamination,
                )
                for source in range(probabilities.shape[1])
            ]
        )
    return {
        "source_advantage_upper": [
            float(value.normalized_advantage) for value in privacy
        ],
        "worst_conditional_error_upper": float(
            max(
                value.error_probability
                for label_values in utility
                for value in label_values
            )
        ),
    }


def optimize_randomized_response(
    counts: np.ndarray,
    radii: np.ndarray,
    bridge: dict[str, Any],
) -> dict[str, Any]:
    probabilities = counts / counts.sum(axis=2, keepdims=True)
    candidates = []
    for task_map in itertools.product((0, 1), repeat=counts.shape[2]):
        channel = randomized_response_channel(task_map)
        certificate = certify_fixed_channel(
            probabilities=probabilities,
            radii=radii,
            bridge=bridge,
            channel=channel,
        )
        candidates.append(
            {
                "task_map": list(task_map),
                "release_channel": channel.tolist(),
                **certificate,
            }
        )
    selected = min(
        candidates,
        key=lambda value: (
            float(value["worst_conditional_error_upper"]),
            tuple(value["task_map"]),
        ),
    )
    if max(selected["source_advantage_upper"]) > SOURCE_THRESHOLD + 1e-10:
        raise RuntimeError("contract-matched randomized response exceeds its bound")
    selected["decision"] = (
        "deploy"
        if selected["worst_conditional_error_upper"] <= UTILITY_THRESHOLD
        else "abstain"
    )
    return selected


def biasbios_jobs() -> list[dict[str, Any]]:
    jobs = []
    for strict_path in sorted(BIASBIOS.glob("BiasBios-Clinical__seed*.json")):
        strict = load(strict_path)
        selection = strict["primary_selection"]
        if selection["decision"] != "deploy":
            continue
        candidate = selection["candidate"]
        strict_row = next(
            row for row in strict["results"] if row["candidate"] == candidate
        )
        original_path = ROOT / strict["original_receipt"]
        original = load(original_path)
        original_row = next(
            row for row in original["results"] if row["candidate"] == candidate
        )
        jobs.append(
            {
                "domain": "BiasBios-Clinical",
                "job": f"BiasBios-Clinical__seed{strict['seed']}",
                "seed": int(strict["seed"]),
                "candidate": candidate,
                "reference_counts": original_row["reference_token_counts"],
                "reference_radii": original_row["reference_l1_radii"],
                "bridge": strict_row["bridge_membership"],
                "mosaic_error": float(
                    strict_row["release_l2"][
                        "certified_worst_conditional_error_upper"
                    ]
                ),
                "inputs": [
                    str(strict_path.relative_to(ROOT)),
                    str(original_path.relative_to(ROOT)),
                ],
            }
        )
    return jobs


def acs_jobs() -> list[dict[str, Any]]:
    jobs = []
    for path in sorted(ACS.glob("ACS-*-CA-*__seed*.json")):
        receipt = load(path)
        alphabet = receipt["alphabets"]["4"]
        selection = alphabet["primary_selection"]["mosaic"]
        if selection["decision"] != "deploy":
            continue
        row = next(
            value for value in alphabet["rows"]
            if value["candidate"] == selection["candidate"]
        )
        jobs.append(
            {
                "domain": "ACS-geographic",
                "job": path.stem,
                "seed": int(receipt["seed"]),
                "candidate": selection["candidate"],
                "reference_counts": row["reference_table"]["token_counts"],
                "reference_radii": row["reference_table"]["l1_radii"],
                "bridge": row["bridge_membership"],
                "mosaic_error": float(
                    row["mosaic_release"][
                        "certified_worst_conditional_error_upper"
                    ]
                ),
                "inputs": [str(path.relative_to(ROOT))],
            }
        )
    return jobs


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = [
        float(row["local_dp"]["worst_conditional_error_upper"])
        - float(row["mosaic_certified_worst_conditional_error_upper"])
        for row in rows
    ]
    return {
        "jobs": len(rows),
        "domains": sorted({str(row["domain"]) for row in rows}),
        "mosaic_deployments": sum(
            row["mosaic_certified_worst_conditional_error_upper"]
            <= UTILITY_THRESHOLD
            for row in rows
        ),
        "local_dp_deployments": sum(
            row["local_dp"]["decision"] == "deploy" for row in rows
        ),
        "mosaic_strictly_lower_error_jobs": sum(gap > 1e-10 for gap in gaps),
        "ties": sum(abs(gap) <= 1e-10 for gap in gaps),
        "local_dp_strictly_lower_error_jobs": sum(gap < -1e-10 for gap in gaps),
        "median_local_dp_minus_mosaic_error": median(gaps),
        "minimum_local_dp_minus_mosaic_error": min(gaps),
        "maximum_local_dp_minus_mosaic_error": max(gaps),
        "local_dp_epsilon": LOCAL_DP_EPSILON,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock = validate_lock(args.lock)
    jobs = biasbios_jobs() + acs_jobs()
    if len(jobs) != 35:
        raise RuntimeError(f"expected 35 frozen MOSAIC deployments, found {len(jobs)}")
    rows = []
    for job in jobs:
        counts = np.asarray(job.pop("reference_counts"), dtype=np.int64)
        radii = np.asarray(job.pop("reference_radii"), dtype=np.float64)
        bridge = job.pop("bridge")
        mosaic_error = float(job.pop("mosaic_error"))
        local_dp = optimize_randomized_response(
            counts,
            radii,
            bridge,
        )
        rows.append(
            {
                **job,
                "mosaic_certified_worst_conditional_error_upper": mosaic_error,
                "local_dp": local_dp,
            }
        )
    summary = summarize(rows)
    payload = {
        "name": "MOSAIC matched local-DP baseline v1",
        "status": "complete_locked_local_dp_baseline",
        "lock_sha256": sha256(args.lock),
        "protocol": expected_protocol(),
        "rows": rows,
        "summary": summary,
        "claim_boundary": lock["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
