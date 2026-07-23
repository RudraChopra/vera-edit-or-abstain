#!/usr/bin/env python3
"""Independently audit the matched local-DP comparison artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "research/artifacts/mosaic_local_dp_baseline_v1.json"
OUTPUT = ROOT / "research/artifacts/mosaic_local_dp_baseline_audit_v1.json"
SOURCE_THRESHOLD = 0.35
UTILITY_THRESHOLD = 0.40
KEEP_PROBABILITY = 0.675
EPSILON = math.log(KEEP_PROBABILITY / (1.0 - KEEP_PROBABILITY))


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


def source_mosaic_error(row: dict[str, Any]) -> float:
    first = ROOT / row["inputs"][0]
    payload = load(first)
    candidate = row["candidate"]
    if row["domain"] == "BiasBios-Clinical":
        selected = next(
            value for value in payload["results"]
            if value["candidate"] == candidate
        )
        return float(
            selected["release_l2"]["certified_worst_conditional_error_upper"]
        )
    alphabet = payload["alphabets"]["4"]
    selected = next(
        value for value in alphabet["rows"]
        if value["candidate"] == candidate
    )
    return float(
        selected["mosaic_release"]["certified_worst_conditional_error_upper"]
    )


def audit_row(row: dict[str, Any]) -> list[str]:
    failures = []
    local_dp = row["local_dp"]
    channel = np.asarray(local_dp["release_channel"], dtype=np.float64)
    task_map = tuple(int(value) for value in local_dp["task_map"])
    if channel.shape != (4, 2) or not np.allclose(channel.sum(axis=1), 1.0):
        failures.append("channel is not a 4-by-2 stochastic matrix")
    expected = np.empty((4, 2), dtype=np.float64)
    for token, label in enumerate(task_map):
        expected[token, label] = KEEP_PROBABILITY
        expected[token, 1 - label] = 1.0 - KEEP_PROBABILITY
    if not np.allclose(channel, expected):
        failures.append("channel differs from registered randomized response")
    for output in range(2):
        ratios = channel[:, output, None] / channel[None, :, output]
        if float(np.max(ratios)) > math.exp(EPSILON) + 1e-12:
            failures.append("channel violates epsilon-local-DP")
    if max(float(value) for value in local_dp["source_advantage_upper"]) > (
        SOURCE_THRESHOLD + 1e-10
    ):
        failures.append("certified source advantage exceeds the contract")
    expected_decision = (
        "deploy"
        if float(local_dp["worst_conditional_error_upper"]) <= UTILITY_THRESHOLD
        else "abstain"
    )
    if local_dp["decision"] != expected_decision:
        failures.append("local-DP decision differs from its certified error")
    source_error = source_mosaic_error(row)
    if not math.isclose(
        source_error,
        float(row["mosaic_certified_worst_conditional_error_upper"]),
        abs_tol=1e-12,
    ):
        failures.append("MOSAIC error differs from the frozen source receipt")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    report = load(args.report)
    lock = ROOT / "research/mosaic/prereg_mosaic_local_dp_baseline_v1.json"
    if report["lock_sha256"] != sha256(lock):
        raise ValueError("report lock hash differs")
    failures = []
    for index, row in enumerate(report["rows"]):
        for failure in audit_row(row):
            failures.append(f"row {index}: {failure}")
    rows = report["rows"]
    gaps = [
        float(row["local_dp"]["worst_conditional_error_upper"])
        - float(row["mosaic_certified_worst_conditional_error_upper"])
        for row in rows
    ]
    recomputed = {
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
        "local_dp_epsilon": EPSILON,
    }
    if recomputed != report["summary"]:
        failures.append("summary differs from independent recomputation")
    if len(rows) != 35:
        failures.append(f"expected 35 rows, found {len(rows)}")
    payload = {
        "name": "MOSAIC matched local-DP baseline audit v1",
        "passed": not failures,
        "report_sha256": sha256(args.report),
        "checks": {
            "rows": len(rows),
            "epsilon_local_dp_channels": len(rows),
            "source_receipt_cross_checks": len(rows),
            "summary_recomputed": True,
        },
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
