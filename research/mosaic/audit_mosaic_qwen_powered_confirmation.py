#!/usr/bin/env python3
"""Replay and audit the locked powered Qwen temporal confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = (
    ROOT / "research/data/civilcomments_qwen25_powered_confirmation"
)
DEFAULT_PREREG = (
    ROOT / "research/mosaic/prereg_mosaic_qwen_powered_confirmation_v1.json"
)
DEFAULT_RESULTS = (
    ROOT / "research/artifacts/mosaic_qwen_powered_confirmation_v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "research/artifacts/"
    "mosaic_qwen_powered_confirmation_audit_v1.json"
)
RUNNER = ROOT / "research/mosaic/run_mosaic_qwen_powered_confirmation.py"


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


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("audit encountered a nonfinite value")
        return round(value, 12)
    return value


def compare_receipts(expected: Path, replayed: Path) -> None:
    left = normalize(load(expected))
    right = normalize(load(replayed))
    if left != right:
        raise RuntimeError(f"Qwen replay differs for {expected.name}")


def validate_summary(
    prereg: dict[str, Any],
    result_dir: Path,
) -> dict[str, int]:
    seeds = [int(value) for value in prereg["seeds"]]
    receipts = [load(result_dir / f"seed-{seed}.json") for seed in seeds]
    complete = [row for row in receipts if row["status"] == "complete"]
    primary = [row for row in complete if row.get("primary_release")]
    heldout = sum(
        bool(row["heldout_primary_violation"]) for row in primary
    )
    operational_trials = sum(
        len(row["operational_replays"]) for row in primary
    )
    operational_violations = sum(
        int(row["operational_violation_count"]) for row in primary
    )
    threshold_counts = {
        f"{float(threshold):.2f}": int(
            sum(
                bool(
                    row["threshold_decisions"][
                        f"{float(threshold):.2f}"
                    ]
                )
                for row in complete
            )
        )
        for threshold in prereg["utility_thresholds"]
    }
    summary = load(result_dir / "summary.json")
    expected = {
        "registered_jobs": len(seeds),
        "completed_jobs": len(complete),
        "error_jobs": len(seeds) - len(complete),
        "primary_releases": len(primary),
        "primary_abstentions": len(seeds) - len(primary),
        "heldout_primary_violations": heldout,
        "operational_primary_trials": operational_trials,
        "operational_primary_violations": operational_violations,
    }
    for key, value in expected.items():
        if summary[key] != value:
            raise RuntimeError(f"Qwen summary field differs: {key}")
    minimum = int(
        prereg["main_paper_inclusion_gate"]["minimum_primary_releases"]
    )
    gate = bool(
        len(complete) == len(seeds)
        and len(primary) >= minimum
        and heldout == 0
        and operational_violations == 0
    )
    if bool(summary["main_paper_inclusion_gate_pass"]) != gate:
        raise RuntimeError("Qwen inclusion gate was not recomputed correctly")
    expected_receipts = {
        f"seed-{seed}.json": sha256(result_dir / f"seed-{seed}.json")
        for seed in seeds
    }
    if summary["receipt_sha256"] != expected_receipts:
        raise RuntimeError("Qwen receipt hash manifest differs")
    return {
        "registered_seeds": len(seeds),
        "completed_seeds": len(complete),
        "primary_releases": len(primary),
        "heldout_violations": heldout,
        "operational_trials": operational_trials,
        "operational_violations": operational_violations,
        "utility_threshold_release_counts": threshold_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prereg = load(args.prereg)
    lock_digest = sha256(args.prereg)
    sidecar = args.prereg.with_suffix(args.prereg.suffix + ".sha256")
    failures: list[str] = []
    checks: dict[str, Any] = {}
    try:
        if sidecar.read_text(encoding="utf-8").split()[0] != lock_digest:
            raise RuntimeError("Qwen lock sidecar mismatch")
        manifest = load(args.store / "manifest.json")
        if manifest["preregistration_sha256"] != lock_digest:
            raise RuntimeError("Qwen store belongs to another lock")
        if manifest["source_csv_sha256"] != prereg["source_csv"]["sha256"]:
            raise RuntimeError("Qwen source CSV hash differs")
        if int(manifest["n_examples"]) != 56_000:
            raise RuntimeError("Qwen store row count differs")
        if int(manifest["dimension"]) != 1_536:
            raise RuntimeError("Qwen hidden dimension differs")
        arrays = {
            key: args.store / value
            for key, value in {
                **manifest["arrays"],
                **manifest["auxiliary_arrays"],
            }.items()
        }
        array_hashes = {key: sha256(path) for key, path in arrays.items()}
        ids = np.load(arrays["ids"], mmap_mode="r")
        if np.any(np.asarray(ids) % 4 == 0):
            raise RuntimeError("Qwen confirmation store contains pilot IDs")
        checks["store"] = {
            "rows": int(manifest["n_examples"]),
            "dimension": int(manifest["dimension"]),
            "array_sha256": array_hashes,
            "pilot_overlap": 0,
        }
        checks["summary"] = validate_summary(prereg, args.results)

        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [
                str(ROOT / "research/mosaic"),
                str(ROOT / "research/scripts"),
            ]
        )
        with tempfile.TemporaryDirectory(
            prefix="mosaic-qwen-audit-"
        ) as temporary:
            replay = Path(temporary) / "replay"
            subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--store",
                    str(args.store),
                    "--prereg",
                    str(args.prereg),
                    "--output",
                    str(replay),
                ],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            names = [
                *(f"seed-{int(seed)}.json" for seed in prereg["seeds"]),
                "summary.json",
            ]
            for name in names:
                compare_receipts(args.results / name, replay / name)
            checks["deterministic_replay_files"] = len(names)
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")

    report = {
        "name": "MOSAIC Qwen2.5 powered confirmation audit v1",
        "pass": not failures,
        "lock_sha256": lock_digest,
        "checks": checks,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
