#!/usr/bin/env python3
"""Lock the paired MOSAIC baseline extension before running its seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_OUTPUT = ROOT / "prereg_mosaic_baseline_extension_v1.json"
CODE_FILES = (
    "research/mosaic/run_mosaic_baseline_extension.py",
    "research/mosaic/lock_mosaic_baseline_extension.py",
    "research/mosaic/run_mosaic_synthetic_pilot.py",
    "research/mosaic/run_mosaic_synthetic_confirmation.py",
    "research/mosaic/mosaic_channel.py",
    "research/mosaic/mosaic_envelope.py",
    "research/mosaic/mosaic_exact.py",
    "research/mosaic/mosaic_invariant.py",
    "research/mosaic/mosaic_optimizer.py",
    "research/tests/test_mosaic_baseline_extension.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite an existing baseline lock")
    original_prereg = ROOT / "prereg_mosaic_synthetic_v1.json"
    payload = {
        "project": "MOSAIC paired structure-aware baseline extension",
        "status": "locked_before_outcomes",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_lock": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "original_synthetic_preregistration_sha256": sha256(original_prereg),
        "familywise_delta": 0.05,
        "replicates_per_cell": 1000,
        "seed_base": 910000000,
        "seed_exclusion": (
            "All prior synthetic pilot, confirmation, transform-exact, and "
            "theory-alignment seeds are excluded."
        ),
        "scenarios": [
            {
                "name": "hard_safety_boundary",
                "sample_size_per_stratum": 125,
                "contamination": 0.20,
                "source_advantage_threshold": 0.30,
                "utility_threshold": 0.40,
            },
            {
                "name": "retention_and_stochastic_value",
                "sample_size_per_stratum": 250,
                "contamination": 0.10,
                "source_advantage_threshold": 0.35,
                "utility_threshold": 0.45,
            },
        ],
        "candidate_family": {
            "channels": "fixed 5^3 binary-output stochastic grid",
            "decoders": "all 2^2 binary decoders",
            "candidate_pairs": 500,
        },
        "methods": {
            "mosaic_continuum": (
                "The globally optimized stochastic continuum under one "
                "pre-channel multinomial confidence event."
            ),
            "table_region_grid": (
                "The same simultaneous confidence event and structured-shift "
                "certificate, restricted to the fixed 500 channel-decoder pairs."
            ),
            "holm_ltt_grid": (
                "Candidate-level composite Hoeffding p-values for the unsafe "
                "union null, followed by Holm step-down control across all 500 pairs."
            ),
            "fare_style_deterministic": (
                "A matched restricted finite-encoder adaptation using all "
                "deterministic 3-to-2 maps and the same shift-aware certificate."
            ),
        },
        "fare_scope": (
            "The deterministic row tests FARE's restricted finite-encoder "
            "principle in this token/shift setting. It is not the official FARE "
            "fair-tree training pipeline, whose raw-input demographic-parity "
            "setting is not commensurate with this conditional shift witness."
        ),
        "pass_conditions": {
            "complete_rows": 8000,
            "maximum_false_acceptance_rate": 0.05,
            "confidence_event_failures": 0,
            "comparison_gate": (
                "None. Report every paired rate and interval regardless of ordering."
            ),
        },
        "claim_boundary": (
            "This extension tests whether MOSAIC's reported retention advantage "
            "survives a matched confidence-region grid and a structure-aware Holm "
            "LTT baseline on untouched paired synthetic tables. It does not turn "
            "the FARE-style adaptation into an official FARE end-to-end experiment."
        ),
        "stopping_rule": (
            "Run all 2,000 tables and all four methods. No seed replacement, "
            "candidate change, threshold change, or selective omission."
        ),
        "code_sha256": {
            relative: sha256(REPOSITORY / relative) for relative in CODE_FILES
        },
    }
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(data, encoding="utf-8")
    digest = hashlib.sha256(data.encode("utf-8")).hexdigest()
    sidecar.write_text(digest + "\n", encoding="utf-8")
    print(digest)


if __name__ == "__main__":
    main()
