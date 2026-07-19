#!/usr/bin/env python3
"""Lock the MOSAIC bridge misspecification study before confirmation outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from run_mosaic_bridge_misspecification import (
    DEFAULT_OUTPUT,
    exact_population_retained_masses,
    scenario_registry,
)


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_LOCK = ROOT / "prereg_mosaic_bridge_misspecification_v1.json"
AUDIT_OUTPUT = (
    REPOSITORY / "research/artifacts/mosaic_bridge_misspecification_audit_v1.json"
)
CODE_FILES = (
    "research/mosaic/run_mosaic_bridge_misspecification.py",
    "research/mosaic/audit_mosaic_bridge_misspecification.py",
    "research/mosaic/lock_mosaic_bridge_misspecification.py",
    "research/mosaic/mosaic_bridge.py",
    "research/mosaic/mosaic_channel.py",
    "research/mosaic/mosaic_envelope.py",
    "research/mosaic/run_mosaic_synthetic_pilot.py",
    "research/tests/test_mosaic_bridge_misspecification.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_LOCK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite an existing lock")
    if DEFAULT_OUTPUT.exists() or AUDIT_OUTPUT.exists():
        raise FileExistsError("misspecification outcomes already exist")
    scenarios = []
    for scenario in scenario_registry():
        retained = exact_population_retained_masses(scenario.target)
        scenarios.append(
            {
                "name": scenario.name,
                "description": scenario.description,
                "target_laws": scenario.target.tolist(),
                "population_retained_masses": list(retained),
                "population_minimum_retained_mass": min(retained),
            }
        )
    payload = {
        "project": "MOSAIC bridge model-misspecification confirmation",
        "status": "locked_before_confirmation_outcomes",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_lock": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "familywise_delta": 0.05,
        "confidence_allocation": (
            "delta/2 to simultaneous reference strata and delta/2 to "
            "simultaneous target strata"
        ),
        "minimum_retained_mass": 0.80,
        "maximum_declared_contamination": 0.20,
        "numerical_margin": 1e-7,
        "replicates_per_cell": 1000,
        "sample_sizes_per_stratum": [500, 1000, 2000, 5000],
        "seed_base": 930000000,
        "seed_formula": (
            "seed_base + scenario_index*100000000 + "
            "sample_index*1000000 + replicate"
        ),
        "pilot_disclosure": (
            "Twenty exploratory tables per scenario-size cell with seeds "
            "100*n+replicate were used to choose the locked scenarios and "
            "sample sizes. Those seeds are excluded from confirmation."
        ),
        "reference_laws": [
            [[0.80, 0.20], [0.60, 0.40]],
            [[0.20, 0.80], [0.40, 0.60]],
        ],
        "scenarios": scenarios,
        "pass_conditions": {
            "complete_rows": 12000,
            "maximum_invalid_false_acceptance_rate": 0.05,
            "confidence_event_failures": 0,
            "comparison_gate": (
                "None. Report every acceptance curve and confidence interval "
                "regardless of ordering."
            ),
        },
        "claim_boundary": (
            "This synthetic stress study tests finite-sample bridge behavior "
            "under one correctly specified and two deliberately misspecified "
            "target laws. It does not prove that arbitrary real-world "
            "misspecification is detectable."
        ),
        "stopping_rule": (
            "Run all 12,000 locked tables with no seed replacement, scenario "
            "change, threshold change, or selective omission."
        ),
        "audit_scope": (
            "A separately executed, prelocked audit regenerates every table "
            "from its seed and recomputes all bridge decisions and aggregates."
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
