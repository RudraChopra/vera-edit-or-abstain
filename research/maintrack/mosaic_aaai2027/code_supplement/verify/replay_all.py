#!/usr/bin/env python3
"""Run every paper-result replay and the exact-rational certificate audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "run_synthetic_safety.py",
    "run_matched_baselines.py",
    "run_real_100jobs.py",
    "run_admitted_shift_stress.py",
    "run_utility_table.py",
    "run_scaling.py",
    "run_acs_ca_tx.py",
    "run_bridge_power.py",
)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--skip-exact-rational", action="store_true")
    args = parser.parse_args()
    for script in SCRIPTS:
        run([sys.executable, str(ROOT / "scripts" / script)])
    if not args.skip_exact_rational:
        run([sys.executable, str(ROOT / "verify" / "audit_exact_rational.py"), "--workers", str(args.workers)])
    summary = {
        "status": "pass",
        "entry_points_replayed": len(SCRIPTS),
        "exact_rational_replayed": not args.skip_exact_rational,
    }
    output = ROOT / "artifacts" / "reproduced" / "replay_all.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
