#!/usr/bin/env python3
"""Regenerate the full locked synthetic confirmation from source."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = Path(__file__).resolve().parent / "full_generator"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "reproduced" / "synthetic_confirmation_full.json")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    runner = GENERATOR / "research" / "mosaic" / "run_mosaic_synthetic_confirmation.py"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(GENERATOR / "research" / "mosaic")
    subprocess.run(
        [sys.executable, str(runner), "--workers", str(args.workers), "--output", str(args.output.resolve())],
        cwd=GENERATOR,
        env=environment,
        check=True,
    )
    report = json.loads(args.output.read_text(encoding="utf-8"))
    cells = {
        (row["scenario"], int(row["sample_size_per_stratum"]), row["method"]): row
        for row in report["cells"]
    }
    naive = cells[("hard_safety_boundary", 125, "plugin_continuum")]
    mosaic = cells[("hard_safety_boundary", 125, "mosaic")]
    if int(naive["false_acceptances"]) != 421 or int(mosaic["false_acceptances"]) != 0:
        raise SystemExit("regenerated primary safety cell does not match the locked result")
    print("full locked synthetic regeneration: pass")


if __name__ == "__main__":
    main()
