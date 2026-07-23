#!/usr/bin/env python3
"""Freeze the matched randomized-response comparison before computing outcomes."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from run_mosaic_local_dp_baseline import (
    ACS,
    BIASBIOS,
    LOCK,
    ROOT,
    expected_protocol,
    load,
    sha256,
)


CODE = (
    "research/mosaic/run_mosaic_local_dp_baseline.py",
    "research/mosaic/mosaic_channel.py",
    "research/mosaic/mosaic_transform_exact.py",
)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def inputs() -> list[Path]:
    paths = sorted(BIASBIOS.glob("BiasBios-Clinical__seed*.json"))
    for strict_path in list(paths):
        original = ROOT / load(strict_path)["original_receipt"]
        paths.append(original)
    paths.extend(sorted(ACS.glob("ACS-*-CA-*__seed*.json")))
    return sorted(set(paths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=LOCK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite an existing local-DP lock")
    paths = inputs()
    payload = {
        "name": "MOSAIC matched local-DP baseline preregistration v1",
        "status": "locked_before_local_dp_outcomes",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_lock": git_head(),
        "protocol": expected_protocol(),
        "code_sha256": {relative: sha256(ROOT / relative) for relative in CODE},
        "input_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in paths
        },
        "stopping_rule": (
            "Report every frozen primary MOSAIC deployment, every one of the "
            "16 task maps, both domain summaries, all ties, and either method's "
            "utility advantage."
        ),
        "claim_boundary": (
            "Randomized response is epsilon-local-DP with respect to the "
            "four-token input and therefore controls every downstream source "
            "attacker without a shift model. MOSAIC enforces the narrower "
            "source-inference contract over its learned bridge class. The "
            "comparison matches output size and attacker-advantage threshold, "
            "but the guarantees are intentionally not identical."
        ),
        "timing": (
            "This baseline was specified after the MOSAIC study outcomes were "
            "known and before any randomized-response comparison was computed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(args.output)
    sidecar.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
