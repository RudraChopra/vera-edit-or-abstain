#!/usr/bin/env python3
"""Freeze the ACS pandemic panel before any 2021 scientific outcome is read."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from run_mosaic_acs_pandemic_panel import (
    DATA_LOCK,
    LOCK,
    RECEIPTS,
    ROOT,
    STATE_FIPS,
    TARGET_STATES,
    expected_protocol,
    sha256,
)


CODE = (
    "research/mosaic/run_mosaic_acs_pandemic_panel.py",
    "research/mosaic/mosaic_channel.py",
    "research/mosaic/mosaic_real.py",
    "research/mosaic/run_mosaic_acs_natural_shift.py",
    "research/mosaic/run_mosaic_official_frontier_exact_confirmation.py",
    "research/scripts/prepare_acs_natural_shift_stores.py",
    "research/scripts/run_official_eraser_frontier.py",
    "research/scripts/official_eraser_adapters.py",
)


def input_set_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=LOCK)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("data/acs_pums/2018/1-Year/psam_p06.csv"),
    )
    parser.add_argument("--future-raw-root", type=Path, default=Path("data/acs_pums"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite an existing pandemic-panel lock")
    future_paths = [
        args.future_raw_root
        / "2021"
        / "1-Year"
        / f"psam_p{STATE_FIPS[state]}.csv"
        for state in TARGET_STATES
    ]
    present = [str(path) for path in future_paths if path.exists()]
    if present:
        raise RuntimeError(
            "2021 raw files must be absent when this lock is created: "
            + ", ".join(present)
        )
    receipts = sorted(RECEIPTS.glob("ACS-*-CA-*__seed*.json"))
    if len(receipts) != 60:
        raise RuntimeError(f"expected 60 frozen ACS receipts, found {len(receipts)}")
    if not args.reference_csv.is_file():
        raise FileNotFoundError(args.reference_csv)
    payload = {
        "name": "MOSAIC ACS pandemic-discontinuity panel preregistration v1",
        "status": "locked_before_2021_download",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_lock": git_head(),
        "protocol": expected_protocol(),
        "code_sha256": {relative: sha256(ROOT / relative) for relative in CODE},
        "input_sha256": {
            str(DATA_LOCK.relative_to(ROOT)): sha256(DATA_LOCK),
        },
        "receipt_set_sha256": input_set_sha256(receipts),
        "reference_raw_asset": {
            "year": "2018",
            "state": "CA",
            "bytes": args.reference_csv.stat().st_size,
            "sha256": sha256(args.reference_csv),
        },
        "raw_2021_assets_absent_at_lock": True,
        "stopping_rule": (
            "Report all 60 state-task-seed jobs, both frozen rules, every "
            "reconstruction failure, every membership decision, every diagnostic "
            "interval, and every natural-failure witness."
        ),
        "claim_boundary": (
            "This prospective panel reuses frozen 2018 interfaces and evaluates "
            "the first standard post-2020 ACS 1-year PUMS population. The Census "
            "Bureau did not publish a standard 2020 ACS 1-year PUMS product, so "
            "the study measures a 2018-to-2021 natural temporal discontinuity, "
            "not a literal standard-product 2020-to-2021 pair. Each 2021 state's "
            "whole PUMAs are split into disjoint membership and diagnostic folds."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(args.output)
    sidecar.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest,
                "raw_2021_assets_absent_at_lock": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
