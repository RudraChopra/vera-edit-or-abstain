#!/usr/bin/env python3
"""Lock the ACS 2022 confirmation before any confirmation asset is accessed."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_mosaic_acs_temporal_replication import (
    DISCOVERY,
    LOCK,
    OUTPUT,
    PANDEMIC_LOCK,
    STATE_FIPS,
    WITNESSES,
    expected_protocol,
    receipt_path,
)


ROOT = Path(__file__).resolve().parents[2]
CODE = (
    "research/mosaic/mosaic_channel.py",
    "research/mosaic/mosaic_real.py",
    "research/mosaic/run_mosaic_acs_pandemic_panel.py",
    "research/mosaic/run_mosaic_acs_temporal_replication.py",
    "research/mosaic/run_mosaic_official_frontier_exact_confirmation.py",
    "research/scripts/official_eraser_adapters.py",
    "research/scripts/prepare_acs_natural_shift_stores.py",
    "research/scripts/run_official_eraser_frontier.py",
)


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


def discovery_rows() -> list[dict[str, Any]]:
    report = load(DISCOVERY)
    rows = []
    for witness in WITNESSES:
        direct = next(
            row
            for row in report["rows"]
            if row["target_state"] == witness["target_state"]
            and row["task"] == witness["task"]
            and row["seed"] == witness["seed"]
            and row["rule"] == "direct"
        )
        if not direct["future_diagnostic"]["utility_contract_violation_empirical"]:
            raise ValueError(f"{witness} is not a discovery-fold utility failure")
        rows.append(
            {
                **witness,
                "discovery_worst_conditional_error_empirical": direct[
                    "future_diagnostic"
                ]["worst_conditional_error_empirical"],
                "discovery_diagnostic_rows": direct["future_diagnostic_rows"],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--future-raw-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if LOCK.exists() or LOCK.with_suffix(LOCK.suffix + ".sha256").exists():
        raise FileExistsError("temporal-replication lock already exists")
    if OUTPUT.exists():
        raise FileExistsError("temporal-replication outcome already exists")
    present = [
        str(
            args.future_raw_root
            / "2022"
            / "1-Year"
            / f"psam_p{STATE_FIPS[state]}.csv"
        )
        for state in sorted(STATE_FIPS)
        if (
            args.future_raw_root
            / "2022"
            / "1-Year"
            / f"psam_p{STATE_FIPS[state]}.csv"
        ).exists()
    ]
    if present:
        raise ValueError(f"2022 confirmation assets are already present: {present}")
    if not DISCOVERY.exists():
        raise FileNotFoundError(DISCOVERY)
    for path in CODE:
        if not (ROOT / path).exists():
            raise FileNotFoundError(ROOT / path)
    inputs = {
        str(DISCOVERY.relative_to(ROOT)): sha256(DISCOVERY),
        str(PANDEMIC_LOCK.relative_to(ROOT)): sha256(PANDEMIC_LOCK),
        "research/mosaic/prereg_mosaic_acs_natural_shift_data_v1.json": sha256(
            ROOT / "research/mosaic/prereg_mosaic_acs_natural_shift_data_v1.json"
        ),
    }
    for witness in WITNESSES:
        path = receipt_path(witness)
        inputs[str(path.relative_to(ROOT))] = sha256(path)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "name": "MOSAIC ACS temporal failure replication preregistration v1",
        "status": "locked_before_2022_download",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_lock": head,
        "claim_boundary": (
            "The 2021 panel is a registered discovery census with empirical but "
            "not familywise-confirmed failures. This second lock tests the exact "
            "three discovered frozen direct interfaces on untouched 2022 ACS "
            "data. It is a prospective replication across years, not a "
            "prospectively selected 2021 hypothesis."
        ),
        "protocol": expected_protocol(),
        "discovery_rows": discovery_rows(),
        "code_sha256": {
            path: sha256(ROOT / path) for path in CODE
        },
        "input_sha256": inputs,
        "reference_raw_asset": load(PANDEMIC_LOCK)["reference_raw_asset"],
        "raw_2022_assets_absent_at_lock": True,
        "stopping_rule": (
            "Report all three frozen interfaces, every reconstruction failure, "
            "all simultaneous lower and upper bounds, and every confirmed or "
            "unconfirmed replication. No replacement witnesses are permitted."
        ),
    }
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(LOCK)
    LOCK.with_suffix(LOCK.suffix + ".sha256").write_text(
        f"{digest}  {LOCK.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"lock": str(LOCK), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
