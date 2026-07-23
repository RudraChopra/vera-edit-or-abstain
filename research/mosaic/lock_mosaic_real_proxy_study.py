#!/usr/bin/env python3
"""Freeze the ACS real-proxy study before any study outcome is computed."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "research/mosaic/prereg_mosaic_real_proxy_v1.json"
RAW = Path(
    "/Users/rudrachopra/Documents/Science Fair/data/"
    "acs_pums/2018/1-Year/psam_p06.csv"
)
CODE = (
    "research/mosaic/run_mosaic_real_proxy_study.py",
    "research/mosaic/mosaic_proxy_bridge.py",
    "research/mosaic/mosaic_transform_exact_optimizer.py",
    "research/mosaic/mosaic_real.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    payload = {
        "name": "MOSAIC real ACS geography-ancestry proxy study v1",
        "status": "locked_before_outcomes",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data": {
            "path": str(RAW),
            "sha256": sha256(RAW),
            "bytes": RAW.stat().st_size,
            "source": "2018 ACS 1-Year California person PUMS",
        },
        "protocol": {
            "task": "Folktables ACSIncome",
            "source": "RAC1P white-alone versus all other codes",
            "proxy": (
                "held-out geography, birthplace, ancestry, Hispanic-origin, "
                "language, nativity, and citizenship imputer"
            ),
            "partition_fractions": {
                "task_train": 0.20,
                "proxy_train": 0.20,
                "calibration": 0.20,
                "target_proxy": 0.25,
                "diagnostic": 0.15,
            },
            "seed": 20270723,
            "fine_token_count": 4,
            "released_token_count": 2,
            "privacy_advantage_threshold": 0.35,
            "utility_error_threshold": 0.40,
            "family_failure_probability": 0.05,
            "calibration_model": (
                "task-label, true-source, and fine-token specific "
                "source-to-proxy confusion tensor"
            ),
            "primary_gate": (
                "release, zero diagnostic contract violations, and proxy "
                "balanced accuracy at least .60"
            ),
        },
        "code_sha256": {
            relative: sha256(ROOT / relative) for relative in CODE
        },
        "claim_boundary": (
            "This is a real attribute imputer on ACS microdata, not BISG: "
            "public ACS lacks surnames. True source labels are used only in "
            "disjoint proxy training, calibration, and diagnostic partitions. "
            "The target certificate uses proxy labels and a separately "
            "estimated token-dependent confusion tensor. The diagnostic "
            "partition is not used for selection."
        ),
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")
    sidecar.write_text(sha256(OUTPUT) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "sha256": sha256(OUTPUT)}))


if __name__ == "__main__":
    main()
