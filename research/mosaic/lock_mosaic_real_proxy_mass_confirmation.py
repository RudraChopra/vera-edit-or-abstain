#!/usr/bin/env python3
"""Lock the source-mass proxy confirmation before computing its outcomes."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import run_mosaic_real_proxy_mass_confirmation as confirmation
import run_mosaic_real_proxy_powered as powered


ROOT = Path(__file__).resolve().parents[2]
CODE = (
    "research/mosaic/mosaic_envelope.py",
    "research/mosaic/mosaic_proxy_bridge.py",
    "research/mosaic/mosaic_real.py",
    "research/mosaic/mosaic_transform_exact_optimizer.py",
    "research/mosaic/run_mosaic_real_proxy_study.py",
    "research/mosaic/run_mosaic_real_proxy_powered.py",
    "research/mosaic/run_mosaic_real_proxy_mass_confirmation.py",
)
POWERED_OUTCOME = (
    ROOT / "research/artifacts/mosaic_real_proxy_powered_v1.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    lock = confirmation.LOCK
    output = confirmation.OUTPUT
    sidecar = lock.with_suffix(lock.suffix + ".sha256")
    if lock.exists() or sidecar.exists():
        raise FileExistsError("proxy mass-confirmation lock already exists")
    if output.exists():
        raise FileExistsError("proxy mass-confirmation outcome already exists")
    if not POWERED_OUTCOME.exists():
        raise FileNotFoundError(POWERED_OUTCOME)
    prior_lock = json.loads(powered.PRIOR_LOCK.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "name": "MOSAIC real ACS proxy source-mass preregistration v1",
        "status": "locked_before_powered_proxy_outcomes",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_lock": head,
        "claim_boundary": (
            "This is a locked method follow-up after both earlier proxy "
            "studies abstained, not an independent first attempt. The source-"
            "mass constraint was developed on the disclosed powered-v1 "
            "calibration roles. This new seed, all four calibration sizes, "
            "the primary release decision, and the diagnostic result are "
            "reported without replacement."
        ),
        "protocol": confirmation.expected_protocol(),
        "raw_data": prior_lock["raw_data"],
        "raw_transport": {
            "url": powered.REFERENCE_URL,
            "verification": (
                "require the streamed uncompressed member to match the prior "
                "locked byte count and SHA-256"
            ),
        },
        "prior_powered_abstention_sha256": sha256(POWERED_OUTCOME),
        "code_sha256": {
            relative: sha256(ROOT / relative) for relative in CODE
        },
        "stopping_rule": (
            "Report every nested calibration point and the fixed full-size "
            "primary decision. Do not replace the seed, imputer, token count, "
            "source-mass allocation, confidence region, thresholds, or "
            "diagnostic fold."
        ),
    }
    lock.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(lock)
    sidecar.write_text(f"{digest}  {lock.name}\n", encoding="utf-8")
    print(json.dumps({"lock": str(lock), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
