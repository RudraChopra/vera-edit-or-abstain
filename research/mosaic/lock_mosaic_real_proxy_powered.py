#!/usr/bin/env python3
"""Lock the powered real-proxy follow-up before its outcomes are computed."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from run_mosaic_real_proxy_powered import (
    LOCK,
    OUTPUT,
    PRIOR_LOCK,
    REFERENCE_URL,
    expected_protocol,
)


ROOT = Path(__file__).resolve().parents[2]
CODE = (
    "research/mosaic/mosaic_envelope.py",
    "research/mosaic/mosaic_proxy_bridge.py",
    "research/mosaic/mosaic_real.py",
    "research/mosaic/mosaic_transform_exact_optimizer.py",
    "research/mosaic/run_mosaic_real_proxy_study.py",
    "research/mosaic/run_mosaic_real_proxy_powered.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    sidecar = LOCK.with_suffix(LOCK.suffix + ".sha256")
    if LOCK.exists() or sidecar.exists():
        raise FileExistsError("powered proxy lock already exists")
    if OUTPUT.exists():
        raise FileExistsError("powered proxy outcome already exists")
    prior = json.loads(PRIOR_LOCK.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "name": "MOSAIC powered real ACS proxy preregistration v1",
        "status": "locked_before_powered_proxy_outcomes",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_lock": head,
        "claim_boundary": (
            "This is a locked powered follow-up motivated by the v1 abstention, "
            "not an independent first attempt. True source labels occur only in "
            "proxy training, calibration, and the untouched diagnostic fold. "
            "The target certificate receives task labels, tokens, and imputed "
            "source labels but not true target source labels."
        ),
        "protocol": expected_protocol(),
        "raw_data": prior["raw_data"],
        "raw_transport": {
            "url": REFERENCE_URL,
            "verification": (
                "require the streamed uncompressed member to match the prior "
                "locked byte count and SHA-256"
            ),
        },
        "prior_negative_result_sha256": sha256(
            ROOT / "research/artifacts/mosaic_real_proxy_v1.json"
        ),
        "code_sha256": {relative: sha256(ROOT / relative) for relative in CODE},
        "stopping_rule": (
            "Report all four nested calibration sizes and the fixed full-size "
            "primary decision. Do not replace the imputer, token count, "
            "partitions, confidence region, thresholds, or diagnostic fold."
        ),
    }
    LOCK.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(LOCK)
    sidecar.write_text(f"{digest}  {LOCK.name}\n", encoding="utf-8")
    print(json.dumps({"lock": str(LOCK), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
