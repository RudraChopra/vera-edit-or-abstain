#!/usr/bin/env python3
"""Lock the ACS 2023 header-only transport amendment before reading data rows."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_LOCK = (
    ROOT / "research/mosaic/prereg_mosaic_acs_scalar_confirmation_v1.json"
)
OUTPUT = ROOT / "research/artifacts/mosaic_acs_scalar_confirmation_v1.json"
RUNNER = (
    ROOT / "research/mosaic/run_mosaic_acs_scalar_confirmation_transport.py"
)
AMENDMENT = ROOT / (
    "research/mosaic/"
    "prereg_mosaic_acs_scalar_confirmation_transport_v1.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    sidecar = AMENDMENT.with_suffix(AMENDMENT.suffix + ".sha256")
    if AMENDMENT.exists() or sidecar.exists():
        raise FileExistsError("scalar transport amendment already exists")
    if OUTPUT.exists():
        raise FileExistsError("scalar confirmation outcome already exists")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runner_relative = RUNNER.relative_to(ROOT).as_posix()
    payload = {
        "name": "MOSAIC ACS scalar confirmation transport amendment v1",
        "status": "locked_after_header_before_2023_rows",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_amendment": head,
        "failure": {
            "stage": "2023 Florida CSV header validation",
            "exception": (
                "ValueError: ACS file lacks required columns: ['ST']"
            ),
            "rows_read": 0,
            "outcomes_read": False,
            "cause": (
                "the official state-specific 2023 PUMS file omits the "
                "redundant constant ST column"
            ),
        },
        "amendment": (
            "When ST is absent from the official Florida state file, append "
            "the fixed Census FIPS value 12 after selecting all other locked "
            "columns."
        ),
        "unchanged": [
            "two frozen interfaces",
            "2018 reconstruction",
            "2023 confirmation population",
            "balanced sampling rule and row cap",
            "familywise allocation",
            "utility threshold",
            "fixed per-token loss functional",
            "one-sided Hoeffding bounds",
            "confirmation criterion",
            "stopping rule",
        ],
        "original_lock_sha256": sha256(ORIGINAL_LOCK),
        "code_sha256": {runner_relative: sha256(RUNNER)},
    }
    AMENDMENT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(AMENDMENT)
    sidecar.write_text(
        f"{digest}  {AMENDMENT.name}\n",
        encoding="utf-8",
    )
    print(json.dumps({"lock": str(AMENDMENT), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
