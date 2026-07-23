#!/usr/bin/env python3
"""Run the locked ACS scalar confirmation through a header-only transport fix."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import run_mosaic_acs_scalar_confirmation as confirmation


ROOT = Path(__file__).resolve().parents[2]
AMENDMENT = ROOT / (
    "research/mosaic/"
    "prereg_mosaic_acs_scalar_confirmation_transport_v1.json"
)
ORIGINAL_LOCK = (
    ROOT / "research/mosaic/prereg_mosaic_acs_scalar_confirmation_v1.json"
)
STATE_FIPS = {"FL": 12}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_state_specific_csv(
    source: Any,
    columns: set[str],
    *,
    state: str,
):
    """Read a state file and restore its redundant state-code feature."""
    import pandas as pd

    header = pd.read_csv(source, nrows=0)
    available = set(header.columns)
    selected = set(columns)
    infer_state = "ST" in selected and "ST" not in available
    if infer_state:
        selected.remove("ST")
    if "RELP" not in available and "RELSHIPP" in available:
        selected.remove("RELP")
        selected.add("RELSHIPP")
    missing = selected - available
    if missing:
        raise ValueError(f"ACS file lacks required columns: {sorted(missing)}")
    if hasattr(source, "seek"):
        source.seek(0)
    frame = pd.read_csv(source, usecols=sorted(selected), low_memory=False)
    if infer_state:
        frame["ST"] = STATE_FIPS[state]
    return frame


def validate_amendment() -> dict[str, Any]:
    sidecar = AMENDMENT.with_suffix(AMENDMENT.suffix + ".sha256")
    if not AMENDMENT.exists() or not sidecar.exists():
        raise ValueError("scalar-confirmation transport amendment is missing")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(AMENDMENT):
        raise ValueError("scalar-confirmation transport sidecar mismatch")
    payload = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if payload.get("status") != "locked_after_header_before_2023_rows":
        raise ValueError("scalar-confirmation transport status differs")
    if payload.get("original_lock_sha256") != sha256(ORIGINAL_LOCK):
        raise ValueError("scalar-confirmation original lock differs")
    relative = Path(
        "research/mosaic/run_mosaic_acs_scalar_confirmation_transport.py"
    )
    if payload.get("code_sha256", {}).get(relative.as_posix()) != sha256(
        ROOT / relative
    ):
        raise ValueError("scalar-confirmation transport code differs")
    for local in (AMENDMENT, sidecar):
        relative = local.relative_to(ROOT)
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if committed != local.read_bytes():
            raise ValueError(f"{relative} is not the committed amendment")
    return payload


def main() -> None:
    validate_amendment()
    original_reader = confirmation.read_selected_csv

    def amended_reader(source: Any, columns: set[str]):
        return read_state_specific_csv(source, columns, state="FL")

    confirmation.read_selected_csv = amended_reader
    try:
        confirmation.main()
    finally:
        confirmation.read_selected_csv = original_reader


if __name__ == "__main__":
    main()
