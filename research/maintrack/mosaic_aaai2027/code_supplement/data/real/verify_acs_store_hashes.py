#!/usr/bin/env python3
"""Verify all twelve recreated ACS stores against the pre-outcome data lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stores",
        type=Path,
        default=Path("data/real/processed/acs_natural_shift"),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=HERE / "acs_natural_shift_data_lock.json",
    )
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    failures = []
    files = 0
    for key, expected in lock["stores"].items():
        task, transition = key.split(":", 1)
        target = transition.split("->", 1)[1].lower()
        store = args.stores / f"acs_{task}_ca_{target}_natural_store"
        manifest = store / "manifest.json"
        if not manifest.is_file() or sha256(manifest) != expected["manifest_sha256"]:
            failures.append(f"{key}: manifest hash mismatch")
        else:
            files += 1
        for name, receipt in expected["arrays"].items():
            path = store / name
            if not path.is_file() or sha256(path) != receipt["sha256"]:
                failures.append(f"{key}:{name}: array hash mismatch")
            else:
                files += 1
    if failures:
        raise SystemExit("ACS store verification failed\n" + "\n".join(failures))
    print(f"ACS store verification: pass ({len(lock['stores'])} stores, {files} files)")


if __name__ == "__main__":
    main()
