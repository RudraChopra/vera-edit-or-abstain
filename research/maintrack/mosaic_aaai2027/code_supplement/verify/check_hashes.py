#!/usr/bin/env python3
"""Verify every immutable member listed in MANIFEST_SHA256.txt."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST_SHA256.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures = []
    rows = MANIFEST.read_text(encoding="utf-8").splitlines()
    for row in rows:
        expected, relative = row.split("  ", 1)
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"hash mismatch: {relative}")
    if failures:
        raise SystemExit("manifest verification failed\n" + "\n".join(failures))
    print(f"manifest verification: pass ({len(rows)} immutable files)")


if __name__ == "__main__":
    main()
