#!/usr/bin/env python3
"""Freeze the transform-exact refinement protocol before confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_TEMPLATE = ROOT / "prereg_mosaic_transform_exact_v1.template.json"
DEFAULT_OUTPUT = ROOT / "prereg_mosaic_transform_exact_v1.json"
DEFAULT_SIDECAR = ROOT / "prereg_mosaic_transform_exact_v1.sha256"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("template must contain a JSON object")
    return payload


def atomic_json(payload: dict[str, Any], output: Path) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.sidecar.exists():
        raise FileExistsError("refusing to overwrite an existing preregistration lock")
    payload = load_json(args.template)
    if int(payload["replicates_per_cell"]) < 1000:
        raise ValueError("confirmation requires at least 1000 replicates per cell")
    if payload["methods"] != ["capacity_transfer", "transform_exact"]:
        raise ValueError("method order changed")
    code_hashes = {}
    for relative in payload.pop("locked_files"):
        path = REPOSITORY / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        code_hashes[relative] = sha256(path)
    pilot_hashes = {}
    for relative in payload["pilot_exclusion"]["artifacts"]:
        path = REPOSITORY / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        pilot_hashes[relative] = sha256(path)
    payload["code_sha256"] = code_hashes
    payload["pilot_artifact_sha256"] = pilot_hashes
    payload["template_sha256"] = sha256(args.template)
    payload["repository_head_at_lock"] = git_head()
    payload["locked_at"] = datetime.now(timezone.utc).isoformat()
    payload["status"] = str(
        payload.pop("lock_status", "locked_before_confirmatory_outcomes")
    )
    atomic_json(payload, args.output)
    digest = sha256(args.output)
    try:
        display_path = args.output.relative_to(REPOSITORY)
    except ValueError:
        display_path = args.output
    args.sidecar.write_text(f"{digest}  {display_path}\n", encoding="utf-8")
    print(json.dumps({"preregistration": str(args.output), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
