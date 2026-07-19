#!/usr/bin/env python3
"""Freeze MOSAIC's paired exact real-feature protocol before new outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import scipy
import sklearn


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_TEMPLATE = ROOT / "prereg_mosaic_real_exact_v1.template.json"
DEFAULT_OUTPUT = ROOT / "prereg_mosaic_real_exact_v1.json"
CODE_PATHS = (
    "research/mosaic/audit_mosaic_real_exact_frontier.py",
    "research/mosaic/lock_mosaic_real_exact_prereg.py",
    "research/mosaic/mosaic_channel.py",
    "research/mosaic/mosaic_envelope.py",
    "research/mosaic/mosaic_exact.py",
    "research/mosaic/mosaic_invariant.py",
    "research/mosaic/mosaic_optimizer.py",
    "research/mosaic/mosaic_real.py",
    "research/mosaic/mosaic_transform_exact.py",
    "research/mosaic/mosaic_transform_exact_optimizer.py",
    "research/mosaic/run_mosaic_official_frontier_exact_confirmation.py",
    "research/mosaic/run_mosaic_real_exact_confirmation.py",
    "research/mosaic/run_mosaic_real_pilot.py",
    "research/scripts/official_eraser_adapters.py",
    "research/scripts/run_official_eraser_frontier.py",
    "research/tests/test_mosaic_real.py",
    "research/tests/test_mosaic_real_exact_confirmation.py",
    "research/tests/test_mosaic_transform_exact.py",
    "research/tests/test_mosaic_transform_exact_optimizer.py",
)
STORES = {
    "Waterbirds": Path("/Volumes/Backups/FARO/artifacts/waterbirds_official_numpy_store"),
    "Camelyon17-WILDS": Path("/Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_center_numpy_store"),
    "CivilComments-WILDS": Path("/Volumes/Backups/FARO/artifacts/civilcomments_lexical_numpy_store"),
    "BiasBios-Clinical": Path("/Volumes/Backups/FARO/artifacts/bios_rlace_numpy_store"),
    "GaitPDB": Path("/Volumes/Backups/FARO/artifacts/gaitpdb_numpy_store"),
}
OFFICIAL_REPOSITORIES = {
    "INLP": Path("/Volumes/Backups/FARO/external/nullspace_projection"),
    "LEACE": Path("/Volumes/Backups/FARO/external/concept-erasure"),
    "R-LACE": Path("/Volumes/Backups/FARO/external/rlace-icml"),
    "TaCo": Path("/Volumes/Backups/FARO/external/TaCo"),
    "MANCE++": Path("/Volumes/Backups/FARO/external/mance"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def official_metadata(name: str, path: Path) -> dict[str, object]:
    dirty = git(path, "status", "--porcelain")
    if dirty:
        raise ValueError(f"official repository must be clean before lock: {name}\n{dirty}")
    return {
        "path": str(path),
        "commit": git(path, "rev-parse", "HEAD"),
        "remote": git(path, "remote", "get-url", "origin"),
        "clean": True,
    }


def validate_template(template: dict[str, object]) -> None:
    seeds = [int(value) for value in template["confirmation_seeds"]]
    excluded = {int(value) for value in template["pilot_exclusion"]["excluded_seeds"]}
    if seeds != [105, 106, 107, 108, 109] or excluded.intersection(seeds):
        raise ValueError("confirmation seeds are not the registered untouched set")
    candidates = template["frontier"]["candidates"]
    if len(candidates) != 13 or len(set(candidates)) != 13:
        raise ValueError("frontier must contain 13 unique candidates")
    if set(template["datasets"]) != set(STORES):
        raise ValueError("template dataset set differs from the frozen stores")
    if template["status"] != "draft_template":
        raise ValueError("input is not a draft preregistration template")


def development_hashes() -> dict[str, str]:
    paths = [
        REPOSITORY / "research/artifacts/mosaic_real_confirmation_manifest_v1.json",
        REPOSITORY / "research/artifacts/mosaic_real_confirmation_audit_v1.json",
        REPOSITORY / "research/artifacts/mosaic_real_transform_exact_exploratory_v1.json",
        REPOSITORY / "research/artifacts/mosaic_real_transform_exact_exploratory_audit_v1.json",
        REPOSITORY / "research/artifacts/mosaic_real_exact_design_smoke_v1.json",
        REPOSITORY / "research/artifacts/mosaic_real_exact_design_smoke_audit_v1.json",
    ]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"development history is missing: {missing}")
    return {
        str(path.relative_to(REPOSITORY)): sha256(path)
        for path in paths
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".sha256").exists():
        raise FileExistsError("refusing to overwrite an existing real-data preregistration")
    confirmation_dir = REPOSITORY / "research" / "artifacts" / "mosaic_real_exact_confirmation_v1"
    if confirmation_dir.exists() and any(confirmation_dir.iterdir()):
        raise FileExistsError("confirmation outcomes already exist; lock is no longer pre-outcome")
    template = json.loads(args.template.read_text(encoding="utf-8"))
    validate_template(template)
    locked = dict(template)
    locked.update(
        {
            "status": "locked_before_confirmatory_outcomes",
            "locked_at": datetime.now(timezone.utc).isoformat(),
            "template_sha256": sha256(args.template),
            "code_sha256": {
                relative: sha256(REPOSITORY / relative) for relative in CODE_PATHS
            },
            "development_artifact_sha256": development_hashes(),
            "frozen_stores": {
                name: {
                    "path": str(path),
                    "manifest_sha256": sha256(path / "manifest.json"),
                }
                for name, path in STORES.items()
            },
            "official_repositories": {
                name: official_metadata(name, path)
                for name, path in OFFICIAL_REPOSITORIES.items()
            },
            "runtime_environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "executable": sys.executable,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "repository_head_at_lock": git(REPOSITORY, "rev-parse", "HEAD"),
        }
    )
    atomic_json_dump(locked, args.output)
    digest = sha256(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        digest + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
