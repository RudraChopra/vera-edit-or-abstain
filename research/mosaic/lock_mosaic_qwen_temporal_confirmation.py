#!/usr/bin/env python3
"""Bind a passing Qwen pilot to the prewritten temporal confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_PILOT = REPOSITORY / "research/artifacts/mosaic_qwen_pilot_v1.json"
DEFAULT_STORE_ROOT = Path("/Volumes/Backups/FARO/artifacts/civilcomments_qwen25_pilot")
DEFAULT_OUTPUT = (
    REPOSITORY / "research/mosaic/prereg_mosaic_qwen_temporal_confirmation_v1.json"
)
LOCKED_FILES = (
    "research/mosaic/MOSAIC_QWEN_TEMPORAL_CONFIRMATION_DESIGN.md",
    "research/mosaic/prepare_civilcomments_qwen_confirmation_store.py",
    "research/mosaic/run_mosaic_qwen_temporal_confirmation.py",
    "research/mosaic/audit_mosaic_qwen_temporal_confirmation.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--pilot-store-root", type=Path, default=DEFAULT_STORE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sidecar = args.output.with_suffix(".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("refusing to replace an existing Qwen confirmation lock")
    pilot = json.loads(args.pilot.read_text(encoding="utf-8"))
    if not pilot.get("go_to_locked_confirmation"):
        raise RuntimeError("pilot did not pass; confirmation cannot be locked")
    candidate = pilot["selected_candidate"]
    key = str(candidate["candidate"])
    representation, token_text = key.split("::K=", 1)
    token_count = int(token_text)
    manifest_path = args.pilot_store_root / representation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(manifest_path) != candidate["representation_manifest_sha256"]:
        raise RuntimeError("selected pilot representation manifest differs from its receipt")
    if manifest["representation"] != representation:
        raise RuntimeError("selected representation key and manifest differ")
    if token_count != int(candidate["token_count"]):
        raise RuntimeError("selected token count differs within the pilot receipt")

    code_hashes = {relative: sha256(REPOSITORY / relative) for relative in LOCKED_FILES}
    repository_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    prereg = {
        "name": "MOSAIC Qwen2.5 temporal confirmation preregistration v1",
        "status": "locked_confirmation_authorized",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "repository_head_at_lock": repository_head,
        "pilot_artifact": str(args.pilot.relative_to(REPOSITORY)),
        "pilot_artifact_sha256": sha256(args.pilot),
        "pilot_go_rule_passed": True,
        "pilot_ids": "integer dataset id modulo 4 equals 0",
        "confirmation_ids": "integer dataset id modulo 4 is nonzero",
        "source_csv_sha256": manifest["source_csv_sha256"],
        "model": {
            "model_id": manifest["model"],
            "revision": manifest["model_revision"],
            "layer_count": manifest["model_layers"],
            "prompt_prefix": manifest["prompt_prefix"],
            "max_length": manifest["max_length"],
            "batch_size": 16,
            "padding_side": "right",
        },
        "selected_candidate": {
            "key": key,
            "representation": representation,
            "hidden_layer": manifest["hidden_layer"],
            "pooling": manifest["pooling"],
            "token_count": token_count,
            "selection_rule": (
                "lowest certified error, then highest minimum retained mass, "
                "then lexical candidate key among prewritten pilot-go candidates"
            ),
        },
        "temporal_roles": {
            "construction_end_exclusive_utc": "2016-10-01",
            "reference_start_utc": "2016-10-01",
            "target_start_utc": "2017-04-01",
            "balanced_role_caps": {
                "construction": 4000,
                "reference": 8000,
                "target": 12000,
            },
            "selection_seed": 3199,
            "target_bridge_fraction": "2/3 within each source-label stratum",
            "target_diagnostic_fraction": "1/3 within each source-label stratum",
        },
        "seeds": [3201, 3202, 3203, 3204, 3205],
        "familywise_delta": 0.05,
        "privacy_advantage_threshold": 0.35,
        "utility_thresholds": [0.30, 0.35, 0.40, 0.45, 0.49],
        "primary_utility_threshold": 0.40,
        "released_token_count": 2,
        "persistent_release_per_item": True,
        "operational_draws_per_primary_release": 100,
        "solver_time_limit_seconds": 300.0,
        "attacker_constraint_generation": False,
        "main_paper_inclusion_gate": {
            "minimum_primary_releases": 3,
            "maximum_heldout_primary_violations": 0,
            "maximum_operational_primary_violations": 0,
        },
        "reporting_rule": (
            "Report every seed, threshold decision, abstention, error, certified bound, "
            "retained mass, held-out diagnostic, and operational replay."
        ),
        "source_semantics": (
            "identity_any >= 0.5 denotes an identity mention in the comment, "
            "not the author's demographic identity"
        ),
        "prelock_information": (
            "Only timestamps and binary source-label support counts from confirmation IDs "
            "were inspected before the pilot outcome; no confirmation representation or "
            "model outcome was computed."
        ),
        "code_sha256": code_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = sha256(args.output)
    sidecar.write_text(digest + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": digest,
                "selected_candidate": key,
                "repository_head_at_lock": repository_head,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
