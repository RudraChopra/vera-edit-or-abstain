#!/usr/bin/env python3
"""Lock the powered Qwen temporal confirmation before model download."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCK = (
    ROOT / "research/mosaic/prereg_mosaic_qwen_powered_confirmation_v1.json"
)
SOURCE_CSV = (
    ROOT
    / "research/data/wilds/civilcomments_v1.0/"
    "all_data_with_identities.csv"
)
STORE = ROOT / "research/data/civilcomments_qwen25_powered_confirmation"
OUTPUT = (
    ROOT / "research/artifacts/mosaic_qwen_powered_confirmation_v1"
)
PILOT = ROOT / "research/artifacts/mosaic_qwen_pilot_v1.json"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
CODE = (
    "research/mosaic/mosaic_bridge.py",
    "research/mosaic/mosaic_channel.py",
    "research/mosaic/mosaic_envelope.py",
    "research/mosaic/mosaic_real.py",
    "research/mosaic/mosaic_transform_exact_optimizer.py",
    "research/mosaic/prepare_civilcomments_qwen_confirmation_store.py",
    "research/mosaic/prepare_civilcomments_qwen_powered_store.py",
    "research/mosaic/run_mosaic_bridge_frontier.py",
    "research/mosaic/run_mosaic_qwen_temporal_confirmation.py",
    "research/mosaic/run_mosaic_qwen_powered_confirmation.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol() -> dict[str, object]:
    return {
        "design": (
            "locked powered temporal confirmation after the architecture "
            "pilot missed its .49 eligibility gate"
        ),
        "candidate": "Qwen2.5-1.5B-Instruct::layer28_mean::K=2",
        "fine_token_count": 2,
        "released_token_count": 2,
        "seeds": [4201, 4202, 4203, 4204, 4205],
        "privacy_advantage_threshold": 0.35,
        "utility_thresholds": [0.35, 0.40, 0.45, 0.49],
        "primary_utility_threshold": 0.40,
        "familywise_delta": 0.05,
        "family": "two token tables for each of five registered seeds",
        "target_split": (
            "within each toxicity by identity-mention stratum, two thirds "
            "bridge and one third untouched diagnostic"
        ),
        "operational_draws_per_primary_release": 100,
        "solver_time_limit_seconds": 300.0,
        "attacker_constraint_generation": True,
        "runtime_semantics": (
            "one persistent sampled release token per immutable item"
        ),
        "selection": (
            "the last-layer mean representation is inherited from the best "
            "pilot candidate; K=2 is fixed because the pilot K=4 optimum "
            "collapsed its four fine tokens into two identical channel rows "
            "per side of the median threshold"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", type=Path, default=SOURCE_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_csv = args.source_csv
    sidecar = LOCK.with_suffix(LOCK.suffix + ".sha256")
    if LOCK.exists() or sidecar.exists():
        raise FileExistsError("Qwen powered lock already exists")
    if STORE.exists() or OUTPUT.exists():
        raise FileExistsError("Qwen powered store or outcome already exists")
    if not source_csv.exists() or not PILOT.exists():
        raise FileNotFoundError("CivilComments source or Qwen pilot is missing")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/"
        "models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
        / MODEL_REVISION
    )
    payload = {
        "name": "MOSAIC Qwen2.5 powered temporal confirmation lock v1",
        "status": "locked_before_model_and_outcomes",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_head_before_lock": head,
        "claim_boundary": (
            "This is a registered powered follow-up after an unlocked pilot, "
            "not an independent first attempt. The source variable records "
            "whether a comment mentions an annotated identity, not the "
            "author's demographic identity. Every seed, abstention, solver "
            "error, held-out diagnostic, and operational replay is reportable."
        ),
        "source_csv": {
            "bytes": source_csv.stat().st_size,
            "sha256": sha256(source_csv),
        },
        "pilot_artifact": str(PILOT.relative_to(ROOT)),
        "pilot_artifact_sha256": sha256(PILOT),
        "pilot_result": {
            "go_to_locked_confirmation": False,
            "best_candidate": "layer28_mean::K=4",
            "best_certified_error": 0.49011716452534876,
            "best_minimum_retained_mass": 0.6557498169410484,
        },
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "parameter_count": 1_540_000_000,
            "layer_count": 28,
            "hidden_size": 1536,
            "representation": "last transformer layer",
            "pooling": "attention-mask-aware mean",
            "prompt_prefix": "Moderate this comment for toxicity:\n",
            "max_length": 96,
            "batch_size": 16,
        },
        "temporal_roles": {
            "pilot_partition": "integer dataset id modulo 4 equals 0",
            "confirmation_partition": "integer dataset id modulo 4 is nonzero",
            "construction": "created before 2016-10-01 UTC",
            "reference": (
                "created from 2016-10-01 through 2017-03-31 UTC"
            ),
            "target": "created on or after 2017-04-01 UTC",
            "reference_start_utc": "2016-10-01",
            "target_start_utc": "2017-04-01",
            "balanced_role_caps": {
                "construction": 6000,
                "reference": 18000,
                "target": 32000,
            },
            "selection_seed": 20270726,
        },
        **protocol(),
        "main_paper_inclusion_gate": {
            "minimum_primary_releases": 3,
            "maximum_heldout_primary_violations": 0,
            "maximum_operational_primary_violations": 0,
        },
        "code_sha256": {
            relative: sha256(ROOT / relative) for relative in CODE
        },
        "model_snapshot_absent_at_lock": not snapshot.exists(),
        "stopping_rule": (
            "Run all five seeds after one locked extraction. Do not replace "
            "the model, revision, representation, pooling, alphabet, temporal "
            "cutoffs, row caps, seeds, thresholds, confidence allocation, or "
            "diagnostic folds."
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
