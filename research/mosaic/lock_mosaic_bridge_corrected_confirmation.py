#!/usr/bin/env python3
"""Lock a fresh MOSAIC confirmation of the corrected strict-v2 pipeline."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import scipy
import sklearn

from mosaic_real import sha256
from run_mosaic_bridge_frontier import expected_protocol
from run_mosaic_real_pilot import DATASETS


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_OUTPUT = ROOT / "prereg_mosaic_bridge_corrected_confirmation_v2.json"
DEFAULT_SEED_START = 1220
DEFAULT_SEED_COUNT = 5
STRICT_AMENDMENT = ROOT / "prereg_mosaic_bridge_strict_amendment_v2.json"
RATIONAL_LOCK = ROOT / "prereg_mosaic_bridge_rational_audit_v2.json"
CODE_PATHS = (
    "research/mosaic/BRIDGE_MEMBERSHIP_THEOREM.md",
    "research/mosaic/REPEATED_QUERY_THEOREM.md",
    "research/mosaic/audit_mosaic_bridge_rational.py",
    "research/mosaic/audit_mosaic_bridge_strict.py",
    "research/mosaic/audit_mosaic_bridge_strict_v2.py",
    "research/mosaic/lock_mosaic_bridge_corrected_confirmation.py",
    "research/mosaic/mosaic_bridge.py",
    "research/mosaic/mosaic_channel.py",
    "research/mosaic/mosaic_envelope.py",
    "research/mosaic/mosaic_rational_certificate.py",
    "research/mosaic/mosaic_real.py",
    "research/mosaic/mosaic_strict_certification.py",
    "research/mosaic/mosaic_strict_certification_v2.py",
    "research/mosaic/mosaic_transform_exact.py",
    "research/mosaic/mosaic_transform_exact_optimizer.py",
    "research/mosaic/replay_mosaic_bridge_strict.py",
    "research/mosaic/replay_mosaic_bridge_strict_v2.py",
    "research/mosaic/run_mosaic_bridge_frontier.py",
    "research/mosaic/run_mosaic_official_frontier_exact_confirmation.py",
    "research/mosaic/run_mosaic_real_pilot.py",
    "research/mosaic/summarize_mosaic_bridge_grouped.py",
    "research/mosaic/verify_mosaic_bridge.py",
    "research/scripts/official_eraser_adapters.py",
    "research/scripts/run_official_eraser_frontier.py",
    "research/tests/test_mosaic_grouped_summary.py",
)
OFFICIAL_REPOSITORIES = {
    "INLP": Path("/Volumes/Backups/FARO/external/nullspace_projection"),
    "LEACE": Path("/Volumes/Backups/FARO/external/concept-erasure"),
    "R-LACE": Path("/Volumes/Backups/FARO/external/rlace-icml"),
    "TaCo": Path("/Volumes/Backups/FARO/external/TaCo"),
    "MANCE++": Path("/Volumes/Backups/FARO/external/mance"),
}


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def git(*arguments: str, cwd: Path = REPOSITORY) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def committed_code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in CODE_PATHS:
        path = REPOSITORY / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        ).stdout
        if committed != path.read_bytes():
            raise RuntimeError(f"{relative} differs from committed HEAD")
        hashes[relative] = sha256(path)
    return hashes


def official_repository_receipts() -> dict[str, object]:
    receipts: dict[str, object] = {}
    for name, path in OFFICIAL_REPOSITORIES.items():
        status = git("status", "--porcelain", cwd=path)
        if status:
            raise RuntimeError(f"official repository is dirty: {name}")
        receipts[name] = {
            "path": str(path),
            "commit": git("rev-parse", "HEAD", cwd=path),
            "remote": git("remote", "get-url", "origin", cwd=path),
            "clean": True,
        }
    return receipts


def locked_hash(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    digest = sha256(path)
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != digest:
        raise RuntimeError(f"lock sidecar mismatch: {path}")
    return digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed_count < 1:
        raise ValueError("seed count must be positive")
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite a corrected confirmation lock")
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    dataset_receipts = {}
    for name, config in DATASETS.items():
        path = Path(config["path"])
        dataset_receipts[name] = {
            "path": str(path),
            "modality": config["modality"],
            "target_mode": config["target_mode"],
            "manifest_sha256": sha256(path / "manifest.json"),
        }
    strict_amendment_hash = locked_hash(STRICT_AMENDMENT)
    rational_lock_hash = locked_hash(RATIONAL_LOCK)
    job_count = len(dataset_receipts) * len(seeds)
    payload: dict[str, object] = {
        "project": "MOSAIC: Minimax-Optimized Source-Agnostic Invariant Channels",
        "phase": "fresh corrected strict-v2 bridge confirmation",
        "status": "locked_before_confirmatory_outcomes",
        "supersedes_preflight_only": {
            "path": "research/mosaic/prereg_mosaic_bridge_corrected_confirmation_v1.json",
            "reason": (
                "The v1 status label was incompatible with the frozen frontier runner. "
                "The runner rejected it before loading data or writing an outcome."
            ),
        },
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "repository_head_at_lock": git("rev-parse", "HEAD"),
        "confirmation_seeds": seeds,
        "datasets": dataset_receipts,
        "protocol": expected_protocol(),
        "numerical_policy": {
            "execution": (
                "Materialize raw frontier receipts, then deterministically apply the "
                "already locked strict-v2 structural-zero correction before any "
                "selection is reported."
            ),
            "strict_amendment_path": str(STRICT_AMENDMENT.relative_to(REPOSITORY)),
            "strict_amendment_sha256": strict_amendment_hash,
            "rational_audit_lock_path": str(RATIONAL_LOCK.relative_to(REPOSITORY)),
            "rational_audit_lock_sha256": rational_lock_hash,
            "strict_repair_version": "v2_structural_zero_columns",
            "decision_tolerance": 0.0,
        },
        "frontier": {
            "candidate_count": 13,
            "official_methods": ["INLP", "LEACE", "R-LACE", "TaCo", "MANCE++"],
            "proxy_rows_permitted": False,
            "primary_selection": (
                "For each dataset and seed, select the L=2 strict-v2 candidate "
                "with minimum certified worst source-label error among candidates "
                "satisfying source advantage <=0.35 and utility error <=0.40; "
                "break exact ties lexically and otherwise abstain."
            ),
            "l4_followup": (
                "Reoptimize L=4 only for the minimum-error L=2 candidate, without "
                "using diagnostic outcomes."
            ),
        },
        "data_protocol": {
            "maximum_eraser_train": 8000,
            "maximum_tokenizer_construction": 2000,
            "maximum_balanced_reference": 8000,
            "maximum_balanced_external": 12000,
            "external_split": (
                "Within each represented source-label stratum, a seeded two-thirds "
                "bridge and one-third untouched diagnostic partition."
            ),
            "tokenizer": (
                "Balanced logistic task score fit only on construction features and "
                "discretized at construction-score quartiles."
            ),
        },
        "claim_boundary": (
            "This confirmation tests the already locked corrected strict-v2 pipeline "
            "on new seed-specific bridge and diagnostic partitions. It is a fresh "
            "confirmation of the implementation, not evidence of 25 independent "
            "deployment domains: jobs on a shared dataset remain statistically "
            "clustered. Results do not imply clinical safety, protect unregistered "
            "side channels, or cover later target-population drift."
        ),
        "decision_gates": {
            "required_files": job_count,
            "required_candidate_rows": job_count * 13,
            "required_global_optimization_replays": job_count * 14,
            "maximum_optimization_errors": 0,
            "maximum_primary_false_acceptances": 0,
            "maximum_bridge_membership_violation": 0.0,
            "require_camelyon_missing_support_abstention": True,
            "require_l4_pointwise_no_worse": True,
        },
        "pass_conditions": {
            "complete_execution": (
                f"All {job_count} dataset-seed jobs contain 13 official candidate "
                "rows and complete strict-v2 replay and rational audits."
            ),
            "corrected_replay": (
                "Every strict-v2 receipt exactly equals an independently regenerated "
                "receipt under zero decision tolerance."
            ),
            "outward_certificate": (
                "Every serialized bridge slack is nonnegative and every reported "
                "source and utility bound is verified by exact-rational recomputation."
            ),
            "complete_reporting": (
                "Report all dataset-level deployments, diagnostics, abstentions, "
                "missing-support cases, retained masses, and dataset-clustered "
                "sensitivity summaries regardless of outcome."
            ),
        },
        "pilot_exclusion": {
            "excluded_real_seed_ranges": ["0-1219"],
            "use": (
                "No earlier real-data seed, including raw, strict-v1, strict-v2, or "
                "transform-exact runs, enters the confirmation estimates."
            ),
        },
        "stopping_rule": (
            "Run every registered dataset and seed. No outcome-based replacement, "
            "threshold change, candidate change, early stopping, or selective omission."
        ),
        "code_sha256": committed_code_hashes(),
        "official_repositories": official_repository_receipts(),
        "runtime_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    atomic_write(args.output, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = sha256(args.output)
    atomic_write(sidecar, digest + "\n")
    print(json.dumps({"path": str(args.output), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
