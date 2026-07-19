#!/usr/bin/env python3
"""Lock the multi-state, multi-task ACS natural-shift confirmation."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from mosaic_real import sha256
from run_mosaic_acs_natural_shift import expected_protocol


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_OUTPUT = ROOT / "prereg_mosaic_acs_natural_shift_v1.json"
RAW_ROOT = Path("/Volumes/Backups/FARO/artifacts/acs_folktables_raw")
STATE_CODES = {"WA": "53", "IL": "17", "NY": "36", "FL": "12"}
OFFICIAL_REPOSITORIES = {
    "INLP": Path("/Volumes/Backups/FARO/external/nullspace_projection"),
    "LEACE": Path("/Volumes/Backups/FARO/external/concept-erasure"),
    "R-LACE": Path("/Volumes/Backups/FARO/external/rlace-icml"),
    "TaCo": Path("/Volumes/Backups/FARO/external/TaCo"),
    "MANCE++": Path("/Volumes/Backups/FARO/external/mance"),
}
CODE_PATHS = (
    "research/mosaic/mosaic_bridge.py",
    "research/mosaic/mosaic_channel.py",
    "research/mosaic/mosaic_envelope.py",
    "research/mosaic/mosaic_exact.py",
    "research/mosaic/mosaic_invariant.py",
    "research/mosaic/mosaic_optimizer.py",
    "research/mosaic/mosaic_real.py",
    "research/mosaic/mosaic_strict_certification.py",
    "research/mosaic/mosaic_transform_exact.py",
    "research/mosaic/mosaic_transform_exact_optimizer.py",
    "research/mosaic/replay_mosaic_bridge_strict.py",
    "research/mosaic/run_mosaic_acs_natural_shift.py",
    "research/mosaic/run_mosaic_official_frontier_exact_confirmation.py",
    "research/scripts/official_eraser_adapters.py",
    "research/scripts/prepare_acs_natural_shift_stores.py",
    "research/scripts/run_official_eraser_frontier.py",
)


def git(*arguments: str, cwd: Path = REPOSITORY) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def committed_code_hashes() -> dict[str, str]:
    hashes = {}
    for relative in CODE_PATHS:
        path = REPOSITORY / relative
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


def official_repositories() -> dict[str, Any]:
    receipts = {}
    for name, path in OFFICIAL_REPOSITORIES.items():
        if git("status", "--porcelain", "--untracked-files=no", cwd=path):
            raise RuntimeError(f"official repository is dirty: {name}")
        receipts[name] = {
            "path": str(path),
            "commit": git("rev-parse", "HEAD", cwd=path),
            "remote": git("remote", "get-url", "origin", cwd=path),
            "clean": True,
        }
    return receipts


def assert_fresh_targets() -> None:
    present = [
        state
        for state, code in STATE_CODES.items()
        if (RAW_ROOT / "2018" / "1-Year" / f"psam_p{code}.csv").exists()
    ]
    if present:
        raise RuntimeError(f"fresh target files already exist before lock: {present}")


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite the natural-shift lock")
    assert_fresh_targets()
    protocol = expected_protocol()
    jobs = [
        {"task": task, "target_state": state, "seed": seed}
        for task in protocol["tasks"]
        for state in protocol["target_states"]
        for seed in protocol["confirmation_seeds"]
    ]
    payload: dict[str, Any] = {
        "project": "MOSAIC: Minimax-Optimized Source-Agnostic Invariant Channels",
        "phase": "fresh multi-state, multi-task ACS natural geographic-shift confirmation",
        "status": "locked_before_fresh_state_downloads",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "repository_head_at_lock": git("rev-parse", "HEAD"),
        "protocol": protocol,
        "jobs": jobs,
        "job_count": len(jobs),
        "code_sha256": committed_code_hashes(),
        "official_repositories": official_repositories(),
        "data_protocol": {
            "source_concept": "binary sex, excluded from the released feature matrix",
            "reference_population": "California ACS 2018 one-year PUMS",
            "target_populations": [
                "Washington ACS 2018 one-year PUMS",
                "Illinois ACS 2018 one-year PUMS",
                "New York ACS 2018 one-year PUMS",
                "Florida ACS 2018 one-year PUMS",
            ],
            "tasks": ["income", "employment", "public coverage"],
            "external_split": (
                "Seeded two-thirds versus one-third partition of entire target-state "
                "PUMAs before tokenization, editing, optimization, or inspection of labels."
            ),
            "frontier": (
                "Identity plus 12 official INLP, LEACE, R-LACE, TaCo, and MANCE++ "
                "edits for every job; no proxy rows."
            ),
            "diagnostic_policy": (
                "The held-out PUMA fold is used only after selection and never repairs, "
                "retunes, or replaces a registered result."
            ),
        },
        "primary_estimands": {
            "mosaic": (
                "Release and false-acceptance rates for the strict structured-shift "
                "certificate at utility threshold 0.40."
            ),
            "direct": (
                "Matched direct target-table release and false-acceptance rates at 0.40."
            ),
            "operational": (
                "For each primary release, 100 draws with one sampled public token per "
                "held-out person and the fixed certified decoder."
            ),
        },
        "secondary_estimands": {
            "threshold_frontier": list(protocol["utility_thresholds"]),
            "alphabet_comparison": list(protocol["fine_token_counts"]),
            "task_and_state_breakdowns": True,
        },
        "pass_conditions": {
            "complete_execution": f"Run all {len(jobs)} registered jobs at both alphabets and all 13 candidates.",
            "complete_reporting": "Report every job, optimization failure, release, abstention, and diagnostic outcome.",
            "certificate_integrity": "Every released row must satisfy the serialized strict bridge and release inequalities.",
            "split_integrity": "Bridge and diagnostic PUMA sets must be nonempty and disjoint for every job.",
        },
        "stopping_rule": (
            "Run all registered states, tasks, seeds, candidates, alphabets, and thresholds. "
            "No state replacement, threshold changes, split changes, early stopping, or "
            "selective omission after any target outcome is observed."
        ),
        "claim_boundary": (
            "This confirmation measures natural 2018 state and whole-PUMA transfer for "
            "the registered ACS tasks. It does not establish a guarantee for every state, "
            "future year, unregistered task, or unregistered downstream use."
        ),
    }
    atomic_write(args.output, payload)
    sidecar.write_text(sha256(args.output) + "\n", encoding="utf-8")
    print(json.dumps({"lock": str(args.output), "jobs": len(jobs)}, indent=2))


if __name__ == "__main__":
    main()
