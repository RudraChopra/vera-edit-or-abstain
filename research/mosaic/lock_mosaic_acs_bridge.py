#!/usr/bin/env python3
"""Lock the California-to-Texas ACSIncome MOSAIC bridge confirmation."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from mosaic_real import sha256
from run_mosaic_bridge_frontier import expected_protocol
from run_mosaic_real_pilot import DATASETS


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DATASET = "ACSIncome-CA-TX"
DEFAULT_OUTPUT = ROOT / "prereg_mosaic_acs_bridge_v1.json"
OFFICIAL_REPOSITORIES = {
    "INLP": Path("/Volumes/Backups/FARO/external/nullspace_projection"),
    "LEACE": Path("/Volumes/Backups/FARO/external/concept-erasure"),
    "R-LACE": Path("/Volumes/Backups/FARO/external/rlace-icml"),
    "TaCo": Path("/Volumes/Backups/FARO/external/TaCo"),
    "MANCE++": Path("/Volumes/Backups/FARO/external/mance"),
}
CODE_PATHS = (
    "research/mosaic/mosaic_bridge.py",
    "research/mosaic/mosaic_envelope.py",
    "research/mosaic/mosaic_real.py",
    "research/mosaic/mosaic_transform_exact_optimizer.py",
    "research/mosaic/run_mosaic_bridge_frontier.py",
    "research/mosaic/run_mosaic_official_frontier_exact_confirmation.py",
    "research/mosaic/run_mosaic_real_pilot.py",
    "research/scripts/official_eraser_adapters.py",
    "research/scripts/run_official_eraser_frontier.py",
    "research/scripts/prepare_acs_income_store.py",
)


def git(*arguments: str, cwd: Path = REPOSITORY) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=cwd, check=True, capture_output=True, text=True
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


def official_repositories() -> dict[str, Any]:
    receipts: dict[str, Any] = {}
    for name, path in OFFICIAL_REPOSITORIES.items():
        status = git("status", "--porcelain", "--untracked-files=no", cwd=path)
        if status:
            raise RuntimeError(f"official repository is dirty: {name}")
        receipts[name] = {
            "path": str(path),
            "commit": git("rev-parse", "HEAD", cwd=path),
            "remote": git("remote", "get-url", "origin", cwd=path),
            "clean": True,
        }
    return receipts


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-start", type=int, default=1300)
    parser.add_argument("--seed-count", type=int, default=5)
    args = parser.parse_args()
    if args.seed_count < 1:
        raise ValueError("seed count must be positive")
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite the ACS lock")
    config = DATASETS[DATASET]
    store = Path(config["path"])
    manifest = store / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_payload.get("excluded_feature") != "SEX":
        raise ValueError("the ACS source feature must be excluded from the release matrix")
    if int(manifest_payload.get("split_counts", {}).get("2", 0)) < 12_000:
        raise ValueError("the ACS target split is too small for the registered cap")

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    payload: dict[str, Any] = {
        "project": "MOSAIC: Minimax-Optimized Source-Agnostic Invariant Channels",
        "phase": "fresh ACSIncome California-to-Texas bridge confirmation",
        "status": "locked_before_confirmatory_outcomes",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "repository_head_at_lock": git("rev-parse", "HEAD"),
        "confirmation_seeds": seeds,
        "datasets": {
            DATASET: {
                "path": str(store),
                "modality": str(config["modality"]),
                "target_mode": str(config["target_mode"]),
                "manifest_sha256": sha256(manifest),
                "reference_population": "California ACS 2018 1-Year PUMS",
                "target_population": "Texas ACS 2018 1-Year PUMS",
                "task": "income above $50,000",
                "audit_source": "binary sex; excluded from the released feature matrix",
            }
        },
        "protocol": expected_protocol(),
        "code_sha256": committed_code_hashes(),
        "official_repositories": official_repositories(),
        "data_protocol": {
            "maximum_eraser_train": 8000,
            "maximum_tokenizer_construction": 2000,
            "maximum_balanced_reference": 8000,
            "maximum_balanced_external": 12000,
            "external_split": "Seeded two-thirds bridge and one-third untouched diagnostic fold within each source-label stratum.",
            "preprocessing": "Train-only standardization and PCA to 128 dimensions or the input dimension, whichever is smaller.",
            "frontier": "Identity plus 12 official INLP, LEACE, R-LACE, TaCo, and MANCE++ edits; no proxy rows.",
        },
        "selection": {
            "primary": "At utility threshold 0.40, select the feasible L=2 candidate with lowest strict bridge-certified worst source-label error; lexical tie break; otherwise abstain.",
            "secondary": "Reoptimize L=4 only for that selected L=2 candidate without using diagnostics.",
        },
        "pass_conditions": {
            "complete_execution": f"All {len(seeds)} seeds run every one of 13 official candidates.",
            "complete_reporting": "Report every seed, candidate failure, abstention, diagnostic result, and selected certificate regardless of outcome.",
            "external_diagnostic": "Treat Texas diagnostic results as held-out checks; do not use them to select a candidate or to estimate theorem coverage.",
        },
        "claim_boundary": "This is a fresh, preregistered confirmation on one public tabular geographic shift. The finite-sample statement is per registered reference/bridge event, not a population-wide privacy or fairness guarantee; it does not cover post-certification drift or unregistered outputs.",
        "stopping_rule": "Run all five seeds and all candidates. No outcome-based threshold changes, data replacement, early stopping, or selective reporting.",
    }
    atomic_write(args.output, payload)
    sidecar.write_text(sha256(args.output) + "\n", encoding="utf-8")
    print(json.dumps({"lock": str(args.output), "seeds": seeds}, indent=2))


if __name__ == "__main__":
    main()
