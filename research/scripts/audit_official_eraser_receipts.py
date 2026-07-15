"""Fail-closed audit for a locked official VERA run matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
DEFAULT_PREREG = ROOT / "prereg_real.json"
DEFAULT_HASH = ROOT / "prereg_real.sha256"
DEFAULT_RECEIPTS = ROOT / "artifacts" / "real_study_receipts"
DEFAULT_REPORT = ROOT / "artifacts" / "official_eraser_receipt_audit.json"
REQUIRED_ATTACKERS = {"linear", "rbf", "forest", "mlp"}
MATERIAL_RUNNER_FILES = (
    "research/scripts/run_official_eraser_frontier.py",
    "research/scripts/run_parallel_real_study_matrix.py",
    "research/scripts/official_eraser_adapters.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def git_blob_sha256(commit: str, relative_path: str) -> str | None:
    try:
        blob = subprocess.run(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    return hashlib.sha256(blob).hexdigest()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    prereg = load_json(args.prereg)
    study = prereg.get("real_study", {}) if isinstance(prereg, dict) else {}
    study = study if isinstance(study, dict) else {}
    datasets = study.get("datasets", {})
    methods = study.get("methods", {})
    datasets = datasets if isinstance(datasets, dict) else {}
    methods = methods if isinstance(methods, dict) else {}
    seeds = [int(value) for value in study.get("seeds", [])]
    expected_hash = args.hash_file.read_text(encoding="utf-8").split()[0] if args.hash_file.is_file() else ""
    actual_hash = sha256(args.prereg) if args.prereg.is_file() else ""

    errors: list[str] = []
    missing: list[str] = []
    invalid: list[str] = []
    proxy_rows = 0
    valid_receipts = 0
    runner_commits: set[str] = set()
    split_signatures: dict[tuple[str, int], set[str]] = {}
    upstream_repositories: dict[str, tuple[str, str]] = {}

    if prereg.get("status") != "locked_before_claim_grade_runs":
        errors.append("preregistration is not locked")
    if not expected_hash or expected_hash != actual_hash:
        errors.append("preregistration hash sidecar mismatch")

    for dataset_name, dataset_config in datasets.items():
        if not isinstance(dataset_config, dict):
            errors.append(f"invalid dataset config: {dataset_name}")
            continue
        for method_key, method_config in methods.items():
            if not isinstance(method_config, dict):
                errors.append(f"invalid method config: {method_key}")
                continue
            for seed in seeds:
                run_key = f"{dataset_name}__{method_key}__seed-{seed}"
                path = args.receipt_dir / f"{run_key}.json"
                if not path.is_file():
                    missing.append(run_key)
                    continue
                receipt = load_json(path)
                receipt_errors: list[str] = []
                expected_values = {
                    "claim_grade": True,
                    "smoke": False,
                    "dataset": dataset_name,
                    "method_family": method_key,
                    "method_name": method_config.get("display_name"),
                    "seed": seed,
                    "prereg_sha256": actual_hash,
                    "claim_configuration_verified": True,
                    "external_labels_locked_during_edit_construction": True,
                    "store_manifest_sha256": dataset_config.get("manifest_sha256"),
                }
                for key, expected in expected_values.items():
                    if receipt.get(key) != expected:
                        receipt_errors.append(f"{key}={receipt.get(key)!r}, expected {expected!r}")
                candidates = receipt.get("candidates", [])
                candidates = candidates if isinstance(candidates, list) else []
                if len(candidates) != int(method_config.get("candidate_count", -1)):
                    receipt_errors.append("candidate count mismatch")
                candidate_keys: list[str] = []
                for candidate_index, candidate in enumerate(candidates):
                    if not isinstance(candidate, dict):
                        receipt_errors.append(f"candidate {candidate_index} is not an object")
                        continue
                    provenance = candidate.get("provenance", {})
                    provenance = provenance if isinstance(provenance, dict) else {}
                    if provenance.get("commit") != method_config.get("upstream_commit"):
                        receipt_errors.append(f"candidate {candidate_index} upstream commit mismatch")
                    if provenance.get("remote") != method_config.get("upstream_remote"):
                        receipt_errors.append(f"candidate {candidate_index} upstream remote mismatch")
                    repository = str(provenance.get("repository") or "")
                    if not repository:
                        receipt_errors.append(f"candidate {candidate_index} upstream repository missing")
                    else:
                        upstream_repositories[repository] = (
                            str(provenance.get("commit") or ""),
                            str(provenance.get("remote") or ""),
                        )
                    entrypoints = provenance.get("official_entrypoint") or provenance.get(
                        "official_entrypoints"
                    )
                    if not entrypoints:
                        receipt_errors.append(f"candidate {candidate_index} official entrypoint missing")
                    candidate_key = str(candidate.get("candidate_key") or "")
                    candidate_keys.append(candidate_key)
                    if not candidate_key or not str(candidate.get("strength") or ""):
                        receipt_errors.append(f"candidate {candidate_index} key or strength missing")
                    serialized = json.dumps(candidate).lower()
                    if "proxy" in serialized:
                        proxy_rows += 1
                        receipt_errors.append(f"candidate {candidate_index} is marked as a proxy")
                    audit_path = Path(str(candidate.get("audit_npz", "")))
                    if not audit_path.is_file():
                        receipt_errors.append(f"candidate {candidate_index} audit NPZ missing")
                    elif sha256(audit_path) != candidate.get("audit_npz_sha256"):
                        receipt_errors.append(f"candidate {candidate_index} audit NPZ hash mismatch")
                    else:
                        try:
                            with np.load(audit_path) as archive:
                                required = {
                                    "target_harm_certification",
                                    "target_harm_external",
                                    "source_certification",
                                    "source_external",
                                    "environment_certification",
                                    "environment_external",
                                    "target_certification",
                                    "target_external",
                                }
                                leakage_certification = {
                                    name.removeprefix("leakage_correct_certification__")
                                    for name in archive.files
                                    if name.startswith("leakage_correct_certification__")
                                }
                                leakage_external = {
                                    name.removeprefix("leakage_correct_external__")
                                    for name in archive.files
                                    if name.startswith("leakage_correct_external__")
                                }
                                if not required.issubset(archive.files):
                                    receipt_errors.append(
                                        f"candidate {candidate_index} audit NPZ required arrays missing"
                                    )
                                elif leakage_certification != REQUIRED_ATTACKERS or leakage_external != REQUIRED_ATTACKERS:
                                    receipt_errors.append(
                                        f"candidate {candidate_index} attacker arrays differ from locked portfolio"
                                    )
                                else:
                                    certification_n = int(receipt.get("indices", {}).get("certification", {}).get("n", -1))
                                    external_n = int(receipt.get("indices", {}).get("external", {}).get("n", -1))
                                    certification_names = [
                                        name
                                        for name in archive.files
                                        if name.endswith("_certification")
                                        or "_certification__" in name
                                    ]
                                    external_names = [
                                        name
                                        for name in archive.files
                                        if name.endswith("_external") or "_external__" in name
                                    ]
                                    if any(len(archive[name]) != certification_n for name in certification_names):
                                        receipt_errors.append(
                                            f"candidate {candidate_index} certification array length mismatch"
                                        )
                                    if any(len(archive[name]) != external_n for name in external_names):
                                        receipt_errors.append(
                                            f"candidate {candidate_index} external array length mismatch"
                                        )
                                    if not np.isin(
                                        archive["target_harm_certification"], (-1, 0, 1)
                                    ).all() or not np.isin(
                                        archive["target_harm_external"], (-1, 0, 1)
                                    ).all():
                                        receipt_errors.append(
                                            f"candidate {candidate_index} paired harm leaves declared support"
                                        )
                                    leakage_names = [
                                        name
                                        for name in archive.files
                                        if name.startswith("leakage_correct_")
                                    ]
                                    if any(
                                        not np.isin(archive[name], (0, 1)).all()
                                        for name in leakage_names
                                    ):
                                        receipt_errors.append(
                                            f"candidate {candidate_index} leakage correctness is non-binary"
                                        )
                        except (OSError, ValueError, KeyError) as exc:
                            receipt_errors.append(
                                f"candidate {candidate_index} audit NPZ unreadable: {exc}"
                            )
                if len(candidate_keys) != len(set(candidate_keys)):
                    receipt_errors.append("candidate keys are not unique")

                indices = receipt.get("indices", {})
                indices = indices if isinstance(indices, dict) else {}
                try:
                    signature = json.dumps(
                        {
                            name: indices[name]["sha256"]
                            for name in ("train", "construction", "certification", "external")
                        },
                        sort_keys=True,
                    )
                    split_signatures.setdefault((dataset_name, seed), set()).add(signature)
                except (KeyError, TypeError):
                    receipt_errors.append("split index hashes are incomplete")
                preprocessing = receipt.get("preprocessing", {})
                if not isinstance(preprocessing, dict) or not (
                    preprocessing.get("standardized_from_train_only") is True
                    and preprocessing.get("pca_fit_on_train_only") is True
                    and int(preprocessing.get("output_dimension", -1))
                    <= int(study.get("pca_dimension", -2))
                ):
                    receipt_errors.append("shared train-only preprocessing is invalid")
                commit = receipt.get("git_commit")
                if isinstance(commit, str) and len(commit) == 40:
                    runner_commits.add(commit)
                else:
                    receipt_errors.append("runner Git commit is missing")

                if receipt_errors:
                    invalid.append(f"{run_key}: " + "; ".join(receipt_errors))
                else:
                    valid_receipts += 1

    mismatched_splits = [
        f"{dataset}/seed-{seed}"
        for (dataset, seed), signatures in split_signatures.items()
        if len(signatures) != 1
    ]
    if mismatched_splits:
        errors.append(f"method families used different split indices: {mismatched_splits}")
    material_runner_hashes: dict[str, dict[str, str | None]] = {}
    material_runner_equivalent = False
    if runner_commits:
        for commit in sorted(runner_commits):
            material_runner_hashes[commit] = {
                relative_path: git_blob_sha256(commit, relative_path)
                for relative_path in MATERIAL_RUNNER_FILES
            }
        material_runner_equivalent = all(
            len({hashes.get(relative_path) for hashes in material_runner_hashes.values()}) == 1
            and next(iter({hashes.get(relative_path) for hashes in material_runner_hashes.values()})) is not None
            for relative_path in MATERIAL_RUNNER_FILES
        )
    if len(runner_commits) > 1 and not material_runner_equivalent:
        errors.append(
            "claim-grade runs span materially different runner code commits: "
            f"{sorted(runner_commits)}"
        )
    for repository, (commit, remote) in sorted(upstream_repositories.items()):
        path = Path(repository)
        try:
            observed_commit = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            observed_remote = subprocess.run(
                ["git", "-C", str(path), "remote", "get-url", "origin"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
            material_dirty = [
                line
                for line in dirty
                if "__pycache__" not in line
                and not line.endswith((".pyc", ".DS_Store"))
            ]
            if observed_commit != commit or observed_remote != remote or material_dirty:
                errors.append(
                    f"upstream checkout is not clean and pinned: {repository}"
                )
        except (OSError, subprocess.CalledProcessError) as exc:
            errors.append(f"could not verify upstream checkout {repository}: {exc}")

    runner_commit_contains_prereg = False
    if runner_commits:
        try:
            prereg_relative = args.prereg.resolve().relative_to(REPOSITORY.resolve()).as_posix()
        except ValueError:
            prereg_relative = ""
        prereg_hashes = {
            commit: git_blob_sha256(commit, prereg_relative)
            for commit in sorted(runner_commits)
        }
        runner_commit_contains_prereg = bool(prereg_relative) and all(
            value == actual_hash for value in prereg_hashes.values()
        )
        if not runner_commit_contains_prereg:
            errors.append("runner commits do not all contain the locked preregistration")

    expected_count = len(datasets) * len(methods) * len(seeds)
    passed = (
        bool(prereg)
        and expected_count > 0
        and valid_receipts == expected_count
        and not missing
        and not invalid
        and not errors
        and proxy_rows == 0
        and bool(runner_commits)
        and material_runner_equivalent
        and runner_commit_contains_prereg
    )
    return {
        "name": "VERA official eraser receipt audit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "prereg_sha256": actual_hash,
        "datasets": list(datasets),
        "erasers": [
            config.get("display_name")
            for config in methods.values()
            if isinstance(config, dict)
        ],
        "seeds": seeds,
        "expected_run_receipt_count": expected_count,
        "official_run_receipt_count": valid_receipts,
        "missing_run_receipt_count": len(missing),
        "invalid_receipt_count": len(invalid),
        "proxy_row_count": proxy_rows,
        "all_upstream_commits_pinned": bool(methods) and all(
            isinstance(config, dict)
            and len(str(config.get("upstream_commit", ""))) == 40
            and str(config.get("upstream_remote", "")).startswith("https://github.com/")
            for config in methods.values()
        ),
        "shared_protocol_verified": not mismatched_splits and not invalid,
        "runner_commits": sorted(runner_commits),
        "runner_commit_contains_locked_preregistration": runner_commit_contains_prereg,
        "runner_material_files": list(MATERIAL_RUNNER_FILES),
        "runner_material_file_sha256_by_commit": material_runner_hashes,
        "runner_material_code_equivalent_across_commits": material_runner_equivalent,
        "upstream_repository_count": len(upstream_repositories),
        "missing_receipts": missing,
        "invalid_receipts": invalid,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--hash-file", type=Path, default=DEFAULT_HASH)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-fail", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "valid": report["official_run_receipt_count"],
        "missing": report["missing_run_receipt_count"],
        "invalid": report["invalid_receipt_count"],
        "output": str(args.output),
    }, indent=2))
    return 0 if args.no_fail or report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
