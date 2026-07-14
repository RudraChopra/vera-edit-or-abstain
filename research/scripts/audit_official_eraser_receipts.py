"""Fail-closed audit for the official five-by-five-by-five VERA run matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "prereg_real.json"
DEFAULT_HASH = ROOT / "prereg_real.sha256"
DEFAULT_RECEIPTS = ROOT / "artifacts" / "real_study_receipts"
DEFAULT_REPORT = ROOT / "artifacts" / "official_eraser_receipt_audit.json"


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
                    serialized = json.dumps(candidate).lower()
                    if "proxy" in serialized:
                        proxy_rows += 1
                        receipt_errors.append(f"candidate {candidate_index} is marked as a proxy")
                    audit_path = Path(str(candidate.get("audit_npz", "")))
                    if not audit_path.is_file():
                        receipt_errors.append(f"candidate {candidate_index} audit NPZ missing")
                    elif sha256(audit_path) != candidate.get("audit_npz_sha256"):
                        receipt_errors.append(f"candidate {candidate_index} audit NPZ hash mismatch")

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
    if len(runner_commits) > 1:
        errors.append(f"claim-grade runs span multiple runner commits: {sorted(runner_commits)}")

    expected_count = len(datasets) * len(methods) * len(seeds)
    passed = (
        bool(prereg)
        and expected_count == 125
        and valid_receipts == expected_count
        and not missing
        and not invalid
        and not errors
        and proxy_rows == 0
        and len(runner_commits) == 1
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
