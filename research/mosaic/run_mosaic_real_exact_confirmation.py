#!/usr/bin/env python3
"""Execute the hash-locked paired exact MOSAIC confirmation end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_PREREG = ROOT / "prereg_mosaic_real_exact_v1.json"
DEFAULT_OUTPUT_DIR = REPOSITORY / "research" / "artifacts" / "mosaic_real_exact_confirmation_v1"
DEFAULT_MANIFEST = REPOSITORY / "research" / "artifacts" / "mosaic_real_exact_confirmation_manifest_v1.json"
DEFAULT_AUDIT = REPOSITORY / "research" / "artifacts" / "mosaic_real_exact_confirmation_audit_v1.json"
RUNNER = ROOT / "run_mosaic_official_frontier_exact_confirmation.py"
AUDITOR = ROOT / "audit_mosaic_real_exact_frontier.py"
VARIANTS = ("capacity_transfer", "transform_exact")


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


def checked_git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_lock(path: Path) -> tuple[dict[str, object], str]:
    prereg_sha = sha256(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists() or sidecar.read_text(encoding="utf-8").strip() != prereg_sha:
        raise ValueError("preregistration sidecar does not match")
    prereg = json.loads(path.read_text(encoding="utf-8"))
    if prereg.get("status") != "locked_before_confirmatory_outcomes":
        raise ValueError("preregistration is not locked")
    for relative, expected in prereg["code_sha256"].items():
        candidate = REPOSITORY / relative
        if not candidate.exists() or sha256(candidate) != expected:
            raise ValueError(f"locked code mismatch: {relative}")
    for name, record in prereg["frozen_stores"].items():
        manifest = Path(record["path"]) / "manifest.json"
        if sha256(manifest) != record["manifest_sha256"]:
            raise ValueError(f"frozen-store manifest mismatch: {name}")
    for method, record in prereg["official_repositories"].items():
        repo = Path(record["path"])
        if checked_git(repo, "rev-parse", "HEAD") != record["commit"]:
            raise ValueError(f"official repository revision mismatch: {method}")
        if checked_git(repo, "status", "--porcelain"):
            raise ValueError(f"official repository is dirty: {method}")
    return prereg, prereg_sha


def output_name(dataset: str, seed: int) -> str:
    slug = dataset.lower().replace("-", "_").replace(" ", "_")
    return f"mosaic_real_exact_confirmation_{slug}_seed{seed}.json"


def validate_existing_output(path: Path, prereg_sha: str, dataset: str, seed: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "confirmatory_locked_protocol"
        or payload.get("prereg_sha256") != prereg_sha
        or payload.get("dataset") != dataset
        or payload.get("seed") != seed
        or len(payload.get("results", [])) != 13
        or set(payload.get("selection", {})) != set(VARIANTS)
    ):
        raise ValueError(f"existing output is not a valid resumable receipt: {path}")


def run_confirmation(
    prereg: dict[str, object],
    prereg_path: Path,
    prereg_sha: str,
    output_dir: Path,
    *,
    resume: bool,
    workers: int,
) -> list[Path]:
    if workers < 1:
        raise ValueError("workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPOSITORY / "research" / "mosaic"),
            str(REPOSITORY / "research" / "scripts"),
        ]
    )
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    outputs = []
    pending: list[tuple[str, int, Path]] = []
    for dataset in prereg["datasets"]:
        for seed in prereg["confirmation_seeds"]:
            output = output_dir / output_name(str(dataset), int(seed))
            outputs.append(output)
            if output.exists():
                if not resume:
                    raise FileExistsError(f"refusing to overwrite confirmation output: {output}")
                validate_existing_output(output, prereg_sha, str(dataset), int(seed))
                print(f"validated existing {dataset} seed {seed}", flush=True)
                continue
            pending.append((str(dataset), int(seed), output))

    def run_one(dataset: str, seed: int, output: Path) -> None:
        print(f"running {dataset} seed {seed}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--dataset",
                str(dataset),
                "--seed",
                str(seed),
                "--output",
                str(output),
                "--confirmation-prereg",
                str(prereg_path),
            ],
            cwd=REPOSITORY,
            env=environment,
            check=True,
        )
        print(f"completed {dataset} seed {seed}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one, dataset, seed, output): (dataset, seed)
            for dataset, seed, output in pending
        }
        for future in as_completed(futures):
            dataset, seed = futures[future]
            try:
                future.result()
            except Exception as error:
                raise RuntimeError(f"confirmation failed: {dataset} seed {seed}") from error
    return outputs


def run_audit(outputs: list[Path], audit: Path, *, workers: int) -> dict[str, object]:
    if audit.exists():
        audit.unlink()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPOSITORY / "research" / "mosaic"),
            str(REPOSITORY / "research" / "scripts"),
        ]
    )
    subprocess.run(
        [
            sys.executable,
            str(AUDITOR),
            *(str(path) for path in outputs),
            "--output",
            str(audit),
            "--workers",
            str(workers),
        ],
        cwd=REPOSITORY,
        env=environment,
        check=True,
    )
    return json.loads(audit.read_text(encoding="utf-8"))


def build_manifest(
    prereg: dict[str, object],
    prereg_sha: str,
    outputs: list[Path],
    audit: dict[str, object],
    audit_path: Path,
) -> dict[str, object]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in outputs]
    selections = {
        variant: [row["selection"][variant] for row in rows]
        for variant in VARIANTS
    }
    deployed = {
        variant: [
            selection
            for selection in selections[variant]
            if selection["decision"] == "deploy"
        ]
        for variant in VARIANTS
    }
    estimable_deployments = {
        variant: [
            selection
            for selection in deployed[variant]
            if selection["external_estimable"]
        ]
        for variant in VARIANTS
    }
    bias_exact = [
        row["selection"]["transform_exact"]
        for row in rows
        if row["dataset"] == "BiasBios-Clinical"
    ]
    additional_exact = [
        row
        for row in rows
        if row["selection"]["capacity_transfer"]["decision"] == "abstain"
        and row["selection"]["transform_exact"]["decision"] == "deploy"
    ]
    safe_additional_exact = [
        row
        for row in additional_exact
        if row["selection"]["transform_exact"]["external_estimable"]
        and row["selection"]["transform_exact"]["external_safe"]
    ]
    successful_results = [
        (row, result)
        for row in rows
        for result in row["results"]
        if all(isinstance(result.get(variant), dict) for variant in VARIANTS)
    ]
    missing_support_rows = [
        {
            "dataset": row["dataset"],
            "seed": row["seed"],
            "candidate_count": sum(
                not bool(result["transform_exact"]["external_estimable"])
                for result in row["results"]
                if isinstance(result.get("transform_exact"), dict)
            ),
        }
        for row in rows
        if any(
            isinstance(result.get("transform_exact"), dict)
            and not bool(result["transform_exact"]["external_estimable"])
            for result in row["results"]
        )
    ]
    registered_gates = prereg["decision_gates"]
    strict_improvements = int(audit["strict_objective_improvements"])
    gates = {
        "complete_execution": len(rows) == 25
        and all(len(row["results"]) == 13 for row in rows)
        and not any(
            "optimization_error" in result for row in rows for result in row["results"]
        ),
        "receipt_replay": bool(audit["passed"])
        and int(audit["candidate_rows_replayed"]) == 325
        and int(audit["optimization_replays"]) == 650,
        "pointwise_dominance": bool(audit["pointwise_dominance"]),
        "strict_improvement_replication": strict_improvements
        >= int(registered_gates["minimum_strict_candidate_improvements"]),
        "additional_safe_deployment": len(safe_additional_exact)
        >= int(registered_gates["minimum_additional_safe_selected_deployments"]),
        "positive_usefulness": sum(
            value["decision"] == "deploy" for value in bias_exact
        )
        >= int(registered_gates["minimum_biasbios_exact_deployments"])
        and all(
            value["external_safe"]
            for value in bias_exact
            if value["external_estimable"]
        ),
        "empirical_safety": sum(
            bool(value["false_acceptance"])
            for value in estimable_deployments["transform_exact"]
        )
        <= int(registered_gates["maximum_selected_false_acceptances"]),
        "support_discipline": all(
            not result[variant]["external_safe"]
            for _, result in successful_results
            for variant in VARIANTS
            if not result[variant]["external_estimable"]
        ),
        "modality_coverage": set(row["dataset"] for row in rows)
        == set(prereg["datasets"]),
        "claim_discipline": prereg["claim_boundary"].startswith(
            "This confirmation evaluates"
        ),
    }
    return {
        "name": "MOSAIC paired exact real-feature confirmation manifest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "prereg_sha256": prereg_sha,
        "output_count": len(outputs),
        "candidate_row_count": sum(len(row["results"]) for row in rows),
        "selected_deployments": {
            variant: len(deployed[variant]) for variant in VARIANTS
        },
        "estimable_selected_deployments": {
            variant: len(estimable_deployments[variant]) for variant in VARIANTS
        },
        "selected_false_acceptances": {
            variant: sum(
                bool(value["false_acceptance"])
                for value in estimable_deployments[variant]
            )
            for variant in VARIANTS
        },
        "strict_candidate_improvements": strict_improvements,
        "additional_exact_selected_deployments": len(additional_exact),
        "additional_safe_exact_selected_deployments": len(safe_additional_exact),
        "biasbios_exact_deployments": sum(
            value["decision"] == "deploy" for value in bias_exact
        ),
        "missing_support_rows": missing_support_rows,
        "gates": gates,
        "all_pass": all(gates.values()),
        "audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        "outputs": [
            {
                "path": str(path),
                "sha256": sha256(path),
                "dataset": row["dataset"],
                "seed": row["seed"],
                "selection": row["selection"],
            }
            for path, row in zip(outputs, rows, strict=True)
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prereg, prereg_sha = verify_lock(args.prereg)
    if args.verify_only:
        print(
            json.dumps(
                {
                    "verified": True,
                    "prereg": str(args.prereg),
                    "prereg_sha256": prereg_sha,
                    "datasets": list(prereg["datasets"]),
                    "confirmation_seeds": prereg["confirmation_seeds"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.manifest.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite confirmation manifest: {args.manifest}")
    outputs = run_confirmation(
        prereg,
        args.prereg,
        prereg_sha,
        args.output_dir,
        resume=args.resume,
        workers=args.workers,
    )
    audit = run_audit(outputs, args.audit, workers=args.workers)
    manifest = build_manifest(prereg, prereg_sha, outputs, audit, args.audit)
    atomic_json_dump(manifest, args.manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not manifest["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
