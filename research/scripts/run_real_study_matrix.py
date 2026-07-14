"""Run or resume every claim-grade cell in the locked official-method matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_official_eraser_frontier.py"
DEFAULT_PREREG = ROOT / "prereg_real.json"
DEFAULT_HASH = ROOT / "prereg_real.sha256"
DEFAULT_RECEIPTS = ROOT / "artifacts" / "real_study_receipts"
DEFAULT_EXTERNAL = Path("/Volumes/Backups/FARO/artifacts/vera_real_study")
DEFAULT_LOGS = Path("/Volumes/Backups/FARO/artifacts/vera_real_study_logs")
DEFAULT_PROGRESS = Path("/Volumes/Backups/FARO/artifacts/vera_real_study_progress.json")
METHOD_ORDER = ("leace", "inlp", "taco", "mance", "rlace")


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


def receipt_matches(path: Path, prereg_hash: str, dataset: str, method: str, seed: int) -> bool:
    receipt = load_json(path)
    return (
        receipt.get("claim_grade") is True
        and receipt.get("smoke") is False
        and receipt.get("prereg_sha256") == prereg_hash
        and receipt.get("dataset") == dataset
        and receipt.get("method_family") == method
        and receipt.get("seed") == seed
        and receipt.get("claim_configuration_verified") is True
    )


def write_progress(path: Path, records: list[dict[str, Any]], total: int) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "complete": sum(record["status"] in {"completed", "reused"} for record in records),
        "failed": sum(record["status"] == "failed" for record in records),
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--hash-file", type=Path, default=DEFAULT_HASH)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--external-output-dir", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOGS)
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prereg = load_json(args.prereg)
    study = prereg.get("real_study", {})
    if not isinstance(study, dict) or prereg.get("status") != "locked_before_claim_grade_runs":
        raise RuntimeError("real-study preregistration is absent or unlocked")
    prereg_hash = sha256(args.prereg)
    expected_hash = args.hash_file.read_text(encoding="utf-8").split()[0]
    if prereg_hash != expected_hash:
        raise RuntimeError("real-study preregistration hash mismatch")
    datasets = study["datasets"]
    methods = study["methods"]
    seeds = [int(value) for value in study["seeds"]]
    tasks = [
        (dataset, method, seed)
        for method in METHOD_ORDER
        if method in methods
        for dataset in datasets
        for seed in seeds
    ]
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "VECLIB_MAXIMUM_THREADS": "4",
    })
    for index, (dataset, method, seed) in enumerate(tasks, start=1):
        run_key = f"{dataset}__{method}__seed-{seed}"
        receipt_path = args.receipt_dir / f"{run_key}.json"
        if receipt_matches(receipt_path, prereg_hash, dataset, method, seed):
            records.append({"run_key": run_key, "status": "reused", "seconds": 0.0})
            write_progress(args.progress, records, len(tasks))
            continue
        dataset_config = datasets[dataset]
        command = [
            sys.executable,
            str(RUNNER),
            "--dataset",
            dataset,
            "--store-dir",
            str(dataset_config["store_dir"]),
            "--method",
            method,
            "--seed",
            str(seed),
            "--max-train",
            str(study["max_train"]),
            "--max-construction",
            str(study["max_construction"]),
            "--max-certification",
            str(study["max_certification"]),
            "--max-external",
            str(study["max_external"]),
            "--dimension",
            str(study["pca_dimension"]),
            "--external-output-dir",
            str(args.external_output_dir),
            "--receipt-dir",
            str(args.receipt_dir),
            "--prereg",
            str(args.prereg),
            "--claim-grade",
        ]
        log_path = args.log_dir / f"{run_key}.log"
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"task={index}/{len(tasks)}\ncommand={json.dumps(command)}\n")
            log.flush()
            result = subprocess.run(
                command,
                cwd=ROOT.parent,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed = time.monotonic() - started
        status = "completed" if result.returncode == 0 and receipt_matches(
            receipt_path, prereg_hash, dataset, method, seed
        ) else "failed"
        records.append({
            "run_key": run_key,
            "status": status,
            "seconds": elapsed,
            "returncode": result.returncode,
            "log": str(log_path),
        })
        write_progress(args.progress, records, len(tasks))
        print(f"[{index}/{len(tasks)}] {run_key}: {status} ({elapsed:.1f}s)", flush=True)
        if status == "failed" and not args.keep_going:
            return 1
    return 0 if all(record["status"] in {"completed", "reused"} for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
