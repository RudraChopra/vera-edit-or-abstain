"""Run or resume a locked official-method matrix with bounded parallelism."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_official_eraser_frontier.py"
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
    ordered = sorted(records, key=lambda record: int(record["task_index"]))
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total": total,
        "complete": sum(record["status"] in {"completed", "reused"} for record in ordered),
        "failed": sum(record["status"] == "failed" for record in ordered),
        "records": ordered,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--hash-file", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--external-output-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--keep-going", action="store_true")
    return parser.parse_args()


def run_task(
    task_index: int,
    total: int,
    dataset: str,
    method: str,
    seed: int,
    study: dict[str, Any],
    prereg_hash: str,
    args: argparse.Namespace,
    environment: dict[str, str],
) -> dict[str, Any]:
    run_key = f"{dataset}__{method}__seed-{seed}"
    receipt_path = args.receipt_dir / f"{run_key}.json"
    if receipt_matches(receipt_path, prereg_hash, dataset, method, seed):
        return {
            "task_index": task_index,
            "run_key": run_key,
            "status": "reused",
            "seconds": 0.0,
        }

    dataset_config = study["datasets"][dataset]
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
        log.write(f"task={task_index}/{total}\ncommand={json.dumps(command)}\n")
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
    return {
        "task_index": task_index,
        "run_key": run_key,
        "status": status,
        "seconds": elapsed,
        "returncode": result.returncode,
        "log": str(log_path),
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.threads_per_worker < 1:
        raise ValueError("workers and threads-per-worker must be positive")

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
    args.external_output_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    thread_count = str(args.threads_per_worker)
    environment.update({
        "OMP_NUM_THREADS": thread_count,
        "MKL_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
        "VECLIB_MAXIMUM_THREADS": thread_count,
    })

    records: list[dict[str, Any]] = []
    failed = False
    futures: dict[Future[dict[str, Any]], int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for task_index, (dataset, method, seed) in enumerate(tasks, start=1):
            future = executor.submit(
                run_task,
                task_index,
                len(tasks),
                dataset,
                method,
                seed,
                study,
                prereg_hash,
                args,
                environment,
            )
            futures[future] = task_index

        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            write_progress(args.progress, records, len(tasks))
            print(
                f"[{len(records)}/{len(tasks)}] {record['run_key']}: "
                f"{record['status']} ({record['seconds']:.1f}s)",
                flush=True,
            )
            if record["status"] == "failed":
                failed = True
                if not args.keep_going:
                    for pending in futures:
                        pending.cancel()
                    break

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
