"""Run the official MANCE reference implementation on a VERA NumPy store.

This script is intentionally separate from the older "MANCE-style" proxy
baseline. It imports the upstream MANCE package from a checked-out repository
and records whether the run is claim-grade or only diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
DEFAULT_CAMELYON_STORE = Path(
    "/Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_full_numpy_store"
)
DEFAULT_MANCE_REPO = Path("/Volumes/Backups/FARO/external/mance")
DEFAULT_EXTERNAL_DIR = Path("/Volumes/Backups/FARO/artifacts/mance_reference")

SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
SPLIT_EXTERNAL = 2


@dataclass(frozen=True)
class Store:
    store_dir: Path
    manifest: dict[str, object]
    z: np.ndarray
    y: np.ndarray
    s: np.ndarray
    split: np.ndarray


@dataclass(frozen=True)
class LinearProbe:
    classes: tuple[int, int]
    coef: np.ndarray
    intercept: float


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_store(store_dir: Path) -> Store:
    manifest_path = store_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays = manifest.get("arrays", {})
    if not isinstance(arrays, dict):
        raise ValueError(f"{manifest_path} missing arrays map")
    z = np.load(store_dir / str(arrays.get("z", "z.npy")), mmap_mode="r")
    y = np.load(store_dir / str(arrays.get("y", "y.npy")), mmap_mode="r")
    s = np.load(store_dir / str(arrays.get("s", "s.npy")), mmap_mode="r")
    split = np.load(store_dir / str(arrays.get("split", "split.npy")), mmap_mode="r")
    n_examples = int(manifest.get("n_examples", -1))
    if not (z.shape[0] == y.shape[0] == s.shape[0] == split.shape[0] == n_examples):
        raise ValueError("store arrays do not match manifest n_examples")
    return Store(store_dir=store_dir, manifest=manifest, z=z, y=y, s=s, split=split)


def split_indices(split: np.ndarray, code: int) -> np.ndarray:
    return np.flatnonzero(split == code)


def value_counts(values: np.ndarray) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in zip(*np.unique(values, return_counts=True))}


def group_counts(y: np.ndarray, s: np.ndarray) -> dict[str, int]:
    counts: Counter[tuple[int, int]] = Counter((int(a), int(b)) for a, b in zip(y, s))
    return {f"y={a},s={b}": int(c) for (a, b), c in sorted(counts.items())}


def stratified_sample(
    indices: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    cap: int | None,
    seed: int,
) -> np.ndarray:
    if cap is None or cap <= 0 or cap >= len(indices):
        return np.array(indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx in indices:
        groups[(int(y[idx]), int(s[idx]))].append(int(idx))
    if not groups:
        return np.array([], dtype=np.int64)

    shuffled: dict[tuple[int, int], list[int]] = {}
    for key, values in groups.items():
        arr = np.array(values, dtype=np.int64)
        rng.shuffle(arr)
        shuffled[key] = arr.tolist()

    selected: list[int] = []
    quota = max(1, cap // len(shuffled))
    leftovers: list[int] = []
    for key in sorted(shuffled):
        values = shuffled[key]
        take = min(quota, len(values), cap - len(selected))
        selected.extend(values[:take])
        leftovers.extend(values[take:])
    if len(selected) < cap and leftovers:
        leftovers_arr = np.array(leftovers, dtype=np.int64)
        rng.shuffle(leftovers_arr)
        selected.extend(leftovers_arr[: cap - len(selected)].tolist())
    return np.sort(np.array(selected[:cap], dtype=np.int64))


def materialize_split(store: Store, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(store.z[indices], dtype=np.float32).copy()
    y = np.asarray(store.y[indices], dtype=np.int64).copy()
    s = np.asarray(store.s[indices], dtype=np.int64).copy()
    return x, y, s


def standardize(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[(~np.isfinite(std)) | (std < 1e-6)] = 1.0
    for array in (x_train, x_val, x_test):
        array -= mean
        array /= std
        np.nan_to_num(array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(array, -10.0, 10.0, out=array)
    return x_train, x_val, x_test


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = Counter(int(item) for item in labels)
    n = len(labels)
    k = max(len(counts), 1)
    weights = np.array([n / (k * counts[int(item)]) for item in labels], dtype=np.float64)
    return np.clip(weights, 0.2, 5.0)


def fit_probe(x_train: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None) -> LinearProbe:
    classes = tuple(sorted({int(item) for item in labels}))
    if len(classes) != 2:
        raise ValueError(f"weighted ridge probe expects binary labels, got {classes}")
    y_signed = np.where(labels == classes[1], 1.0, -1.0).astype(np.float64)
    if weights is None:
        sample_weight = np.ones(len(labels), dtype=np.float64)
    else:
        sample_weight = np.asarray(weights, dtype=np.float64)
    sample_weight = np.clip(sample_weight, 1e-6, np.inf)
    sample_weight = sample_weight / float(sample_weight.mean())

    d = int(x_train.shape[1])
    alpha = 10.0
    gram = np.zeros((d, d), dtype=np.float64)
    cross = np.zeros(d, dtype=np.float64)
    feature_sum = np.zeros(d, dtype=np.float64)
    label_sum = 0.0
    weight_sum = 0.0
    chunk_size = 32768
    for start in range(0, x_train.shape[0], chunk_size):
        stop = min(start + chunk_size, x_train.shape[0])
        x_chunk = np.asarray(x_train[start:stop], dtype=np.float64)
        w_chunk = sample_weight[start:stop]
        yw_chunk = y_signed[start:stop] * w_chunk
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gram += x_chunk.T @ (x_chunk * w_chunk[:, None])
            cross += x_chunk.T @ yw_chunk
            feature_sum += x_chunk.T @ w_chunk
        label_sum += float(np.sum(yw_chunk))
        weight_sum += float(np.sum(w_chunk))
    if not bool(np.isfinite(gram).all() and np.isfinite(cross).all() and np.isfinite(feature_sum).all()):
        raise ValueError("weighted ridge probe produced non-finite sufficient statistics")

    system = np.zeros((d + 1, d + 1), dtype=np.float64)
    system[:d, :d] = gram + alpha * np.eye(d, dtype=np.float64)
    system[:d, d] = feature_sum
    system[d, :d] = feature_sum
    system[d, d] = weight_sum
    rhs = np.empty(d + 1, dtype=np.float64)
    rhs[:d] = cross
    rhs[d] = label_sum
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(system, rhs, rcond=1e-8)[0]
    if not bool(np.isfinite(solution).all()):
        raise ValueError("weighted ridge probe produced non-finite coefficients")
    return LinearProbe(classes=classes, coef=solution[:d], intercept=float(solution[d]))


def predict_probe(model: LinearProbe, x_values: np.ndarray) -> np.ndarray:
    classes = np.array(model.classes, dtype=np.int64)
    output = np.empty(x_values.shape[0], dtype=np.int64)
    chunk_size = 32768
    for start in range(0, x_values.shape[0], chunk_size):
        stop = min(start + chunk_size, x_values.shape[0])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            scores = np.asarray(x_values[start:stop], dtype=np.float64) @ model.coef + model.intercept
        if not bool(np.isfinite(scores).all()):
            raise ValueError("weighted ridge probe produced non-finite decision scores")
        output[start:stop] = np.where(scores > 0.0, classes[1], classes[0])
    return output


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    classes = sorted({int(item) for item in y_true})
    if len(classes) < 2:
        return float("nan")
    recalls = []
    for cls in classes:
        mask = y_true == cls
        recalls.append(float(np.mean(y_pred[mask] == y_true[mask])))
    return float(np.mean(recalls))


def worst_target_source_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sources: np.ndarray,
) -> float:
    values: list[float] = []
    for y_value in sorted({int(item) for item in y_true}):
        for s_value in sorted({int(item) for item in sources}):
            mask = (y_true == y_value) & (sources == s_value)
            if int(mask.sum()) > 0:
                values.append(float(np.mean(y_pred[mask] == y_true[mask])))
    return min(values) if values else float("nan")


def evaluate_representation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    s_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    s_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
) -> dict[str, float | None]:
    target_probe = fit_probe(x_train, y_train, class_weights(y_train))
    pred_y_val = predict_probe(target_probe, x_val)
    pred_y_test = predict_probe(target_probe, x_test)

    source_probe = fit_probe(x_train, s_train, class_weights(s_train))
    pred_s_val = predict_probe(source_probe, x_val)
    pred_s_test = predict_probe(source_probe, x_test)
    external_source_bacc = balanced_accuracy(s_test, pred_s_test)

    return {
        "validation_target_balanced_accuracy": balanced_accuracy(y_val, pred_y_val),
        "external_target_balanced_accuracy": balanced_accuracy(y_test, pred_y_test),
        "external_worst_target_source_accuracy": worst_target_source_accuracy(
            y_test, pred_y_test, s_test
        ),
        "validation_source_leakage_balanced_accuracy": balanced_accuracy(s_val, pred_s_val),
        "external_source_leakage_balanced_accuracy": external_source_bacc
        if math.isfinite(external_source_bacc)
        else None,
    }


def metric_delta(after: dict[str, float | None], before: dict[str, float | None]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key, before_value in before.items():
        after_value = after.get(key)
        if before_value is None or after_value is None:
            out[key] = None
        else:
            out[key] = float(after_value) - float(before_value)
    return out


def git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def write_markdown(path: Path, receipt: dict[str, object]) -> None:
    metrics = receipt["metrics"]
    before = metrics["before"]
    after = metrics["after"]
    delta = metrics["delta"]
    lines = [
        "# MANCE Reference Run",
        "",
        f"- Dataset: `{receipt['dataset_name']}`",
        f"- Variant: `{receipt['reference_method']['variant']}`",
        f"- Claim-grade reference row: `{receipt['claim_grade_reference_row']}`",
        f"- Diagnostic reason: {receipt['diagnostic_reason'] or 'none'}",
        f"- Train/val/test examples: {receipt['sample']['counts']}",
        "",
        "## Metrics",
        "",
        "| Metric | Before | After | Delta |",
        "|---|---:|---:|---:|",
    ]
    for key in sorted(before):
        b = before.get(key)
        a = after.get(key)
        d = delta.get(key)
        fmt = lambda value: "" if value is None else f"{float(value):.6f}"
        lines.append(f"| `{key}` | {fmt(b)} | {fmt(a)} | {fmt(d)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(receipt["interpretation"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_CAMELYON_STORE)
    parser.add_argument("--mance-repo", type=Path, default=DEFAULT_MANCE_REPO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    parser.add_argument("--project-artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--dataset-name", default="camelyon17")
    parser.add_argument("--output-stem", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", default="mance++", choices=["mance", "mance+", "mance++"])
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--n-steps", type=int, default=3)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--scorer-hidden", type=int, default=128)
    parser.add_argument("--scorer-steps", type=int, default=120)
    parser.add_argument("--eval-hidden", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=80)
    parser.add_argument("--lambda-max", type=float, default=64.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train", type=int, default=2000)
    parser.add_argument("--max-val", type=int, default=800)
    parser.add_argument("--max-test", type=int, default=800)
    parser.add_argument("--standardize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--claim-grade",
        action="store_true",
        help="Mark the run as a claim-grade reference row. Use only for full matched runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    if not args.mance_repo.exists():
        raise FileNotFoundError(f"MANCE repo not found: {args.mance_repo}")
    sys.path.insert(0, str(args.mance_repo))
    from mance import MANCE  # noqa: PLC0415

    store = load_store(args.store_dir)
    train_all = split_indices(store.split, SPLIT_TRAIN)
    val_all = split_indices(store.split, SPLIT_VALIDATION)
    test_all = split_indices(store.split, SPLIT_EXTERNAL)
    train_idx = stratified_sample(train_all, store.y, store.s, args.max_train, args.seed + 11)
    val_idx = stratified_sample(val_all, store.y, store.s, args.max_val, args.seed + 13)
    test_idx = stratified_sample(test_all, store.y, store.s, args.max_test, args.seed + 17)

    x_train, y_train, s_train = materialize_split(store, train_idx)
    x_val, y_val, s_val = materialize_split(store, val_idx)
    x_test, y_test, s_test = materialize_split(store, test_idx)
    if args.standardize:
        x_train, x_val, x_test = standardize(x_train, x_val, x_test)

    before = evaluate_representation(
        x_train, y_train, s_train, x_val, y_val, s_val, x_test, y_test, s_test
    )

    eraser = MANCE(
        variant=args.variant,
        epsilon=args.epsilon,
        n_steps=args.n_steps,
        lambda_max=args.lambda_max,
        alpha=args.alpha,
        n_neighbors=args.n_neighbors,
        scorer_hidden=args.scorer_hidden,
        scorer_steps=args.scorer_steps,
        eval_hidden=args.eval_hidden,
        eval_steps=args.eval_steps,
        seed=args.seed,
        device=args.device,
        stop_at_floor=False,
        verbose=args.verbose,
    )
    result = eraser.fit_erase(
        x_train,
        s_train,
        x_val,
        s_val,
        x_test,
        s_test,
        control_train=y_train,
        control_val=y_val,
        control_test=y_test,
    )
    after = evaluate_representation(
        result.train, y_train, s_train, result.val, y_val, s_val, result.test, y_test, s_test
    )

    full_counts = {
        "train": int(len(train_all)),
        "validation": int(len(val_all)),
        "external": int(len(test_all)),
    }
    sampled_counts = {
        "train": int(len(train_idx)),
        "validation": int(len(val_idx)),
        "external": int(len(test_idx)),
    }
    sampled = sampled_counts != full_counts
    diagnostic_reason = ""
    if sampled:
        diagnostic_reason = "stratified subset run, not a full matched benchmark row"
    elif args.n_steps < 5:
        diagnostic_reason = "short MANCE edit schedule"

    claim_grade = bool(args.claim_grade and not diagnostic_reason)
    interpretation = (
        "This is an official-code MANCE reference diagnostic. It proves integration "
        "against the upstream implementation, but it should not be cited as a full "
        "claim-grade baseline unless rerun without caps and with a finalized schedule."
        if not claim_grade
        else "This run is labeled as a claim-grade official-code MANCE reference row."
    )

    receipt = {
        "schema": "faro_mance_reference_numpy_store_v1",
        "created_at_utc": utc_now(),
        "dataset_name": args.dataset_name,
        "store_dir": str(args.store_dir),
        "store_manifest": {
            "name": store.manifest.get("name"),
            "n_examples": store.manifest.get("n_examples"),
            "feature_count": store.manifest.get("feature_count"),
            "format": store.manifest.get("format"),
        },
        "reference_method": {
            "name": "MANCE",
            "variant": args.variant,
            "repo": str(args.mance_repo),
            "repo_url": "https://github.com/MatanAvitan/mance",
            "commit": git_commit(args.mance_repo),
            "paper_url": "https://arxiv.org/abs/2607.03973",
            "epsilon": args.epsilon,
            "n_steps": args.n_steps,
            "n_neighbors": args.n_neighbors,
            "scorer_hidden": args.scorer_hidden,
            "scorer_steps": args.scorer_steps,
            "eval_hidden": args.eval_hidden,
            "eval_steps": args.eval_steps,
            "device": args.device,
            "seed": args.seed,
        },
        "sample": {
            "strategy": "stratified_by_target_and_source_within_each_split",
            "seed": args.seed,
            "full_counts": full_counts,
            "counts": sampled_counts,
            "sampled": sampled,
            "train_y_counts": value_counts(y_train),
            "train_s_counts": value_counts(s_train),
            "train_group_counts": group_counts(y_train, s_train),
            "validation_group_counts": group_counts(y_val, s_val),
            "external_group_counts": group_counts(y_test, s_test),
        },
        "metrics": {
            "before": before,
            "after": after,
            "delta": metric_delta(after, before),
            "mance_history": result.history,
            "mance_n_neighbors": result.n_neighbors,
            "mance_rank": result.rank,
        },
        "claim_grade_reference_row": claim_grade,
        "diagnostic_reason": diagnostic_reason,
        "interpretation": interpretation,
        "runtime_seconds": round(time.time() - start, 3),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.project_artifact_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem.strip()
    if not stem:
        stem = f"{args.dataset_name}_{args.variant.replace('+', 'p')}_reference"
    if not claim_grade:
        stem += "_diagnostic"
    external_json = args.output_dir / f"{stem}_receipt.json"
    external_md = args.output_dir / f"{stem}_receipt.md"
    project_json = args.project_artifact_dir / f"{stem}_receipt.json"
    project_md = args.project_artifact_dir / f"{stem}_receipt.md"
    for path in (external_json, project_json):
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for path in (external_md, project_md):
        write_markdown(path, receipt)
    print(json.dumps({"receipt": str(project_json), "claim_grade": claim_grade}, indent=2))


if __name__ == "__main__":
    main()
