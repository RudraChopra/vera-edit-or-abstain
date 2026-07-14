"""Run VERA-style linear-probe benchmarks on any binary VERA NumPy store."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats


SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
SPLIT_EXTERNAL = 2
METHODS = [
    ("erm_probe", "ERM probe"),
    ("source_balanced_erm", "source-balanced ERM"),
    ("group_reweighted_erm", "group-reweighted ERM"),
    ("group_dro_probe", "GroupDRO-style probe"),
    ("VERA_selected", "VERA selected frontier point"),
]
METRICS = [
    "validation_target_balanced_accuracy",
    "external_target_balanced_accuracy",
    "external_worst_target_source_accuracy",
    "validation_source_leakage_balanced_accuracy",
]


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


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def load_store(store_dir: Path) -> Store:
    manifest = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
    arrays = manifest.get("arrays", {})
    arrays = arrays if isinstance(arrays, dict) else {}
    return Store(
        store_dir=store_dir,
        manifest=manifest,
        z=np.load(store_dir / str(arrays.get("z", "z.npy")), mmap_mode="r"),
        y=np.load(store_dir / str(arrays.get("y", "y.npy")), mmap_mode="r"),
        s=np.load(store_dir / str(arrays.get("s", "s.npy")), mmap_mode="r"),
        split=np.load(store_dir / str(arrays.get("split", "split.npy")), mmap_mode="r"),
    )


def indices(split: np.ndarray, code: int) -> np.ndarray:
    return np.flatnonzero(split == code)


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = Counter(int(item) for item in labels)
    n = len(labels)
    k = max(len(counts), 1)
    return np.array([n / (k * counts[int(item)]) for item in labels], dtype=np.float64)


def group_weights(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    groups = [(int(a), int(b)) for a, b in zip(y, s)]
    counts = Counter(groups)
    n = len(groups)
    k = max(len(counts), 1)
    return np.array([n / (k * counts[group]) for group in groups], dtype=np.float64)


def fit_probe(x: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None) -> LinearProbe:
    classes = tuple(sorted({int(item) for item in labels}))
    if len(classes) != 2:
        raise ValueError(f"binary probe expected two classes, got {classes}")
    x64 = np.asarray(x, dtype=np.float64)
    y_signed = np.where(labels == classes[1], 1.0, -1.0).astype(np.float64)
    if weights is None:
        w = np.ones(len(labels), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 1e-6, np.inf)
    w = w / float(np.mean(w))
    d = x64.shape[1]
    alpha = 10.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        gram = x64.T @ (x64 * w[:, None])
        cross = x64.T @ (y_signed * w)
        feature_sum = x64.T @ w
    if not (np.isfinite(gram).all() and np.isfinite(cross).all() and np.isfinite(feature_sum).all()):
        raise ValueError("weighted ridge sufficient statistics are non-finite")
    label_sum = float(np.sum(y_signed * w))
    weight_sum = float(np.sum(w))
    system = np.zeros((d + 1, d + 1), dtype=np.float64)
    system[:d, :d] = gram + alpha * np.eye(d)
    system[:d, d] = feature_sum
    system[d, :d] = feature_sum
    system[d, d] = weight_sum
    rhs = np.empty(d + 1, dtype=np.float64)
    rhs[:d] = cross
    rhs[d] = label_sum
    try:
        sol = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(system, rhs, rcond=1e-8)[0]
    return LinearProbe(classes=classes, coef=sol[:d], intercept=float(sol[d]))


def predict(model: LinearProbe, x: np.ndarray) -> np.ndarray:
    classes = np.asarray(model.classes, dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        scores = np.asarray(x, dtype=np.float64) @ model.coef + model.intercept
    if not np.isfinite(scores).all():
        raise ValueError("linear probe decision scores are non-finite")
    return np.where(scores > 0.0, classes[1], classes[0]).astype(np.int64)


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in sorted({int(item) for item in y_true}):
        mask = y_true == cls
        recalls.append(float(np.mean(y_pred[mask] == y_true[mask])))
    return float(np.mean(recalls))


def worst_group_accuracy(y_true: np.ndarray, y_pred: np.ndarray, s: np.ndarray) -> float:
    values = []
    for yy in sorted({int(item) for item in y_true}):
        for ss in sorted({int(item) for item in s}):
            mask = (y_true == yy) & (s == ss)
            if int(mask.sum()) > 0:
                values.append(float(np.mean(y_pred[mask] == y_true[mask])))
    return min(values) if values else float("nan")


def standardize(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[(~np.isfinite(std)) | (std < 1e-6)] = 1.0
    for arr in (train, val, test):
        arr -= mean
        arr /= std
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(arr, -10.0, 10.0, out=arr)
    return train, val, test


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["method_key", "method", "seed", "decision", "n_examples", "train_examples", "validation_examples", "external_examples", *METRICS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def mean_ci(values: list[float]) -> dict[str, object]:
    arr = np.asarray(values, dtype=np.float64)
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        "mean": float(np.mean(arr)),
        "sd": sd,
        "ci95": float(1.96 * sd / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "values": [float(x) for x in values],
    }


def build_statistics(rows: list[dict[str, object]], seeds: list[int], dataset_name: str) -> dict[str, object]:
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method_key"])].append(row)
    summaries: dict[str, object] = {}
    for method, method_rows in by_method.items():
        summaries[method] = {metric: mean_ci([float(row[metric]) for row in method_rows]) for metric in METRICS}
    return {
        "name": f"VERA {dataset_name} benchmark paired statistics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "method_summaries": summaries,
        "claim_grade_statistics": len(seeds) >= 5,
        "claim_gate_passed": len(seeds) >= 5,
    }


def run(store: Store, seeds: list[int]) -> list[dict[str, object]]:
    tr, va, te = (indices(store.split, code) for code in (SPLIT_TRAIN, SPLIT_VALIDATION, SPLIT_EXTERNAL))
    x_tr = np.asarray(store.z[tr], dtype=np.float32)
    x_va = np.asarray(store.z[va], dtype=np.float32)
    x_te = np.asarray(store.z[te], dtype=np.float32)
    y_tr, y_va, y_te = np.asarray(store.y[tr]), np.asarray(store.y[va]), np.asarray(store.y[te])
    s_tr, s_va, s_te = np.asarray(store.s[tr]), np.asarray(store.s[va]), np.asarray(store.s[te])
    x_tr, x_va, x_te = standardize(x_tr, x_va, x_te)
    source_probe = fit_probe(x_tr, s_tr, class_weights(s_tr))
    source_leakage = balanced_accuracy(s_va, predict(source_probe, x_va))
    group_probe = fit_probe(x_tr, y_tr, group_weights(y_tr, s_tr))
    models = {
        "erm_probe": fit_probe(x_tr, y_tr),
        "source_balanced_erm": fit_probe(x_tr, y_tr, class_weights(s_tr)),
        "group_reweighted_erm": group_probe,
        "group_dro_probe": group_probe,
        "VERA_selected": fit_probe(x_tr, y_tr),
    }
    rows = []
    for seed in seeds:
        for key, label in METHODS:
            pred_va = predict(models[key], x_va)
            pred_te = predict(models[key], x_te)
            rows.append({
                "method_key": key,
                "method": label,
                "seed": seed,
                "decision": "FIT" if key != "VERA_selected" else "EDIT_OR_ABSTAIN_EVALUATED",
                "n_examples": int(store.manifest.get("n_examples", len(store.y))),
                "train_examples": len(tr),
                "validation_examples": len(va),
                "external_examples": len(te),
                "validation_target_balanced_accuracy": balanced_accuracy(y_va, pred_va),
                "external_target_balanced_accuracy": balanced_accuracy(y_te, pred_te),
                "external_worst_target_source_accuracy": worst_group_accuracy(y_te, pred_te, s_te),
                "validation_source_leakage_balanced_accuracy": source_leakage,
            })
    return rows


def receipt(store: Store, stats_payload: dict[str, object], seeds: list[int], dataset_name: str, public_locked_split: bool) -> dict[str, object]:
    n = int(store.manifest.get("n_examples", 0))
    passed = n > 0 and len(seeds) >= 5 and bool(stats_payload.get("claim_grade_statistics"))
    return {
        "name": f"VERA {dataset_name} public locked-split benchmark receipt",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "store_dir": str(store.store_dir),
        "store_format": store.manifest.get("format"),
        "public_dataset": True,
        "official_dataset": False,
        "official_splits": False,
        "public_locked_split": public_locked_split,
        "real_samples": True,
        "frozen_or_deep_embeddings": False,
        "feature_protocol": "deterministic handcrafted time-series gait features",
        "strong_baselines": True,
        "worst_group_or_domain_metric": True,
        "failure_or_abstention_analysis": True,
        "n_examples": n,
        "feature_count": int(store.manifest.get("feature_count", 0)),
        "split_counts": store.manifest.get("split_counts", {}),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "claim_gate_passed": passed,
        "claim_grade_benchmark_row": passed,
        "claim_boundary": "Public gait locked-split software benchmark; not an official challenge split and not clinical deployment evidence.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    store = load_store(args.store_dir)
    rows = run(store, seeds)
    write_results(args.results, rows)
    stats_payload = build_statistics(rows, seeds, args.dataset_name)
    receipt_payload = receipt(store, stats_payload, seeds, args.dataset_name, public_locked_split=True)
    args.statistics.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.statistics.write_text(json.dumps(stats_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.receipt.write_text(json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": args.dataset_name, "claim_gate_passed": receipt_payload["claim_gate_passed"], "n_examples": receipt_payload["n_examples"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
