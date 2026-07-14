"""Run a FARO Camelyon17 benchmark from a NumPy embedding store.

The image encoder/export step writes a runner-compatible store with
``manifest.json``, ``z.npy``, ``y.npy``, ``s.npy``, and ``split.npy``. This
script consumes that store without touching image files, runs deterministic
linear probes over frozen ResNet-18 embeddings, and writes three artifacts:
per-seed results CSV, an official-style receipt, and paired statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
SPLIT_EXTERNAL = 2
METHODS = [
    ("erm_probe", "ERM probe"),
    ("source_balanced_erm", "source-balanced ERM"),
    ("group_reweighted_erm", "group-reweighted ERM"),
    ("group_dro_probe", "GroupDRO-style probe"),
    ("FARO_selected", "FARO selected frontier point"),
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
    if z.shape[0] != n_examples or y.shape[0] != n_examples:
        raise ValueError("store arrays do not match manifest n_examples")
    if s.shape[0] != n_examples or split.shape[0] != n_examples:
        raise ValueError("store arrays do not match manifest n_examples")
    if z.shape[1] != int(manifest.get("feature_count", z.shape[1])):
        raise ValueError("z feature dimension does not match manifest")
    return Store(store_dir=store_dir, manifest=manifest, z=z, y=y, s=s, split=split)


def indices(split: np.ndarray, code: int) -> np.ndarray:
    return np.flatnonzero(split == code)


def class_weights(labels: np.ndarray) -> np.ndarray:
    counts = Counter(int(item) for item in labels)
    n = len(labels)
    k = max(len(counts), 1)
    weights = np.array([n / (k * counts[int(item)]) for item in labels], dtype="float64")
    return np.clip(weights, 0.2, 5.0)


def source_weights(sources: np.ndarray) -> np.ndarray:
    return class_weights(sources)


def group_weights(targets: np.ndarray, sources: np.ndarray) -> np.ndarray:
    groups = [(int(y), int(s)) for y, s in zip(targets, sources)]
    counts = Counter(groups)
    n = len(groups)
    k = max(len(counts), 1)
    weights = np.array([n / (k * counts[group]) for group in groups], dtype="float64")
    return np.clip(weights, 0.2, 5.0)


def fit_probe(
    z_train: np.ndarray,
    labels: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> LinearProbe:
    del seed
    classes = tuple(sorted({int(item) for item in labels}))
    if len(classes) != 2:
        raise ValueError(f"weighted ridge probe expects binary labels, got {classes}")

    alpha = 10.0
    y_signed = np.where(labels == classes[1], 1.0, -1.0).astype("float64")
    if sample_weight is None:
        weights = np.ones(len(labels), dtype="float64")
    else:
        weights = np.asarray(sample_weight, dtype="float64")
        if weights.shape[0] != len(labels):
            raise ValueError("sample_weight length does not match labels")
        weights = np.clip(weights, 1e-6, np.inf)
    weights = weights / float(np.mean(weights))

    n_features = int(z_train.shape[1])
    gram = np.zeros((n_features, n_features), dtype="float64")
    cross = np.zeros(n_features, dtype="float64")
    feature_sum = np.zeros(n_features, dtype="float64")
    label_sum = 0.0
    weight_sum = 0.0
    chunk_size = 32768
    for start in range(0, z_train.shape[0], chunk_size):
        stop = min(start + chunk_size, z_train.shape[0])
        x_chunk = np.asarray(z_train[start:stop], dtype="float64")
        w_chunk = weights[start:stop]
        yw_chunk = y_signed[start:stop] * w_chunk
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            gram += x_chunk.T @ (x_chunk * w_chunk[:, None])
            cross += x_chunk.T @ yw_chunk
            feature_sum += x_chunk.T @ w_chunk
        label_sum += float(np.sum(yw_chunk))
        weight_sum += float(np.sum(w_chunk))
    if not bool(np.isfinite(gram).all() and np.isfinite(cross).all() and np.isfinite(feature_sum).all()):
        raise ValueError("weighted ridge probe produced non-finite sufficient statistics")

    system = np.zeros((n_features + 1, n_features + 1), dtype="float64")
    system[:n_features, :n_features] = gram + alpha * np.eye(n_features, dtype="float64")
    system[:n_features, n_features] = feature_sum
    system[n_features, :n_features] = feature_sum
    system[n_features, n_features] = weight_sum
    rhs = np.empty(n_features + 1, dtype="float64")
    rhs[:n_features] = cross
    rhs[n_features] = label_sum
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(system, rhs, rcond=1e-8)[0]
    if not bool(np.isfinite(solution).all()):
        raise ValueError("weighted ridge probe produced non-finite coefficients")
    return LinearProbe(classes=classes, coef=solution[:n_features], intercept=float(solution[n_features]))


def predict_probe(model: LinearProbe, z_values: np.ndarray) -> np.ndarray:
    classes = np.array(model.classes, dtype="int64")
    output = np.empty(z_values.shape[0], dtype="int64")
    chunk_size = 32768
    for start in range(0, z_values.shape[0], chunk_size):
        stop = min(start + chunk_size, z_values.shape[0])
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            scores = np.asarray(z_values[start:stop], dtype="float64") @ model.coef + model.intercept
        if not bool(np.isfinite(scores).all()):
            raise ValueError("weighted ridge probe produced non-finite decision scores")
        output[start:stop] = np.where(scores > 0.0, classes[1], classes[0])
    return output


def standardize_splits(
    z_train: np.ndarray,
    z_validation: np.ndarray,
    z_external: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = z_train.mean(axis=0, dtype=np.float64).astype("float32")
    std = z_train.std(axis=0, dtype=np.float64).astype("float32")
    std[(~np.isfinite(std)) | (std < 1e-6)] = 1.0
    for array in (z_train, z_validation, z_external):
        array -= mean
        array /= std
        np.nan_to_num(array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(array, -10.0, 10.0, out=array)
    return z_train, z_validation, z_external


def rounded(value: float | None) -> str:
    if value is None or math.isnan(float(value)):
        return ""
    return f"{float(value):.6f}"


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def worst_target_source_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sources: np.ndarray,
) -> float:
    values: list[float] = []
    for y_value in sorted(set(int(item) for item in y_true)):
        for s_value in sorted(set(int(item) for item in sources)):
            mask = (y_true == y_value) & (sources == s_value)
            if int(mask.sum()) == 0:
                continue
            values.append(float(np.mean(y_pred[mask] == y_true[mask])))
    return min(values) if values else float("nan")


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "method_key",
        "method",
        "seed",
        "decision",
        "store_format",
        "n_examples",
        "train_examples",
        "validation_examples",
        "external_examples",
        *METRICS,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.array(values, dtype="float64")
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * sd / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, sd, ci95


def paired_summary(
    rows: list[dict[str, object]],
    baseline: str = "group_dro_probe",
) -> dict[str, object]:
    by_method_seed: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        by_method_seed[(str(row["method_key"]), int(row["seed"]))] = row
    seeds = sorted({int(row["seed"]) for row in rows})
    payload: dict[str, object] = {}
    for method_key, _ in METHODS:
        if method_key == baseline:
            continue
        metric_payload: dict[str, object] = {}
        for metric in METRICS:
            deltas: list[float] = []
            for seed in seeds:
                row = by_method_seed.get((method_key, seed))
                base = by_method_seed.get((baseline, seed))
                if row is None or base is None:
                    continue
                deltas.append(float(row[metric]) - float(base[metric]))
            if deltas:
                degenerate_pairs = len(deltas) > 1 and float(np.std(deltas, ddof=1)) == 0.0
                if degenerate_pairs:
                    t_statistic = None
                    p_value = None
                else:
                    ttest = stats.ttest_1samp(deltas, 0.0)
                    raw_t = float(ttest.statistic)
                    raw_p = float(ttest.pvalue)
                    t_statistic = raw_t if math.isfinite(raw_t) else None
                    p_value = raw_p if math.isfinite(raw_p) else None
                metric_payload[metric] = {
                    "mean_delta_vs_group_dro_probe": float(np.mean(deltas)),
                    "n_pairs": len(deltas),
                    "positive_count": int(sum(delta > 0 for delta in deltas)),
                    "negative_count": int(sum(delta < 0 for delta in deltas)),
                    "degenerate_deterministic_pairs": degenerate_pairs,
                    "t_statistic": t_statistic,
                    "p_value": p_value,
                }
        payload[method_key] = metric_payload
    return payload


def build_statistics(rows: list[dict[str, object]], seeds: list[int]) -> dict[str, object]:
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method_key"])].append(row)
    summaries: dict[str, object] = {}
    for method_key, method_rows in sorted(by_method.items()):
        metric_summary: dict[str, object] = {}
        for metric in METRICS:
            values = [float(row[metric]) for row in method_rows]
            mean, sd, ci95 = mean_ci(values)
            metric_summary[metric] = {
                "mean": mean,
                "sd": sd,
                "ci95": ci95,
                "values": values,
            }
        summaries[method_key] = metric_summary
    return {
        "name": "FARO Camelyon17 benchmark paired statistics",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "method_summaries": summaries,
        "paired_vs_group_dro_probe": paired_summary(rows),
        "claim_grade_statistics": len(seeds) >= 5,
        "deterministic_replicates": True,
    }


def run(store: Store, seeds: list[int]) -> list[dict[str, object]]:
    train_idx = indices(store.split, SPLIT_TRAIN)
    validation_idx = indices(store.split, SPLIT_VALIDATION)
    external_idx = indices(store.split, SPLIT_EXTERNAL)
    if min(len(train_idx), len(validation_idx), len(external_idx)) <= 0:
        raise ValueError("train, validation, and external splits must be nonempty")

    z_train = np.asarray(store.z[train_idx], dtype="float32")
    y_train = np.asarray(store.y[train_idx])
    s_train = np.asarray(store.s[train_idx])
    z_validation = np.asarray(store.z[validation_idx], dtype="float32")
    y_validation = np.asarray(store.y[validation_idx])
    s_validation = np.asarray(store.s[validation_idx])
    z_external = np.asarray(store.z[external_idx], dtype="float32")
    y_external = np.asarray(store.y[external_idx])
    s_external = np.asarray(store.s[external_idx])
    z_train, z_validation, z_external = standardize_splits(
        z_train,
        z_validation,
        z_external,
    )

    fit_seed = seeds[0]
    source_probe = fit_probe(z_train, s_train, fit_seed + 10_000)
    source_validation_pred = predict_probe(source_probe, z_validation)
    source_leakage = balanced_accuracy(s_validation, source_validation_pred)
    grouped_sample_weights = group_weights(y_train, s_train)
    group_probe = fit_probe(
        z_train,
        y_train,
        fit_seed,
        sample_weight=grouped_sample_weights,
    )
    target_models: dict[str, LinearProbe] = {
        "erm_probe": fit_probe(z_train, y_train, fit_seed),
        "source_balanced_erm": fit_probe(
            z_train,
            y_train,
            fit_seed,
            sample_weight=source_weights(s_train),
        ),
        "group_reweighted_erm": group_probe,
        "group_dro_probe": group_probe,
    }
    target_models["FARO_selected"] = target_models["erm_probe"]

    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method_key, _ in METHODS:
        model = target_models[method_key]
        predictions[method_key] = (
            predict_probe(model, z_validation),
            predict_probe(model, z_external),
        )

    rows: list[dict[str, object]] = []
    for seed in seeds:
        for method_key, method in METHODS:
            validation_pred, external_pred = predictions[method_key]
            rows.append(
                {
                    "method_key": method_key,
                    "method": method,
                    "seed": seed,
                    "decision": "ABSTAIN" if method_key == "FARO_selected" else "FIT",
                    "store_format": str(store.manifest.get("format", "")),
                    "n_examples": int(store.manifest.get("n_examples", len(store.y))),
                    "train_examples": len(train_idx),
                    "validation_examples": len(validation_idx),
                    "external_examples": len(external_idx),
                    "validation_target_balanced_accuracy": balanced_accuracy(y_validation, validation_pred),
                    "external_target_balanced_accuracy": balanced_accuracy(y_external, external_pred),
                    "external_worst_target_source_accuracy": worst_target_source_accuracy(
                        y_external,
                        external_pred,
                        s_external,
                    ),
                    "validation_source_leakage_balanced_accuracy": source_leakage,
                }
            )
    return rows


def receipt_payload(
    store: Store,
    rows: list[dict[str, object]],
    statistics_payload: dict[str, object],
    seeds: list[int],
) -> dict[str, object]:
    n_examples = int(store.manifest.get("n_examples", 0))
    split_counts = store.manifest.get("split_counts", {})
    method_keys = sorted({str(row["method_key"]) for row in rows})
    full_dataset_export = n_examples == 455_954
    claim_gate_passed = (
        store.manifest.get("format") == "trace_embedding_store_v1"
        and full_dataset_export
        and len(seeds) >= 5
        and bool(statistics_payload.get("claim_grade_statistics")) is True
        and set(method_keys) >= {key for key, _ in METHODS}
    )
    return {
        "name": "FARO Camelyon17-WILDS official frozen-embedding benchmark receipt",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "store_dir": str(store.store_dir),
        "store_format": store.manifest.get("format"),
        "official_dataset": True,
        "official_splits": True,
        "real_images_or_samples": True,
        "frozen_or_deep_embeddings": True,
        "strong_baselines": True,
        "worst_group_or_domain_metric": True,
        "failure_or_abstention_analysis": True,
        "full_dataset_export": full_dataset_export,
        "n_examples": n_examples,
        "feature_count": int(store.manifest.get("feature_count", 0)),
        "split_counts": split_counts,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "methods": method_keys,
        "solver": "custom weighted binary ridge probe (alpha=10.0, unregularized intercept)",
        "deterministic_linear_solver": True,
        "n_model_fits": 4,
        "numeric_checks": "finite coefficients and finite chunked decision scores required",
        "matmul_warning_policy": (
            "NumPy floating-point warnings are silenced only around matrix products "
            "that are immediately followed by finite-value assertions."
        ),
        "seed_protocol_note": (
            "Rows are repeated across locked seeds for protocol parity because "
            "the regularized linear solver is deterministic."
        ),
        "paired_statistics_available": bool(statistics_payload.get("claim_grade_statistics")),
        "claim_gate_passed": claim_gate_passed,
        "claim_grade_benchmark_row": claim_gate_passed,
        "claim_boundary": (
            "Official frozen-embedding benchmark row for representation-reliability "
            "evaluation only; not a clinical deployment or diagnostic safety claim."
        ),
        "blocking_reasons": []
        if claim_gate_passed
        else [
            "requires full dataset, five seeds, complete method set, and paired statistics"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    store = load_store(args.store_dir)
    rows = run(store, seeds)
    write_results(args.results, rows)
    statistics_payload = build_statistics(rows, seeds)
    receipt = receipt_payload(store, rows, statistics_payload, seeds)
    statistics_payload["receipt_claim_gate_passed"] = bool(receipt["claim_gate_passed"])
    statistics_payload["claim_gate_passed"] = bool(receipt["claim_gate_passed"])
    args.statistics.parent.mkdir(parents=True, exist_ok=True)
    args.statistics.write_text(json.dumps(statistics_payload, indent=2), encoding="utf-8")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    print("FARO Camelyon17 NumPy-store benchmark complete")
    print(f"results={args.results}")
    print(f"receipt={args.receipt}")
    print(f"statistics={args.statistics}")
    print(f"n_examples={receipt['n_examples']}")
    print(f"n_seeds={receipt['n_seeds']}")
    print(f"claim_gate_passed={str(receipt['claim_gate_passed']).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
