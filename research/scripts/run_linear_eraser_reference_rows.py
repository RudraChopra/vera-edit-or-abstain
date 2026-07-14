"""Run real linear-erasure reference rows on VERA frozen-feature stores.

This script adds the cheap rows that a reviewer would reasonably expect on
frozen representations: no edit, INLP, and the official LEACE implementation.
It intentionally does not label R-LACE as complete; R-LACE remains a pinned
upstream baseline until its official minimax optimization is run under matched
splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
DEFAULT_LEACE_REPO = Path("/Volumes/Backups/FARO/external/concept-erasure")
SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
SPLIT_EXTERNAL = 2
METRICS = [
    "validation_target_balanced_accuracy",
    "external_target_balanced_accuracy",
    "external_worst_target_source_accuracy",
    "validation_source_leakage_balanced_accuracy",
    "external_source_leakage_balanced_accuracy",
]


@dataclass(frozen=True)
class Store:
    name: str
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


def load_leace(leace_repo: Path):
    sys.path.insert(0, str(leace_repo))
    from concept_erasure import LeaceFitter  # type: ignore

    return LeaceFitter


def load_store(name: str, store_dir: Path) -> Store:
    manifest = json.loads((store_dir / "manifest.json").read_text(encoding="utf-8"))
    arrays = manifest.get("arrays", {})
    arrays = arrays if isinstance(arrays, dict) else {}
    return Store(
        name=name,
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
    weights = np.array([n / (k * counts[int(item)]) for item in labels], dtype=np.float64)
    return np.clip(weights, 1e-6, np.inf)


def fit_probe(x: np.ndarray, labels: np.ndarray, weights: np.ndarray | None = None) -> LinearProbe:
    classes = tuple(sorted({int(item) for item in labels}))
    if len(classes) != 2:
        raise ValueError(f"binary probe expected two classes, got {classes}")
    x64 = np.asarray(x, dtype=np.float64)
    signed = np.where(labels == classes[1], 1.0, -1.0).astype(np.float64)
    if weights is None:
        w = np.ones(len(labels), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
    w = w / float(np.mean(w))
    d = int(x64.shape[1])
    alpha = 10.0
    gram = x64.T @ (x64 * w[:, None])
    cross = x64.T @ (signed * w)
    feature_sum = x64.T @ w
    label_sum = float(np.sum(signed * w))
    weight_sum = float(np.sum(w))
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
    if not np.isfinite(solution).all():
        raise ValueError("linear probe produced non-finite coefficients")
    return LinearProbe(classes=classes, coef=solution[:d], intercept=float(solution[d]))


def predict(model: LinearProbe, x: np.ndarray, chunk_size: int = 32768) -> np.ndarray:
    classes = np.asarray(model.classes, dtype=np.int64)
    out = np.empty(x.shape[0], dtype=np.int64)
    for start in range(0, x.shape[0], chunk_size):
        stop = min(start + chunk_size, x.shape[0])
        scores = np.asarray(x[start:stop], dtype=np.float64) @ model.coef + model.intercept
        if not np.isfinite(scores).all():
            raise ValueError("linear probe produced non-finite scores")
        out[start:stop] = np.where(scores > 0.0, classes[1], classes[0])
    return out


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in sorted({int(item) for item in y_true}):
        mask = y_true == cls
        if int(mask.sum()) > 0:
            recalls.append(float(np.mean(y_pred[mask] == y_true[mask])))
    return float(np.mean(recalls)) if recalls else float("nan")


def worst_group_accuracy(y_true: np.ndarray, y_pred: np.ndarray, source: np.ndarray) -> float:
    values = []
    for yy in sorted({int(item) for item in y_true}):
        for ss in sorted({int(item) for item in source}):
            mask = (y_true == yy) & (source == ss)
            if int(mask.sum()) > 0:
                values.append(float(np.mean(y_pred[mask] == y_true[mask])))
    return float(min(values)) if values else float("nan")


def standardize(train: np.ndarray, validation: np.ndarray, external: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[(~np.isfinite(std)) | (std < 1e-6)] = 1.0
    for array in (train, validation, external):
        array -= mean
        array /= std
        np.nan_to_num(array, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(array, -10.0, 10.0, out=array)
    return train, validation, external


def projection_from_vector(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.eye(v.shape[0], dtype=np.float64)
    u = v / norm
    return np.eye(v.shape[0], dtype=np.float64) - np.outer(u, u)


def apply_projection(x: np.ndarray, projection: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float64) @ projection).astype(np.float32)


def inlp_projection(train: np.ndarray, source: np.ndarray, rank: int) -> np.ndarray:
    projection = np.eye(train.shape[1], dtype=np.float64)
    edited = np.asarray(train, dtype=np.float64).copy()
    for _ in range(rank):
        source_probe = fit_probe(edited, source, class_weights(source))
        step = projection_from_vector(source_probe.coef)
        projection = projection @ step
        edited = edited @ step
    return projection


def one_hot(labels: np.ndarray) -> torch.Tensor:
    classes = sorted({int(item) for item in labels})
    index = {cls: i for i, cls in enumerate(classes)}
    arr = np.zeros((len(labels), len(classes)), dtype=np.float32)
    for row, label in enumerate(labels):
        arr[row, index[int(label)]] = 1.0
    return torch.from_numpy(arr)


def leace_apply(
    fitter_cls,
    train: np.ndarray,
    validation: np.ndarray,
    external: np.ndarray,
    source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_train = torch.from_numpy(np.asarray(train, dtype=np.float32))
    z_source = one_hot(source)
    eraser = fitter_cls.fit(x_train, z_source, method="leace").eraser
    outputs = []
    with torch.no_grad():
        for array in (train, validation, external):
            outputs.append(eraser(torch.from_numpy(np.asarray(array, dtype=np.float32))).cpu().numpy().astype(np.float32))
    return outputs[0], outputs[1], outputs[2]


def external_source_leakage(train: np.ndarray, external: np.ndarray, train_source: np.ndarray, external_source: np.ndarray) -> float:
    if len(set(int(item) for item in external_source)) < 2:
        return float("nan")
    source_probe = fit_probe(train, train_source, class_weights(train_source))
    return balanced_accuracy(external_source, predict(source_probe, external))


def evaluate_method(
    dataset: str,
    method_key: str,
    method: str,
    train: np.ndarray,
    validation: np.ndarray,
    external: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    y_external: np.ndarray,
    s_train: np.ndarray,
    s_validation: np.ndarray,
    s_external: np.ndarray,
    metadata: dict[str, object],
) -> dict[str, object]:
    target_probe = fit_probe(train, y_train)
    source_probe = fit_probe(train, s_train, class_weights(s_train))
    validation_pred = predict(target_probe, validation)
    external_pred = predict(target_probe, external)
    source_validation_pred = predict(source_probe, validation)
    ext_source = external_source_leakage(train, external, s_train, s_external)
    return {
        "dataset": dataset,
        "method_key": method_key,
        "method": method,
        "official_code": bool(metadata.get("official_code", False)),
        "rank": metadata.get("rank", ""),
        "selection_role": metadata.get("selection_role", "reference_eraser"),
        "validation_target_balanced_accuracy": balanced_accuracy(y_validation, validation_pred),
        "external_target_balanced_accuracy": balanced_accuracy(y_external, external_pred),
        "external_worst_target_source_accuracy": worst_group_accuracy(y_external, external_pred, s_external),
        "validation_source_leakage_balanced_accuracy": balanced_accuracy(s_validation, source_validation_pred),
        "external_source_leakage_balanced_accuracy": ext_source,
    }


def run_store(store: Store, fitter_cls, inlp_ranks: Iterable[int]) -> list[dict[str, object]]:
    train_idx = indices(store.split, SPLIT_TRAIN)
    validation_idx = indices(store.split, SPLIT_VALIDATION)
    external_idx = indices(store.split, SPLIT_EXTERNAL)
    train = np.asarray(store.z[train_idx], dtype=np.float32)
    validation = np.asarray(store.z[validation_idx], dtype=np.float32)
    external = np.asarray(store.z[external_idx], dtype=np.float32)
    y_train = np.asarray(store.y[train_idx])
    y_validation = np.asarray(store.y[validation_idx])
    y_external = np.asarray(store.y[external_idx])
    s_train = np.asarray(store.s[train_idx])
    s_validation = np.asarray(store.s[validation_idx])
    s_external = np.asarray(store.s[external_idx])
    train, validation, external = standardize(train, validation, external)

    rows = [
        evaluate_method(
            store.name,
            "no_edit",
            "No edit",
            train,
            validation,
            external,
            y_train,
            y_validation,
            y_external,
            s_train,
            s_validation,
            s_external,
            {"selection_role": "reference_no_edit"},
        )
    ]
    for rank in inlp_ranks:
        projection = inlp_projection(train, s_train, int(rank))
        rows.append(
            evaluate_method(
                store.name,
                f"inlp_rank{rank}",
                f"INLP rank {rank}",
                apply_projection(train, projection),
                apply_projection(validation, projection),
                apply_projection(external, projection),
                y_train,
                y_validation,
                y_external,
                s_train,
                s_validation,
                s_external,
                {"rank": int(rank), "official_code": False},
            )
        )
    leace_train, leace_validation, leace_external = leace_apply(
        fitter_cls,
        train,
        validation,
        external,
        s_train,
    )
    rows.append(
        evaluate_method(
            store.name,
            "leace_official",
            "LEACE official",
            leace_train,
            leace_validation,
            leace_external,
            y_train,
            y_validation,
            y_external,
            s_train,
            s_validation,
            s_external,
            {"official_code": True},
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["dataset", "method_key", "method", "official_code", "rank", "selection_role", *METRICS]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def write_markdown(path: Path, report: dict[str, object]) -> None:
    lines = [
        "# VERA Linear Eraser Reference Rows",
        "",
        f"Generated at UTC: `{report['created_at_utc']}`",
        f"Reference rows ready: `{report['reference_rows_ready']}`",
        "",
        "| Dataset | Method | Target BA ext. | Worst group ext. | Source leak val. | Source leak ext. |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["rows"]:
        ext_source = finite_or_none(row.get("external_source_leakage_balanced_accuracy"))
        lines.append(
            "| {dataset} | {method} | {target:.4f} | {worst:.4f} | {leak:.4f} | {ext_leak} |".format(
                dataset=row["dataset"],
                method=row["method"],
                target=float(row["external_target_balanced_accuracy"]),
                worst=float(row["external_worst_target_source_accuracy"]),
                leak=float(row["validation_source_leakage_balanced_accuracy"]),
                ext_leak="" if ext_source is None else f"{ext_source:.4f}",
            )
        )
    lines.extend(
        [
            "",
            "R-LACE is intentionally not marked complete in this report. The upstream",
            "repository is pinned locally, but a matched official minimax run is still",
            "required before claiming a real R-LACE reference row.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leace-repo", type=Path, default=DEFAULT_LEACE_REPO)
    parser.add_argument("--waterbirds-store", type=Path, default=Path("/Volumes/Backups/FARO/artifacts/waterbirds_official_numpy_store"))
    parser.add_argument("--camelyon-store", type=Path, default=Path("/Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_full_numpy_store"))
    parser.add_argument("--gait-store", type=Path, default=Path("/Volumes/Backups/FARO/artifacts/gaitpdb_numpy_store"))
    parser.add_argument("--datasets", default="waterbirds,camelyon17,gaitpdb")
    parser.add_argument("--inlp-ranks", default="1,4")
    parser.add_argument("--csv-out", type=Path, default=ARTIFACT_DIR / "linear_eraser_reference_rows.csv")
    parser.add_argument("--json-out", type=Path, default=ARTIFACT_DIR / "linear_eraser_reference_report.json")
    parser.add_argument("--markdown-out", type=Path, default=ARTIFACT_DIR / "linear_eraser_reference_report.md")
    args = parser.parse_args()

    fitter_cls = load_leace(args.leace_repo)
    ranks = [int(item.strip()) for item in args.inlp_ranks.split(",") if item.strip()]
    store_map = {
        "waterbirds": args.waterbirds_store,
        "camelyon17": args.camelyon_store,
        "gaitpdb": args.gait_store,
    }
    rows: list[dict[str, object]] = []
    for dataset in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        rows.extend(run_store(load_store(dataset, store_map[dataset]), fitter_cls, ranks))
    datasets = sorted({str(row["dataset"]) for row in rows})
    methods_by_dataset: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        methods_by_dataset[str(row["dataset"])].add(str(row["method_key"]))
    ready = all({"no_edit", "inlp_rank1", "inlp_rank4", "leace_official"}.issubset(methods_by_dataset[dataset]) for dataset in datasets)
    report = {
        "name": "VERA linear eraser reference row report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_rows_ready": ready,
        "datasets": datasets,
        "inlp_ranks": ranks,
        "leace_repo": str(args.leace_repo),
        "leace_official_code": True,
        "inlp_real_closed_form": True,
        "rlace_reference_row_ready": False,
        "rlace_boundary": "Pinned upstream repository exists, but no matched official R-LACE minimax receipt is claimed.",
        "rows": rows,
    }
    write_csv(args.csv_out, rows)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.markdown_out, report)
    print("VERA linear eraser reference rows complete")
    print(f"reference_rows_ready={str(ready).lower()}")
    print(f"rows={len(rows)}")
    print(f"report={args.json_out}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
