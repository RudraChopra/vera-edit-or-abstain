"""Build a full Camelyon17 VERA projection-frontier certificate.

This script uses the claim-grade Camelyon17 frozen ResNet-18 NumPy store and
tests a transparent source-direction projection family:

    z(lambda) = z - lambda <z, u_s> u_s

where u_s is the unit linear-probe direction for the hospital/source label on
the training split. VERA selection is performed only on validation metrics. The
external split is reported after the selected decision is fixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run_camelyon17_numpy_store_benchmark import (
    SPLIT_EXTERNAL,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    balanced_accuracy,
    class_weights,
    fit_probe,
    indices,
    load_store,
    predict_probe,
    standardize_splits,
    worst_target_source_accuracy,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
DEFAULT_STORE = Path("/Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_full_numpy_store")
DEFAULT_FRONTIER = ARTIFACT_DIR / "camelyon17_faro_projection_frontier.csv"
DEFAULT_CERTIFICATE = ARTIFACT_DIR / "camelyon17_faro_projection_certificate.json"
DEFAULT_MARKDOWN = ARTIFACT_DIR / "camelyon17_faro_projection_certificate.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_strengths(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("at least one strength is required")
    if 0.0 not in values:
        values.insert(0, 0.0)
    return sorted(dict.fromkeys(values))


def materialize(store, code: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = indices(store.split, code)
    return (
        np.asarray(store.z[idx], dtype="float32").copy(),
        np.asarray(store.y[idx], dtype="int64").copy(),
        np.asarray(store.s[idx], dtype="int64").copy(),
    )


def projection_edit(x: np.ndarray, direction: np.ndarray, strength: float) -> np.ndarray:
    if abs(strength) < 1e-12:
        return x.copy()
    scores = x @ direction
    edited = x - float(strength) * scores[:, None] * direction[None, :]
    np.nan_to_num(edited, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(edited, -10.0, 10.0, out=edited)
    return edited.astype("float32", copy=False)


def grouped_recall_bounds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    delta: float,
    n_candidates: int,
    label_name: str,
) -> dict[str, object]:
    labels = sorted(int(item) for item in np.unique(y_true))
    group_count = max(len(labels), 1)
    log_term = math.log(max(2.0 * n_candidates * group_count / max(delta, 1e-12), 1.0))
    recalls: list[float] = []
    radii: list[float] = []
    groups: dict[str, object] = {}
    for label in labels:
        mask = y_true == label
        n = int(mask.sum())
        recall = float(np.mean(y_pred[mask] == y_true[mask])) if n else float("nan")
        radius = math.sqrt(log_term / (2.0 * max(n, 1)))
        recalls.append(recall)
        radii.append(radius)
        groups[str(label)] = {
            "n": n,
            "recall": recall,
            "radius": radius,
            "lcb": max(0.0, recall - radius),
            "ucb": min(1.0, recall + radius),
        }
    estimate = float(np.mean(recalls)) if recalls else float("nan")
    lcb = float(np.mean([max(0.0, r - e) for r, e in zip(recalls, radii)]))
    ucb = float(np.mean([min(1.0, r + e) for r, e in zip(recalls, radii)]))
    return {
        "label": label_name,
        "estimate": estimate,
        "lcb": lcb,
        "ucb": ucb,
        "groups": groups,
    }


def split_counts(values: np.ndarray) -> dict[str, int]:
    counts = Counter(int(item) for item in values)
    return {str(key): int(counts[key]) for key in sorted(counts)}


def balanced_accuracy_or_none(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return balanced_accuracy(y_true, y_pred)


def evaluate_strength(
    z_train: np.ndarray,
    y_train: np.ndarray,
    s_train: np.ndarray,
    z_val: np.ndarray,
    y_val: np.ndarray,
    s_val: np.ndarray,
    z_test: np.ndarray,
    y_test: np.ndarray,
    s_test: np.ndarray,
    direction: np.ndarray,
    strength: float,
    seed: int,
    delta: float,
    n_candidates: int,
) -> dict[str, object]:
    x_train = projection_edit(z_train, direction, strength)
    x_val = projection_edit(z_val, direction, strength)
    x_test = projection_edit(z_test, direction, strength)

    target_probe = fit_probe(x_train, y_train, seed=seed, sample_weight=class_weights(y_train))
    source_probe = fit_probe(x_train, s_train, seed=seed + 10_000, sample_weight=class_weights(s_train))

    pred_y_val = predict_probe(target_probe, x_val)
    pred_y_test = predict_probe(target_probe, x_test)
    pred_s_val = predict_probe(source_probe, x_val)
    pred_s_test = predict_probe(source_probe, x_test)

    target_val_bounds = grouped_recall_bounds(
        y_val,
        pred_y_val,
        delta=delta,
        n_candidates=n_candidates,
        label_name="target_validation_balanced_accuracy",
    )
    source_val_bounds = grouped_recall_bounds(
        s_val,
        pred_s_val,
        delta=delta,
        n_candidates=n_candidates,
        label_name="source_validation_balanced_accuracy",
    )

    return {
        "strength": float(strength),
        "validation_target_balanced_accuracy": balanced_accuracy(y_val, pred_y_val),
        "validation_target_lcb": target_val_bounds["lcb"],
        "validation_target_ucb": target_val_bounds["ucb"],
        "validation_source_leakage_balanced_accuracy": balanced_accuracy(s_val, pred_s_val),
        "validation_source_leakage_lcb": source_val_bounds["lcb"],
        "validation_source_leakage_ucb": source_val_bounds["ucb"],
        "external_target_balanced_accuracy": balanced_accuracy(y_test, pred_y_test),
        "external_worst_target_source_accuracy": worst_target_source_accuracy(
            y_test,
            pred_y_test,
            s_test,
        ),
        "external_source_leakage_balanced_accuracy": balanced_accuracy_or_none(s_test, pred_s_test),
        "target_validation_bounds": target_val_bounds,
        "source_validation_bounds": source_val_bounds,
    }


def write_frontier(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "strength",
        "decision",
        "certified_safe",
        "target_safe",
        "source_sufficient",
        "target_loss_ucb95",
        "source_reduction_lcb95",
        "validation_target_balanced_accuracy",
        "validation_target_lcb",
        "validation_target_ucb",
        "validation_source_leakage_balanced_accuracy",
        "validation_source_leakage_lcb",
        "validation_source_leakage_ucb",
        "external_target_balanced_accuracy",
        "external_worst_target_source_accuracy",
        "external_source_leakage_balanced_accuracy",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "" if row.get(key, "") is None else row.get(key, "")
                    for key in fieldnames
                }
            )


def write_markdown(path: Path, certificate: dict[str, object]) -> None:
    selected = certificate["selected_candidate"]
    lines = [
        "# Camelyon17 VERA Projection Frontier Certificate",
        "",
        f"- Decision: `{certificate['decision']}`",
        f"- Selected strength: `{selected.get('strength')}`",
        f"- Full dataset examples: `{certificate['n_examples']}`",
        f"- Claim-grade certificate: `{certificate['claim_grade_frontier_certificate']}`",
        f"- Validation target-loss budget: `{certificate['epsilon_target_loss']}`",
        f"- Validation source-reduction target: `{certificate['delta_source_reduction']}`",
        "",
        "| Strength | Safe | Target loss UCB95 | Source reduction LCB95 | Val target BA | Val source BA | Ext target BA | Ext worst group | Ext source BA |",
        "|---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in certificate["frontier"]:
        external_source = row.get("external_source_leakage_balanced_accuracy")
        external_source_text = "" if external_source is None else f"{float(external_source):.6f}"
        lines.append(
            "| "
            f"{float(row['strength']):.3f} | "
            f"{row['certified_safe']} | "
            f"{float(row['target_loss_ucb95']):.6f} | "
            f"{float(row['source_reduction_lcb95']):.6f} | "
            f"{float(row['validation_target_balanced_accuracy']):.6f} | "
            f"{float(row['validation_source_leakage_balanced_accuracy']):.6f} | "
            f"{float(row['external_target_balanced_accuracy']):.6f} | "
            f"{float(row['external_worst_target_source_accuracy']):.6f} | "
            f"{external_source_text} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            str(certificate["claim_boundary"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--frontier", type=Path, default=DEFAULT_FRONTIER)
    parser.add_argument("--certificate", type=Path, default=DEFAULT_CERTIFICATE)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--strengths", default="0,0.1,0.2,0.35,0.5,0.75,1.0")
    parser.add_argument("--epsilon-target-loss", type=float, default=0.02)
    parser.add_argument("--delta-source-reduction", type=float, default=0.05)
    parser.add_argument("--confidence-delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    strengths = parse_strengths(args.strengths)
    store = load_store(args.store_dir)
    z_train, y_train, s_train = materialize(store, SPLIT_TRAIN)
    z_val, y_val, s_val = materialize(store, SPLIT_VALIDATION)
    z_test, y_test, s_test = materialize(store, SPLIT_EXTERNAL)
    z_train, z_val, z_test = standardize_splits(z_train, z_val, z_test)

    source_probe = fit_probe(
        z_train,
        s_train,
        seed=args.seed + 20_000,
        sample_weight=class_weights(s_train),
    )
    direction = np.asarray(source_probe.coef, dtype="float64")
    direction_norm = float(np.linalg.norm(direction))
    if not math.isfinite(direction_norm) or direction_norm < 1e-12:
        raise ValueError("source direction has zero or non-finite norm")
    direction = (direction / direction_norm).astype("float32")

    raw_rows = [
        evaluate_strength(
            z_train,
            y_train,
            s_train,
            z_val,
            y_val,
            s_val,
            z_test,
            y_test,
            s_test,
            direction,
            strength,
            args.seed,
            args.confidence_delta,
            len(strengths),
        )
        for strength in strengths
    ]

    baseline = next(row for row in raw_rows if abs(float(row["strength"])) < 1e-12)
    frontier: list[dict[str, object]] = []
    safe_rows: list[dict[str, object]] = []
    for row in raw_rows:
        target_loss_ucb = max(
            0.0,
            float(baseline["validation_target_ucb"]) - float(row["validation_target_lcb"]),
        )
        source_reduction_lcb = max(
            0.0,
            float(baseline["validation_source_leakage_lcb"])
            - float(row["validation_source_leakage_ucb"]),
        )
        target_safe = target_loss_ucb <= float(args.epsilon_target_loss)
        source_sufficient = source_reduction_lcb >= float(args.delta_source_reduction)
        certified_safe = bool(target_safe and source_sufficient and float(row["strength"]) > 0.0)
        public_row = {
            key: value
            for key, value in row.items()
            if not key.endswith("_bounds")
        }
        public_row.update(
            {
                "target_loss_ucb95": target_loss_ucb,
                "source_reduction_lcb95": source_reduction_lcb,
                "target_safe": target_safe,
                "source_sufficient": source_sufficient,
                "certified_safe": certified_safe,
                "decision": "",
            }
        )
        if certified_safe:
            safe_rows.append(public_row)
        frontier.append(public_row)

    if safe_rows:
        selected = max(
            safe_rows,
            key=lambda row: (
                float(row["source_reduction_lcb95"]),
                float(row["strength"]),
                -float(row["target_loss_ucb95"]),
            ),
        )
        decision = "EDIT"
    else:
        selected = dict(baseline)
        selected.update(
            {
                "target_loss_ucb95": 0.0,
                "source_reduction_lcb95": 0.0,
                "target_safe": True,
                "source_sufficient": False,
                "certified_safe": False,
            }
        )
        decision = "ABSTAIN"

    for row in frontier:
        if float(row["strength"]) == float(selected["strength"]):
            row["decision"] = decision

    certificate = {
        "schema": "faro_camelyon17_projection_frontier_certificate_v1",
        "created_at_utc": utc_now(),
        "store_dir": str(args.store_dir),
        "store_format": store.manifest.get("format"),
        "official_dataset": True,
        "official_splits": True,
        "n_examples": int(store.manifest.get("n_examples", len(store.y))),
        "feature_count": int(store.manifest.get("feature_count", z_train.shape[1])),
        "split_counts": store.manifest.get("split_counts"),
        "train_label_counts": {
            "target": split_counts(y_train),
            "source": split_counts(s_train),
        },
        "validation_label_counts": {
            "target": split_counts(y_val),
            "source": split_counts(s_val),
        },
        "external_label_counts": {
            "target": split_counts(y_test),
            "source": split_counts(s_test),
        },
        "external_source_leakage_interpretable": len(np.unique(s_test)) >= 2,
        "external_source_leakage_note": (
            "External source leakage is omitted because the Camelyon17 external "
            "split in this binary source encoding contains a single source class."
            if len(np.unique(s_test)) < 2
            else "External source leakage has both source classes."
        ),
        "frontier_family": "linear source-direction projection over frozen standardized ResNet-18 embeddings",
        "strengths": strengths,
        "selection_uses_external_metrics": False,
        "epsilon_target_loss": float(args.epsilon_target_loss),
        "delta_source_reduction": float(args.delta_source_reduction),
        "confidence_delta": float(args.confidence_delta),
        "baseline_validation": {
            "target_balanced_accuracy": baseline["validation_target_balanced_accuracy"],
            "target_lcb": baseline["validation_target_lcb"],
            "target_ucb": baseline["validation_target_ucb"],
            "source_leakage_balanced_accuracy": baseline["validation_source_leakage_balanced_accuracy"],
            "source_leakage_lcb": baseline["validation_source_leakage_lcb"],
            "source_leakage_ucb": baseline["validation_source_leakage_ucb"],
        },
        "decision": decision,
        "selected_candidate": selected,
        "safe_candidate_count": len(safe_rows),
        "frontier": frontier,
        "claim_grade_frontier_certificate": (
            int(store.manifest.get("n_examples", 0)) == 455_954
            and store.manifest.get("format") == "trace_embedding_store_v1"
            and len(strengths) >= 5
        ),
        "claim_boundary": (
            "This is a full-dataset representation-reliability certificate for "
            "a transparent linear projection family on frozen Camelyon17-WILDS "
            "embeddings. It is not a clinical safety, diagnostic, or deployment claim."
        ),
    }

    write_frontier(args.frontier, frontier)
    args.certificate.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.markdown, certificate)

    print("Camelyon17 VERA projection frontier certificate complete")
    print(f"decision={decision}")
    print(f"safe_candidate_count={len(safe_rows)}")
    print(f"selected_strength={selected.get('strength')}")
    print(f"certificate={args.certificate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
