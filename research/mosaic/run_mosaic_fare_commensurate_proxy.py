#!/usr/bin/env python3
"""Compare official FARE representations under MOSAIC's identical proxy contract."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from folktables import ACSIncome
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier

from mosaic_channel import (
    adaptive_channel_attacker_confidence_bound,
    selected_decoder_error_confidence_bound,
)
from mosaic_proxy_bridge import certify_proxy_label_conditionals
from mosaic_real import evaluate_external_channel
from run_mosaic_real_proxy_study import (
    PROXY_COLUMNS,
    RAW,
    calibration_counts,
    stratified_partitions,
)


ROOT = Path(__file__).resolve().parents[2]
PREREG = ROOT / "research/mosaic/prereg_mosaic_fare_proxy_v1.json"
OUTPUT = ROOT / "research/artifacts/mosaic_fare_proxy_comparison_v1.json"
PRIVACY_THRESHOLD = 0.35
UTILITY_THRESHOLD = 0.40
FAMILY_FAILURE = 0.05
GRID = tuple(
    (leaves, alpha)
    for leaves in (2, 4)
    for alpha in (0.80, 0.90, 0.95, 0.99)
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_lock() -> dict[str, object]:
    sidecar = PREREG.with_suffix(PREREG.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").strip() != sha256(PREREG):
        raise ValueError("FARE comparison preregistration sidecar mismatch")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    if prereg["status"] != "locked_before_fare_outcomes":
        raise ValueError("FARE comparison is not locked")
    for relative, expected in prereg["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"locked code mismatch: {relative}")
    if sha256(RAW) != prereg["raw_data_sha256"]:
        raise ValueError("raw ACS data mismatch")
    for path in (PREREG, sidecar):
        relative = path.relative_to(ROOT)
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if committed != path.read_bytes():
            raise ValueError(f"{relative} is not the committed lock")
    return prereg


def remap_leaves(tree: DecisionTreeClassifier, values: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    raw = tree.apply(values)
    mapping = {
        int(value): index for index, value in enumerate(sorted(np.unique(raw)))
    }
    return np.asarray([mapping[int(value)] for value in raw], dtype=np.int16), mapping


def proxy_counts(
    labels: np.ndarray,
    proxies: np.ndarray,
    tokens: np.ndarray,
    token_count: int,
) -> np.ndarray:
    counts = np.zeros((2, 2, token_count), dtype=np.int64)
    np.add.at(counts, (labels, proxies, tokens), 1)
    return counts


def proxy_calibration_counts(
    labels: np.ndarray,
    sources: np.ndarray,
    tokens: np.ndarray,
    proxies: np.ndarray,
    token_count: int,
) -> np.ndarray:
    counts = np.zeros((2, 2, token_count, 2), dtype=np.int64)
    np.add.at(counts, (labels, sources, tokens, proxies), 1)
    return counts


def main() -> None:
    prereg = validate_lock()
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    import pandas as pd

    frame = pd.read_csv(RAW)
    filtered = ACSIncome._preprocess(frame).copy()
    labels = np.asarray(
        ACSIncome.target_transform(filtered[ACSIncome.target]), dtype=np.int16
    )
    sources = np.asarray(filtered["RAC1P"] != 1, dtype=np.int16)
    task_columns = tuple(
        column for column in ACSIncome.features if column != "RAC1P"
    )
    features = np.nan_to_num(
        filtered.loc[:, task_columns].to_numpy(dtype=np.float64),
        nan=-1.0,
    )
    proxy_features = np.nan_to_num(
        filtered.loc[:, PROXY_COLUMNS].to_numpy(dtype=np.float64),
        nan=-1.0,
    )
    partitions = stratified_partitions(labels, sources, seed=20270723)
    proxy_model = HistGradientBoostingClassifier(
        max_iter=160,
        max_leaf_nodes=31,
        learning_rate=0.07,
        class_weight="balanced",
        random_state=20270724,
    )
    proxy_model.fit(
        proxy_features[partitions["proxy_train"]],
        sources[partitions["proxy_train"]],
    )
    proxies = proxy_model.predict(proxy_features).astype(np.int16)
    train = partitions["task_train"]
    reference = partitions["target_proxy"]
    calibration = partitions["calibration"]
    diagnostic = partitions["diagnostic"]
    candidates = []
    for maximum_leaves, alpha in GRID:
        tree = DecisionTreeClassifier(
            criterion="fair_gini_eo",
            max_leaf_nodes=maximum_leaves,
            min_samples_leaf=1000,
            random_state=43,
        )
        tree.fit(
            features[train],
            labels[train].reshape(-1, 1),
            sources[train].reshape(-1, 1),
            cat_pos=np.asarray([], dtype=np.int32),
            alpha=alpha,
        )
        all_tokens, mapping = remap_leaves(tree, features)
        token_count = len(mapping)
        joint = proxy_counts(
            labels[reference],
            proxies[reference],
            all_tokens[reference],
            token_count,
        )
        confusion = proxy_calibration_counts(
            labels[calibration],
            sources[calibration],
            all_tokens[calibration],
            proxies[calibration],
            token_count,
        )
        certificate = certify_proxy_label_conditionals(
            joint,
            family_failure_probability=FAMILY_FAILURE / len(GRID),
            calibration_confusion_counts=confusion,
            confidence_region="l1_weissman",
        )
        identity = np.eye(token_count)
        source_bounds = [
            adaptive_channel_attacker_confidence_bound(
                certificate.conditional_empirical_distributions[label],
                identity,
                l1_radii=certificate.conditional_l1_radii[label],
            ).normalized_advantage
            for label in (0, 1)
        ]
        decoder = []
        for token in range(token_count):
            cell_labels = labels[train][all_tokens[train] == token]
            decoder.append(
                int(cell_labels.mean() >= 0.5) if len(cell_labels) else 0
            )
        errors = []
        for label in (0, 1):
            for source in (0, 1):
                errors.append(
                    selected_decoder_error_confidence_bound(
                        certificate.conditional_empirical_distributions[
                            label, source
                        ],
                        identity,
                        decoder,
                        true_label=label,
                        l1_radius=certificate.conditional_l1_radii[
                            label, source
                        ],
                    )
                )
        diagnostic_risk = evaluate_external_channel(
            all_tokens[diagnostic],
            labels[diagnostic],
            sources[diagnostic],
            identity,
            tuple(decoder),
        )
        deployed = bool(
            max(source_bounds) <= PRIVACY_THRESHOLD
            and max(errors) <= UTILITY_THRESHOLD
        )
        diagnostic_safe = bool(
            diagnostic_risk.estimable
            and diagnostic_risk.worst_privacy_advantage is not None
            and diagnostic_risk.worst_conditional_error is not None
            and diagnostic_risk.worst_privacy_advantage <= PRIVACY_THRESHOLD
            and diagnostic_risk.worst_conditional_error <= UTILITY_THRESHOLD
        )
        candidates.append(
            {
                "candidate": f"FARE::leaves={maximum_leaves},alpha={alpha:.2f}",
                "realized_leaf_count": token_count,
                "certified_source_advantage_upper": source_bounds,
                "certified_worst_conditional_error_upper": max(errors),
                "decision": "deploy" if deployed else "abstain",
                "diagnostic_source_advantage": diagnostic_risk.worst_privacy_advantage,
                "diagnostic_worst_conditional_error": diagnostic_risk.worst_conditional_error,
                "diagnostic_safe": diagnostic_safe,
                "false_acceptance": bool(deployed and not diagnostic_safe),
                "diagnostic_balanced_accuracy": balanced_accuracy_score(
                    labels[diagnostic],
                    np.asarray(decoder)[all_tokens[diagnostic]],
                ),
                "decoder": decoder,
                "calibration_cells_observed": bool(
                    np.all(confusion.sum(axis=-1) > 0)
                ),
            }
        )
    eligible = [value for value in candidates if value["decision"] == "deploy"]
    selected = (
        min(
            eligible,
            key=lambda value: (
                value["certified_worst_conditional_error_upper"],
                value["candidate"],
            ),
        )
        if eligible
        else None
    )
    payload = {
        "name": "Official FARE representation under the MOSAIC proxy contract",
        "preregistration_sha256": sha256(PREREG),
        "protocol": prereg["protocol"],
        "candidates": candidates,
        "selected": selected,
        "summary": {
            "candidate_count": len(candidates),
            "certified_candidates": len(eligible),
            "selected_candidate": None if selected is None else selected["candidate"],
            "false_acceptances": sum(
                value["false_acceptance"] for value in candidates
            ),
        },
        "claim_boundary": (
            "This is a commensurate representation comparison: official FARE "
            "constructs each tree representation, and the identical proxy, "
            "source-inference, and task-error contract judges its public leaf "
            "token. It does not convert MOSAIC's source-inference contract into "
            "FARE's original downstream demographic-parity theorem."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
