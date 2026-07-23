#!/usr/bin/env python3
"""Run the locked ACS geography-ancestry proxy certification study."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from folktables import ACSIncome
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

from mosaic_proxy_bridge import certify_proxy_label_conditionals
from mosaic_real import evaluate_external_channel
from mosaic_transform_exact_optimizer import optimize_transform_exact_channel


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "research/mosaic/prereg_mosaic_real_proxy_v1.json"
AMENDMENT = ROOT / (
    "research/mosaic/prereg_mosaic_real_proxy_v1_amendment.json"
)
RAW = Path(
    "/Users/rudrachopra/Documents/Science Fair/data/"
    "acs_pums/2018/1-Year/psam_p06.csv"
)
OUTPUT = ROOT / "research/artifacts/mosaic_real_proxy_v1.json"
FINE_TOKEN_COUNT = 4
PRIVACY_THRESHOLD = 0.35
UTILITY_THRESHOLD = 0.40
FAMILY_FAILURE = 0.05
SEED = 20270723
PROXY_COLUMNS = (
    "PUMA",
    "POBP",
    "ANC",
    "ANC1P",
    "ANC2P",
    "HISP",
    "LANP",
    "ENG",
    "NATIVITY",
    "CIT",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_lock() -> dict[str, Any]:
    sidecar = LOCK.with_suffix(LOCK.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").strip() != sha256(LOCK):
        raise ValueError("proxy preregistration sidecar mismatch")
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "locked_before_outcomes":
        raise ValueError("proxy preregistration status mismatch")
    if sha256(RAW) != lock["raw_data"]["sha256"]:
        raise ValueError("ACS raw-data hash mismatch")
    amendment_sidecar = AMENDMENT.with_suffix(
        AMENDMENT.suffix + ".sha256"
    )
    if amendment_sidecar.read_text(encoding="utf-8").strip() != sha256(
        AMENDMENT
    ):
        raise ValueError("proxy preregistration amendment sidecar mismatch")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if amendment["status"] != "locked_before_outcomes_after_hash_only_failure":
        raise ValueError("proxy preregistration amendment status mismatch")
    for relative, expected in amendment["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"amended locked code mismatch: {relative}")
    for path in (LOCK, sidecar, AMENDMENT, amendment_sidecar):
        relative = path.relative_to(ROOT)
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if committed != path.read_bytes():
            raise ValueError(f"{relative} is not the committed lock")
    lock["amendment"] = amendment
    return lock


def stratified_partitions(
    labels: np.ndarray,
    sources: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    fractions = {
        "task_train": 0.20,
        "proxy_train": 0.20,
        "calibration": 0.20,
        "target_proxy": 0.25,
        "diagnostic": 0.15,
    }
    names = tuple(fractions)
    output: dict[str, list[int]] = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    cumulative = np.cumsum([fractions[name] for name in names])
    for label in (0, 1):
        for source in (0, 1):
            indices = np.flatnonzero(
                (labels == label) & (sources == source)
            )
            rng.shuffle(indices)
            boundaries = np.floor(cumulative * len(indices)).astype(int)
            boundaries[-1] = len(indices)
            start = 0
            for name, end in zip(names, boundaries, strict=True):
                output[name].extend(indices[start:end].tolist())
                start = int(end)
    return {
        name: np.sort(np.asarray(values, dtype=np.int64))
        for name, values in output.items()
    }


def token_counts(
    labels: np.ndarray,
    sources: np.ndarray,
    tokens: np.ndarray,
    *,
    source_count: int = 2,
) -> np.ndarray:
    counts = np.zeros((2, source_count, FINE_TOKEN_COUNT), dtype=np.int64)
    np.add.at(counts, (labels, sources, tokens), 1)
    return counts


def calibration_counts(
    labels: np.ndarray,
    sources: np.ndarray,
    tokens: np.ndarray,
    proxies: np.ndarray,
) -> np.ndarray:
    counts = np.zeros((2, 2, FINE_TOKEN_COUNT, 2), dtype=np.int64)
    np.add.at(counts, (labels, sources, tokens, proxies), 1)
    return counts


def encode_tokens(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.searchsorted(thresholds, scores, side="right").astype(np.int16)


def expected_balanced_accuracy(
    tokens: np.ndarray,
    labels: np.ndarray,
    channel: np.ndarray,
    decoder: tuple[int, ...],
) -> float:
    decoder_array = np.asarray(decoder)
    values = []
    for label in (0, 1):
        current = tokens[labels == label]
        correct = (decoder_array == label).astype(float)
        values.append(float(np.mean(channel[current] @ correct)))
    return float(np.mean(values))


def serialize_proxy(certificate: Any) -> dict[str, Any]:
    return {
        "calibration_mode": certificate.calibration_mode,
        "proxy_sample_size": certificate.proxy_sample_size,
        "calibration_sample_size": certificate.calibration_sample_size,
        "per_event_failure_probability": (
            certificate.per_event_failure_probability
        ),
        "conditional_centers": (
            certificate.conditional_empirical_distributions.tolist()
        ),
        "conditional_l1_radii": (
            certificate.conditional_l1_radii.tolist()
        ),
        "labels": [
            {
                "confusion_matrix": label.confusion_matrix.tolist(),
                "confusion_row_l1_radii": list(
                    label.confusion_row_l1_radii
                ),
                "effective_proxy_l1_radius": (
                    label.effective_proxy_l1_radius
                ),
                "source_mass_lower_bounds": list(
                    label.source_mass_lower_bounds
                ),
                "conditional_lp_solves": label.conditional_lp_solves,
            }
            for label in certificate.labels
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock = validate_lock()

    import pandas as pd

    frame = pd.read_csv(args.raw)
    filtered = ACSIncome._preprocess(frame).copy()
    labels = np.asarray(
        ACSIncome.target_transform(filtered[ACSIncome.target]), dtype=np.int16
    )
    sources = np.asarray(filtered["RAC1P"] != 1, dtype=np.int16)
    task_columns = tuple(
        column for column in ACSIncome.features if column != "RAC1P"
    )
    task_features = np.nan_to_num(
        filtered.loc[:, task_columns].to_numpy(dtype=np.float64),
        nan=-1.0,
    )
    proxy_features = np.nan_to_num(
        filtered.loc[:, PROXY_COLUMNS].to_numpy(dtype=np.float64),
        nan=-1.0,
    )
    partitions = stratified_partitions(labels, sources, seed=SEED)

    task_model = HistGradientBoostingClassifier(
        max_iter=120,
        max_leaf_nodes=31,
        learning_rate=0.08,
        random_state=SEED,
    )
    task_train = partitions["task_train"]
    task_model.fit(task_features[task_train], labels[task_train])
    task_scores = task_model.predict_proba(task_features)[:, 1]
    thresholds = np.quantile(
        task_scores[task_train], [0.25, 0.50, 0.75]
    )
    tokens = encode_tokens(task_scores, thresholds)

    proxy_model = HistGradientBoostingClassifier(
        max_iter=160,
        max_leaf_nodes=31,
        learning_rate=0.07,
        class_weight="balanced",
        random_state=SEED + 1,
    )
    proxy_train = partitions["proxy_train"]
    proxy_model.fit(proxy_features[proxy_train], sources[proxy_train])
    proxies = proxy_model.predict(proxy_features).astype(np.int16)

    calibration = partitions["calibration"]
    target_proxy = partitions["target_proxy"]
    diagnostic = partitions["diagnostic"]
    proxy_joint = token_counts(
        labels[target_proxy],
        proxies[target_proxy],
        tokens[target_proxy],
    )
    confusion = calibration_counts(
        labels[calibration],
        sources[calibration],
        tokens[calibration],
        proxies[calibration],
    )
    certificate = certify_proxy_label_conditionals(
        proxy_joint,
        family_failure_probability=FAMILY_FAILURE,
        calibration_confusion_counts=confusion,
        confidence_region="l1_weissman",
    )
    identity = np.stack(
        [np.eye(FINE_TOKEN_COUNT), np.eye(FINE_TOKEN_COUNT)]
    )
    solution = optimize_transform_exact_channel(
        certificate.conditional_empirical_distributions,
        l1_radii=certificate.conditional_l1_radii,
        common_channels_by_label=identity,
        contaminations=(0.0, 0.0),
        privacy_advantage_thresholds=(
            PRIVACY_THRESHOLD,
            PRIVACY_THRESHOLD,
        ),
        released_token_count=2,
        maximum_worst_conditional_error=UTILITY_THRESHOLD,
        solver_time_limit_seconds=300.0,
    )
    diagnostic_risk = evaluate_external_channel(
        tokens[diagnostic],
        labels[diagnostic],
        sources[diagnostic],
        solution.release_channel,
        solution.decoder,
    )
    raw_predictions = (task_scores[diagnostic] >= 0.5).astype(np.int16)
    raw_balanced_accuracy = balanced_accuracy_score(
        labels[diagnostic], raw_predictions
    )
    released_balanced_accuracy = expected_balanced_accuracy(
        tokens[diagnostic],
        labels[diagnostic],
        solution.release_channel,
        solution.decoder,
    )
    source_bounds = tuple(
        float(value.normalized_advantage)
        for value in solution.privacy_certificates
    )
    deployed = bool(
        max(source_bounds) <= PRIVACY_THRESHOLD + 1e-10
        and solution.certified_worst_conditional_error
        <= UTILITY_THRESHOLD + 1e-10
    )
    diagnostic_safe = bool(
        diagnostic_risk.estimable
        and diagnostic_risk.worst_privacy_advantage is not None
        and diagnostic_risk.worst_conditional_error is not None
        and diagnostic_risk.worst_privacy_advantage
        <= PRIVACY_THRESHOLD + 1e-10
        and diagnostic_risk.worst_conditional_error
        <= UTILITY_THRESHOLD + 1e-10
    )
    proxy_accuracy = balanced_accuracy_score(
        sources[diagnostic], proxies[diagnostic]
    )
    nonempty_calibration = bool(
        np.all(confusion.sum(axis=-1) > 0)
    )
    gates = {
        "all_calibration_cells_observed": nonempty_calibration,
        "proxy_is_nontrivial": proxy_accuracy >= 0.60,
        "certificate_releases": deployed,
        "released_interface_is_diagnostically_safe": (
            not deployed or diagnostic_safe
        ),
        "no_false_acceptance": not (deployed and not diagnostic_safe),
    }
    payload = {
        "name": "MOSAIC real ACS proxy certification v1",
        "status": "complete_locked_study",
        "preregistration_sha256": sha256(LOCK),
        "raw_data_sha256": sha256(args.raw),
        "protocol": lock["protocol"],
        "sample_counts": {
            name: len(indices) for name, indices in partitions.items()
        },
        "task_columns": list(task_columns),
        "proxy_columns": list(PROXY_COLUMNS),
        "token_thresholds": thresholds.tolist(),
        "proxy_balanced_accuracy": float(proxy_accuracy),
        "proxy_certificate": serialize_proxy(certificate),
        "release": {
            "decision": "deploy" if deployed else "abstain",
            "certified_source_advantage_upper": list(source_bounds),
            "certified_worst_conditional_error_upper": float(
                solution.certified_worst_conditional_error
            ),
            "release_channel": solution.release_channel.tolist(),
            "decoder": list(solution.decoder),
            "diagnostic_source_advantage": (
                diagnostic_risk.worst_privacy_advantage
            ),
            "diagnostic_worst_conditional_error": (
                diagnostic_risk.worst_conditional_error
            ),
            "diagnostic_safe": diagnostic_safe,
        },
        "utility": {
            "unedited_task_score_balanced_accuracy": float(
                raw_balanced_accuracy
            ),
            "released_interface_expected_balanced_accuracy": float(
                released_balanced_accuracy
            ),
            "absolute_gap": float(
                raw_balanced_accuracy - released_balanced_accuracy
            ),
        },
        "gates": gates,
        "passed": all(gates.values()),
        "claim_boundary": lock["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "passed": payload["passed"],
                "gates": gates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
