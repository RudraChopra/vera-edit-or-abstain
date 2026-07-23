#!/usr/bin/env python3
"""Run the locked powered ACS proxy-label certification follow-up."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from folktables import ACSIncome
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

from mosaic_proxy_bridge import certify_proxy_label_conditionals
from mosaic_real import evaluate_external_channel
from mosaic_transform_exact_optimizer import optimize_transform_exact_channel
from run_mosaic_real_proxy_study import (
    PROXY_COLUMNS,
    expected_balanced_accuracy,
    serialize_proxy,
)


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "research/mosaic/prereg_mosaic_real_proxy_powered_v1.json"
OUTPUT = ROOT / "research/artifacts/mosaic_real_proxy_powered_v1.json"
PRIOR_LOCK = ROOT / "research/mosaic/prereg_mosaic_real_proxy_v1.json"
REFERENCE_URL = (
    "https://www2.census.gov/programs-surveys/acs/data/pums/"
    "2018/1-Year/csv_pca.zip"
)
TOKEN_COUNT = 2
PRIVACY_THRESHOLD = 0.35
UTILITY_THRESHOLD = 0.40
FAMILY_FAILURE = 0.05
SEED = 20270724
PARTITION_FRACTIONS = {
    "task_train": 0.15,
    "proxy_train": 0.20,
    "calibration": 0.40,
    "target_proxy": 0.20,
    "diagnostic": 0.05,
}
CALIBRATION_CURVE_FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def expected_protocol() -> dict[str, Any]:
    return {
        "design": (
            "locked powered follow-up after v1 abstained; no diagnostic outcome "
            "selects the primary design"
        ),
        "task": "Folktables ACSIncome",
        "source": "RAC1P white-alone versus all other codes",
        "seed": SEED,
        "fine_token_count": TOKEN_COUNT,
        "released_token_count": 2,
        "task_token_threshold": 0.5,
        "partition_fractions": PARTITION_FRACTIONS,
        "proxy_imputer": (
            "fixed equal-weight soft ensemble of class-balanced "
            "HistGradientBoosting and ExtraTrees"
        ),
        "calibration_model": (
            "task-label, true-source, and two-token-specific source-to-proxy "
            "confusion tensor"
        ),
        "confidence_region": "coordinate_clopper_pearson",
        "family_failure_probability": FAMILY_FAILURE,
        "privacy_advantage_threshold": PRIVACY_THRESHOLD,
        "utility_error_threshold": UTILITY_THRESHOLD,
        "calibration_curve_fractions": list(CALIBRATION_CURVE_FRACTIONS),
        "curve_role": (
            "descriptive nested calibration-size sensitivity; only the fixed "
            "full-calibration point is the primary release decision"
        ),
        "primary_gate": (
            "full-calibration release, proxy balanced accuracy at least .60, "
            "and zero diagnostic contract violations"
        ),
    }


def validate_lock() -> dict[str, Any]:
    sidecar = LOCK.with_suffix(LOCK.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(LOCK):
        raise ValueError("powered proxy lock sidecar mismatch")
    lock = load(LOCK)
    if lock.get("status") != "locked_before_powered_proxy_outcomes":
        raise ValueError("powered proxy lock status mismatch")
    if lock.get("protocol") != expected_protocol():
        raise ValueError("powered proxy protocol differs from its lock")
    for relative, expected in lock["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"powered proxy code mismatch: {relative}")
    if lock["raw_data"] != load(PRIOR_LOCK)["raw_data"]:
        raise ValueError("powered proxy raw asset differs from the prior lock")
    for path in (LOCK, sidecar):
        relative = path.relative_to(ROOT)
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if committed != path.read_bytes():
            raise ValueError(f"{relative} is not the committed lock")
    return lock


def load_raw_from_url(url: str, expected: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    import pandas as pd

    request = urllib.request.Request(
        url, headers={"User-Agent": "MOSAIC-research/1.0"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        members = [
            name
            for name in bundle.namelist()
            if name.lower().endswith(".csv") and "psam_p" in name.lower()
        ]
        if len(members) != 1:
            raise ValueError(f"unexpected ACS archive members: {members}")
        raw = bundle.read(members[0])
    if len(raw) != expected["bytes"] or sha256_bytes(raw) != expected["sha256"]:
        raise ValueError("streamed ACS member differs from the locked raw asset")
    frame = pd.read_csv(io.BytesIO(raw), low_memory=False)
    return frame, {
        "source": "official_archive_streamed_in_memory",
        "url": url,
        "archive_member": members[0],
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "compressed_bytes": len(archive),
        "compressed_sha256": sha256_bytes(archive),
    }


def stratified_partitions(
    labels: np.ndarray,
    sources: np.ndarray,
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    names = tuple(PARTITION_FRACTIONS)
    cumulative = np.cumsum([PARTITION_FRACTIONS[name] for name in names])
    output: dict[str, list[int]] = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        for source in (0, 1):
            indices = np.flatnonzero((labels == label) & (sources == source))
            rng.shuffle(indices)
            boundaries = np.floor(cumulative * len(indices)).astype(int)
            boundaries[-1] = len(indices)
            start = 0
            for name, end in zip(names, boundaries, strict=True):
                output[name].extend(indices[start:end].tolist())
                start = int(end)
    return {
        name: np.sort(np.asarray(indices, dtype=np.int64))
        for name, indices in output.items()
    }


def token_counts(
    labels: np.ndarray,
    sources: np.ndarray,
    tokens: np.ndarray,
) -> np.ndarray:
    counts = np.zeros((2, 2, TOKEN_COUNT), dtype=np.int64)
    np.add.at(counts, (labels, sources, tokens), 1)
    return counts


def calibration_counts(
    labels: np.ndarray,
    sources: np.ndarray,
    tokens: np.ndarray,
    proxies: np.ndarray,
) -> np.ndarray:
    counts = np.zeros((2, 2, TOKEN_COUNT, 2), dtype=np.int64)
    np.add.at(counts, (labels, sources, tokens, proxies), 1)
    return counts


def optimize(certificate: Any) -> tuple[Any | None, str | None]:
    identity = ((np.eye(TOKEN_COUNT),), (np.eye(TOKEN_COUNT),))
    try:
        return (
            optimize_transform_exact_channel(
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
            ),
            None,
        )
    except RuntimeError as error:
        return None, str(error)


def curve_row(
    *,
    fraction: float,
    calibration_indices: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    tokens: np.ndarray,
    proxies: np.ndarray,
    proxy_joint: np.ndarray,
) -> tuple[dict[str, Any], Any | None]:
    count = max(1, int(np.floor(fraction * len(calibration_indices))))
    selected = calibration_indices[:count]
    confusion = calibration_counts(
        labels[selected],
        sources[selected],
        tokens[selected],
        proxies[selected],
    )
    try:
        certificate = certify_proxy_label_conditionals(
            proxy_joint,
            family_failure_probability=FAMILY_FAILURE,
            calibration_confusion_counts=confusion,
            confidence_region="coordinate_clopper_pearson",
        )
        solution, reason = optimize(certificate)
        return {
            "fraction": fraction,
            "calibration_rows": len(selected),
            "all_calibration_cells_observed": bool(
                np.all(confusion.sum(axis=-1) > 0)
            ),
            "maximum_conditional_l1_radius": float(
                np.max(certificate.conditional_l1_radii)
            ),
            "maximum_effective_proxy_l1_radius": float(
                max(label.effective_proxy_l1_radius for label in certificate.labels)
            ),
            "minimum_source_mass_lower_bound": float(
                min(
                    min(label.source_mass_lower_bounds)
                    for label in certificate.labels
                )
            ),
            "decision": "deploy" if solution is not None else "abstain",
            "reason": reason,
            "certified_source_advantage_upper": (
                []
                if solution is None
                else [
                    float(value.normalized_advantage)
                    for value in solution.privacy_certificates
                ]
            ),
            "certified_worst_conditional_error_upper": (
                None
                if solution is None
                else float(solution.certified_worst_conditional_error)
            ),
        }, certificate
    except (RuntimeError, ValueError) as error:
        return {
            "fraction": fraction,
            "calibration_rows": len(selected),
            "all_calibration_cells_observed": bool(
                np.all(confusion.sum(axis=-1) > 0)
            ),
            "maximum_conditional_l1_radius": None,
            "maximum_effective_proxy_l1_radius": None,
            "minimum_source_mass_lower_bound": None,
            "decision": "abstain",
            "reason": str(error),
            "certified_source_advantage_upper": [],
            "certified_worst_conditional_error_upper": None,
        }, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-url", default=REFERENCE_URL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock = validate_lock()
    frame, raw_asset = load_raw_from_url(args.raw_url, lock["raw_data"])
    filtered = ACSIncome._preprocess(frame).copy()
    labels = np.asarray(
        ACSIncome.target_transform(filtered[ACSIncome.target]), dtype=np.int16
    )
    sources = np.asarray(filtered["RAC1P"] != 1, dtype=np.int16)
    task_columns = tuple(
        column for column in ACSIncome.features if column != "RAC1P"
    )
    task_features = np.nan_to_num(
        filtered.loc[:, task_columns].to_numpy(dtype=np.float64), nan=-1.0
    )
    proxy_features = np.nan_to_num(
        filtered.loc[:, PROXY_COLUMNS].to_numpy(dtype=np.float64), nan=-1.0
    )
    partitions = stratified_partitions(labels, sources, seed=SEED)

    task_model = HistGradientBoostingClassifier(
        max_iter=180,
        max_leaf_nodes=31,
        learning_rate=0.06,
        l2_regularization=0.5,
        random_state=SEED,
    )
    task_train = partitions["task_train"]
    task_model.fit(task_features[task_train], labels[task_train])
    task_scores = task_model.predict_proba(task_features)[:, 1]
    tokens = (task_scores >= 0.5).astype(np.int16)

    proxy_train = partitions["proxy_train"]
    gradient = HistGradientBoostingClassifier(
        max_iter=300,
        max_leaf_nodes=63,
        learning_rate=0.05,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=SEED + 1,
    )
    trees = ExtraTreesClassifier(
        n_estimators=320,
        min_samples_leaf=2,
        max_features=0.8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED + 2,
    )
    gradient.fit(proxy_features[proxy_train], sources[proxy_train])
    trees.fit(proxy_features[proxy_train], sources[proxy_train])
    proxy_scores = 0.5 * (
        gradient.predict_proba(proxy_features)[:, 1]
        + trees.predict_proba(proxy_features)[:, 1]
    )
    proxies = (proxy_scores >= 0.5).astype(np.int16)

    target_proxy = partitions["target_proxy"]
    proxy_joint = token_counts(
        labels[target_proxy], proxies[target_proxy], tokens[target_proxy]
    )
    calibration = partitions["calibration"].copy()
    np.random.default_rng(SEED + 3).shuffle(calibration)
    curve = []
    primary_certificate = None
    for fraction in CALIBRATION_CURVE_FRACTIONS:
        row, certificate = curve_row(
            fraction=fraction,
            calibration_indices=calibration,
            labels=labels,
            sources=sources,
            tokens=tokens,
            proxies=proxies,
            proxy_joint=proxy_joint,
        )
        curve.append(row)
        if fraction == 1.0:
            primary_certificate = certificate

    primary = curve[-1]
    solution = None
    if primary_certificate is not None and primary["decision"] == "deploy":
        solution, _ = optimize(primary_certificate)
    diagnostic = partitions["diagnostic"]
    proxy_accuracy = balanced_accuracy_score(
        sources[diagnostic], proxies[diagnostic]
    )
    raw_predictions = (task_scores[diagnostic] >= 0.5).astype(np.int16)
    raw_balanced_accuracy = balanced_accuracy_score(
        labels[diagnostic], raw_predictions
    )
    if solution is None:
        diagnostic_risk = None
        released_balanced_accuracy = None
        diagnostic_safe = True
    else:
        diagnostic_risk = evaluate_external_channel(
            tokens[diagnostic],
            labels[diagnostic],
            sources[diagnostic],
            solution.release_channel,
            solution.decoder,
        )
        released_balanced_accuracy = expected_balanced_accuracy(
            tokens[diagnostic],
            labels[diagnostic],
            solution.release_channel,
            solution.decoder,
        )
        diagnostic_safe = bool(
            diagnostic_risk.estimable
            and diagnostic_risk.worst_privacy_advantage is not None
            and diagnostic_risk.worst_conditional_error is not None
            and diagnostic_risk.worst_privacy_advantage <= PRIVACY_THRESHOLD
            and diagnostic_risk.worst_conditional_error <= UTILITY_THRESHOLD
        )
    deployed = solution is not None
    gates = {
        "proxy_is_nontrivial": proxy_accuracy >= 0.60,
        "full_calibration_certificate_releases": deployed,
        "released_interface_is_diagnostically_safe": (
            not deployed or diagnostic_safe
        ),
        "no_false_acceptance": not (deployed and not diagnostic_safe),
    }
    payload = {
        "name": "MOSAIC powered real ACS proxy certification v1",
        "status": "complete_locked_powered_followup",
        "lock_sha256": sha256(LOCK),
        "raw_data": raw_asset,
        "protocol": lock["protocol"],
        "sample_counts": {
            name: len(indices) for name, indices in partitions.items()
        },
        "task_columns": list(task_columns),
        "proxy_columns": list(PROXY_COLUMNS),
        "proxy_balanced_accuracy": float(proxy_accuracy),
        "calibration_curve": curve,
        "primary_proxy_certificate": (
            None
            if primary_certificate is None
            else serialize_proxy(primary_certificate)
        ),
        "release": {
            "decision": "deploy" if deployed else "abstain",
            "reason": primary["reason"],
            "certified_source_advantage_upper": primary[
                "certified_source_advantage_upper"
            ],
            "certified_worst_conditional_error_upper": primary[
                "certified_worst_conditional_error_upper"
            ],
            "release_channel": (
                None if solution is None else solution.release_channel.tolist()
            ),
            "decoder": None if solution is None else list(solution.decoder),
            "diagnostic_source_advantage": (
                None
                if diagnostic_risk is None
                else diagnostic_risk.worst_privacy_advantage
            ),
            "diagnostic_worst_conditional_error": (
                None
                if diagnostic_risk is None
                else diagnostic_risk.worst_conditional_error
            ),
            "diagnostic_safe": diagnostic_safe,
        },
        "utility": {
            "unedited_task_score_balanced_accuracy": float(raw_balanced_accuracy),
            "released_interface_expected_balanced_accuracy": (
                None
                if released_balanced_accuracy is None
                else float(released_balanced_accuracy)
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
                "proxy_balanced_accuracy": payload["proxy_balanced_accuracy"],
                "release": payload["release"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
