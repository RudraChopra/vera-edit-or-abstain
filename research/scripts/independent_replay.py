"""Independent finite-reference replay for the VERA controlled-shift protocol.

This implementation reads only the locked protocol, receipts, and candidate
arrays. It intentionally has no project-module imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import beta, binomtest


ATTACKERS = ("linear", "rbf", "forest", "mlp")
DATASETS = ("Waterbirds", "CivilComments-WILDS", "Bios", "GaitPDB")
RULES = (
    "always_deploy",
    "validation_point_selection",
    "iid_ltt",
    "robust_point_estimate",
    "generic_scalar_robust_certificate",
    "vera_fixed_profile",
    "vera_vector_envelope",
    "vera_common_radius",
    "external_oracle",
)
GAMMAS = (1.1, 1.25, 1.5)
BUDGETS = (1000, 2000, 4000, 8000)
SEEDS = tuple(range(45, 109))
PRIMARY_GAMMA = 1.1
PRIMARY_BUDGET = 4000
PRIMARY_ALLOCATION = "targeted_floor_0.15"
GAMMA_CAP = 8.0
SENTINELS = ("Bios", "CivilComments-WILDS", "GaitPDB", "Waterbirds")
CANONICAL_METHOD = {
    "inlp": "INLP",
    "rlace": "R-LACE",
    "leace": "LEACE",
    "taco": "TaCo",
    "mance": "MANCE++",
}
EXPECTED_ARRAYS = {
    f"{field}_{split}"
    for field in (
        "target_harm",
        "identity_target_error",
        "edited_target_error",
        "source",
        "environment",
        "target",
    )
    for split in ("certification", "external")
} | {
    f"leakage_correct_{split}__{attacker}"
    for split in ("certification", "external")
    for attacker in ATTACKERS
} | {
    f"heldout_leakage_correct_{split}__boosted_tree"
    for split in ("certification", "external")
}


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_array(array: np.ndarray, domain: str) -> str:
    contiguous = np.ascontiguousarray(array)
    header = json.dumps(
        {
            "domain": domain,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def hash_array_bytes(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root is not an object: {path.name}")
    return value


def cp_upper(successes: int, n: int, alpha: float) -> float:
    if not 0.0 < alpha < 1.0 or not 0 <= successes <= n or n <= 0:
        raise ValueError("invalid one-sided binomial upper-bound inputs")
    return 1.0 if successes == n else float(
        beta.ppf(1.0 - alpha, successes + 1, n - successes)
    )


def cp_lower(successes: int, n: int, alpha: float) -> float:
    if not 0.0 < alpha < 1.0 or not 0 <= successes <= n or n <= 0:
        raise ValueError("invalid one-sided binomial lower-bound inputs")
    return 0.0 if successes == 0 else float(
        beta.ppf(alpha, successes, n - successes + 1)
    )


def empirical_worst_case(values: np.ndarray, gamma: float) -> float:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or gamma < 1.0:
        raise ValueError("invalid empirical robust-risk inputs")
    ordered = np.sort(array)[::-1]
    remaining = 1.0
    capacity = gamma / len(ordered)
    total = 0.0
    for value in ordered:
        mass = min(capacity, remaining)
        total += mass * float(value)
        remaining -= mass
        if remaining <= 1e-15:
            break
    return float(total)


def target_curve(values: np.ndarray, gamma: float, alpha: float) -> dict[str, Any]:
    array = np.asarray(values)
    if array.ndim != 1 or not len(array) or not np.isin(array, (-1, 0, 1)).all():
        raise ValueError("paired target-harm sample leaves {-1,0,1}")
    n = len(array)
    positive = int(np.sum(array == 1))
    negative = int(np.sum(array == -1))
    positive_upper = cp_upper(positive, n, alpha / 2.0)
    negative_lower = cp_lower(negative, n, alpha / 2.0)
    positive_probability = min(positive_upper, 1.0 - negative_lower)
    zero_probability = max(0.0, 1.0 - negative_lower - positive_probability)
    positive_mass = min(1.0, gamma * positive_probability)
    remaining = 1.0 - positive_mass
    zero_mass = min(remaining, gamma * zero_probability)
    negative_mass = max(0.0, remaining - zero_mass)
    return {
        "n": n,
        "positive_count": positive,
        "zero_count": int(n - positive - negative),
        "negative_count": negative,
        "positive_probability_upper": positive_upper,
        "negative_probability_lower": negative_lower,
        "gamma": float(gamma),
        "empirical_robust_risk": empirical_worst_case(array, gamma),
        "upper_confidence_bound": float(
            min(1.0, max(-1.0, positive_mass - negative_mass))
        ),
    }


def leakage_curve(
    correct: np.ndarray,
    source: np.ndarray,
    profile: Mapping[str, float],
    alpha: float,
) -> dict[str, Any]:
    correctness = np.asarray(correct)
    source_array = np.asarray(source)
    if correctness.ndim != 1 or len(correctness) != len(source_array):
        raise ValueError("leakage correctness and source are not aligned")
    if not np.isin(correctness, (0, 1)).all() or set(np.unique(source_array)) != {0, 1}:
        raise ValueError("invalid balanced-leakage support")
    components: dict[str, Any] = {}
    upper = 0.0
    empirical = 0.0
    for source_class in (0, 1):
        values = correctness[source_array == source_class]
        n = len(values)
        successes = int(np.sum(values))
        probability_upper = cp_upper(successes, n, alpha / 2.0)
        gamma = float(profile[str(source_class)])
        components[str(source_class)] = {
            "n": n,
            "correct_count": successes,
            "incorrect_count": int(n - successes),
            "probability_upper": probability_upper,
            "gamma": gamma,
        }
        upper += 0.5 * min(1.0, gamma * probability_upper)
        empirical += 0.5 * empirical_worst_case(values, gamma)
    return {
        "classes": components,
        "empirical_robust_risk": float(empirical),
        "upper_confidence_bound": float(upper),
    }


def evaluate_profile(
    target_samples: Mapping[str, np.ndarray],
    leakage_samples: Mapping[str, np.ndarray],
    source: np.ndarray,
    target_profile: Mapping[str, float],
    source_profile: Mapping[str, float],
    *,
    alpha: float,
    target_threshold: float,
    leakage_threshold: float,
) -> dict[str, Any]:
    contracts: dict[str, Any] = {}
    for key, values in target_samples.items():
        curve = target_curve(values, float(target_profile[key]), alpha)
        curve["threshold"] = target_threshold
        contracts[key] = curve
    for attacker, values in leakage_samples.items():
        key = f"balanced_leakage::{attacker}"
        curve = leakage_curve(values, source, source_profile, alpha)
        curve["threshold"] = leakage_threshold
        contracts[key] = curve
    margins = {
        key: float(record["threshold"] - record["upper_confidence_bound"])
        for key, record in contracts.items()
    }
    worst = min(margins.values())
    return {
        "passed": all(margin >= 0.0 for margin in margins.values()),
        "contracts": contracts,
        "margins": margins,
        "limiting_contracts": sorted(
            key for key, margin in margins.items() if margin <= worst + 1e-12
        ),
    }


def bisect_radius(predicate, cap: float, tolerance: float = 1e-4) -> tuple[float, bool]:
    if not predicate(1.0):
        return 0.0, False
    if predicate(cap):
        return float(cap), True
    lower, upper = 1.0, float(cap)
    while upper - lower > tolerance:
        midpoint = (lower + upper) / 2.0
        if predicate(midpoint):
            lower = midpoint
        else:
            upper = midpoint
    return float(lower), False


def envelope_geometry(
    target_samples: Mapping[str, np.ndarray],
    leakage_samples: Mapping[str, np.ndarray],
    source: np.ndarray,
    requested_target_profile: Mapping[str, float],
    requested_source_profile: Mapping[str, float],
    *,
    alpha: float,
    target_threshold: float,
    leakage_threshold: float,
    cap: float,
) -> dict[str, Any]:
    iid_target = {key: 1.0 for key in target_samples}
    iid_source = {"0": 1.0, "1": 1.0}
    iid = evaluate_profile(
        target_samples,
        leakage_samples,
        source,
        iid_target,
        iid_source,
        alpha=alpha,
        target_threshold=target_threshold,
        leakage_threshold=leakage_threshold,
    )
    target_intercepts = {key.split("=", 1)[1]: 0.0 for key in target_samples}
    source_intercepts = {"0": 0.0, "1": 0.0}
    right_censored_coordinates: list[str] = []
    if iid["passed"]:
        for key, values in target_samples.items():
            group = key.split("=", 1)[1]
            value, censored = bisect_radius(
                lambda gamma, values=values: target_curve(values, gamma, alpha)[
                    "upper_confidence_bound"
                ]
                <= target_threshold,
                cap,
            )
            target_intercepts[group] = value
            if censored:
                right_censored_coordinates.append(f"target::environment={group}")
        iid_leakage = {
            attacker: leakage_curve(values, source, iid_source, alpha)
            for attacker, values in leakage_samples.items()
        }
        probabilities = {
            attacker: {
                source_class: record["classes"][str(source_class)][
                    "probability_upper"
                ]
                for source_class in (0, 1)
            }
            for attacker, record in iid_leakage.items()
        }
        for source_class in (0, 1):
            other = 1 - source_class

            def source_passes(gamma: float) -> bool:
                return all(
                    0.5
                    * (
                        min(1.0, gamma * values[source_class])
                        + values[other]
                    )
                    <= leakage_threshold
                    for values in probabilities.values()
                )

            value, censored = bisect_radius(source_passes, cap)
            source_intercepts[str(source_class)] = value
            if censored:
                right_censored_coordinates.append(f"source_class={source_class}")

    def common_passes(gamma: float) -> bool:
        profile = evaluate_profile(
            target_samples,
            leakage_samples,
            source,
            {key: gamma for key in target_samples},
            {"0": gamma, "1": gamma},
            alpha=alpha,
            target_threshold=target_threshold,
            leakage_threshold=leakage_threshold,
        )
        return bool(profile["passed"])

    common_radius, common_censored = bisect_radius(common_passes, cap)
    common_gamma = 1.0 if common_radius <= 0.0 else common_radius
    common_profile = evaluate_profile(
        target_samples,
        leakage_samples,
        source,
        {key: common_gamma for key in target_samples},
        {"0": common_gamma, "1": common_gamma},
        alpha=alpha,
        target_threshold=target_threshold,
        leakage_threshold=leakage_threshold,
    )
    worst_common_margin = min(common_profile["margins"].values())
    limiter_tolerance = 1e-4 if common_radius > 0.0 and not common_censored else 1e-12
    common_limiters = sorted(
        key
        for key, margin in common_profile["margins"].items()
        if margin <= worst_common_margin + limiter_tolerance
    )
    target_profile_passes = all(
        float(requested_target_profile[group]) <= radius + 1e-12
        for group, radius in target_intercepts.items()
    )
    leakage_profile_passes = all(
        leakage_curve(values, source, requested_source_profile, alpha)[
            "upper_confidence_bound"
        ]
        <= leakage_threshold + 1e-12
        for values in leakage_samples.values()
    )
    requested_profile_passes = target_profile_passes and leakage_profile_passes
    common_budget = max(
        *requested_target_profile.values(), *requested_source_profile.values()
    )
    return {
        "curve_parameters": iid["contracts"],
        "target_coordinate_axis_intercepts": target_intercepts,
        "source_coordinate_axis_intercepts": source_intercepts,
        "right_censored_coordinates": sorted(right_censored_coordinates),
        "coupled_common_radius": common_radius,
        "common_radius_right_censored": common_censored,
        "common_radius_limiting_contracts": common_limiters,
        "common_radius_contract_margins": common_profile["margins"],
        "requested_profile_in_envelope": requested_profile_passes,
        "requested_common_profile_in_envelope": common_radius >= common_budget,
    }


def conditional_profile(weights: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    output: dict[str, float] = {}
    for group in sorted(map(int, np.unique(groups))):
        local = weights[groups == group]
        output[str(group)] = max(1.0, float(local.max() / local.mean()))
    return output


def construct_shift(
    environment: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    design_environment: np.ndarray,
    design_source: np.ndarray,
    design_target: np.ndarray,
    requested_gamma: float,
    minimum_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    options: list[tuple[float, int, int, int, int]] = []
    for environment_value in sorted(map(int, np.unique(design_environment))):
        environment_mask = design_environment == environment_value
        environment_count = int(np.sum(environment_mask))
        for source_value in sorted(map(int, np.unique(design_source[environment_mask]))):
            for target_value in sorted(map(int, np.unique(design_target[environment_mask]))):
                mask = (
                    environment_mask
                    & (design_source == source_value)
                    & (design_target == target_value)
                )
                count = int(np.sum(mask))
                full_environment = environment == environment_value
                full_focus = (
                    full_environment
                    & (source == source_value)
                    & (target == target_value)
                )
                if count < minimum_count or not np.any(full_environment):
                    continue
                probability = float(np.sum(full_focus) / np.sum(full_environment))
                if probability < 1.0 and requested_gamma * probability <= 1.0 + 1e-12:
                    options.append(
                        (
                            count / environment_count,
                            environment_value,
                            source_value,
                            target_value,
                            count,
                        )
                    )
    if not options:
        raise RuntimeError("no design-supported focus cell realizes the shift")
    _, focus_environment, focus_source, focus_target, _ = min(options)
    environment_mask = environment == focus_environment
    focus_mask = (
        environment_mask & (source == focus_source) & (target == focus_target)
    )
    focus_probability = float(np.sum(focus_mask) / np.sum(environment_mask))
    nonfocus = (1.0 - focus_probability * requested_gamma) / (1.0 - focus_probability)
    weights = np.ones(len(environment), dtype=float)
    weights[environment_mask & ~focus_mask] = max(0.0, nonfocus)
    weights[focus_mask] = requested_gamma
    weights /= weights.mean()
    probabilities = weights / weights.sum()
    profile = {
        "requested_gamma": requested_gamma,
        "focus_environment": focus_environment,
        "focus_source": focus_source,
        "focus_target": focus_target,
        "focus_probability_within_environment": focus_probability,
        "nonfocus_weight": max(0.0, nonfocus),
        "global_density_ratio_cap": float(weights.max()),
        "target_profile": conditional_profile(weights, environment),
        "source_profile": conditional_profile(weights, source),
        "reference_size": len(environment),
        "design_size": len(design_environment),
    }
    if (
        not np.isclose(probabilities.sum(), 1.0)
        or np.any(probabilities < 0.0)
        or float((probabilities * len(probabilities)).max())
        > requested_gamma + 1e-10
    ):
        raise RuntimeError("constructed shift violates its density-ratio cap")
    return probabilities, profile


def integer_allocation(
    scores: Mapping[str, float], total: int, minimum: int
) -> dict[str, int]:
    if total < minimum * len(scores):
        raise ValueError("budget is below the cell minimum")
    remaining = total - minimum * len(scores)
    total_score = sum(scores.values())
    ideals = {key: remaining * value / total_score for key, value in scores.items()}
    allocation = {
        key: minimum + int(np.floor(value)) for key, value in ideals.items()
    }
    leftover = total - sum(allocation.values())
    order = sorted(
        scores,
        key=lambda key: (-(ideals[key] - np.floor(ideals[key])), key),
    )
    for key in order[:leftover]:
        allocation[key] += 1
    if sum(allocation.values()) != total:
        raise RuntimeError("integer allocation does not exhaust the budget")
    return allocation


def registered_strengths(method_key: str, config: Mapping[str, Any]) -> tuple[str, ...]:
    candidate = config["candidate_configuration"]
    if method_key in {"inlp", "rlace"}:
        return tuple(f"rank={int(value)}" for value in candidate["ranks"])
    if method_key == "leace":
        return (str(candidate["candidate"]),)
    if method_key == "taco":
        return tuple(
            f"components_removed={int(value)}" for value in candidate["removals"]
        )
    if method_key == "mance":
        return (
            f"epsilon={float(candidate['epsilon']):g},steps={int(candidate['steps'])}",
        )
    raise RuntimeError(f"unknown method key: {method_key}")


def split_view(arrays: Mapping[str, np.ndarray], split: str) -> dict[str, np.ndarray]:
    output = {
        "target_harm": arrays[f"target_harm_{split}"],
        "source": arrays[f"source_{split}"],
        "environment": arrays[f"environment_{split}"],
        "target": arrays[f"target_{split}"],
    }
    for attacker in ATTACKERS:
        output[f"leakage::{attacker}"] = arrays[
            f"leakage_correct_{split}__{attacker}"
        ]
    return output


def load_candidate_frontier(
    receipt_root: Path,
    study: Mapping[str, Any],
    dataset: str,
    seed: int,
    prereg_hash: str,
) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    shared_metadata: dict[str, np.ndarray] | None = None
    split_signature: str | None = None
    for method_key, config in study["methods"].items():
        method = CANONICAL_METHOD.get(method_key)
        if method is None:
            raise RuntimeError(f"unregistered method key: {method_key}")
        receipt_path = receipt_root / f"{dataset}__{method_key}__seed-{seed}.json"
        receipt = load_object(receipt_path)
        if (
            receipt.get("dataset") != dataset
            or int(receipt.get("seed", -1)) != seed
            or receipt.get("method_family") != method_key
            or receipt.get("prereg_sha256") != prereg_hash
            or receipt.get("claim_grade") is not True
            or receipt.get("smoke") is not False
            or receipt.get("claim_configuration_verified") is not True
            or receipt.get("external_labels_locked_during_edit_construction")
            is not True
            or receipt.get("heldout_attacker") != study["heldout_attacker"]
        ):
            raise RuntimeError(f"receipt identity mismatch: {receipt_path.name}")
        observed_signature = json.dumps(
            {
                split: receipt["indices"][split]["sha256"]
                for split in ("train", "construction", "certification", "external")
            },
            sort_keys=True,
        )
        if split_signature is None:
            split_signature = observed_signature
        elif split_signature != observed_signature:
            raise RuntimeError(f"receipt split hashes disagree: {dataset}/seed-{seed}")
        expected_keys = {
            f"{method}::{strength}" for strength in registered_strengths(method_key, config)
        }
        candidates = receipt.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != len(expected_keys):
            raise RuntimeError(f"candidate count mismatch: {receipt_path.name}")
        observed: set[str] = set()
        for record in candidates:
            candidate_method = str(record.get("method", ""))
            strength = str(record.get("strength", ""))
            key = str(record.get("candidate_key", ""))
            if candidate_method != method or key != f"{candidate_method}::{strength}":
                raise RuntimeError(f"candidate key mismatch: {receipt_path.name}")
            if key not in expected_keys or key in observed:
                raise RuntimeError(f"unexpected or duplicate candidate key: {key}")
            if "proxy" in json.dumps(record, sort_keys=True).lower():
                raise RuntimeError(f"proxy candidate is forbidden: {key}")
            observed.add(key)
            archive_path = Path(str(record.get("audit_npz", "")))
            if archive_path.is_symlink():
                raise RuntimeError(f"candidate archive is a symlink: {key}")
            if hash_file(archive_path) != record.get("audit_npz_sha256"):
                raise RuntimeError(f"candidate archive hash mismatch: {key}")
            with np.load(archive_path, allow_pickle=False) as archive:
                if set(archive.files) != EXPECTED_ARRAYS:
                    raise RuntimeError(f"candidate array contract mismatch: {key}")
                arrays = {name: np.asarray(archive[name]) for name in archive.files}
            certification_n = int(receipt["indices"]["certification"]["n"])
            external_n = int(receipt["indices"]["external"]["n"])
            for name, array in arrays.items():
                expected_n = certification_n if "certification" in name else external_n
                if (
                    array.ndim != 1
                    or len(array) != expected_n
                    or array.dtype.kind not in "biuf"
                    or not np.isfinite(array).all()
                ):
                    raise RuntimeError(f"candidate array shape/type mismatch: {key}/{name}")
                if (
                    name.startswith("leakage_correct_")
                    or name.startswith("heldout_leakage_correct_")
                ) and not np.isin(array, (0, 1)).all():
                    raise RuntimeError(f"correctness array leaves binary support: {key}/{name}")
            for split in ("certification", "external"):
                identity = arrays[f"identity_target_error_{split}"]
                edited = arrays[f"edited_target_error_{split}"]
                harm = arrays[f"target_harm_{split}"]
                if (
                    not np.isin(identity, (0, 1)).all()
                    or not np.isin(edited, (0, 1)).all()
                    or not np.isin(harm, (-1, 0, 1)).all()
                    or not np.array_equal(edited - identity, harm)
                ):
                    raise RuntimeError(f"paired-harm reconstruction failed: {key}/{split}")
                source_values = arrays[f"source_{split}"]
                environment_values = arrays[f"environment_{split}"]
                target_values = arrays[f"target_{split}"]
                if (
                    not np.isin(source_values, receipt["source_classes"][split]).all()
                    or not np.isin(
                        environment_values, receipt["environment_classes"][split]
                    ).all()
                    or not np.array_equal(target_values, np.rint(target_values))
                ):
                    raise RuntimeError(f"metadata leaves declared support: {key}/{split}")
            metadata = {
                name: arrays[name]
                for name in (
                    "source_certification",
                    "environment_certification",
                    "target_certification",
                    "source_external",
                    "environment_external",
                    "target_external",
                    "identity_target_error_certification",
                    "identity_target_error_external",
                )
            }
            if shared_metadata is None:
                shared_metadata = metadata
            elif any(
                not np.array_equal(array, shared_metadata[name])
                for name, array in metadata.items()
            ):
                raise RuntimeError(f"candidate shared-array mismatch: {dataset}/seed-{seed}")
            loaded.append(
                {
                    "canonical_candidate_key": key,
                    "legacy_cap4_candidate_key": (
                        key.replace("R-LACE::", "RLACE::", 1)
                        if key.startswith("R-LACE::")
                        else key
                    ),
                    "method": candidate_method,
                    "strength": strength,
                    "arrays": arrays,
                    "certification": split_view(arrays, "certification"),
                    "design": split_view(arrays, "external"),
                    "audit_npz_sha256": str(record["audit_npz_sha256"]),
                    "_audit_npz_path": str(archive_path.resolve()),
                    "receipt_certification_split_sha256": str(
                        receipt["indices"]["certification"]["sha256"]
                    ),
                }
            )
        if observed != expected_keys:
            raise RuntimeError(f"candidate frontier mismatch: {receipt_path.name}")
    loaded.sort(key=lambda candidate: candidate["canonical_candidate_key"])
    if len(loaded) != 12:
        raise RuntimeError(f"dataset-seed frontier is not 12 candidates: {dataset}/{seed}")
    return loaded


def exact_shifted_metrics(
    candidate: Mapping[str, np.ndarray], probabilities: np.ndarray
) -> tuple[float, float]:
    target_risks = []
    for environment in sorted(map(int, np.unique(candidate["environment"]))):
        mask = candidate["environment"] == environment
        conditional = probabilities[mask] / probabilities[mask].sum()
        target_risks.append(float(np.dot(conditional, candidate["target_harm"][mask])))
    leakage_risks = []
    for attacker in ATTACKERS:
        recalls = []
        for source_class in (0, 1):
            mask = candidate["source"] == source_class
            conditional = probabilities[mask] / probabilities[mask].sum()
            recalls.append(
                float(np.dot(conditional, candidate[f"leakage::{attacker}"][mask]))
            )
        leakage_risks.append(float(np.mean(recalls)))
    return max(target_risks), max(leakage_risks)


def sampled_shifted_metrics(
    candidate: Mapping[str, np.ndarray], indices: np.ndarray
) -> tuple[float, float]:
    target_risks = [
        float(
            candidate["target_harm"]
            [indices[candidate["environment"][indices] == environment]].mean()
        )
        for environment in sorted(map(int, np.unique(candidate["environment"])))
    ]
    leakage_risks = []
    for attacker in ATTACKERS:
        recalls = [
            float(
                candidate[f"leakage::{attacker}"]
                [indices[candidate["source"][indices] == source_class]].mean()
            )
            for source_class in (0, 1)
        ]
        leakage_risks.append(float(np.mean(recalls)))
    return max(target_risks), max(leakage_risks)


def design_risks(
    candidate: Mapping[str, np.ndarray],
    indices: np.ndarray,
    target_profile: Mapping[str, float],
    source_profile: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float], float, float]:
    target_risk = {
        environment: empirical_worst_case(
            candidate["target_harm"]
            [indices[candidate["environment"][indices] == int(environment)]],
            gamma,
        )
        for environment, gamma in target_profile.items()
    }
    leakage_risk: dict[str, float] = {}
    for attacker in ATTACKERS:
        recalls = []
        for source_class, gamma in source_profile.items():
            values = candidate[f"leakage::{attacker}"][
                indices[candidate["source"][indices] == int(source_class)]
            ]
            recalls.append(empirical_worst_case(values, gamma))
        leakage_risk[attacker] = float(np.mean(recalls))
    return (
        target_risk,
        leakage_risk,
        max(target_risk.values()),
        max(leakage_risk.values()),
    )


def allocation_plan(
    candidates: list[dict[str, Any]],
    design_indices: np.ndarray,
    target_profile: Mapping[str, float],
    source_profile: Mapping[str, float],
    target_threshold: float,
    leakage_threshold: float,
) -> tuple[dict[str, float], str]:
    ranked: list[tuple[Any, ...]] = []
    for candidate in candidates:
        target, leakage, target_max, leakage_max = design_risks(
            candidate["design"], design_indices, target_profile, source_profile
        )
        margin = min(
            *((target_threshold - value) / 2.0 for value in target.values()),
            *(leakage_threshold - value for value in leakage.values()),
        )
        ranked.append(
            (
                margin,
                -leakage_max,
                -target_max,
                candidate["canonical_candidate_key"],
                target,
                leakage,
            )
        )
    _, _, _, selected_key, target, leakage = max(ranked)
    leakage_margin = max(0.01, leakage_threshold - max(leakage.values()))
    scores = {
        f"target::{environment}": (
            2.0 * gamma / max(0.01, target_threshold - target[environment])
        )
        ** 2
        for environment, gamma in target_profile.items()
    }
    scores.update(
        {
            f"source::{source_class}": (0.5 * gamma / leakage_margin) ** 2
            for source_class, gamma in source_profile.items()
        }
    )
    return scores, str(selected_key)


def draw_certification_streams(
    metadata: Mapping[str, np.ndarray],
    allocation: Mapping[str, int],
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[int, np.ndarray]]:
    target_indices: dict[str, np.ndarray] = {}
    source_indices: dict[int, np.ndarray] = {}
    for cell, count in allocation.items():
        if cell.startswith("target::"):
            environment = int(cell.split("::", 1)[1])
            key = f"target::environment={environment}"
            target_indices[key] = rng.choice(
                np.flatnonzero(metadata["environment"] == environment),
                size=count,
                replace=True,
            )
    for cell, count in allocation.items():
        if cell.startswith("source::"):
            source_class = int(cell.split("::", 1)[1])
            source_indices[source_class] = rng.choice(
                np.flatnonzero(metadata["source"] == source_class),
                size=count,
                replace=True,
            )
    return target_indices, source_indices


def sampled_contracts(
    candidate: Mapping[str, np.ndarray],
    target_indices: Mapping[str, np.ndarray],
    source_indices: Mapping[int, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    target = {
        key: candidate["target_harm"][indices]
        for key, indices in target_indices.items()
    }
    ordered = [source_indices[source_class] for source_class in (0, 1)]
    source = np.concatenate(
        [
            np.full(len(indices), source_class, dtype=np.int64)
            for source_class, indices in zip((0, 1), ordered)
        ]
    )
    leakage = {
        attacker: np.concatenate(
            [candidate[f"leakage::{attacker}"][indices] for indices in ordered]
        )
        for attacker in ATTACKERS
    }
    return target, leakage, source


def point_risks(
    target: Mapping[str, np.ndarray],
    leakage: Mapping[str, np.ndarray],
    source: np.ndarray,
) -> tuple[float, float]:
    target_max = max(float(values.mean()) for values in target.values())
    leakage_max = max(
        0.5
        * sum(
            float(values[source == source_class].mean())
            for source_class in (0, 1)
        )
        for values in leakage.values()
    )
    return target_max, leakage_max


def select_candidate(
    records: list[dict[str, Any]], eligibility: str | None
) -> dict[str, Any] | None:
    eligible = records if eligibility is None else [
        record for record in records if record[eligibility]
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda record: (
            record["point_leakage"],
            record["point_target"],
            record["canonical_candidate_key"],
        ),
    )


def evaluate_configuration(
    candidates: list[dict[str, Any]],
    metadata: Mapping[str, np.ndarray],
    target_profile: Mapping[str, float],
    source_profile: Mapping[str, float],
    allocation: Mapping[str, int],
    rng: np.random.Generator,
    *,
    delta: float,
    target_threshold: float,
    leakage_threshold: float,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    target_indices, source_indices = draw_certification_streams(
        metadata, allocation, rng
    )
    candidate_count = len(candidates)
    family_size = candidate_count * (len(target_profile) + len(ATTACKERS))
    local_alpha = delta / family_size
    candidate_alpha = delta / candidate_count
    target_key_profile = {
        f"target::environment={environment}": gamma
        for environment, gamma in target_profile.items()
    }
    evaluated: list[dict[str, Any]] = []
    detail_records: list[dict[str, Any]] = []
    for candidate in candidates:
        target, leakage, source = sampled_contracts(
            candidate["certification"], target_indices, source_indices
        )
        point_target, point_leakage = point_risks(target, leakage, source)
        iid = evaluate_profile(
            target,
            leakage,
            source,
            {key: 1.0 for key in target},
            {"0": 1.0, "1": 1.0},
            alpha=candidate_alpha,
            target_threshold=target_threshold,
            leakage_threshold=leakage_threshold,
        )
        fixed = evaluate_profile(
            target,
            leakage,
            source,
            target_key_profile,
            source_profile,
            alpha=candidate_alpha,
            target_threshold=target_threshold,
            leakage_threshold=leakage_threshold,
        )
        profile = evaluate_profile(
            target,
            leakage,
            source,
            target_key_profile,
            source_profile,
            alpha=local_alpha,
            target_threshold=target_threshold,
            leakage_threshold=leakage_threshold,
        )
        geometry = envelope_geometry(
            target,
            leakage,
            source,
            target_profile,
            source_profile,
            alpha=local_alpha,
            target_threshold=target_threshold,
            leakage_threshold=leakage_threshold,
            cap=GAMMA_CAP,
        )
        robust_point_eligible = all(
            contract["empirical_robust_risk"] <= contract["threshold"]
            for contract in profile["contracts"].values()
        )
        scalar_score = float(
            np.mean(
                [
                    contract["upper_confidence_bound"] / contract["threshold"]
                    for contract in profile["contracts"].values()
                ]
            )
        )
        q_target, q_leakage = candidate["q_metrics"]
        evaluation_target, evaluation_leakage = candidate["evaluation_metrics"]
        q_safe = q_target <= target_threshold and q_leakage <= leakage_threshold
        evaluation_safe = (
            evaluation_target <= target_threshold
            and evaluation_leakage <= leakage_threshold
        )
        coordinate_radii = {
            **{
                f"target::environment={key}": value
                for key, value in geometry[
                    "target_coordinate_axis_intercepts"
                ].items()
            },
            **{
                f"source::{key}": value
                for key, value in geometry[
                    "source_coordinate_axis_intercepts"
                ].items()
            },
        }
        minimum_intercept = min(coordinate_radii.values())
        axis_limiters = sorted(
            key
            for key, value in coordinate_radii.items()
            if value <= minimum_intercept + 1e-12
        )
        record = {
            "canonical_candidate_key": candidate["canonical_candidate_key"],
            "legacy_cap4_candidate_key": candidate["legacy_cap4_candidate_key"],
            "method": candidate["method"],
            "point_target": point_target,
            "point_leakage": point_leakage,
            "point_feasible": (
                point_target <= target_threshold and point_leakage <= leakage_threshold
            ),
            "iid_eligible": iid["passed"],
            "robust_point_eligible": robust_point_eligible,
            "scalar_eligible": scalar_score <= 1.0,
            "fixed_eligible": fixed["passed"],
            "vector_eligible": geometry["requested_profile_in_envelope"],
            "common_eligible": geometry["requested_common_profile_in_envelope"],
            "q_safe": q_safe,
            "evaluation_safe": evaluation_safe,
            "q_target": q_target,
            "q_leakage": q_leakage,
            "evaluation_target": evaluation_target,
            "evaluation_leakage": evaluation_leakage,
            "envelope_radius": geometry["coupled_common_radius"],
            "target_environment_radii": geometry[
                "target_coordinate_axis_intercepts"
            ],
            "source_class_radii": geometry["source_coordinate_axis_intercepts"],
            "limiting_coordinates": axis_limiters,
            "axis_limiting_coordinates": axis_limiters,
            "common_limiting_contracts": geometry[
                "common_radius_limiting_contracts"
            ],
            "common_radius_right_censored": geometry[
                "common_radius_right_censored"
            ],
            "fixed_profile_limiting_contracts": fixed["limiting_contracts"],
        }
        evaluated.append(record)
        detail_records.append(
            {
                **record,
                "eraser_family": candidate["method"],
                "family_size": family_size,
                "local_error_budget": local_alpha,
                "target_threshold": target_threshold,
                "leakage_threshold": leakage_threshold,
                "gamma_cap": GAMMA_CAP,
                "requested_target_profile": dict(target_profile),
                "requested_source_profile": dict(source_profile),
                "requested_profile_in_envelope": geometry[
                    "requested_profile_in_envelope"
                ],
                "target_sufficient_statistics": {
                    key: {
                        "n": len(value),
                        "positive_count": int(np.sum(value == 1)),
                        "zero_count": int(np.sum(value == 0)),
                        "negative_count": int(np.sum(value == -1)),
                    }
                    for key, value in target.items()
                },
                "leakage_sufficient_statistics": {
                    attacker: {
                        str(source_class): {
                            "n": int(np.sum(source == source_class)),
                            "correct_count": int(
                                np.sum(leakage[attacker][source == source_class])
                            ),
                            "incorrect_count": int(
                                np.sum(source == source_class)
                                - np.sum(leakage[attacker][source == source_class])
                            ),
                        }
                        for source_class in (0, 1)
                    }
                    for attacker in ATTACKERS
                },
                "curve_parameters": geometry["curve_parameters"],
                "target_coordinate_axis_intercepts": geometry[
                    "target_coordinate_axis_intercepts"
                ],
                "source_coordinate_axis_intercepts": geometry[
                    "source_coordinate_axis_intercepts"
                ],
                "right_censored_coordinates": geometry[
                    "right_censored_coordinates"
                ],
                "coupled_common_radius": geometry["coupled_common_radius"],
                "common_radius_contract_margins": geometry[
                    "common_radius_contract_margins"
                ],
                "certification_index_sha256": {
                    "target": {
                        key: hash_array(value, f"target indices::{key}")
                        for key, value in target_indices.items()
                    },
                    "source": {
                        str(source_class): hash_array(
                            value, f"source indices::{source_class}"
                        )
                        for source_class, value in source_indices.items()
                    },
                },
                "sampled_source_sha256": hash_array(
                    source, "sampled certification source labels"
                ),
                "certification_source_sha256": hash_array(
                    np.asarray(metadata["source"]),
                    "full certification source labels",
                ),
                "receipt_certification_split_sha256": candidate[
                    "receipt_certification_split_sha256"
                ],
                "audit_npz_sha256": candidate["audit_npz_sha256"],
            }
        )

    selections = {
        "always_deploy": select_candidate(evaluated, None),
        "validation_point_selection": select_candidate(evaluated, "point_feasible"),
        "iid_ltt": select_candidate(evaluated, "iid_eligible"),
        "robust_point_estimate": select_candidate(
            evaluated, "robust_point_eligible"
        ),
        "generic_scalar_robust_certificate": select_candidate(
            evaluated, "scalar_eligible"
        ),
        "vera_fixed_profile": select_candidate(evaluated, "fixed_eligible"),
        "vera_vector_envelope": select_candidate(evaluated, "vector_eligible"),
        "vera_common_radius": select_candidate(evaluated, "common_eligible"),
        "external_oracle": select_candidate(evaluated, "q_safe"),
    }
    decisions: dict[str, dict[str, Any]] = {}
    for rule, selected in selections.items():
        decisions[rule] = {
            "deployed": selected is not None,
            "safe": bool(selected is not None and selected["q_safe"]),
            "violation": bool(selected is not None and not selected["q_safe"]),
            "evaluation_violation": bool(
                selected is not None and not selected["evaluation_safe"]
            ),
            "selected_candidate": (
                "" if selected is None else selected["canonical_candidate_key"]
            ),
            "canonical_candidate_key": (
                "" if selected is None else selected["canonical_candidate_key"]
            ),
            "legacy_cap4_candidate_key": (
                "" if selected is None else selected["legacy_cap4_candidate_key"]
            ),
            "selected_method": "" if selected is None else selected["method"],
            "q_target": None if selected is None else selected["q_target"],
            "q_leakage": None if selected is None else selected["q_leakage"],
            "evaluation_target": (
                None if selected is None else selected["evaluation_target"]
            ),
            "evaluation_leakage": (
                None if selected is None else selected["evaluation_leakage"]
            ),
            "certified_common_radius": (
                0.0 if selected is None else selected["envelope_radius"]
            ),
            "target_environment_radii": (
                {} if selected is None else selected["target_environment_radii"]
            ),
            "source_class_radii": (
                {} if selected is None else selected["source_class_radii"]
            ),
            "limiting_coordinates": (
                [] if selected is None else selected["limiting_coordinates"]
            ),
            "axis_limiting_coordinates": (
                [] if selected is None else selected["axis_limiting_coordinates"]
            ),
            "common_limiting_contracts": (
                [] if selected is None else selected["common_limiting_contracts"]
            ),
            "common_radius_right_censored": bool(
                selected is not None and selected["common_radius_right_censored"]
            ),
            "fixed_profile_limiting_contracts": (
                []
                if selected is None
                else selected["fixed_profile_limiting_contracts"]
            ),
        }
    return decisions, detail_records


def heldout_balanced_accuracy(
    arrays: Mapping[str, np.ndarray], probabilities: np.ndarray
) -> float:
    correct = arrays["heldout_leakage_correct_certification__boosted_tree"]
    source = arrays["source_certification"]
    recalls = []
    for source_class in (0, 1):
        mask = source == source_class
        conditional = probabilities[mask] / probabilities[mask].sum()
        recalls.append(float(np.dot(conditional, correct[mask])))
    return float(np.mean(recalls))


def per_attacker_shifted_accuracy(
    candidate: Mapping[str, np.ndarray], probabilities: np.ndarray
) -> dict[str, float]:
    output: dict[str, float] = {}
    for attacker in ATTACKERS:
        recalls = []
        for source_class in (0, 1):
            mask = candidate["source"] == source_class
            conditional = probabilities[mask] / probabilities[mask].sum()
            recalls.append(
                float(
                    np.dot(
                        conditional,
                        candidate[f"leakage::{attacker}"][mask],
                    )
                )
            )
        output[attacker] = float(np.mean(recalls))
    return output


def attach_postselection_stress(
    decisions: dict[str, dict[str, Any]],
    candidates: list[dict[str, Any]],
    probabilities: np.ndarray,
    leakage_threshold: float,
) -> None:
    by_key = {
        candidate["canonical_candidate_key"]: candidate for candidate in candidates
    }
    for decision in decisions.values():
        selected_key = str(decision["canonical_candidate_key"])
        if not selected_key:
            decision["heldout_leakage"] = None
            decision["heldout_stress_violation"] = False
            decision["registered_attacker_q"] = {}
            continue
        candidate = by_key[selected_key]
        heldout = heldout_balanced_accuracy(candidate["arrays"], probabilities)
        decision["heldout_leakage"] = heldout
        decision["heldout_stress_violation"] = heldout > leakage_threshold
        decision["registered_attacker_q"] = per_attacker_shifted_accuracy(
            candidate["certification"], probabilities
        )


def exact_interval(events: int, n: int, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if events == 0 else float(
        beta.ppf(alpha / 2.0, events, n - events + 1)
    )
    upper = 1.0 if events == n else float(
        beta.ppf(1.0 - alpha / 2.0, events + 1, n - events)
    )
    return [lower, upper]


def holm_adjustment(values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda key: (values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * values[key]))
        adjusted[key] = running
    return adjusted


def bootstrap_primary(
    rows: list[dict[str, Any]], replicates: int = 20_000
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_key = {
        (int(row["seed"]), str(row["dataset"]), str(row["rule"])): row
        for row in rows
    }
    seeds = np.asarray(SEEDS, dtype=int)

    def statistic(sample: np.ndarray) -> tuple[dict[str, float], dict[str, int]]:
        opportunities = vector_safe = common_safe = 0
        point_violations = vector_violations = 0
        point_deployments = vector_deployments = 0
        for seed in sample:
            for dataset in DATASETS:
                oracle = by_key[(int(seed), dataset, "external_oracle")]
                point = by_key[(int(seed), dataset, "validation_point_selection")]
                vector = by_key[(int(seed), dataset, "vera_vector_envelope")]
                common = by_key[(int(seed), dataset, "vera_common_radius")]
                opportunities += int(bool(oracle["deployed"]))
                vector_safe += int(bool(vector["safe"]))
                common_safe += int(bool(common["safe"]))
                point_violations += int(bool(point["violation"]))
                vector_violations += int(bool(vector["violation"]))
                point_deployments += int(bool(point["deployed"]))
                vector_deployments += int(bool(vector["deployed"]))
        decisions = len(sample) * len(DATASETS)
        values = {
            "vector_safe_retention": (
                np.nan if opportunities == 0 else vector_safe / opportunities
            ),
            "common_safe_retention": (
                np.nan if opportunities == 0 else common_safe / opportunities
            ),
            "vector_to_common_ratio": (
                np.nan if common_safe == 0 else vector_safe / common_safe
            ),
            "point_violation_rate": point_violations / decisions,
            "vector_violation_rate": vector_violations / decisions,
            "absolute_violation_reduction": (
                point_violations - vector_violations
            )
            / decisions,
            "point_deployment_rate": point_deployments / decisions,
            "vector_deployment_rate": vector_deployments / decisions,
        }
        counts = {
            "opportunities": opportunities,
            "vector_safe": vector_safe,
            "common_safe": common_safe,
        }
        return values, counts

    point, point_counts = statistic(seeds)
    rng = np.random.default_rng(2_027_071_601)
    samples: list[dict[str, float]] = []
    sample_counts: list[dict[str, int]] = []
    for _ in range(replicates):
        values, counts = statistic(rng.choice(seeds, size=len(seeds), replace=True))
        samples.append(values)
        sample_counts.append(counts)
    intervals: dict[str, list[float | None]] = {}
    for key in point:
        values = np.asarray([sample[key] for sample in samples], dtype=float)
        finite = values[np.isfinite(values)]
        intervals[key] = (
            [None, None]
            if not len(finite)
            else [
                float(np.quantile(finite, 0.025)),
                float(np.quantile(finite, 0.975)),
            ]
        )
    ordinary = {
        "unit": "seed cluster across all supported datasets",
        "replicates": replicates,
        "random_seed": 2_027_071_601,
        "point_estimates": {
            key: None if not np.isfinite(value) else value
            for key, value in point.items()
        },
        "confidence_intervals_95": intervals,
    }
    zero_opportunity = sum(counts["opportunities"] == 0 for counts in sample_counts)
    positive_opportunity = replicates - zero_opportunity
    impossible = sum(
        counts["opportunities"] == 0 and counts["vector_safe"] > 0
        for counts in sample_counts
    )
    completed_retention = np.asarray(
        [
            0.0
            if counts["opportunities"] == 0
            else counts["vector_safe"] / counts["opportunities"]
            for counts in sample_counts
        ],
        dtype=float,
    )
    division_free = np.asarray(
        [counts["vector_safe"] - 0.20 * counts["opportunities"] for counts in sample_counts],
        dtype=float,
    )
    sensitivity = {
        "positive_opportunity_replicates": positive_opportunity,
        "zero_opportunity_replicates": zero_opportunity,
        "impossible_positive_retention_without_opportunity": impossible,
        "completed_retention_interval_95": [
            float(np.quantile(completed_retention, 0.025)),
            float(np.quantile(completed_retention, 0.975)),
        ],
        "division_free_margin_interval_95": [
            float(np.quantile(division_free, 0.025)),
            float(np.quantile(division_free, 0.975)),
        ],
        "point_counts": point_counts,
    }
    if positive_opportunity + zero_opportunity != replicates or impossible != 0:
        raise RuntimeError("bootstrap zero-denominator accounting failed")
    return ordinary, sensitivity


def infer_primary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 2_304:
        raise RuntimeError(f"primary row count mismatch: {len(rows)}")
    by_key = {
        (int(row["seed"]), str(row["dataset"]), str(row["rule"])): row
        for row in rows
    }
    if len(by_key) != len(rows):
        raise RuntimeError("duplicate primary decision key")
    differences = []
    for seed in SEEDS:
        point_count = sum(
            bool(by_key[(seed, dataset, "validation_point_selection")]["violation"])
            for dataset in DATASETS
        )
        vector_count = sum(
            bool(by_key[(seed, dataset, "vera_vector_envelope")]["violation"])
            for dataset in DATASETS
        )
        differences.append(point_count - vector_count)
    favorable = sum(value > 0 for value in differences)
    adverse = sum(value < 0 for value in differences)
    non_ties = favorable + adverse
    sign_p = 1.0 if non_ties == 0 else float(
        binomtest(favorable, non_ties, p=0.5, alternative="two-sided").pvalue
    )
    sentinel_events = [
        bool(
            by_key[
                (
                    seed,
                    SENTINELS[(seed - SEEDS[0]) % len(SENTINELS)],
                    "vera_vector_envelope",
                )
            ]["violation"]
        )
        for seed in SEEDS
    ]
    event_count = sum(sentinel_events)
    bootstrap, _usefulness_sensitivity = bootstrap_primary(rows)
    estimates = bootstrap["point_estimates"]
    intervals = bootstrap["confidence_intervals_95"]
    point_rate = float(estimates["point_violation_rate"])
    vector_rate = float(estimates["vector_violation_rate"])
    relative_reduction = (
        None if point_rate == 0.0 else (point_rate - vector_rate) / point_rate
    )
    per_dataset: dict[str, Any] = {}
    raw_p: dict[str, float] = {}
    for dataset in DATASETS:
        point_only = vector_only = 0
        for seed in SEEDS:
            point_event = bool(
                by_key[(seed, dataset, "validation_point_selection")]["violation"]
            )
            vector_event = bool(
                by_key[(seed, dataset, "vera_vector_envelope")]["violation"]
            )
            point_only += int(point_event and not vector_event)
            vector_only += int(vector_event and not point_event)
        discordant = point_only + vector_only
        value = 1.0 if discordant == 0 else float(
            binomtest(point_only, discordant, p=0.5, alternative="two-sided").pvalue
        )
        raw_p[dataset] = value
        per_dataset[dataset] = {
            "point_only_violation_seeds": point_only,
            "vector_only_violation_seeds": vector_only,
            "raw_exact_p": value,
        }
    adjusted = holm_adjustment(raw_p)
    for dataset in DATASETS:
        per_dataset[dataset]["holm_adjusted_p"] = adjusted[dataset]
    radii = [
        float(row["certified_common_radius"])
        for row in rows
        if row["rule"] == "vera_vector_envelope" and row["deployed"]
    ]
    axis_limiting = Counter(
        coordinate
        for row in rows
        if row["rule"] == "vera_vector_envelope" and row["deployed"]
        for coordinate in row["limiting_coordinates"]
    )
    common_limiting = Counter(
        contract
        for row in rows
        if row["rule"] == "vera_vector_envelope" and row["deployed"]
        for contract in row["common_limiting_contracts"]
    )
    vector_when_common_abstains = sum(
        bool(by_key[(seed, dataset, "vera_vector_envelope")]["deployed"])
        and not bool(by_key[(seed, dataset, "vera_common_radius")]["deployed"])
        for seed in SEEDS
        for dataset in DATASETS
    )
    vector_deployments = [
        row
        for row in rows
        if row["rule"] == "vera_vector_envelope" and row["deployed"]
    ]
    portfolio_safe = [row for row in vector_deployments if row["safe"]]
    heldout_failures = sum(
        bool(row["heldout_stress_violation"]) for row in portfolio_safe
    )
    safety_upper = cp_upper(event_count, len(SEEDS), 0.05)
    paired_pass = favorable > adverse and sign_p < 0.05
    safety_pass = safety_upper <= 0.05
    usefulness_lower = intervals["vector_safe_retention"][0]
    usefulness_pass = usefulness_lower is not None and usefulness_lower >= 0.20
    ratio = estimates["vector_to_common_ratio"]
    vector_advantage_pass = ratio is not None and ratio >= 2.0
    return {
        "paired_reduction": {
            "favorable_seed_clusters": favorable,
            "adverse_seed_clusters": adverse,
            "ties": len(SEEDS) - non_ties,
            "exact_two_sided_p": sign_p,
            "passed": paired_pass,
        },
        "safety": {
            "sentinel_event_count": event_count,
            "sentinel_decision_count": len(SEEDS),
            "one_sided_cp95_upper": safety_upper,
            "exact_two_sided_interval": exact_interval(event_count, len(SEEDS)),
            "passed": safety_pass,
        },
        "usefulness": {
            "point_estimate": estimates["vector_safe_retention"],
            "confidence_interval_95": intervals["vector_safe_retention"],
            "passed": usefulness_pass,
        },
        "vector_advantage": {
            "point_ratio": ratio,
            "confidence_interval_95": intervals["vector_to_common_ratio"],
            "vector_deploys_when_common_abstains": vector_when_common_abstains,
            "passed": vector_advantage_pass,
        },
        "overall_confirmatory_success": bool(
            paired_pass and safety_pass and usefulness_pass and vector_advantage_pass
        ),
        "effect_sizes": {
            **bootstrap,
            "relative_violation_reduction": relative_reduction,
        },
        "per_dataset_paired_effects": per_dataset,
        "common_radius_distribution_on_vector_deployments": {
            "count": len(radii),
            "minimum": None if not radii else min(radii),
            "median": None if not radii else float(np.median(radii)),
            "maximum": None if not radii else max(radii),
        },
        "limiting_coordinate_counts": dict(sorted(axis_limiting.items())),
        "common_limiting_contract_counts": dict(sorted(common_limiting.items())),
        "heldout_attacker_stress": {
            "all_vector_deployment_count": len(vector_deployments),
            "portfolio_safe_deployment_count": len(portfolio_safe),
            "heldout_violation_count": heldout_failures,
            "heldout_safe_fraction": (
                None
                if not portfolio_safe
                else 1.0 - heldout_failures / len(portfolio_safe)
            ),
            "formal_guarantee": False,
        },
    }


def validate_normalized_records(
    profiles: list[dict[str, Any]],
    allocations: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    details: list[dict[str, Any]],
) -> None:
    profile_keys = [
        (row["dataset"], row["seed"], row["requested_gamma"])
        for row in profiles
    ]
    allocation_keys = [
        (
            row["dataset"],
            row["seed"],
            row["requested_gamma"],
            row["total_budget"],
            row["allocation"],
        )
        for row in allocations
    ]
    row_keys = [
        (
            row["dataset"],
            row["seed"],
            row["requested_gamma"],
            row["total_budget"],
            row["allocation"],
            row["rule"],
        )
        for row in rows
    ]
    detail_keys = [
        (
            row["dataset"],
            row["seed"],
            row["requested_gamma"],
            row["total_budget"],
            row["allocation"],
            row["canonical_candidate_key"],
        )
        for row in details
    ]
    expected_counts = (
        (profile_keys, 768, "profile"),
        (allocation_keys, 6_144, "allocation"),
        (row_keys, 55_296, "decision"),
        (detail_keys, 73_728, "envelope detail"),
    )
    for keys, expected, label in expected_counts:
        if len(keys) != expected or len(set(keys)) != expected:
            raise RuntimeError(f"{label} cardinality or uniqueness mismatch")
    details_by_config: dict[tuple[Any, ...], set[str]] = {}
    for key in detail_keys:
        details_by_config.setdefault(key[:-1], set()).add(str(key[-1]))
    rows_by_key = {key: row for key, row in zip(row_keys, rows)}
    for key, row in rows_by_key.items():
        deployed = bool(row["deployed"])
        selected = str(row["canonical_candidate_key"])
        safe = bool(row["safe"])
        violation = bool(row["violation"])
        if not deployed and (selected or safe or violation):
            raise RuntimeError(f"invalid abstention state: {key}")
        if deployed and (
            not selected
            or safe == violation
            or selected not in details_by_config.get(key[:-1], set())
        ):
            raise RuntimeError(f"invalid deployment state: {key}")
        if row["rule"] == "external_oracle" and violation:
            raise RuntimeError(f"opportunity oracle selected an unsafe candidate: {key}")
        if row["rule"] in {"vera_vector_envelope", "vera_common_radius"} and safe:
            oracle_key = (*key[:-1], "external_oracle")
            if not bool(rows_by_key[oracle_key]["deployed"]):
                raise RuntimeError(f"safe deployment lacks an opportunity: {key}")
    for detail in details:
        if detail["gamma_cap"] != GAMMA_CAP:
            raise RuntimeError("wrong cap in envelope detail")
        if not np.isclose(
            detail["local_error_budget"], 0.05 / detail["family_size"]
        ):
            raise RuntimeError("wrong local error budget in envelope detail")
        margins = detail["common_radius_contract_margins"]
        minimum = min(margins.values())
        tolerance = (
            1e-4
            if detail["coupled_common_radius"] > 0.0
            and not detail["common_radius_right_censored"]
            else 1e-12
        )
        expected_limiters = sorted(
            key for key, value in margins.items() if value <= minimum + tolerance
        )
        if expected_limiters != detail["common_limiting_contracts"]:
            raise RuntimeError("common-radius limiter does not attain minimum margin")
        common_budget = max(
            *detail["requested_target_profile"].values(),
            *detail["requested_source_profile"].values(),
        )
        if bool(detail["common_eligible"]) != bool(
            detail["coupled_common_radius"] >= common_budget
        ):
            raise RuntimeError("common eligibility disagrees with common radius")
        if bool(detail["vector_eligible"]) != bool(
            detail["requested_profile_in_envelope"]
        ):
            raise RuntimeError("vector eligibility disagrees with coupled envelope")


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    prereg_hash = hash_file(args.prereg)
    expected_hash = args.hash_file.read_text(encoding="utf-8").split()[0]
    if prereg_hash != expected_hash:
        raise RuntimeError("locked protocol hash mismatch")
    prereg = load_object(args.prereg)
    if prereg.get("status") != "locked_before_claim_grade_runs":
        raise RuntimeError("protocol is not locked")
    study = prereg["real_study"]
    if (
        tuple(int(value) for value in study["seeds"]) != SEEDS
        or set(study["datasets"]) != set(DATASETS)
        or (
            float(study["deployment_gamma"]),
            *tuple(
                float(value)
                for value in study["controlled_shift_protocol"][
                    "secondary_requested_gammas"
                ]
            ),
        )
        != GAMMAS
        or tuple(
            sorted(
                [
                    int(
                        study["evidence_allocation"][
                            "primary_total_contract_observation_budget"
                        ]
                    ),
                    *[
                        int(value)
                        for value in study["evidence_allocation"][
                            "sensitivity_budgets"
                        ]
                    ],
                ]
            )
        )
        != BUDGETS
        or float(study["gamma_cap"]) != GAMMA_CAP
        or set(study["leakage_attackers"]) != set(ATTACKERS)
        or tuple(study["deployment_rules"]) != RULES
    ):
        raise RuntimeError("protocol constants differ from the replay contract")
    expected_receipts = {
        f"{dataset}__{method}__seed-{seed}.json"
        for dataset in DATASETS
        for method in study["methods"]
        for seed in SEEDS
    }
    observed_receipts = {
        path.name for path in args.receipt_dir.iterdir() if path.is_file()
    }
    if any(path.is_symlink() for path in args.receipt_dir.iterdir()):
        raise RuntimeError("receipt root contains a symlink")
    if observed_receipts != expected_receipts or len(observed_receipts) != 1_280:
        raise RuntimeError("receipt file set differs from the exact 1,280-run matrix")

    profiles: list[dict[str, Any]] = []
    allocations: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    archive_paths: set[str] = set()
    runner_commits: set[str] = set()
    for dataset in DATASETS:
        contract = study["locked_dataset_contracts"][dataset]
        target_threshold = float(contract["target_harm_threshold"])
        leakage_threshold = float(contract["balanced_leakage_threshold"])
        for seed in SEEDS:
            candidates = load_candidate_frontier(
                args.receipt_dir, study, dataset, seed, prereg_hash
            )
            archive_paths.update(
                candidate["_audit_npz_path"] for candidate in candidates
            )
            for method_key in study["methods"]:
                receipt = load_object(
                    args.receipt_dir / f"{dataset}__{method_key}__seed-{seed}.json"
                )
                commit = str(receipt.get("git_commit", ""))
                if len(commit) != 40:
                    raise RuntimeError("receipt runner commit is malformed")
                runner_commits.add(commit)
                method = study["methods"][method_key]
                for candidate in receipt["candidates"]:
                    provenance = candidate.get("provenance", {})
                    entrypoints = provenance.get("official_entrypoint") or provenance.get(
                        "official_entrypoints"
                    )
                    if (
                        provenance.get("commit") != method["upstream_commit"]
                        or provenance.get("remote") != method["upstream_remote"]
                        or not entrypoints
                    ):
                        raise RuntimeError("candidate upstream provenance mismatch")
            metadata = candidates[0]["certification"]
            design_metadata = candidates[0]["design"]
            design_rng = np.random.default_rng(
                2_027_071_500 + 1009 * seed + sum(map(ord, dataset))
            )
            design_size = min(1000, len(design_metadata["source"]))
            design_indices = np.sort(
                design_rng.choice(
                    len(design_metadata["source"]),
                    size=design_size,
                    replace=False,
                )
            )
            for requested_gamma in GAMMAS:
                probabilities, shift = construct_shift(
                    metadata["environment"],
                    metadata["source"],
                    metadata["target"],
                    design_metadata["environment"][design_indices],
                    design_metadata["source"][design_indices],
                    design_metadata["target"][design_indices],
                    requested_gamma,
                    max(2, min(8, design_size // 20)),
                )
                evaluation_rng = np.random.default_rng(
                    6_000_000_000
                    + 1_000_003 * seed
                    + 10_007 * int(round(100 * requested_gamma))
                    + sum(map(ord, dataset))
                )
                evaluation_indices = evaluation_rng.choice(
                    len(metadata["source"]),
                    size=50_000,
                    replace=True,
                    p=probabilities,
                )
                profiles.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        **shift,
                        "reference_probability_sha256": hash_array_bytes(
                            probabilities
                        ),
                        "design_indices_sha256": hash_array_bytes(design_indices),
                        "evaluation_indices_sha256": hash_array_bytes(
                            evaluation_indices
                        ),
                        "evaluation_size": len(evaluation_indices),
                        "membership_verified": True,
                    }
                )
                for candidate in candidates:
                    candidate["q_metrics"] = exact_shifted_metrics(
                        candidate["certification"], probabilities
                    )
                    candidate["evaluation_metrics"] = sampled_shifted_metrics(
                        candidate["certification"], evaluation_indices
                    )
                scores, pilot_candidate = allocation_plan(
                    candidates,
                    design_indices,
                    shift["target_profile"],
                    shift["source_profile"],
                    target_threshold,
                    leakage_threshold,
                )
                for budget in BUDGETS:
                    plans = {
                        "uniform": integer_allocation(
                            {key: 1.0 for key in scores}, budget, 1
                        ),
                        "targeted_floor_0.15": integer_allocation(
                            scores, budget, max(1, int(np.ceil(0.15 * budget)))
                        ),
                    }
                    for allocation_name, allocation in plans.items():
                        allocations.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "requested_gamma": requested_gamma,
                                "total_budget": budget,
                                "allocation": allocation_name,
                                "pilot_candidate": pilot_candidate,
                                "cell_allocation": allocation,
                                "scores": scores,
                            }
                        )
                        rng = np.random.default_rng(
                            8_000_000_000
                            + 1_000_003 * seed
                            + 10_007 * int(round(100 * requested_gamma))
                            + 101 * budget
                            + sum(map(ord, dataset))
                        )
                        decisions, candidate_details = evaluate_configuration(
                            candidates,
                            metadata,
                            shift["target_profile"],
                            shift["source_profile"],
                            allocation,
                            rng,
                            delta=float(study["delta"]),
                            target_threshold=target_threshold,
                            leakage_threshold=leakage_threshold,
                        )
                        attach_postselection_stress(
                            decisions,
                            candidates,
                            probabilities,
                            leakage_threshold,
                        )
                        configuration = {
                            "dataset": dataset,
                            "seed": seed,
                            "requested_gamma": requested_gamma,
                            "total_budget": budget,
                            "allocation": allocation_name,
                        }
                        details.extend(
                            {**configuration, **detail}
                            for detail in candidate_details
                        )
                        oracle_deployed = decisions["external_oracle"]["deployed"]
                        rows.extend(
                            {
                                **configuration,
                                "rule": rule,
                                "oracle_deployed": oracle_deployed,
                                **decision,
                            }
                            for rule, decision in decisions.items()
                        )
    if len(archive_paths) != 3_072:
        raise RuntimeError("candidate archive cardinality mismatch")
    if len(runner_commits) != 1:
        raise RuntimeError("receipts span more than one runner revision")
    profiles.sort(key=lambda row: (row["dataset"], row["seed"], row["requested_gamma"]))
    allocations.sort(
        key=lambda row: (
            row["dataset"],
            row["seed"],
            row["requested_gamma"],
            row["total_budget"],
            row["allocation"],
        )
    )
    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["seed"],
            row["requested_gamma"],
            row["total_budget"],
            row["allocation"],
            row["rule"],
        )
    )
    details.sort(
        key=lambda row: (
            row["dataset"],
            row["seed"],
            row["requested_gamma"],
            row["total_budget"],
            row["allocation"],
            row["canonical_candidate_key"],
        )
    )
    validate_normalized_records(profiles, allocations, rows, details)
    primary_rows = [
        row
        for row in rows
        if row["requested_gamma"] == PRIMARY_GAMMA
        and row["total_budget"] == PRIMARY_BUDGET
        and row["allocation"] == PRIMARY_ALLOCATION
    ]
    inference = infer_primary(primary_rows)
    return {
        "schema_version": 1,
        "name": "VERA independent finite-reference replay",
        "preregistration_sha256": prereg_hash,
        "prior_source_exposure_disclosed": True,
        "code_path_independent": True,
        "gamma_cap": GAMMA_CAP,
        "counts": {
            "receipts": len(observed_receipts),
            "candidate_archives": len(archive_paths),
            "profiles": len(profiles),
            "allocations": len(allocations),
            "decision_rows": len(rows),
            "primary_rows": len(primary_rows),
            "candidate_envelope_details": len(details),
        },
        "runner_commit_count": len(runner_commits),
        "profiles": profiles,
        "allocation_records": allocations,
        "decision_rows": rows,
        "candidate_envelope_details": details,
        "primary_inference": inference,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, required=True)
    parser.add_argument("--hash-file", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_replay(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        handle.write("\n")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": "completed",
                "normalized_row_count": result["counts"]["decision_rows"],
                "output": str(args.output),
                "byte_size": args.output.stat().st_size,
                "sha256": hash_file(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
