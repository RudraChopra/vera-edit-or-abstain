"""Independently audit VERA confirmatory selections and headline numbers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import beta, binomtest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "prereg_confirmatory_balanced.json"
DEFAULT_HASH = ROOT / "prereg_confirmatory_balanced.sha256"
DEFAULT_RECEIPTS = ROOT / "artifacts" / "confirmatory_balanced_receipt_audit.json"
DEFAULT_RECEIPT_DIR = ROOT / "artifacts" / "confirmatory_balanced_receipts"
DEFAULT_ROWS = ROOT / "artifacts" / "vera_confirmatory_balanced_rule_rows.csv"
DEFAULT_CANDIDATES = ROOT / "artifacts" / "vera_confirmatory_balanced_candidate_rows.csv"
DEFAULT_REPORT = ROOT / "artifacts" / "vera_confirmatory_balanced_report.json"
DEFAULT_ABSTRACT = ROOT / "artifacts" / "vera_confirmatory_abstract_numbers.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "vera_confirmatory_analysis_audit.json"

RULE_PREDICATES = {
    "always_deploy_balanced": None,
    "point_selection_balanced": "point_feasible",
    "vera_balanced_iut": "iut_eligible",
    "vera_balanced_envelope": "envelope_eligible",
    "external_balanced_oracle": "external_contract_satisfied",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: str | bool) -> bool:
    return value if isinstance(value, bool) else value.strip().lower() == "true"


def choose(
    candidates: Iterable[dict[str, str]], predicate: str | None
) -> dict[str, str] | None:
    values = list(candidates)
    if predicate is not None:
        values = [
            candidate
            for candidate in values
            if as_bool(candidate[predicate])
        ]
    if not values:
        return None
    return min(
        values,
        key=lambda candidate: (
            float(candidate["validation_max_balanced_leakage"]),
            float(candidate["validation_max_target_harm"]),
            candidate["candidate"],
        ),
    )


def independent_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    estimable = [row for row in rows if as_bool(row["external_contract_estimable"])]
    deployments = sum(as_bool(row["deployed"]) for row in rows)
    estimable_deployments = sum(as_bool(row["deployed"]) for row in estimable)
    violations = sum(
        as_bool(row["measured_external_contract_violation"]) for row in estimable
    )
    safe = sum(
        as_bool(row["deployed"]) and row["external_contract_satisfied"] == "True"
        for row in estimable
    )
    return {
        "configuration_count": len(rows),
        "estimable_configuration_count": len(estimable),
        "deployment_count": deployments,
        "deployment_rate": deployments / len(rows),
        "estimable_deployment_count": estimable_deployments,
        "safe_deployment_count": safe,
        "measured_external_violation_count": violations,
        "measured_external_violation_rate": (
            None if not estimable else violations / len(estimable)
        ),
        "violation_rate_conditional_on_estimable_deployment": (
            None if estimable_deployments == 0 else violations / estimable_deployments
        ),
        "procedurally_unsupported_deployment_count": sum(
            as_bool(row["procedurally_unsupported_deployment"]) for row in rows
        ),
    }


def values_match(observed: Any, expected: Any) -> bool:
    if observed is None or expected is None:
        return observed is expected
    if isinstance(expected, bool) or isinstance(expected, (str, int)):
        return observed == expected
    return bool(np.isclose(float(observed), float(expected), rtol=1e-12, atol=1e-12))


def compare_summary(
    reported: dict[str, Any],
    expected: dict[str, Any],
    prefix: str,
    failures: list[str],
) -> None:
    for key, value in expected.items():
        if key not in reported or not values_match(reported[key], value):
            failures.append(f"{prefix}.{key} differs from independent replay")


def one_sided_signflip(differences: list[float]) -> float:
    if not differences or not any(abs(value) > 1e-15 for value in differences):
        return 1.0
    observed = float(np.mean(differences))
    if observed <= 0.0:
        return 1.0
    null = [
        float(np.mean([sign * value for sign, value in zip(signs, differences)]))
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return sum(value >= observed - 1e-15 for value in null) / len(null)


def one_sided_sign_test(differences: list[float]) -> float:
    signs = [value for value in differences if abs(value) > 1e-15]
    if not signs:
        return 1.0
    positive = sum(value > 0.0 for value in signs)
    return float(
        binomtest(positive, len(signs), 0.5, alternative="greater").pvalue
    )


def cluster_ratio_interval(
    numerators: dict[int, int],
    denominators: dict[int, int],
    *,
    seed: int = 2_027_071_9,
    replicates: int = 20_000,
) -> tuple[float, float]:
    keys = sorted(set(numerators) | set(denominators))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        denominator = sum(denominators.get(int(key), 0) for key in sampled)
        if denominator:
            values.append(
                sum(numerators.get(int(key), 0) for key in sampled)
                / denominator
            )
    if not values:
        return 0.0, 1.0
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=values.get)
    running = 0.0
    output: dict[str, float] = {}
    total = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * values[key]))
        output[key] = running
    return output


@lru_cache(maxsize=None)
def cp_upper(successes: int, trials: int, alpha: float) -> float:
    if successes >= trials:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, trials - successes))


@lru_cache(maxsize=None)
def cp_lower(successes: int, trials: int, alpha: float) -> float:
    if successes <= 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def independent_target_ucb(
    values: np.ndarray, *, gamma: float, alpha: float
) -> float:
    """Exact paired {-1, 0, 1} upper bound, reimplemented for audit."""

    trials = int(len(values))
    positive = int(np.sum(values == 1))
    negative = int(np.sum(values == -1))
    positive_upper = cp_upper(positive, trials, alpha / 2.0)
    negative_lower = cp_lower(negative, trials, alpha / 2.0)
    positive_probability = min(positive_upper, 1.0 - negative_lower)
    zero_probability = max(
        0.0, 1.0 - negative_lower - positive_probability
    )
    positive_mass = min(1.0, gamma * positive_probability)
    remaining = 1.0 - positive_mass
    zero_mass = min(remaining, gamma * zero_probability)
    negative_mass = max(0.0, remaining - zero_mass)
    return min(1.0, max(-1.0, positive_mass - negative_mass))


def independent_balanced_leakage_ucb(
    correct: np.ndarray,
    source: np.ndarray,
    *,
    gamma: float,
    alpha: float,
) -> float:
    """Equal-weight class-recall bound, reimplemented for raw-data audit."""

    components: list[float] = []
    for source_class in (0, 1):
        values = correct[source == source_class]
        if len(values) == 0:
            raise ValueError("balanced leakage source class is absent")
        probability_upper = cp_upper(
            int(np.sum(values == 1)), int(len(values)), alpha / 2.0
        )
        components.append(min(1.0, gamma * probability_upper))
    return float(np.mean(components))


def independent_nested_indices(
    target: np.ndarray,
    source: np.ndarray,
    environment: np.ndarray,
    fractions: list[float],
    *,
    seed: int,
) -> dict[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    strata: dict[tuple[int, int, int], np.ndarray] = {}
    keys = sorted(
        set(zip(map(int, target), map(int, source), map(int, environment)))
    )
    for key in keys:
        indices = np.flatnonzero(
            (target == key[0]) & (source == key[1]) & (environment == key[2])
        )
        rng.shuffle(indices)
        strata[key] = indices
    outputs: dict[float, np.ndarray] = {}
    for fraction in sorted(fractions):
        selected: list[int] = []
        for indices in strata.values():
            take = (
                len(indices)
                if fraction >= 1.0
                else max(1, int(np.floor(fraction * len(indices))))
            )
            selected.extend(indices[:take].tolist())
        outputs[fraction] = np.asarray(sorted(selected), dtype=np.int64)
    return outputs


def independent_balanced_accuracy(
    correct: np.ndarray, source: np.ndarray
) -> float | None:
    if set(map(int, np.unique(source))) != {0, 1}:
        return None
    return 0.5 * (
        float(correct[source == 0].mean())
        + float(correct[source == 1].mean())
    )


def independent_external_metrics(
    arrays: dict[str, np.ndarray]
) -> tuple[float, float | None]:
    environment = arrays["environment_external"]
    target_harm = arrays["target_harm_external"]
    source = arrays["source_external"]
    target = max(
        float(target_harm[environment == group].mean())
        for group in sorted(map(int, np.unique(environment)))
    )
    leakage = [
        independent_balanced_accuracy(values, source)
        for name, values in arrays.items()
        if name.startswith("leakage_correct_external__")
    ]
    if not leakage or any(value is None for value in leakage):
        return target, None
    return target, max(float(value) for value in leakage if value is not None)


def independent_envelope_radius(
    target: dict[str, np.ndarray],
    leakage: dict[str, np.ndarray],
    source: np.ndarray,
    *,
    alpha: float,
    target_threshold: float,
    leakage_threshold: float,
    gamma_cap: float,
    tolerance: float = 1e-4,
) -> float:
    def passes(gamma: float) -> bool:
        return all(
            independent_target_ucb(values, gamma=gamma, alpha=alpha)
            <= target_threshold
            for values in target.values()
        ) and all(
            independent_balanced_leakage_ucb(
                values, source, gamma=gamma, alpha=alpha
            )
            <= leakage_threshold
            for values in leakage.values()
        )

    if not passes(1.0):
        return 0.0
    if passes(gamma_cap):
        return gamma_cap
    lower, upper = 1.0, gamma_cap
    while upper - lower > tolerance:
        midpoint = (lower + upper) / 2.0
        if passes(midpoint):
            lower = midpoint
        else:
            upper = midpoint
    return lower


def replay_raw_candidates(
    prereg: dict[str, Any],
    receipt_dir: Path,
    failures: list[str],
) -> tuple[dict[str, list[dict[str, Any]]], int, int]:
    """Recompute every candidate flag from receipt NPZs without analysis helpers."""

    study = prereg["real_study"]
    datasets: dict[str, dict[str, Any]] = study["datasets"]
    seeds = [int(value) for value in study["seeds"]]
    fractions = [float(value) for value in study["validation_fractions"]]
    primary_fraction = float(study["primary_validation_fraction"])
    target_thresholds = [float(value) for value in study["target_harm_thresholds"]]
    leakage_thresholds = [float(value) for value in study["leakage_thresholds"]]
    delta = float(study["delta"])
    primary_gamma = float(study["deployment_gamma"])
    shifted_gamma = float(study["shifted_sensitivity_gamma"])
    gamma_cap = float(study["gamma_cap"])
    settings = [(fraction, primary_gamma) for fraction in fractions]
    settings.append((primary_fraction, shifted_gamma))
    by_config: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    npz_count = 0
    checksum_count = 0

    for dataset, dataset_config in datasets.items():
        support_mismatch = bool(
            dataset_config.get("force_abstain_for_unsupported_environment")
        )
        for seed in seeds:
            loaded: list[tuple[str, str, dict[str, np.ndarray]]] = []
            reference: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
            for method_key, method_config in study["methods"].items():
                receipt_path = receipt_dir / f"{dataset}__{method_key}__seed-{seed}.json"
                receipt = load_json(receipt_path)
                for candidate in receipt["candidates"]:
                    npz_path = Path(candidate["audit_npz"])
                    if sha256(npz_path) != candidate["audit_npz_sha256"]:
                        failures.append(f"raw NPZ checksum mismatch: {npz_path}")
                        continue
                    checksum_count += 1
                    with np.load(npz_path) as archive:
                        arrays = {
                            key: np.asarray(archive[key]) for key in archive.files
                        }
                    npz_count += 1
                    labels = (
                        arrays["target_certification"],
                        arrays["source_certification"],
                        arrays["environment_certification"],
                    )
                    if reference is None:
                        reference = labels
                    elif not all(
                        np.array_equal(left, right)
                        for left, right in zip(reference, labels)
                    ):
                        failures.append(
                            f"raw candidate labels differ: {dataset}/seed-{seed}"
                        )
                    loaded.append(
                        (
                            f"{method_config['display_name']}::{candidate['strength']}",
                            str(method_config["display_name"]),
                            arrays,
                        )
                    )
            if reference is None or len(loaded) != 12:
                failures.append(
                    f"raw candidate family has {len(loaded)} rows: {dataset}/seed-{seed}"
                )
                continue
            subsets = independent_nested_indices(
                *reference,
                fractions,
                seed=2_027_071_300 + 1009 * seed + sum(map(ord, dataset)),
            )
            for fraction, gamma in settings:
                subset = subsets[fraction]
                prepared: list[dict[str, Any]] = []
                for candidate_key, method, arrays in loaded:
                    environment = arrays["environment_certification"][subset]
                    target_harm = arrays["target_harm_certification"][subset]
                    source = arrays["source_certification"][subset]
                    target = {
                        f"target::environment={group}": target_harm[
                            environment == group
                        ]
                        for group in sorted(map(int, np.unique(environment)))
                    }
                    leakage = {
                        name.removeprefix("leakage_correct_certification__"): values[
                            subset
                        ]
                        for name, values in arrays.items()
                        if name.startswith("leakage_correct_certification__")
                    }
                    external_target, external_leakage = independent_external_metrics(
                        arrays
                    )
                    prepared.append(
                        {
                            "candidate": candidate_key,
                            "method": method,
                            "target": target,
                            "leakage": leakage,
                            "source": source,
                            "validation_max_target_harm": max(
                                float(values.mean()) for values in target.values()
                            ),
                            "validation_max_balanced_leakage": max(
                                float(
                                    independent_balanced_accuracy(values, source)
                                )
                                for values in leakage.values()
                            ),
                            "external_max_target_harm": external_target,
                            "external_max_balanced_leakage": external_leakage,
                        }
                    )
                family_size = 12 * (
                    len(prepared[0]["target"]) + len(prepared[0]["leakage"])
                )
                tier = (
                    "shifted_sensitivity"
                    if gamma == shifted_gamma
                    else "primary"
                    if fraction == primary_fraction
                    else "learning_curve"
                )
                for target_threshold in target_thresholds:
                    for leakage_threshold in leakage_thresholds:
                        config_id = (
                            f"{tier}|{dataset}|seed={seed}|fraction={fraction:g}|"
                            f"gamma={gamma:g}|tau={target_threshold:g}|"
                            f"lambda={leakage_threshold:g}"
                        )
                        for candidate in prepared:
                            iut_alpha = delta / 12
                            iut_eligible = (
                                not support_mismatch
                                and all(
                                    independent_target_ucb(
                                        values, gamma=gamma, alpha=iut_alpha
                                    )
                                    <= target_threshold
                                    for values in candidate["target"].values()
                                )
                                and all(
                                    independent_balanced_leakage_ucb(
                                        values,
                                        candidate["source"],
                                        gamma=gamma,
                                        alpha=iut_alpha,
                                    )
                                    <= leakage_threshold
                                    for values in candidate["leakage"].values()
                                )
                            )
                            envelope_alpha = delta / family_size
                            observed_radius = independent_envelope_radius(
                                candidate["target"],
                                candidate["leakage"],
                                candidate["source"],
                                alpha=envelope_alpha,
                                target_threshold=target_threshold,
                                leakage_threshold=leakage_threshold,
                                gamma_cap=gamma_cap,
                            )
                            envelope_radius = (
                                0.0 if support_mismatch else observed_radius
                            )
                            external_leakage = candidate[
                                "external_max_balanced_leakage"
                            ]
                            external_estimable = external_leakage is not None
                            external_satisfied = (
                                False
                                if external_leakage is None
                                else candidate["external_max_target_harm"]
                                <= target_threshold
                                and float(external_leakage) <= leakage_threshold
                            )
                            by_config[config_id].append(
                                {
                                    **{
                                        key: candidate[key]
                                        for key in (
                                            "candidate",
                                            "method",
                                            "validation_max_target_harm",
                                            "validation_max_balanced_leakage",
                                            "external_max_target_harm",
                                            "external_max_balanced_leakage",
                                        )
                                    },
                                    "point_feasible": (
                                        candidate["validation_max_target_harm"]
                                        <= target_threshold
                                        and candidate[
                                            "validation_max_balanced_leakage"
                                        ]
                                        <= leakage_threshold
                                    ),
                                    "iut_eligible": iut_eligible,
                                    "envelope_eligible": envelope_radius >= gamma,
                                    "envelope_radius": envelope_radius,
                                    "observed_envelope_radius": observed_radius,
                                    "external_contract_estimable": external_estimable,
                                    "external_contract_satisfied": external_satisfied,
                                    "family_size": family_size,
                                    "certification_n": len(subset),
                                }
                            )
    return dict(by_config), npz_count, checksum_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--hash-file", type=Path, default=DEFAULT_HASH)
    parser.add_argument("--receipt-audit", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPT_DIR)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--abstract", type=Path, default=DEFAULT_ABSTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prereg = load_json(args.prereg)
    receipt_audit = load_json(args.receipt_audit)
    report = load_json(args.report)
    abstract = load_json(args.abstract)
    rows = load_csv(args.rows)
    candidates = load_csv(args.candidates)
    failures: list[str] = []

    prereg_hash = sha256(args.prereg)
    if prereg_hash != args.hash_file.read_text(encoding="utf-8").split()[0]:
        failures.append("parent preregistration hash mismatch")
    if (
        receipt_audit.get("passed") is not True
        or receipt_audit.get("prereg_sha256") != prereg_hash
    ):
        failures.append("receipt audit gate is absent or mismatched")
    if len(rows) != 10_800:
        failures.append(f"expected 10,800 rule rows, found {len(rows)}")
    if len(candidates) != 25_920:
        failures.append(f"expected 25,920 candidate rows, found {len(candidates)}")

    raw_candidates_by_config, raw_npz_count, raw_checksum_count = (
        replay_raw_candidates(prereg, args.receipt_dir, failures)
    )
    if raw_npz_count != 480 or raw_checksum_count != 480:
        failures.append(
            f"expected 480 checksum-verified raw NPZs, found "
            f"{raw_npz_count}/{raw_checksum_count}"
        )

    candidates_by_config: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    rules_by_config: defaultdict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for candidate in candidates:
        candidates_by_config[candidate["config_id"]].append(candidate)
    for row in rows:
        if row["rule"] in rules_by_config[row["config_id"]]:
            failures.append(f"duplicate rule row: {row['config_id']} / {row['rule']}")
        rules_by_config[row["config_id"]][row["rule"]] = row
    if set(candidates_by_config) != set(rules_by_config):
        failures.append("candidate and rule configuration keys differ")
    if set(raw_candidates_by_config) != set(candidates_by_config):
        failures.append("raw-replay and aggregate candidate configuration keys differ")

    raw_candidate_mismatches = 0
    raw_candidate_rows = 0
    for config_id, raw_values in raw_candidates_by_config.items():
        reported = {
            candidate["candidate"]: candidate
            for candidate in candidates_by_config.get(config_id, [])
        }
        for raw in raw_values:
            raw_candidate_rows += 1
            observed = reported.get(raw["candidate"])
            if observed is None:
                raw_candidate_mismatches += 1
                continue
            expected_external_leakage = raw["external_max_balanced_leakage"]
            comparisons = (
                observed["method"] == raw["method"],
                values_match(
                    float(observed["validation_max_target_harm"]),
                    raw["validation_max_target_harm"],
                ),
                values_match(
                    float(observed["validation_max_balanced_leakage"]),
                    raw["validation_max_balanced_leakage"],
                ),
                as_bool(observed["point_feasible"]) == raw["point_feasible"],
                as_bool(observed["iut_eligible"]) == raw["iut_eligible"],
                as_bool(observed["envelope_eligible"])
                == raw["envelope_eligible"],
                values_match(
                    float(observed["envelope_radius"]), raw["envelope_radius"]
                ),
                values_match(
                    float(observed["observed_envelope_radius"]),
                    raw["observed_envelope_radius"],
                ),
                values_match(
                    float(observed["external_max_target_harm"]),
                    raw["external_max_target_harm"],
                ),
                (
                    observed["external_max_balanced_leakage"] == ""
                    if expected_external_leakage is None
                    else values_match(
                        float(observed["external_max_balanced_leakage"]),
                        expected_external_leakage,
                    )
                ),
                as_bool(observed["external_contract_estimable"])
                == raw["external_contract_estimable"],
                (
                    observed["external_contract_satisfied"] == "NA"
                    if not raw["external_contract_estimable"]
                    else as_bool(observed["external_contract_satisfied"])
                    == raw["external_contract_satisfied"]
                ),
                int(observed["family_size"]) == raw["family_size"],
                int(observed["certification_n"]) == raw["certification_n"],
            )
            if not all(comparisons):
                raw_candidate_mismatches += 1
    if raw_candidate_rows != 25_920:
        failures.append(
            f"raw receipt replay generated {raw_candidate_rows} candidate rows"
        )
    if raw_candidate_mismatches:
        failures.append(
            f"{raw_candidate_mismatches} candidate rows differ from raw NPZ replay"
        )

    selection_mismatches = 0
    semantic_mismatches = 0
    for config_id, config_candidates in raw_candidates_by_config.items():
        if len(config_candidates) != 12:
            failures.append(f"{config_id} has {len(config_candidates)} candidates")
            continue
        config_rules = rules_by_config[config_id]
        if set(config_rules) != set(RULE_PREDICATES):
            failures.append(f"{config_id} has the wrong rule family")
            continue
        for rule, predicate in RULE_PREDICATES.items():
            selected = choose(config_candidates, predicate)
            row = config_rules[rule]
            expected_deployed = selected is not None
            expected_candidate = "" if selected is None else selected["candidate"]
            if (
                as_bool(row["deployed"]) != expected_deployed
                or row["selected_candidate"] != expected_candidate
            ):
                selection_mismatches += 1
            if selected is None:
                expected_estimable = row["dataset"] != "Camelyon17-WILDS"
                expected_satisfied = "NA"
            else:
                expected_estimable = as_bool(
                    selected["external_contract_estimable"]
                )
                expected_satisfied = (
                    (
                        "True"
                        if as_bool(selected["external_contract_satisfied"])
                        else "False"
                    )
                    if expected_estimable
                    else "NA"
                )
            expected_violation = bool(
                selected is not None
                and expected_satisfied not in {"NA", "True"}
            )
            if (
                as_bool(row["external_contract_estimable"]) != expected_estimable
                or row["external_contract_satisfied"] != expected_satisfied
                or as_bool(row["measured_external_contract_violation"])
                != expected_violation
            ):
                semantic_mismatches += 1
    if selection_mismatches:
        failures.append(f"{selection_mismatches} rule selections do not replay")
    if semantic_mismatches:
        failures.append(f"{semantic_mismatches} external semantics do not replay")

    primary = [row for row in rows if row["analysis_tier"] == "primary"]
    shifted = [
        row for row in rows if row["analysis_tier"] == "shifted_sensitivity"
    ]
    for tier_name, tier_rows, reported_summaries in (
        ("primary", primary, report.get("primary_summaries", {})),
        (
            "shifted",
            shifted,
            report.get("shifted_sensitivity_summaries", {}),
        ),
    ):
        for rule in RULE_PREDICATES:
            expected = independent_summary(
                [row for row in tier_rows if row["rule"] == rule]
            )
            compare_summary(
                reported_summaries.get(rule, {}),
                expected,
                f"{tier_name}.{rule}",
                failures,
            )

    study = prereg["real_study"]
    supported = [
        dataset
        for dataset, config in study["datasets"].items()
        if not config.get("force_abstain_for_unsupported_environment")
    ]
    point = [
        row
        for row in primary
        if row["rule"] == "point_selection_balanced"
    ]
    iut = [row for row in primary if row["rule"] == "vera_balanced_iut"]
    point_by_id = {row["config_id"]: row for row in point}
    iut_by_id = {row["config_id"]: row for row in iut}
    raw_p: dict[str, float] = {}
    sign_raw_p: dict[str, float] = {}
    mcnemar_raw_p: dict[str, float] = {}
    mcnemar_counts: dict[str, dict[str, int]] = {}
    strict_seed_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in supported:
        differences: list[float] = []
        for seed in map(int, study["seeds"]):
            ids = [
                config_id
                for config_id, row in point_by_id.items()
                if row["dataset"] == dataset and int(row["seed"]) == seed
            ]
            differences.append(
                float(
                    np.mean(
                        [
                            float(
                                as_bool(
                                    point_by_id[config_id][
                                        "measured_external_contract_violation"
                                    ]
                                )
                            )
                            - float(
                                as_bool(
                                    iut_by_id[config_id][
                                        "measured_external_contract_violation"
                                    ]
                                )
                            )
                            for config_id in ids
                        ]
                    )
                )
            )
        raw_p[dataset] = one_sided_signflip(differences)
        sign_raw_p[dataset] = one_sided_sign_test(differences)
        dataset_ids = [
            config_id
            for config_id, row in point_by_id.items()
            if row["dataset"] == dataset
        ]
        point_only = sum(
            as_bool(
                point_by_id[config_id]["measured_external_contract_violation"]
            )
            and not as_bool(
                iut_by_id[config_id]["measured_external_contract_violation"]
            )
            for config_id in dataset_ids
        )
        iut_only = sum(
            not as_bool(
                point_by_id[config_id]["measured_external_contract_violation"]
            )
            and as_bool(
                iut_by_id[config_id]["measured_external_contract_violation"]
            )
            for config_id in dataset_ids
        )
        mcnemar_counts[dataset] = {
            "point_only_violation": point_only,
            "vera_only_violation": iut_only,
        }
        discordant = point_only + iut_only
        mcnemar_raw_p[dataset] = (
            1.0
            if discordant == 0
            else float(
                binomtest(
                    min(point_only, iut_only),
                    discordant,
                    0.5,
                    alternative="two-sided",
                ).pvalue
            )
        )
        strict_seed_summaries[dataset] = {
            str(seed): independent_summary(
                [
                    row
                    for row in iut
                    if row["dataset"] == dataset
                    and int(row["seed"]) == seed
                ]
            )
            for seed in map(int, study["seeds"])
        }
    adjusted = holm(raw_p)
    sign_adjusted = holm(sign_raw_p)
    mcnemar_adjusted = holm(mcnemar_raw_p)
    if any(
        not values_match(
            report.get("seed_blocked_one_sided_signflip_holm_p", {}).get(dataset),
            value,
        )
        for dataset, value in adjusted.items()
    ):
        failures.append("seed-blocked Holm values do not independently replay")
    if any(
        not values_match(
            report.get("seed_blocked_one_sided_sign_test_holm_p", {}).get(
                dataset
            ),
            value,
        )
        for dataset, value in sign_adjusted.items()
    ):
        failures.append("seed-blocked sign-test Holm values do not replay")
    if report.get("mcnemar_discordant_counts") != mcnemar_counts:
        failures.append("McNemar discordant counts do not independently replay")
    if any(
        not values_match(
            report.get("mcnemar_two_sided_holm_p", {}).get(dataset), value
        )
        for dataset, value in mcnemar_adjusted.items()
    ):
        failures.append("McNemar Holm values do not independently replay")
    reported_seed_summaries = report.get(
        "primary_vera_by_supported_dataset_seed", {}
    )
    for dataset, seed_map in strict_seed_summaries.items():
        for seed, expected in seed_map.items():
            compare_summary(
                reported_seed_summaries.get(dataset, {}).get(seed, {}),
                expected,
                f"strict_seed.{dataset}.{seed}",
                failures,
            )
    strict_seed_control = all(
        float(summary["measured_external_violation_rate"])
        <= float(study["delta"])
        for seed_map in strict_seed_summaries.values()
        for summary in seed_map.values()
    )
    maximum_seed_rate = max(
        float(summary["measured_external_violation_rate"])
        for seed_map in strict_seed_summaries.values()
        for summary in seed_map.values()
    )
    if report.get("strict_supported_dataset_seed_control") is not strict_seed_control:
        failures.append("strict dataset-seed control flag does not replay")
    if not values_match(
        report.get("maximum_supported_dataset_seed_violation_rate"),
        maximum_seed_rate,
    ):
        failures.append("maximum dataset-seed violation rate does not replay")

    oracle = {
        row["config_id"]: row
        for row in primary
        if row["rule"] == "external_balanced_oracle"
    }
    numerators: dict[int, int] = {}
    denominators: dict[int, int] = {}
    for seed in map(int, study["seeds"]):
        opportunities = [
            config_id
            for config_id, row in oracle.items()
            if int(row["seed"]) == seed and as_bool(row["deployed"])
        ]
        denominators[seed] = len(opportunities)
        numerators[seed] = sum(
            as_bool(iut_by_id[config_id]["deployed"])
            and iut_by_id[config_id]["external_contract_satisfied"] == "True"
            for config_id in opportunities
        )
    retention_interval = cluster_ratio_interval(numerators, denominators)
    reported_retention = report.get("safe_retention", {})
    if reported_retention.get("seed_cluster_numerators") != {
        str(key): value for key, value in numerators.items()
    }:
        failures.append("retention seed-cluster numerators do not replay")
    if reported_retention.get("seed_cluster_denominators") != {
        str(key): value for key, value in denominators.items()
    }:
        failures.append("retention seed-cluster denominators do not replay")
    if any(
        not values_match(observed, expected)
        for observed, expected in zip(
            reported_retention.get("seed_cluster_bootstrap95", []),
            retention_interval,
        )
    ) or len(reported_retention.get("seed_cluster_bootstrap95", [])) != 2:
        failures.append("retention seed-cluster interval does not replay")

    stress_lookup = {
        (
            record["dataset"],
            float(record["target_harm_threshold"]),
            float(record["leakage_threshold"]),
        )
        for record in study["headline_stress_family"]["regimes"]
    }
    stress_point = [
        row
        for row in point
        if (row["dataset"], float(row["target_threshold"]), float(row["leakage_threshold"]))
        in stress_lookup
    ]
    stress_iut = [
        row
        for row in iut
        if (row["dataset"], float(row["target_threshold"]), float(row["leakage_threshold"]))
        in stress_lookup
    ]
    point_rate = float(
        np.mean(
            [as_bool(row["measured_external_contract_violation"]) for row in stress_point]
        )
    )
    iut_rate = float(
        np.mean(
            [as_bool(row["measured_external_contract_violation"]) for row in stress_iut]
        )
    )
    retention = float(report.get("safe_retention", {}).get("rate", -1.0))
    gap_pass = point_rate - iut_rate >= 0.15
    camelyon_vera = [
        row
        for row in primary
        if row["dataset"] == "Camelyon17-WILDS"
        and row["rule"] in {"vera_balanced_iut", "vera_balanced_envelope"}
    ]
    camelyon_forced_count = sum(
        not as_bool(row["deployed"])
        and as_bool(row["support_mismatch_forced_abstention"])
        for row in camelyon_vera
    )
    camelyon_abstention_pass = bool(camelyon_vera) and (
        camelyon_forced_count == len(camelyon_vera)
    )
    if report.get("passed") is True and gap_pass:
        headline_mode = "empirical_gap"
        sentence = (
            f"Across 32 prespecified stress configurations, validation-only selection "
            f"deployed contract-violating edits in {100 * point_rate:.1f}% of "
            f"configurations versus {100 * iut_rate:.1f}% for VERA, while VERA "
            f"retained {100 * retention:.1f}% of external-oracle opportunities."
        )
    else:
        headline_mode = "theory_forced_abstention"
        sentence = (
            "VERA gives finite-sample false-acceptance control over its declared "
            "shift class and identifies when certification is impossible; on "
            f"Camelyon17 it forced abstention in all {camelyon_forced_count} "
            "registered VERA configurations because the deployment hospital was "
            "outside certification support."
        )
    if len(stress_point) != 32 or len(stress_iut) != 32:
        failures.append("stress family does not contain 32 rows per rule")
    for key, expected in (
        ("point_selection_violation_rate", point_rate),
        ("vera_iut_violation_rate", iut_rate),
        ("safe_retention", retention),
        ("sentence", sentence),
        ("headline_mode", headline_mode),
        ("headline_gap_condition_met", gap_pass),
        (
            "theory_forced_abstention_lead_verified",
            headline_mode == "theory_forced_abstention"
            and camelyon_abstention_pass,
        ),
        ("unsupported_camelyon_abstention_verified", camelyon_abstention_pass),
        (
            "camelyon_forced_abstention_configuration_count",
            camelyon_forced_count,
        ),
        ("registered_pass_conditions_met", report.get("passed") is True),
    ):
        if not values_match(abstract.get(key), expected):
            failures.append(f"abstract field {key} does not replay")
    if abstract.get("verified") is not True:
        failures.append("abstract numbers are not marked as receipt-verified")

    audit = {
        "name": "Independent VERA confirmatory aggregate audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "confirmatory_passed": report.get("passed") is True,
        "abstract_verified": abstract.get("verified") is True,
        "prereg_sha256": prereg_hash,
        "receipt_audit_sha256": sha256(args.receipt_audit),
        "rule_rows_sha256": sha256(args.rows),
        "candidate_rows_sha256": sha256(args.candidates),
        "report_sha256": sha256(args.report),
        "abstract_sha256": sha256(args.abstract),
        "rule_rows_replayed": len(rows),
        "candidate_rows_replayed": len(candidates),
        "raw_candidate_rows_recomputed": raw_candidate_rows,
        "raw_npz_files_recomputed": raw_npz_count,
        "raw_npz_checksums_verified": raw_checksum_count,
        "raw_candidate_mismatches": raw_candidate_mismatches,
        "selection_mismatches": selection_mismatches,
        "semantic_mismatches": semantic_mismatches,
        "stress_point_rate": point_rate,
        "stress_vera_rate": iut_rate,
        "holm_p_replayed": adjusted,
        "sign_test_holm_p_replayed": sign_adjusted,
        "mcnemar_holm_p_replayed": mcnemar_adjusted,
        "strict_supported_dataset_seed_control": strict_seed_control,
        "maximum_supported_dataset_seed_violation_rate": maximum_seed_rate,
        "retention_seed_cluster_bootstrap95": list(retention_interval),
        "failures": failures,
    }
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
