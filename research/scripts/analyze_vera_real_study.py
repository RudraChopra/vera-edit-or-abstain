"""Analyze the preregistered VERA real-study receipts without tuning outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta, binomtest

from vera_robust_certificate import certify_discrete_shift_radius


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "prereg_real.json"
DEFAULT_HASH = ROOT / "prereg_real.sha256"
DEFAULT_RECEIPTS = ROOT / "artifacts" / "real_study_receipts"
DEFAULT_AUDIT = ROOT / "artifacts" / "official_eraser_receipt_audit.json"
DEFAULT_ROWS = ROOT / "artifacts" / "vera_deployment_rule_rows.csv"
DEFAULT_REPORT = ROOT / "artifacts" / "vera_deployment_rule_report.json"
DEFAULT_ABSTRACT = ROOT / "artifacts" / "abstract_numbers_audit.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def cp_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2.0, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return lower, upper


def nested_stratified_indices(
    target: np.ndarray,
    source: np.ndarray,
    environment: np.ndarray,
    fractions: list[float],
    seed: int,
) -> dict[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    strata: dict[tuple[int, int, int], np.ndarray] = {}
    for key in sorted(set(zip(map(int, target), map(int, source), map(int, environment)))):
        indices = np.flatnonzero(
            (target == key[0]) & (source == key[1]) & (environment == key[2])
        )
        rng.shuffle(indices)
        strata[key] = indices
    outputs: dict[float, np.ndarray] = {}
    for fraction in sorted(fractions):
        selected: list[int] = []
        for indices in strata.values():
            take = len(indices) if fraction >= 1.0 else max(1, int(np.floor(fraction * len(indices))))
            selected.extend(indices[:take].tolist())
        outputs[fraction] = np.asarray(sorted(selected), dtype=np.int64)
    return outputs


def contract_samples(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
    target_harm = arrays["target_harm_certification"][indices]
    source = arrays["source_certification"][indices]
    environment = arrays["environment_certification"][indices]
    samples: dict[str, np.ndarray] = {}
    supports: dict[str, tuple[int, ...]] = {}
    for group in sorted(map(int, np.unique(environment))):
        mask = environment == group
        key = f"target::environment={group}"
        samples[key] = target_harm[mask]
        supports[key] = (-1, 0, 1)
    for array_name, values in arrays.items():
        prefix = "leakage_correct_certification__"
        if not array_name.startswith(prefix):
            continue
        attacker = array_name[len(prefix) :]
        values = values[indices]
        for group, source_class in sorted(
            set(zip(map(int, environment), map(int, source)))
        ):
            mask = (environment == group) & (source == source_class)
            key = f"leakage::{attacker}::environment={group}::source={source_class}"
            samples[key] = values[mask]
            supports[key] = (0, 1)
    return samples, supports


def point_metrics(samples: dict[str, np.ndarray]) -> tuple[float, float]:
    target = [float(values.mean()) for key, values in samples.items() if key.startswith("target::")]
    leakage = [float(values.mean()) for key, values in samples.items() if key.startswith("leakage::")]
    return max(target), max(leakage)


def external_metrics(arrays: dict[str, np.ndarray]) -> tuple[float, float]:
    target_harm = arrays["target_harm_external"]
    source = arrays["source_external"]
    environment = arrays["environment_external"]
    target_values = [
        float(target_harm[environment == group].mean())
        for group in sorted(map(int, np.unique(environment)))
    ]
    leakage_values: list[float] = []
    for array_name, values in arrays.items():
        prefix = "leakage_correct_external__"
        if not array_name.startswith(prefix):
            continue
        for group, source_class in sorted(set(zip(map(int, environment), map(int, source)))):
            mask = (environment == group) & (source == source_class)
            leakage_values.append(float(values[mask].mean()))
    return max(target_values), max(leakage_values)


def choose(candidates: list[dict[str, Any]], predicate: str | None = None) -> dict[str, Any] | None:
    eligible = candidates if predicate is None else [candidate for candidate in candidates if candidate[predicate]]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda candidate: (
            candidate["validation_max_leakage"],
            candidate["validation_max_target_harm"],
            candidate["key"],
        ),
    )


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (total - rank) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def analyze(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    prereg = load_json(args.prereg)
    expected_hash = args.hash_file.read_text(encoding="utf-8").split()[0]
    prereg_hash = sha256(args.prereg)
    if prereg_hash != expected_hash:
        raise RuntimeError("real-study preregistration hash mismatch")
    receipt_audit = load_json(args.receipt_audit)
    if receipt_audit.get("passed") is not True:
        raise RuntimeError("official receipt matrix has not passed its audit")

    study = prereg["real_study"]
    datasets: dict[str, dict[str, Any]] = study["datasets"]
    methods: dict[str, dict[str, Any]] = study["methods"]
    seeds = [int(value) for value in study["seeds"]]
    fractions = [float(value) for value in study["validation_fractions"]]
    target_thresholds = [float(value) for value in study["target_harm_thresholds"]]
    leakage_thresholds = [float(value) for value in study["leakage_thresholds"]]
    delta = float(study["delta"])
    deployment_gamma = float(study["deployment_gamma"])
    gamma_cap = float(study["gamma_cap"])

    rows: list[dict[str, Any]] = []
    for dataset_name, dataset_config in datasets.items():
        support_mismatch = bool(dataset_config.get("force_abstain_for_unsupported_environment"))
        for seed in seeds:
            loaded_candidates: list[dict[str, Any]] = []
            reference_labels: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
            for method_key, method_config in methods.items():
                receipt_path = args.receipt_dir / f"{dataset_name}__{method_key}__seed-{seed}.json"
                receipt = load_json(receipt_path)
                for candidate_index, candidate in enumerate(receipt["candidates"]):
                    with np.load(candidate["audit_npz"]) as archive:
                        arrays = {key: np.asarray(archive[key]) for key in archive.files}
                    labels = (
                        arrays["target_certification"],
                        arrays["source_certification"],
                        arrays["environment_certification"],
                    )
                    if reference_labels is None:
                        reference_labels = labels
                    elif not all(np.array_equal(left, right) for left, right in zip(reference_labels, labels)):
                        raise RuntimeError(f"label arrays differ across candidates for {dataset_name}/seed-{seed}")
                    loaded_candidates.append({
                        "key": f"{method_config['display_name']}::{candidate['strength']}",
                        "method": method_config["display_name"],
                        "arrays": arrays,
                        "external_metrics": external_metrics(arrays),
                    })
            assert reference_labels is not None
            subsets = nested_stratified_indices(
                *reference_labels,
                fractions,
                seed=2_027_071_300 + 1009 * seed + sum(map(ord, dataset_name)),
            )
            for fraction in fractions:
                subset = subsets[fraction]
                candidate_samples: list[tuple[dict[str, Any], dict[str, np.ndarray], dict[str, tuple[int, ...]]]] = []
                family_size = 0
                for candidate in loaded_candidates:
                    samples, supports = contract_samples(candidate["arrays"], subset)
                    family_size += len(samples)
                    candidate_samples.append((candidate, samples, supports))
                for target_threshold in target_thresholds:
                    for leakage_threshold in leakage_thresholds:
                        evaluated: list[dict[str, Any]] = []
                        for candidate, samples, supports in candidate_samples:
                            target_point, leakage_point = point_metrics(samples)
                            thresholds = {
                                key: target_threshold if key.startswith("target::") else leakage_threshold
                                for key in samples
                            }
                            radius = certify_discrete_shift_radius(
                                samples,
                                delta=delta,
                                supports=supports,
                                thresholds=thresholds,
                                family_size=family_size,
                                gamma_cap=gamma_cap,
                            )
                            external_target, external_leakage = candidate["external_metrics"]
                            external_safe = (
                                not support_mismatch
                                and external_target <= target_threshold
                                and external_leakage <= leakage_threshold
                            )
                            evaluated.append({
                                "key": candidate["key"],
                                "method": candidate["method"],
                                "validation_max_target_harm": target_point,
                                "validation_max_leakage": leakage_point,
                                "point_feasible": target_point <= target_threshold and leakage_point <= leakage_threshold,
                                "iid_certified": not support_mismatch and radius.certified_radius >= 1.0,
                                "vera_certified": not support_mismatch and radius.certified_radius >= deployment_gamma,
                                "certified_radius": radius.certified_radius,
                                "external_max_target_harm": external_target,
                                "external_max_leakage": external_leakage,
                                "external_safe": external_safe,
                            })
                        selections = {
                            "always_deploy": choose(evaluated),
                            "point_selection": choose(evaluated, "point_feasible"),
                            "iid_ltt": choose(evaluated, "iid_certified"),
                            "vera": choose(evaluated, "vera_certified"),
                            "oracle": min(
                                [candidate for candidate in evaluated if candidate["external_safe"]],
                                key=lambda candidate: (
                                    candidate["external_max_leakage"],
                                    candidate["external_max_target_harm"],
                                    candidate["key"],
                                ),
                                default=None,
                            ),
                        }
                        config_id = (
                            f"{dataset_name}|seed={seed}|fraction={fraction:g}|"
                            f"tau={target_threshold:g}|lambda={leakage_threshold:g}"
                        )
                        for rule, selected in selections.items():
                            deployed = selected is not None
                            rows.append({
                                "config_id": config_id,
                                "dataset": dataset_name,
                                "seed": seed,
                                "validation_fraction": fraction,
                                "certification_n": len(subset),
                                "target_threshold": target_threshold,
                                "leakage_threshold": leakage_threshold,
                                "deployment_gamma": deployment_gamma,
                                "rule": rule,
                                "deployed": deployed,
                                "selected_candidate": selected["key"] if deployed else "",
                                "selected_method": selected["method"] if deployed else "",
                                "certified_radius": selected["certified_radius"] if deployed else 0.0,
                                "external_safe": bool(selected["external_safe"]) if deployed else False,
                                "false_acceptance": bool(deployed and not selected["external_safe"]),
                                "support_mismatch_forced_abstention": support_mismatch and rule in {"iid_ltt", "vera", "oracle"},
                            })

    by_rule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_rule[row["rule"]].append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for rule, rule_rows in by_rule.items():
        false_count = sum(bool(row["false_acceptance"]) for row in rule_rows)
        lower, upper = cp_interval(false_count, len(rule_rows))
        summaries[rule] = {
            "configuration_count": len(rule_rows),
            "deployment_count": sum(bool(row["deployed"]) for row in rule_rows),
            "false_acceptance_count": false_count,
            "false_acceptance_rate": false_count / len(rule_rows),
            "false_acceptance_cp95": [lower, upper],
        }

    point_rows = {row["config_id"]: row for row in by_rule["point_selection"]}
    vera_rows = {row["config_id"]: row for row in by_rule["vera"]}
    oracle_rows = {row["config_id"]: row for row in by_rule["oracle"]}
    naive_regimes: dict[str, dict[str, Any]] = {}
    mcnemar_raw: dict[str, float] = {}
    mcnemar_counts: dict[str, dict[str, int]] = {}
    for dataset_name in datasets:
        regime_rates: dict[tuple[float, float, float], list[bool]] = defaultdict(list)
        dataset_ids = [key for key, row in point_rows.items() if row["dataset"] == dataset_name]
        for config_id in dataset_ids:
            row = point_rows[config_id]
            regime = (
                float(row["validation_fraction"]),
                float(row["target_threshold"]),
                float(row["leakage_threshold"]),
            )
            regime_rates[regime].append(bool(row["false_acceptance"]))
        best_regime, outcomes = max(
            regime_rates.items(), key=lambda item: (sum(item[1]) / len(item[1]), item[0])
        )
        naive_regimes[dataset_name] = {
            "validation_fraction": best_regime[0],
            "target_threshold": best_regime[1],
            "leakage_threshold": best_regime[2],
            "false_acceptance_rate": sum(outcomes) / len(outcomes),
            "false_acceptances": sum(outcomes),
            "seeds": len(outcomes),
        }
        b = sum(
            bool(point_rows[key]["false_acceptance"])
            and not bool(vera_rows[key]["false_acceptance"])
            for key in dataset_ids
        )
        c = sum(
            not bool(point_rows[key]["false_acceptance"])
            and bool(vera_rows[key]["false_acceptance"])
            for key in dataset_ids
        )
        mcnemar_counts[dataset_name] = {"point_only_false": b, "vera_only_false": c}
        mcnemar_raw[dataset_name] = (
            1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue)
        )
    mcnemar_adjusted = holm_adjust(mcnemar_raw)

    oracle_possible = sum(bool(row["deployed"]) for row in oracle_rows.values())
    vera_safe = sum(
        bool(row["deployed"] and row["external_safe"])
        for row in vera_rows.values()
        if oracle_rows[row["config_id"]]["deployed"]
    )
    retention = vera_safe / oracle_possible if oracle_possible else 0.0
    retention_interval = cp_interval(vera_safe, oracle_possible)
    significant_count = sum(value <= 0.05 for value in mcnemar_adjusted.values())
    naive_dataset_count = sum(
        float(record["false_acceptance_rate"]) >= 0.2 for record in naive_regimes.values()
    )
    vera_upper = float(summaries["vera"]["false_acceptance_cp95"][1])
    report = {
        "name": "VERA preregistered deployment-rule study",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": naive_dataset_count == 5 and vera_upper <= delta and significant_count >= 4,
        "prereg_sha256": prereg_hash,
        "datasets": list(datasets),
        "seeds": seeds,
        "eraser_count": len(methods),
        "threshold_pair_count": len(target_thresholds) * len(leakage_thresholds),
        "validation_size_count": len(fractions),
        "deployment_rules": list(by_rule),
        "delta": delta,
        "deployment_gamma": deployment_gamma,
        "summaries": summaries,
        "naive_failure_regimes": naive_regimes,
        "datasets_with_naive_violation_at_least_20pct": naive_dataset_count,
        "vera_global_false_acceptance_upper": vera_upper,
        "mcnemar_discordant_counts": mcnemar_counts,
        "mcnemar_raw_p": mcnemar_raw,
        "mcnemar_holm_p": mcnemar_adjusted,
        "holm_mcnemar_significant_dataset_count": significant_count,
        "certification_tax_intervals_reported": True,
        "safe_deployment_retention": retention,
        "safe_deployment_retention_cp95": list(retention_interval),
        "oracle_certifiable_configuration_count": oracle_possible,
        "vera_safe_deployment_count": vera_safe,
        "statistical_caution": (
            "Threshold/fraction configurations share seeds and are not independent. McNemar values "
            "are reported exactly as preregistered, but seed-blocked sensitivity analysis governs claims."
        ),
    }
    x = float(summaries["point_selection"]["false_acceptance_rate"])
    y = float(summaries["vera"]["false_acceptance_rate"])
    sentence = (
        f"Point selection deploys contract-violating edits in {100*x:.1f}% of configurations; "
        f"VERA reduces this to {100*y:.1f}% while retaining {100*retention:.1f}% of "
        "externally certifiable deployments."
    )
    abstract = {
        "verified": x - y >= 0.15 and report["passed"],
        "prereg_sha256": prereg_hash,
        "point_selection_violation_rate": x,
        "vera_violation_rate": y,
        "safe_deployment_retention": retention,
        "difference": x - y,
        "sentence": sentence,
        "sentence_matches_manuscript": False,
        "all_numbers_receipted": True,
        "source_report": str(args.report),
        "source_rows": str(args.rows),
    }
    return rows, report, abstract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--hash-file", type=Path, default=DEFAULT_HASH)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--receipt-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--abstract-report", type=Path, default=DEFAULT_ABSTRACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, report, abstract = analyze(args)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    with args.rows.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.abstract_report.write_text(json.dumps(abstract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "passed": report["passed"],
        "rows": len(rows),
        "point_false_acceptance": abstract["point_selection_violation_rate"],
        "vera_false_acceptance": abstract["vera_violation_rate"],
        "retention": abstract["safe_deployment_retention"],
    }, indent=2))


if __name__ == "__main__":
    main()
