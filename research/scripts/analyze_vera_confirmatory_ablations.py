"""Run the locked secondary ablations on balanced confirmatory receipts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from analyze_vera_attacker_ablation import load_candidates
from analyze_vera_balanced_existing import (
    balanced_external_metrics,
    balanced_point_metrics,
    balanced_samples,
)
from analyze_vera_real_study import cp_interval
from vera_robust_certificate import balanced_contract_certificates


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
DEFAULT_PREREG = ROOT / "prereg_confirmatory_balanced.json"
DEFAULT_ABLATION = ROOT / "prereg_confirmatory_secondary_ablations.json"
DEFAULT_HASH = ROOT / "prereg_confirmatory_secondary_ablations.sha256"
DEFAULT_AUDIT = ROOT / "artifacts" / "confirmatory_balanced_receipt_audit.json"
DEFAULT_RECEIPTS = ROOT / "artifacts" / "confirmatory_balanced_receipts"
DEFAULT_RULE_ROWS = ROOT / "artifacts" / "vera_confirmatory_balanced_rule_rows.csv"
DEFAULT_CANDIDATES = ROOT / "artifacts" / "vera_confirmatory_balanced_candidate_rows.csv"
DEFAULT_ROWS = ROOT / "artifacts" / "vera_confirmatory_ablation_rows.csv"
DEFAULT_REPORT = ROOT / "artifacts" / "vera_confirmatory_ablation_report.json"
DEFAULT_TEX = (
    ROOT
    / "maintrack"
    / "aaai2027_template"
    / "AuthorKit27"
    / "vera_ablation_results.tex"
)


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


def choose(candidates: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    values = list(candidates)
    if not values:
        return None
    return min(
        values,
        key=lambda candidate: (
            candidate["validation_max_balanced_leakage"],
            candidate["validation_max_target_harm"],
            candidate["candidate"],
        ),
    )


def evaluate_candidates(
    loaded: list[dict[str, Any]],
    indices: np.ndarray,
    gamma: float,
    delta: float,
    candidate_count: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    local_alpha = delta / candidate_count
    for candidate in loaded:
        target, leakage, source = balanced_samples(candidate["arrays"], indices)
        target_point, leakage_point = balanced_point_metrics(target, leakage, source)
        certificates = balanced_contract_certificates(
            target,
            leakage,
            source,
            gamma=gamma,
            local_failure_probability=local_alpha,
        )
        attacker_bounds = {
            key.removeprefix("balanced_leakage::"): certificate.upper_confidence_bound
            for key, certificate in certificates.items()
            if key.startswith("balanced_leakage::")
        }
        external_target, external_leakage = balanced_external_metrics(
            candidate["arrays"]
        )
        output.append(
            {
                "candidate": candidate["candidate"],
                "method": candidate["method"],
                "validation_max_target_harm": target_point,
                "validation_max_balanced_leakage": leakage_point,
                "target_ucb": max(
                    certificate.upper_confidence_bound
                    for key, certificate in certificates.items()
                    if key.startswith("target::")
                ),
                "attacker_ucbs": attacker_bounds,
                "external_max_target_harm": external_target,
                "external_max_balanced_leakage": external_leakage,
            }
        )
    return output


def make_row(
    *,
    dimension: str,
    condition: str,
    dataset: str,
    seed: int,
    target_threshold: float,
    leakage_threshold: float,
    gamma: float,
    support_mismatch: bool,
    candidates: list[dict[str, Any]],
    attackers: set[str],
    available: set[str] | None = None,
) -> dict[str, Any]:
    pool = (
        candidates
        if available is None
        else [candidate for candidate in candidates if candidate["candidate"] in available]
    )
    eligible = [
        candidate
        for candidate in pool
        if candidate["target_ucb"] <= target_threshold
        and max(candidate["attacker_ucbs"][name] for name in attackers)
        <= leakage_threshold
    ]
    selected = None if support_mismatch else choose(eligible)
    oracle = choose(
        candidate
        for candidate in pool
        if candidate["external_max_balanced_leakage"] is not None
        and candidate["external_max_target_harm"] <= target_threshold
        and float(candidate["external_max_balanced_leakage"]) <= leakage_threshold
    )
    deployed = selected is not None
    estimable = not support_mismatch
    satisfied = (
        None
        if selected is None or not estimable
        else selected["external_max_target_harm"] <= target_threshold
        and float(selected["external_max_balanced_leakage"]) <= leakage_threshold
    )
    limiting = ""
    if selected is not None:
        worst = max(selected["attacker_ucbs"][name] for name in attackers)
        limiting = ",".join(
            sorted(
                name
                for name in attackers
                if selected["attacker_ucbs"][name] >= worst - 1e-12
            )
        )
    return {
        "dimension": dimension,
        "condition": condition,
        "dataset": dataset,
        "seed": seed,
        "gamma": gamma,
        "target_threshold": target_threshold,
        "leakage_threshold": leakage_threshold,
        "deployed": deployed,
        "selected_candidate": "" if selected is None else selected["candidate"],
        "selected_method": "" if selected is None else selected["method"],
        "limiting_attackers": limiting,
        "external_contract_estimable": estimable,
        "external_contract_satisfied": (
            "NA" if satisfied is None else bool(satisfied)
        ),
        "measured_external_contract_violation": bool(
            deployed and satisfied is False
        ),
        "safe_deployment": bool(deployed and satisfied is True),
        "oracle_deployed": oracle is not None,
        "procedurally_unsupported": support_mismatch,
        "secondary": True,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    estimable = [row for row in rows if as_bool(row["external_contract_estimable"])]
    deployments = sum(as_bool(row["deployed"]) for row in rows)
    estimable_deployments = sum(as_bool(row["deployed"]) for row in estimable)
    violations = sum(
        as_bool(row["measured_external_contract_violation"]) for row in estimable
    )
    safe = sum(as_bool(row["safe_deployment"]) for row in estimable)
    opportunities = sum(as_bool(row["oracle_deployed"]) for row in rows)
    return {
        "configuration_count": len(rows),
        "estimable_configuration_count": len(estimable),
        "deployment_count": deployments,
        "deployment_rate": deployments / len(rows),
        "measured_external_violation_count": violations,
        "measured_external_violation_rate": (
            None if not estimable else violations / len(estimable)
        ),
        "measured_external_violation_cp95": list(
            cp_interval(violations, len(estimable))
        ),
        "conditional_violation_rate": (
            None if estimable_deployments == 0 else violations / estimable_deployments
        ),
        "safe_deployment_count": safe,
        "oracle_opportunity_count": opportunities,
        "safe_retention": 0.0 if opportunities == 0 else safe / opportunities,
        "selected_method_counts": dict(
            sorted(
                Counter(
                    row["selected_method"]
                    for row in rows
                    if as_bool(row["deployed"])
                ).items()
            )
        ),
        "limiting_attacker_counts": dict(
            sorted(
                Counter(
                    attacker
                    for row in rows
                    for attacker in str(row["limiting_attackers"]).split(",")
                    if attacker
                ).items()
            )
        ),
    }


def derive_validation_and_threshold_rows(
    rule_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    oracle_by_id = {
        row["config_id"]: row
        for row in rule_rows
        if row["rule"] == "external_balanced_oracle"
    }
    for row in rule_rows:
        if row["rule"] not in {
            "point_selection_balanced",
            "vera_balanced_iut",
        }:
            continue
        if not np.isclose(float(row["gamma"]), 1.0):
            continue
        rule_label = (
            "point" if row["rule"] == "point_selection_balanced" else "vera_iut"
        )
        common = {
            "dataset": row["dataset"],
            "seed": int(row["seed"]),
            "gamma": 1.0,
            "target_threshold": float(row["target_threshold"]),
            "leakage_threshold": float(row["leakage_threshold"]),
            "deployed": as_bool(row["deployed"]),
            "selected_candidate": row["selected_candidate"],
            "selected_method": row["selected_method"],
            "limiting_attackers": "",
            "external_contract_estimable": as_bool(
                row["external_contract_estimable"]
            ),
            "external_contract_satisfied": row["external_contract_satisfied"],
            "measured_external_contract_violation": as_bool(
                row["measured_external_contract_violation"]
            ),
            "safe_deployment": bool(
                as_bool(row["deployed"])
                and row["external_contract_satisfied"] == "True"
            ),
            "oracle_deployed": as_bool(oracle_by_id[row["config_id"]]["deployed"]),
            "procedurally_unsupported": row["dataset"] == "Camelyon17-WILDS",
            "secondary": True,
        }
        output.append(
            {
                "dimension": "validation_size",
                "condition": (
                    f"fraction={float(row['validation_fraction']):g}|rule={rule_label}"
                ),
                **common,
            }
        )
        if (
            row["analysis_tier"] == "primary"
            and row["rule"] == "vera_balanced_iut"
        ):
            output.append(
                {
                    "dimension": "contract_threshold",
                    "condition": (
                        f"tau={float(row['target_threshold']):g}|"
                        f"lambda={float(row['leakage_threshold']):g}"
                    ),
                    **common,
                }
            )
    return output


def geometry_summary(candidate_rows: list[dict[str, str]]) -> dict[str, Any]:
    primary = [
        row for row in candidate_rows if row["analysis_tier"] == "primary"
    ]
    target_coordinates: defaultdict[str, list[float]] = defaultdict(list)
    source_coordinates: defaultdict[str, list[float]] = defaultdict(list)
    common: list[float] = []
    limiting = Counter()
    camelyon_zero_ok = True
    for row in primary:
        common.append(float(row["envelope_radius"]))
        target = json.loads(row["target_environment_radii"])
        source = json.loads(row["source_class_radii"])
        for key, value in target.items():
            target_coordinates[f"{row['dataset']}|environment={key}"].append(
                float(value)
            )
        for key, value in source.items():
            source_coordinates[f"{row['dataset']}|source={key}"].append(
                float(value)
            )
        limiting.update(json.loads(row["envelope_limiting_contracts"]))
        if row["dataset"] == "Camelyon17-WILDS":
            camelyon_zero_ok &= float(target.get("2", np.nan)) == 0.0
            camelyon_zero_ok &= float(row["envelope_radius"]) == 0.0
    return {
        "candidate_configuration_count": len(primary),
        "common_radius_median": float(np.median(common)),
        "common_radius_nonzero_rate": float(np.mean(np.asarray(common) > 0.0)),
        "target_coordinate_medians": {
            key: float(np.median(values))
            for key, values in sorted(target_coordinates.items())
        },
        "source_coordinate_medians": {
            key: float(np.median(values))
            for key, values in sorted(source_coordinates.items())
        },
        "limiting_contract_counts": dict(sorted(limiting.items())),
        "camelyon_unsupported_coordinate_zero": bool(camelyon_zero_ok),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float | None) -> str:
    return "NA" if value is None else f"{100.0 * value:.1f}\\%"


def write_tex(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    lines = [
        r"\section{Secondary Sensitivity Analyses}",
        r"\begin{table*}[t]",
        r"\centering\scriptsize",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dimension & Condition & Deploy & Measured violation & Safe retention \\",
        r"\midrule",
    ]
    dimensions = (
        "fixed_profile_shift_budget",
        "attacker_portfolio",
        "eraser_frontier",
        "frontier_granularity",
    )
    for dimension in dimensions:
        records = [
            (condition, summary)
            for key, summary in summaries.items()
            for key_dimension, condition in [key.split("|", 1)]
            if key_dimension == dimension
        ]
        for condition, summary in records:
            lines.append(
                f"{dimension.replace('_', ' ')} & "
                f"{condition.replace('_', r'\_')} & "
                f"{pct(float(summary['deployment_rate']))} & "
                f"{pct(summary['measured_external_violation_rate'])} & "
                f"{pct(float(summary['safe_retention']))} \\\\"
            )
        lines.append(r"\addlinespace")
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Outcome-blindly locked secondary analyses. External safety always uses all four attackers. Candidate alpha and envelope multiplicity remain fixed at the original 12-candidate family, so removing attackers or erasers receives no confidence-allocation windfall.}",
            r"\label{tab:ablations}",
            r"\end{table*}",
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--ablation-prereg", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--hash-file", type=Path, default=DEFAULT_HASH)
    parser.add_argument("--receipt-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--rule-rows", type=Path, default=DEFAULT_RULE_ROWS)
    parser.add_argument("--candidate-rows", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ablation_hash = sha256(args.ablation_prereg)
    if ablation_hash != args.hash_file.read_text(encoding="utf-8").split()[0]:
        raise RuntimeError("confirmatory ablation hash mismatch")
    relative = args.ablation_prereg.relative_to(REPOSITORY).as_posix()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != ablation_hash:
        raise RuntimeError("confirmatory ablation preregistration is not committed")

    prereg = load_json(args.prereg)
    ablation = load_json(args.ablation_prereg)
    if ablation["parent_preregistration_sha256"] != sha256(args.prereg):
        raise RuntimeError("ablation parent preregistration mismatch")
    receipt_audit = load_json(args.receipt_audit)
    if (
        receipt_audit.get("passed") is not True
        or int(receipt_audit.get("official_run_receipt_count", -1)) != 200
    ):
        raise RuntimeError("200-run receipt gate has not passed")

    study = prereg["real_study"]
    seeds = list(map(int, study["seeds"]))
    target_thresholds = list(map(float, study["target_harm_thresholds"]))
    leakage_thresholds = list(map(float, study["leakage_thresholds"]))
    delta = float(study["delta"])
    gamma_values = list(
        map(
            float,
            ablation["ablations"]["fixed_profile_shift_budget"]["gamma_values"],
        )
    )
    portfolios = {
        name: set(attackers)
        for name, attackers in ablation["ablations"]["attacker_portfolio"][
            "portfolios"
        ].items()
    }
    full_attackers = portfolios["full"]
    eraser_conditions = ablation["ablations"]["eraser_frontier"]["conditions"]
    coarse = set(
        ablation["ablations"]["frontier_granularity"]["conditions"]["coarse_5"]
    )
    rule_rows = load_csv(args.rule_rows)
    candidate_rows = load_csv(args.candidate_rows)
    primary_iut = {
        (
            row["dataset"],
            int(row["seed"]),
            float(row["target_threshold"]),
            float(row["leakage_threshold"]),
        ): row
        for row in rule_rows
        if row["analysis_tier"] == "primary"
        and row["rule"] == "vera_balanced_iut"
    }

    rows: list[dict[str, Any]] = []
    consistency_failures: list[str] = []
    for dataset, dataset_config in study["datasets"].items():
        support_mismatch = bool(
            dataset_config.get("force_abstain_for_unsupported_environment")
        )
        for seed in seeds:
            loaded, labels = load_candidates(args.receipt_dir, study, dataset, seed)
            indices = np.arange(len(labels[0]), dtype=np.int64)
            evaluations_by_gamma = {
                gamma: evaluate_candidates(
                    loaded, indices, gamma, delta, candidate_count=12
                )
                for gamma in gamma_values
            }
            base = evaluations_by_gamma[1.0]
            all_candidates = {candidate["candidate"] for candidate in base}
            if not coarse.issubset(all_candidates):
                missing = sorted(coarse - all_candidates)
                raise RuntimeError(f"coarse frontier candidates missing: {missing}")

            for gamma, evaluated in evaluations_by_gamma.items():
                for tau in target_thresholds:
                    for lam in leakage_thresholds:
                        rows.append(
                            make_row(
                                dimension="fixed_profile_shift_budget",
                                condition=f"gamma={gamma:g}",
                                dataset=dataset,
                                seed=seed,
                                target_threshold=tau,
                                leakage_threshold=lam,
                                gamma=gamma,
                                support_mismatch=support_mismatch,
                                candidates=evaluated,
                                attackers=full_attackers,
                            )
                        )
            for portfolio, attackers in portfolios.items():
                for tau in target_thresholds:
                    for lam in leakage_thresholds:
                        row = make_row(
                            dimension="attacker_portfolio",
                            condition=portfolio,
                            dataset=dataset,
                            seed=seed,
                            target_threshold=tau,
                            leakage_threshold=lam,
                            gamma=1.0,
                            support_mismatch=support_mismatch,
                            candidates=base,
                            attackers=attackers,
                        )
                        rows.append(row)
                        if portfolio == "full":
                            expected = primary_iut[(dataset, seed, tau, lam)]
                            if (
                                as_bool(expected["deployed"]) != row["deployed"]
                                or expected["selected_candidate"]
                                != row["selected_candidate"]
                            ):
                                consistency_failures.append(
                                    f"full portfolio mismatch: {dataset}/seed-{seed}/"
                                    f"{tau:g}/{lam:g}"
                                )
            method_aliases = {
                "leave INLP out": "INLP",
                "leave R-LACE out": "RLACE",
                "leave LEACE out": "LEACE",
                "leave TaCo out": "TaCo",
                "leave MANCE++ out": "MANCE++",
            }
            for condition in eraser_conditions:
                available = set(all_candidates)
                if condition != "all five official erasers":
                    excluded = method_aliases[condition]
                    available = {
                        candidate["candidate"]
                        for candidate in base
                        if candidate["method"] != excluded
                    }
                for tau in target_thresholds:
                    for lam in leakage_thresholds:
                        rows.append(
                            make_row(
                                dimension="eraser_frontier",
                                condition=condition,
                                dataset=dataset,
                                seed=seed,
                                target_threshold=tau,
                                leakage_threshold=lam,
                                gamma=1.0,
                                support_mismatch=support_mismatch,
                                candidates=base,
                                attackers=full_attackers,
                                available=available,
                            )
                        )
            for condition, available in (
                ("full_12", all_candidates),
                ("coarse_5", coarse),
            ):
                for tau in target_thresholds:
                    for lam in leakage_thresholds:
                        rows.append(
                            make_row(
                                dimension="frontier_granularity",
                                condition=condition,
                                dataset=dataset,
                                seed=seed,
                                target_threshold=tau,
                                leakage_threshold=lam,
                                gamma=1.0,
                                support_mismatch=support_mismatch,
                                candidates=base,
                                attackers=full_attackers,
                                available=set(available),
                            )
                        )

    rows.extend(derive_validation_and_threshold_rows(rule_rows))
    expected_by_dimension = {
        "fixed_profile_shift_budget": 2880,
        "attacker_portfolio": 2520,
        "eraser_frontier": 2160,
        "frontier_granularity": 720,
        "validation_size": 3600,
        "contract_threshold": 360,
    }
    observed_by_dimension = Counter(row["dimension"] for row in rows)
    coverage_failures = [
        f"{dimension}: expected {expected}, observed {observed_by_dimension[dimension]}"
        for dimension, expected in expected_by_dimension.items()
        if observed_by_dimension[dimension] != expected
    ]
    if consistency_failures or coverage_failures:
        raise RuntimeError("; ".join(consistency_failures + coverage_failures))

    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dimension"], row["condition"])].append(row)
    summaries = {
        f"{dimension}|{condition}": summarize(values)
        for (dimension, condition), values in sorted(grouped.items())
    }
    geometry = geometry_summary(candidate_rows)
    passed = geometry["camelyon_unsupported_coordinate_zero"]
    report = {
        "name": "VERA untouched-seed balanced secondary ablations",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "secondary": True,
        "parent_preregistration_sha256": sha256(args.prereg),
        "ablation_preregistration_sha256": ablation_hash,
        "receipt_audit_sha256": sha256(args.receipt_audit),
        "primary_rule_rows_sha256": sha256(args.rule_rows),
        "candidate_rows_sha256": sha256(args.candidate_rows),
        "row_count": len(rows),
        "rows_by_dimension": dict(sorted(observed_by_dimension.items())),
        "summaries": summaries,
        "certificate_geometry": geometry,
        "consistency_failures": consistency_failures,
        "coverage_failures": coverage_failures,
        "claim_boundary": (
            "These outcome-blindly locked analyses are secondary. They cannot "
            "replace or redefine a failed confirmatory endpoint."
        ),
    }
    write_csv(args.rows, rows)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_tex(args.tex, summaries)
    print(
        json.dumps(
            {
                "passed": passed,
                "row_count": len(rows),
                "dimensions": dict(sorted(observed_by_dimension.items())),
            },
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
