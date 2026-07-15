"""Design the fresh controlled-shift study using inspected seeds only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from analyze_vera_attacker_ablation import load_candidates
from analyze_vera_secondary_ablations import load_json, sha256
from vera_controlled_shift import (
    allocate_integer_budget,
    design_controlled_shift_from_fold,
)
from vera_robust_certificate import (
    balanced_profile_contract_certificates,
    balanced_profile_in_envelope,
    certify_balanced_iut_profile,
    certify_balanced_shift_envelope,
    empirical_reweighting_risk,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "prereg_independent_stress_replication.json"
DEFAULT_HASH = ROOT / "prereg_independent_stress_replication.sha256"
DEFAULT_RECEIPTS = ROOT / "artifacts" / "independent_stress_replication_receipts"
DEFAULT_OUTPUT = ROOT / "artifacts" / "vera_controlled_shift_design.json"
DATASETS = ("Waterbirds", "CivilComments-WILDS", "Bios", "GaitPDB")
ATTACKERS = ("linear", "rbf", "forest", "mlp")
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


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def candidate_arrays(
    arrays: Mapping[str, np.ndarray], split: str
) -> dict[str, np.ndarray]:
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


def validate_shared_metadata(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    reference = candidate_arrays(candidates[0]["arrays"], "certification")
    design = candidate_arrays(candidates[0]["arrays"], "external")
    for candidate in candidates[1:]:
        for split, expected in (("certification", reference), ("external", design)):
            observed = candidate_arrays(candidate["arrays"], split)
            for key in ("source", "environment", "target"):
                if not np.array_equal(expected[key], observed[key]):
                    raise RuntimeError(f"candidate {split} metadata mismatch: {key}")
    return reference, design


def q_metrics(
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
        for source in (0, 1):
            mask = candidate["source"] == source
            conditional = probabilities[mask] / probabilities[mask].sum()
            recalls.append(
                float(np.dot(conditional, candidate[f"leakage::{attacker}"][mask]))
            )
        leakage_risks.append(float(np.mean(recalls)))
    return max(target_risks), max(leakage_risks)


def sampled_metrics(
    candidate: Mapping[str, np.ndarray], indices: np.ndarray
) -> tuple[float, float]:
    target_risks = [
        float(candidate["target_harm"][indices[candidate["environment"][indices] == environment]].mean())
        for environment in sorted(map(int, np.unique(candidate["environment"])))
    ]
    leakage_risks = []
    for attacker in ATTACKERS:
        recalls = [
            float(candidate[f"leakage::{attacker}"][indices[candidate["source"][indices] == source]].mean())
            for source in (0, 1)
        ]
        leakage_risks.append(float(np.mean(recalls)))
    return max(target_risks), max(leakage_risks)


def design_metrics(
    candidate: Mapping[str, np.ndarray],
    indices: np.ndarray,
    target_profile: Mapping[str, float],
    source_profile: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, float], float, float]:
    target_risks: dict[str, float] = {}
    for environment, gamma in target_profile.items():
        values = candidate["target_harm"][
            indices[candidate["environment"][indices] == int(environment)]
        ]
        target_risks[str(environment)] = empirical_reweighting_risk(values, gamma)
    attacker_risks: dict[str, float] = {}
    for attacker in ATTACKERS:
        recalls = []
        for source, gamma in source_profile.items():
            values = candidate[f"leakage::{attacker}"][
                indices[candidate["source"][indices] == int(source)]
            ]
            recalls.append(empirical_reweighting_risk(values, gamma))
        attacker_risks[attacker] = float(np.mean(recalls))
    return (
        target_risks,
        attacker_risks,
        max(target_risks.values()),
        max(attacker_risks.values()),
    )


def allocation_scores(
    candidates: list[dict[str, Any]],
    design_indices: np.ndarray,
    target_profile: Mapping[str, float],
    source_profile: Mapping[str, float],
    *,
    target_threshold: float,
    leakage_threshold: float,
) -> tuple[dict[str, float], str]:
    records = []
    for candidate in candidates:
        target, leakage, target_max, leakage_max = design_metrics(
            candidate["design"], design_indices, target_profile, source_profile
        )
        normalized_margin = min(
            *(
                (target_threshold - value) / 2.0
                for value in target.values()
            ),
            *(
                leakage_threshold - value
                for value in leakage.values()
            ),
        )
        records.append(
            (normalized_margin, -leakage_max, -target_max, candidate, target, leakage)
        )
    _, _, _, selected, target_risks, leakage_risks = max(
        records, key=lambda item: (item[0], item[1], item[2], item[3]["candidate"])
    )
    leakage_margin = max(
        0.01, leakage_threshold - max(leakage_risks.values())
    )
    scores = {
        f"target::{environment}": (
            2.0
            * gamma
            / max(0.01, target_threshold - target_risks[environment])
        )
        ** 2
        for environment, gamma in target_profile.items()
    }
    scores.update(
        {
            f"source::{source}": (0.5 * gamma / leakage_margin) ** 2
            for source, gamma in source_profile.items()
        }
    )
    return scores, str(selected["candidate"])


def sample_streams(
    metadata: Mapping[str, np.ndarray],
    allocation: Mapping[str, int],
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    target_indices = {
        f"target::environment={cell.split('::', 1)[1]}": rng.choice(
            np.flatnonzero(metadata["environment"] == int(cell.split("::", 1)[1])),
            size=count,
            replace=True,
        )
        for cell, count in allocation.items()
        if cell.startswith("target::")
    }
    source_indices = {
        int(cell.split("::", 1)[1]): rng.choice(
            np.flatnonzero(metadata["source"] == int(cell.split("::", 1)[1])),
            size=count,
            replace=True,
        )
        for cell, count in allocation.items()
        if cell.startswith("source::")
    }
    return target_indices, source_indices


def candidate_certification_data(
    candidate: Mapping[str, np.ndarray],
    target_indices: Mapping[str, np.ndarray],
    source_indices: Mapping[int, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    target = {
        key: candidate["target_harm"][indices]
        for key, indices in target_indices.items()
    }
    ordered = [source_indices[source] for source in (0, 1)]
    source = np.concatenate(
        [np.full(len(indices), value, dtype=np.int64) for value, indices in zip((0, 1), ordered)]
    )
    leakage = {
        attacker: np.concatenate(
            [candidate[f"leakage::{attacker}"][indices] for indices in ordered]
        )
        for attacker in ATTACKERS
    }
    return target, leakage, source


def point_metrics(
    target: Mapping[str, np.ndarray],
    leakage: Mapping[str, np.ndarray],
    source: np.ndarray,
) -> tuple[float, float]:
    target_max = max(float(values.mean()) for values in target.values())
    leakage_max = max(
        0.5
        * sum(float(values[source == source_class].mean()) for source_class in (0, 1))
        for values in leakage.values()
    )
    return target_max, leakage_max


def choose(candidates: list[dict[str, Any]], key: str | None) -> dict[str, Any] | None:
    eligible = candidates if key is None else [candidate for candidate in candidates if candidate[key]]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda candidate: (
            candidate["point_leakage"],
            candidate["point_target"],
            candidate["candidate"],
        ),
    )


def evaluate_configuration(
    candidates: list[dict[str, Any]],
    metadata: Mapping[str, np.ndarray],
    probabilities: np.ndarray,
    target_profile: Mapping[str, float],
    source_profile: Mapping[str, float],
    allocation: Mapping[str, int],
    *,
    rng: np.random.Generator,
    delta: float,
    target_threshold: float,
    leakage_threshold: float,
) -> dict[str, dict[str, Any]]:
    target_indices, source_indices = sample_streams(metadata, allocation, rng)
    candidate_count = len(candidates)
    environment_count = len(target_profile)
    family_size = candidate_count * (environment_count + len(ATTACKERS))
    common_budget = max(*target_profile.values(), *source_profile.values())
    evaluated = []
    for candidate in candidates:
        target, leakage, source = candidate_certification_data(
            candidate["reference"], target_indices, source_indices
        )
        point_target, point_leakage = point_metrics(target, leakage, source)
        target_key_profile = {
            f"target::environment={environment}": gamma
            for environment, gamma in target_profile.items()
        }
        iid = certify_balanced_iut_profile(
            target,
            leakage,
            source,
            target_profile={key: 1.0 for key in target},
            source_profile={"0": 1.0, "1": 1.0},
            delta=delta,
            candidate_count=candidate_count,
            target_threshold=target_threshold,
            leakage_threshold=leakage_threshold,
        )
        fixed = certify_balanced_iut_profile(
            target,
            leakage,
            source,
            target_profile=target_key_profile,
            source_profile=source_profile,
            delta=delta,
            candidate_count=candidate_count,
            target_threshold=target_threshold,
            leakage_threshold=leakage_threshold,
        )
        envelope = certify_balanced_shift_envelope(
            target,
            leakage,
            source,
            delta=delta,
            family_size=family_size,
            target_threshold=target_threshold,
            leakage_threshold=leakage_threshold,
            registered_target_environments=sorted(map(int, target_profile)),
            gamma_cap=4.0,
        )
        profile_certificates = balanced_profile_contract_certificates(
            target,
            leakage,
            source,
            target_profile=target_key_profile,
            source_profile=source_profile,
            local_failure_probability=delta / family_size,
        )
        empirical_profile = {
            key: certificate.empirical_robust_risk
            for key, certificate in profile_certificates.items()
        }
        robust_point_feasible = all(
            value
            <= (target_threshold if key.startswith("target::") else leakage_threshold)
            for key, value in empirical_profile.items()
        )
        scalar_score = float(
            np.mean(
                [
                    certificate.upper_confidence_bound
                    / (target_threshold if key.startswith("target::") else leakage_threshold)
                    for key, certificate in profile_certificates.items()
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
        evaluated.append(
            {
                "candidate": candidate["candidate"],
                "method": candidate["method"],
                "point_target": point_target,
                "point_leakage": point_leakage,
                "point_feasible": point_target <= target_threshold and point_leakage <= leakage_threshold,
                "iid_eligible": iid.decision == "EDIT",
                "robust_point_eligible": robust_point_feasible,
                "scalar_eligible": scalar_score <= 1.0,
                "fixed_eligible": fixed.decision == "EDIT",
                "vector_eligible": balanced_profile_in_envelope(
                    envelope,
                    target_profile=target_profile,
                    source_profile=source_profile,
                ),
                "common_eligible": envelope.deployment_common_radius >= common_budget,
                "q_safe": q_safe,
                "evaluation_safe": evaluation_safe,
                "q_target": q_target,
                "q_leakage": q_leakage,
                "evaluation_target": evaluation_target,
                "evaluation_leakage": evaluation_leakage,
                "envelope_radius": envelope.deployment_common_radius,
            }
        )
    selections = {
        "always_deploy": choose(evaluated, None),
        "validation_point_selection": choose(evaluated, "point_feasible"),
        "iid_ltt": choose(evaluated, "iid_eligible"),
        "robust_point_estimate": choose(evaluated, "robust_point_eligible"),
        "generic_scalar_robust_certificate": choose(evaluated, "scalar_eligible"),
        "vera_fixed_profile": choose(evaluated, "fixed_eligible"),
        "vera_vector_envelope": choose(evaluated, "vector_eligible"),
        "vera_common_radius": choose(evaluated, "common_eligible"),
        "external_oracle": choose(evaluated, "q_safe"),
    }
    return {
        rule: {
            "deployed": selected is not None,
            "safe": bool(selected is not None and selected["q_safe"]),
            "violation": bool(selected is not None and not selected["q_safe"]),
            "evaluation_violation": bool(
                selected is not None and not selected["evaluation_safe"]
            ),
            "selected_candidate": "" if selected is None else selected["candidate"],
            "selected_method": "" if selected is None else selected["method"],
            "q_target": None if selected is None else selected["q_target"],
            "q_leakage": None if selected is None else selected["q_leakage"],
            "evaluation_target": (
                None if selected is None else selected["evaluation_target"]
            ),
            "evaluation_leakage": (
                None if selected is None else selected["evaluation_leakage"]
            ),
            "certified_common_radius": 0.0 if selected is None else selected["envelope_radius"],
        }
        for rule, selected in selections.items()
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["requested_gamma"],
            row["total_budget"],
            row["allocation"],
            row["rule"],
        )
        grouped[key].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        oracle_opportunities = sum(row["oracle_deployed"] for row in values)
        safe_on_opportunity = sum(
            row["oracle_deployed"] and row["safe"] for row in values
        )
        output.append(
            {
                "dataset": key[0],
                "requested_gamma": key[1],
                "total_budget": key[2],
                "allocation": key[3],
                "rule": key[4],
                "seed_count": len(values),
                "deployment_count": sum(row["deployed"] for row in values),
                "violation_count": sum(row["violation"] for row in values),
                "violation_rate": sum(row["violation"] for row in values) / len(values),
                "evaluation_violation_count": sum(
                    row["evaluation_violation"] for row in values
                ),
                "evaluation_violation_rate": sum(
                    row["evaluation_violation"] for row in values
                )
                / len(values),
                "oracle_opportunity_count": oracle_opportunities,
                "safe_retention": (
                    None
                    if oracle_opportunities == 0
                    else safe_on_opportunity / oracle_opportunities
                ),
            }
        )
    return output


def global_operating_points(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (
            row["dataset"],
            row["requested_gamma"],
            row["total_budget"],
            row["allocation"],
            row["rule"],
        ): row
        for row in summary
    }
    output = []
    settings = sorted(
        {
            (row["requested_gamma"], row["total_budget"], row["allocation"])
            for row in summary
        }
    )
    for gamma, budget, allocation in settings:
        records: dict[str, dict[str, float | int | None]] = {}
        for rule in RULES:
            rows = [
                by_key[(dataset, gamma, budget, allocation, rule)]
                for dataset in DATASETS
            ]
            opportunities = sum(int(row["oracle_opportunity_count"]) for row in rows)
            retained = sum(
                int(row["deployment_count"]) - int(row["violation_count"])
                for row in rows
            )
            records[rule] = {
                "decision_count": sum(int(row["seed_count"]) for row in rows),
                "deployment_count": sum(int(row["deployment_count"]) for row in rows),
                "violation_count": sum(int(row["violation_count"]) for row in rows),
                "violation_rate": sum(int(row["violation_count"]) for row in rows)
                / sum(int(row["seed_count"]) for row in rows),
                "evaluation_violation_count": sum(
                    int(row["evaluation_violation_count"]) for row in rows
                ),
                "evaluation_violation_rate": sum(
                    int(row["evaluation_violation_count"]) for row in rows
                )
                / sum(int(row["seed_count"]) for row in rows),
                "safe_retention": None if opportunities == 0 else retained / opportunities,
                "oracle_opportunity_count": opportunities,
            }
        vector_retention = records["vera_vector_envelope"]["safe_retention"]
        common_retention = records["vera_common_radius"]["safe_retention"]
        ratio = None
        unbounded_advantage = False
        if isinstance(vector_retention, float) and isinstance(common_retention, float):
            unbounded_advantage = common_retention == 0.0 and vector_retention > 0.0
            ratio = (
                None
                if common_retention == 0.0
                else vector_retention / common_retention
            )
        output.append(
            {
                "requested_gamma": gamma,
                "total_budget": budget,
                "allocation": allocation,
                "rules": records,
                "vector_to_common_safe_retention_ratio": ratio,
                "vector_positive_when_common_zero": unbounded_advantage,
                "candidate_primary": (
                    allocation.startswith("targeted")
                    and records["vera_vector_envelope"]["violation_rate"] <= 0.05
                    and isinstance(vector_retention, float)
                    and vector_retention >= 0.20
                    and (unbounded_advantage or (ratio is not None and ratio >= 2.0))
                ),
            }
        )
    return output


def design(args: argparse.Namespace) -> dict[str, Any]:
    prereg = load_json(args.prereg)
    expected_hash = args.hash_file.read_text(encoding="utf-8").split()[0]
    if sha256(args.prereg) != expected_hash:
        raise RuntimeError("design-source preregistration hash mismatch")
    study = prereg["real_study"]
    seeds = [int(seed) for seed in study["seeds"]]
    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    allocations: list[dict[str, Any]] = []
    for dataset in DATASETS:
        contract = study["locked_dataset_contracts"][dataset]
        target_threshold = float(contract["target_harm_threshold"])
        leakage_threshold = float(contract["balanced_leakage_threshold"])
        for seed in seeds:
            loaded, _ = load_candidates(args.receipt_dir, study, dataset, seed)
            metadata, design_metadata = validate_shared_metadata(loaded)
            candidates = [
                {
                    "candidate": candidate["candidate"],
                    "method": candidate["method"],
                    "reference": candidate_arrays(
                        candidate["arrays"], "certification"
                    ),
                    "design": candidate_arrays(candidate["arrays"], "external"),
                }
                for candidate in loaded
            ]
            design_rng = np.random.default_rng(
                2_027_071_500 + 1009 * seed + sum(map(ord, dataset))
            )
            design_size = min(1000, len(design_metadata["source"]))
            design_indices = np.sort(
                design_rng.choice(
                    len(design_metadata["source"]), size=design_size, replace=False
                )
            )
            for requested_gamma in args.gammas:
                probabilities, shift = design_controlled_shift_from_fold(
                    metadata["environment"],
                    metadata["source"],
                    metadata["target"],
                    design_metadata["environment"][design_indices],
                    design_metadata["source"][design_indices],
                    design_metadata["target"][design_indices],
                    requested_gamma=requested_gamma,
                    minimum_design_cell_count=max(2, min(8, design_size // 20)),
                )
                evaluation_rng = np.random.default_rng(
                    6_000_000_000
                    + 1_000_003 * seed
                    + 10_007 * int(round(100 * requested_gamma))
                    + sum(map(ord, dataset))
                )
                evaluation_indices = evaluation_rng.choice(
                    len(metadata["source"]),
                    size=args.evaluation_size,
                    replace=True,
                    p=probabilities,
                )
                density_ratio = probabilities * len(probabilities)
                profiles.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        **shift.to_dict(),
                        "reference_probability_sha256": array_sha256(probabilities),
                        "design_indices_sha256": array_sha256(design_indices),
                        "evaluation_indices_sha256": array_sha256(evaluation_indices),
                        "evaluation_size": int(len(evaluation_indices)),
                        "membership_verified": bool(
                            np.isclose(probabilities.sum(), 1.0)
                            and np.all(probabilities >= 0.0)
                            and density_ratio.max() <= requested_gamma + 1e-10
                        ),
                    }
                )
                for candidate in candidates:
                    candidate["q_metrics"] = q_metrics(
                        candidate["reference"], probabilities
                    )
                    candidate["evaluation_metrics"] = sampled_metrics(
                        candidate["reference"], evaluation_indices
                    )
                scores, allocation_candidate = allocation_scores(
                    candidates,
                    design_indices,
                    shift.target_profile,
                    shift.source_profile,
                    target_threshold=target_threshold,
                    leakage_threshold=leakage_threshold,
                )
                cell_count = len(scores)
                for total_budget in args.budgets:
                    allocation_plans = {
                        "uniform": allocate_integer_budget(
                            {key: 1.0 for key in scores},
                            total_budget=total_budget,
                            minimum_per_cell=1,
                        ),
                    }
                    for floor_fraction in args.targeted_floor_fractions:
                        minimum = max(1, int(np.ceil(total_budget * floor_fraction)))
                        if minimum * cell_count > total_budget:
                            raise ValueError(
                                "targeted floor fraction leaves no feasible allocation"
                            )
                        allocation_plans[f"targeted_floor_{floor_fraction:g}"] = (
                            allocate_integer_budget(
                                scores,
                                total_budget=total_budget,
                                minimum_per_cell=minimum,
                            )
                        )
                    for allocation_name, allocation in allocation_plans.items():
                        allocations.append(
                            {
                                "dataset": dataset,
                                "seed": seed,
                                "requested_gamma": requested_gamma,
                                "total_budget": total_budget,
                                "allocation": allocation_name,
                                "pilot_candidate": allocation_candidate,
                                "cell_allocation": allocation,
                                "scores": scores,
                            }
                        )
                        rng = np.random.default_rng(
                            8_000_000_000
                            + 1_000_003 * seed
                            + 10_007 * int(round(100 * requested_gamma))
                            + 101 * total_budget
                            + (1 if allocation_name == "targeted" else 0)
                            + sum(map(ord, dataset))
                        )
                        decisions = evaluate_configuration(
                            candidates,
                            metadata,
                            probabilities,
                            shift.target_profile,
                            shift.source_profile,
                            allocation,
                            rng=rng,
                            delta=float(study["delta"]),
                            target_threshold=target_threshold,
                            leakage_threshold=leakage_threshold,
                        )
                        oracle_deployed = decisions["external_oracle"]["deployed"]
                        for rule, decision in decisions.items():
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "requested_gamma": requested_gamma,
                                    "total_budget": total_budget,
                                    "allocation": allocation_name,
                                    "rule": rule,
                                    "oracle_deployed": oracle_deployed,
                                    **decision,
                                }
                            )
    summary = summarize(rows)
    operating_points = global_operating_points(summary)
    candidates = [point for point in operating_points if point["candidate_primary"]]
    recommended = (
        min(
            candidates,
            key=lambda point: (
                point["total_budget"],
                -point["requested_gamma"],
                -float(point["rules"]["vera_vector_envelope"]["safe_retention"]),
            ),
        )
        if candidates
        else None
    )
    return {
        "name": "VERA controlled-shift design analysis",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_tier": "exploratory design on inspected seeds; not confirmatory evidence",
        "source_prereg_sha256": sha256(args.prereg),
        "design_seeds": seeds,
        "future_seed_block": list(range(45, 77)),
        "datasets": list(DATASETS),
        "gammas": args.gammas,
        "budgets": args.budgets,
        "evaluation_size_per_dataset_seed_profile": args.evaluation_size,
        "rules": list(RULES),
        "profile_count": len(profiles),
        "allocation_count": len(allocations),
        "row_count": len(rows),
        "profiles": profiles,
        "allocations": allocations,
        "rows": rows,
        "summaries": summary,
        "operating_points": operating_points,
        "recommended_primary": recommended,
        "claim_boundary": (
            "These outcomes were inspected only to choose the fresh protocol. "
            "No number in this design report is a confirmatory paper result."
        ),
        "split_policy": (
            "Edits, target probes, and attackers use construction data; the official "
            "validation audit arrays define the finite reference law; the disjoint "
            "official external metadata arrays choose the shift cell without reading "
            "their outcomes; external design outcomes set the prospective evidence "
            "allocation; certification and final deployment are independent random "
            "streams from the reference and shifted laws, respectively."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--hash-file", type=Path, default=DEFAULT_HASH)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gammas", type=float, nargs="+", default=[1.1, 1.25, 1.5])
    parser.add_argument("--budgets", type=int, nargs="+", default=[4000, 8000, 16000])
    parser.add_argument("--evaluation-size", type=int, default=50_000)
    parser.add_argument(
        "--targeted-floor-fractions",
        type=float,
        nargs="+",
        default=[0.15],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = design(args)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    recommended = report["recommended_primary"]
    concise_recommendation = None
    if recommended is not None:
        vector = recommended["rules"]["vera_vector_envelope"]
        concise_recommendation = {
            "requested_gamma": recommended["requested_gamma"],
            "total_budget": recommended["total_budget"],
            "allocation": recommended["allocation"],
            "violation_rate": vector["violation_rate"],
            "safe_retention": vector["safe_retention"],
            "vector_to_common_safe_retention_ratio": recommended[
                "vector_to_common_safe_retention_ratio"
            ],
        }
    print(json.dumps({
        "row_count": report["row_count"],
        "recommended_primary": concise_recommendation,
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
