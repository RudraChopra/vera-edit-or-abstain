#!/usr/bin/env python3
"""Summarize MOSAIC real evidence by dataset rather than treating jobs as domains."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np


RULES = (
    "strict_mosaic",
    "bridge_plugin",
    "validation_plugin",
    "always_deploy_validation",
)
RULE_LABELS = {
    "strict_mosaic": "strict_mosaic",
    "bridge_plugin": "bridge_plugin",
    "validation_plugin": "validation_plugin",
    "always_deploy_validation": "unconditional",
}


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def atomic_dump(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)


def key(value: float) -> str:
    return f"{float(value):.2f}"


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def strict_receipts(directory: Path, threshold: float) -> dict[str, list[dict[str, Any]]]:
    receipts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    threshold_name = key(threshold)
    for path in sorted(directory.glob("*.json")):
        payload = load(path)
        selection = payload.get("selection_by_utility_threshold", {}).get(threshold_name)
        if not isinstance(selection, dict):
            raise ValueError(f"{path} has no selection for {threshold_name}")
        payload["_selection"] = selection
        receipts[str(payload["dataset"])].append(payload)
    if not receipts:
        raise ValueError(f"no receipts found in {directory}")
    return dict(receipts)


def has_zero_retained_mass(receipt: dict[str, Any]) -> bool:
    """Detect the strict replay's serialized consequence of missing bridge support."""

    masses = [
        float(value)
        for row in receipt.get("results", [])
        for value in row.get("bridge_membership", {}).get("retained_masses", [])
    ]
    return bool(masses) and all(value <= 1e-12 for value in masses)


def strict_dataset_summary(
    receipts: list[dict[str, Any]], *, privacy_threshold: float, utility_threshold: float
) -> dict[str, Any]:
    selections = [receipt["_selection"] for receipt in receipts]
    deployed = [selection for selection in selections if selection.get("decision") == "deploy"]
    estimable = [selection for selection in deployed if selection.get("diagnostic_estimable")]
    violations = sum(bool(selection.get("false_acceptance")) for selection in estimable)
    retained = [
        1.0 - float(contamination)
        for selection in deployed
        for contamination in selection.get("bridge_contaminations", [])
    ]
    limiting = []
    for selection in deployed:
        privacy_slack = privacy_threshold - max(
            float(value) for value in selection["certified_source_advantage_upper"]
        )
        utility_slack = utility_threshold - float(
            selection["certified_worst_conditional_error_upper"])
        limiting.append("source" if privacy_slack < utility_slack else "utility")
    if limiting:
        limiting_contract = (
            limiting[0] if all(value == limiting[0] for value in limiting) else "mixed"
        )
    elif any(has_zero_retained_mass(receipt) for receipt in receipts):
        limiting_contract = "zero retained mass (audited missing support)"
    else:
        limiting_contract = "no joint source/utility feasibility"
    return {
        "jobs": len(receipts),
        "deployments": len(deployed),
        "estimable_deployments": len(estimable),
        "diagnostic_violations": violations,
        "median_retained_mass": float(np.median(retained)) if retained else None,
        "retained_mass_range": [float(min(retained)), float(max(retained))] if retained else None,
        "limiting_contract": limiting_contract,
        "zero_retained_mass_jobs": sum(has_zero_retained_mass(receipt) for receipt in receipts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-summary", required=True, type=Path)
    parser.add_argument("--strict-receipts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--utility-threshold", type=float, default=0.40)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_719)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_repetitions < 1:
        raise ValueError("bootstrap repetitions must be positive")
    evidence = load(args.evidence_summary)
    cells = evidence.get("cells")
    if not isinstance(cells, dict) or any(rule not in cells for rule in RULES):
        raise ValueError("evidence summary is missing a required deployment rule")
    threshold_name = key(args.utility_threshold)
    privacy_threshold = 0.35
    receipts = strict_receipts(args.strict_receipts, args.utility_threshold)
    datasets = sorted(receipts)
    if set(datasets) != {
        dataset for dataset in cells["strict_mosaic"] if dataset != "all_datasets"
    }:
        raise ValueError("strict receipt datasets do not match the evidence summary")
    per_dataset: dict[str, Any] = {}
    for dataset in datasets:
        strict = strict_dataset_summary(
            receipts[dataset],
            privacy_threshold=privacy_threshold,
            utility_threshold=args.utility_threshold,
        )
        rules: dict[str, Any] = {}
        for rule in RULES:
            cell = cells[rule][dataset][threshold_name]
            rules[RULE_LABELS[rule]] = {
                "deployments": int(cell["deployments"]),
                "estimable_deployments": int(cell["estimable_deployments"]),
                "diagnostic_violations": int(cell["false_acceptances"]),
                "unestimable_deployments": int(cell["unestimable_deployments"]),
            }
        if strict["deployments"] != rules["strict_mosaic"]["deployments"]:
            raise ValueError(f"strict deployment mismatch for {dataset}")
        if strict["diagnostic_violations"] != rules["strict_mosaic"]["diagnostic_violations"]:
            raise ValueError(f"strict diagnostic mismatch for {dataset}")
        per_dataset[dataset] = {"strict_details": strict, "rules": rules}

    strict_rates = np.asarray(
        [
            per_dataset[dataset]["rules"]["strict_mosaic"]["deployments"]
            / per_dataset[dataset]["strict_details"]["jobs"]
            for dataset in datasets
        ],
        dtype=np.float64,
    )
    generator = np.random.default_rng(args.bootstrap_seed)
    indices = generator.integers(
        0, len(datasets), size=(args.bootstrap_repetitions, len(datasets))
    )
    bootstrap_rates = strict_rates[indices].mean(axis=1)
    leave_one_out = {}
    for omitted in datasets:
        retained_datasets = [dataset for dataset in datasets if dataset != omitted]
        deployments = sum(
            per_dataset[dataset]["rules"]["strict_mosaic"]["deployments"]
            for dataset in retained_datasets
        )
        estimable = sum(
            per_dataset[dataset]["rules"]["strict_mosaic"]["estimable_deployments"]
            for dataset in retained_datasets
        )
        violations = sum(
            per_dataset[dataset]["rules"]["strict_mosaic"]["diagnostic_violations"]
            for dataset in retained_datasets
        )
        leave_one_out[omitted] = {
            "strict_deployments": deployments,
            "strict_estimable_deployments": estimable,
            "strict_diagnostic_violations": violations,
        }

    aggregate = {
        RULE_LABELS[rule]: cells[rule]["all_datasets"][threshold_name]
        for rule in RULES
    }
    report = {
        "name": "MOSAIC dataset-grouped real-evidence sensitivity",
        "utility_threshold": args.utility_threshold,
        "privacy_threshold": privacy_threshold,
        "datasets": datasets,
        "per_dataset": per_dataset,
        "aggregate": aggregate,
        "strict_dataset_rate": {
            "estimate": float(strict_rates.mean()),
            "descriptive_dataset_bootstrap_95_interval": percentile_interval(bootstrap_rates),
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "leave_one_dataset_out": leave_one_out,
        "claim_boundary": (
            "Dataset-level bootstrap is descriptive only. The finite-sample MOSAIC "
            "guarantee is per job on its covered reference and bridge event. These "
            "five benchmarks are not five exchangeable deployment populations, and "
            "the aggregate 100-job rate must not be read as an interval over 100 "
            "independent real domains."
        ),
    }
    atomic_dump(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
