#!/usr/bin/env python3
"""Locked deterministic replay audit for the MOSAIC baseline extension."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Sequence

import numpy as np
from scipy.stats import beta

from mosaic_exact import (
    exact_external_attacker_risk,
    exact_external_utility_risk,
)
from mosaic_optimizer import optimize_invariant_channel
from run_mosaic_baseline_extension import (
    holm_ltt_selection,
    table_region_grid_selection,
)
from run_mosaic_synthetic_pilot import (
    Scenario,
    Selection,
    confidence_event,
    deterministic_selection,
    empirical_table,
    selection_from_mosaic_solution,
    simultaneous_radii,
    witness_scenario,
)


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_PREREG = ROOT / "prereg_mosaic_baseline_extension_v1.json"
DEFAULT_AUDIT_LOCK = ROOT / "prereg_mosaic_baseline_extension_audit_v1.json"
DEFAULT_REPORT = (
    REPOSITORY / "research/artifacts/mosaic_baseline_extension_v1.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY / "research/artifacts/mosaic_baseline_extension_audit_v1.json"
)
METHODS = (
    "mosaic_continuum",
    "table_region_grid",
    "holm_ltt_grid",
    "fare_style_deterministic",
)
TOLERANCE = 5e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def scenario_from_config(config: dict[str, Any]) -> Scenario:
    return replace(
        witness_scenario(
            privacy_threshold=float(config["source_advantage_threshold"]),
            utility_threshold=float(config["utility_threshold"]),
            contamination=float(config["contamination"]),
        ),
        name=str(config["name"]),
    )


def exact_row(
    *,
    seed: int,
    n: int,
    method: str,
    scenario: Scenario,
    selection: Selection,
    deployed: bool,
    event: bool | None,
) -> dict[str, object]:
    privacy_by_label = tuple(
        exact_external_attacker_risk(
            scenario.population[label],
            selection.channel,
            scenario.libraries[label],
            contamination=scenario.contaminations[label],
        ).normalized_advantage
        for label in range(scenario.population.shape[0])
    )
    worst_privacy = max(privacy_by_label)
    worst_utility = max(
        exact_external_utility_risk(
            scenario.population[label, source],
            selection.channel,
            selection.decoder,
            true_label=label,
            common_fine_token_channels=scenario.libraries[label],
            contamination=scenario.contaminations[label],
        ).error_probability
        for label in range(scenario.population.shape[0])
        for source in range(scenario.population.shape[1])
    )
    safe = bool(
        all(
            value <= scenario.privacy_thresholds[label] + 1e-9
            for label, value in enumerate(privacy_by_label)
        )
        and worst_utility <= scenario.utility_threshold + 1e-9
    )
    false_acceptance = bool(deployed and not safe)
    return {
        "seed": seed,
        "sample_size_per_stratum": n,
        "method": method,
        "deployed": deployed,
        "exact_safe": safe,
        "false_acceptance": false_acceptance,
        "exact_worst_privacy_advantage": float(worst_privacy),
        "exact_worst_conditional_error": float(worst_utility),
        "selection_criterion": float(selection.criterion),
        "release_channel": [
            [float(value) for value in row] for row in selection.channel
        ],
        "decoder": [int(value) for value in selection.decoder],
        "confidence_event": event,
        "failure_on_confidence_event": (
            None if event is None else bool(false_acceptance and event)
        ),
    }


def replay_table(
    payload: tuple[int, int, Scenario, float]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    seed, n, scenario, delta = payload
    rng = np.random.default_rng(seed)
    empirical = empirical_table(rng, scenario.population, n)
    radii = simultaneous_radii(
        n,
        label_count=empirical.shape[0],
        source_count=empirical.shape[1],
        fine_count=empirical.shape[2],
        delta=delta,
    )
    event = confidence_event(empirical, scenario.population, radii)

    mosaic = selection_from_mosaic_solution(
        optimize_invariant_channel(
            empirical,
            l1_radii=radii,
            common_channels_by_label=scenario.libraries,
            contaminations=scenario.contaminations,
            privacy_advantage_thresholds=scenario.privacy_thresholds,
            released_token_count=scenario.released_token_count,
        )
    )
    grid = table_region_grid_selection(empirical, radii, scenario)
    holm, holm_deployed, holm_p, holm_rejections = holm_ltt_selection(
        empirical, scenario, n=n, delta=delta
    )
    deterministic = deterministic_selection(empirical, radii, scenario)
    selections = (
        ("mosaic_continuum", mosaic, mosaic.criterion <= scenario.utility_threshold, event),
        ("table_region_grid", grid, grid.criterion <= scenario.utility_threshold, event),
        ("holm_ltt_grid", holm, holm_deployed, None),
        (
            "fare_style_deterministic",
            deterministic,
            deterministic.criterion <= scenario.utility_threshold,
            event,
        ),
    )
    rows = [
        exact_row(
            seed=seed,
            n=n,
            method=method,
            scenario=scenario,
            selection=selection,
            deployed=deployed,
            event=selection_event,
        )
        for method, selection, deployed, selection_event in selections
    ]
    diagnostic = {
        "seed": seed,
        "scenario": scenario.name,
        "sample_size_per_stratum": n,
        "holm_selected_p_value": float(holm_p),
        "holm_rejections": int(holm_rejections),
        "confidence_event": event,
    }
    return rows, diagnostic


def cp_interval(successes: int, trials: int) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(
        beta.ppf(0.025, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta.ppf(0.975, successes + 1, trials - successes)
    )
    return lower, upper


def aggregate_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["scenario"]),
                int(row["sample_size_per_stratum"]),
                str(row["method"]),
            )
        ].append(row)
    cells: list[dict[str, object]] = []
    for (scenario, n, method), subset in sorted(grouped.items()):
        count = len(subset)
        deployments = sum(bool(row["deployed"]) for row in subset)
        false_acceptances = sum(bool(row["false_acceptance"]) for row in subset)
        safe_deployments = sum(
            bool(row["deployed"]) and bool(row["exact_safe"]) for row in subset
        )
        cell: dict[str, object] = {
            "scenario": scenario,
            "sample_size_per_stratum": n,
            "method": method,
            "replicates": count,
            "deployments": deployments,
            "deployment_rate": deployments / count,
            "false_acceptances": false_acceptances,
            "false_acceptance_rate": false_acceptances / count,
            "safe_deployments": safe_deployments,
            "safe_deployment_rate": safe_deployments / count,
            "confidence_event_count": sum(
                row["confidence_event"] is True for row in subset
            ),
            "failures_on_confidence_event": sum(
                row["failure_on_confidence_event"] is True for row in subset
            ),
            "mean_exact_privacy_advantage": float(
                np.mean(
                    [float(row["exact_worst_privacy_advantage"]) for row in subset]
                )
            ),
            "mean_exact_worst_error": float(
                np.mean(
                    [float(row["exact_worst_conditional_error"]) for row in subset]
                )
            ),
            "mean_selection_criterion": float(
                np.mean([float(row["selection_criterion"]) for row in subset])
            ),
        }
        for count_field, prefix in (
            ("deployments", "deployment"),
            ("false_acceptances", "false_acceptance"),
            ("safe_deployments", "safe_deployment"),
        ):
            lower, upper = cp_interval(int(cell[count_field]), count)
            cell[f"{prefix}_cp95_lower"] = lower
            cell[f"{prefix}_cp95_upper"] = upper
        cells.append(cell)
    return cells


def assert_scalar_equal(name: str, observed: object, expected: object) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if observed != expected:
            raise AssertionError(f"{name}: expected {expected!r}, found {observed!r}")
        return
    if abs(float(observed) - float(expected)) > TOLERANCE:
        raise AssertionError(f"{name}: expected {expected!r}, found {observed!r}")


def assert_row_equal(observed: dict[str, Any], expected: dict[str, object]) -> None:
    if set(observed) != set(expected) | {"scenario"}:
        raise AssertionError("replicate row has unexpected fields")
    for field, value in expected.items():
        if field == "release_channel":
            np.testing.assert_allclose(
                np.asarray(observed[field], dtype=np.float64),
                np.asarray(value, dtype=np.float64),
                atol=TOLERANCE,
                rtol=0.0,
            )
        elif field == "decoder":
            if tuple(observed[field]) != tuple(value):
                raise AssertionError("decoder replay mismatch")
        else:
            assert_scalar_equal(field, observed[field], value)


def assert_cell_equal(observed: dict[str, Any], expected: dict[str, object]) -> None:
    if set(observed) != set(expected):
        raise AssertionError("aggregate cell has unexpected fields")
    for field, value in expected.items():
        assert_scalar_equal(field, observed[field], value)


def atomic_json_dump(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--audit-lock", type=Path, default=DEFAULT_AUDIT_LOCK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    prereg_hash = sha256(args.prereg)
    if args.prereg.with_suffix(args.prereg.suffix + ".sha256").read_text(
        encoding="utf-8"
    ).strip() != prereg_hash:
        raise AssertionError("baseline-extension preregistration sidecar mismatch")
    audit_lock_hash = sha256(args.audit_lock)
    if args.audit_lock.with_suffix(args.audit_lock.suffix + ".sha256").read_text(
        encoding="utf-8"
    ).strip() != audit_lock_hash:
        raise AssertionError("baseline-extension audit sidecar mismatch")
    prereg = load_json(args.prereg)
    audit_lock = load_json(args.audit_lock)
    report = load_json(args.report)
    if audit_lock["baseline_preregistration_sha256"] != prereg_hash:
        raise AssertionError("audit was locked against a different baseline study")
    for relative, expected_hash in audit_lock["code_sha256"].items():
        if sha256(REPOSITORY / relative) != expected_hash:
            raise AssertionError(f"audit code lock mismatch: {relative}")
    if report["preregistration_sha256"] != prereg_hash:
        raise AssertionError("baseline report uses the wrong preregistration")
    if report["code_sha256"] != prereg["code_sha256"]:
        raise AssertionError("baseline report code lock mismatch")
    if int(report["replicates_per_cell"]) != int(prereg["replicates_per_cell"]):
        raise AssertionError("baseline report uses the wrong replicate count")
    if float(report["familywise_delta"]) != float(prereg["familywise_delta"]):
        raise AssertionError("baseline report uses the wrong familywise level")

    replicate_count = int(prereg["replicates_per_cell"])
    seed_base = int(prereg["seed_base"])
    delta = float(prereg["familywise_delta"])
    scenarios = [scenario_from_config(config) for config in prereg["scenarios"]]
    payloads = [
        (
            seed_base + scenario_index * 10_000_000 + replicate,
            int(config["sample_size_per_stratum"]),
            scenario,
            delta,
        )
        for scenario_index, (config, scenario) in enumerate(
            zip(prereg["scenarios"], scenarios, strict=True)
        )
        for replicate in range(replicate_count)
    ]
    expected_keys = {
        (scenario.name, n, seed, method)
        for seed, n, scenario, _ in payloads
        for method in METHODS
    }
    observed_rows = {
        (
            str(row["scenario"]),
            int(row["sample_size_per_stratum"]),
            int(row["seed"]),
            str(row["method"]),
        ): row
        for row in report["replicate_results"]
    }
    if len(observed_rows) != len(report["replicate_results"]):
        raise AssertionError("duplicate baseline replicate rows")
    if set(observed_rows) != expected_keys:
        raise AssertionError("baseline replicate grid is incomplete or unexpected")
    observed_diagnostics = {
        (
            str(row["scenario"]),
            int(row["sample_size_per_stratum"]),
            int(row["seed"]),
        ): row
        for row in report["diagnostics"]
    }
    if len(observed_diagnostics) != len(payloads):
        raise AssertionError("baseline diagnostic grid is incomplete")

    replayed_rows: list[dict[str, object]] = []
    diagnostic_checks = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for (seed, n, scenario, _), (rows, diagnostic) in zip(
            payloads, executor.map(replay_table, payloads), strict=True
        ):
            for row in rows:
                row["scenario"] = scenario.name
                key = (scenario.name, n, seed, str(row["method"]))
                assert_row_equal(observed_rows[key], row)
                replayed_rows.append(row)
            observed_diagnostic = observed_diagnostics[(scenario.name, n, seed)]
            if set(observed_diagnostic) != set(diagnostic):
                raise AssertionError("diagnostic row has unexpected fields")
            for field, value in diagnostic.items():
                assert_scalar_equal(field, observed_diagnostic[field], value)
                diagnostic_checks += 1

    expected_cells = {
        (
            str(cell["scenario"]),
            int(cell["sample_size_per_stratum"]),
            str(cell["method"]),
        ): cell
        for cell in aggregate_rows(replayed_rows)
    }
    observed_cells = {
        (
            str(cell["scenario"]),
            int(cell["sample_size_per_stratum"]),
            str(cell["method"]),
        ): cell
        for cell in report["cells"]
    }
    if set(observed_cells) != set(expected_cells):
        raise AssertionError("aggregate baseline grid mismatch")
    for key, expected in expected_cells.items():
        assert_cell_equal(observed_cells[key], expected)

    maximum_false_acceptance_rate = max(
        float(cell["false_acceptance_rate"]) for cell in expected_cells.values()
    )
    confidence_event_failures = sum(
        int(cell["failures_on_confidence_event"])
        for cell in expected_cells.values()
    )
    pass_conditions = {
        "complete_rows": len(replayed_rows)
        == int(prereg["pass_conditions"]["complete_rows"]),
        "maximum_false_acceptance_rate": maximum_false_acceptance_rate
        <= float(prereg["pass_conditions"]["maximum_false_acceptance_rate"]),
        "confidence_event_failures": confidence_event_failures
        == int(prereg["pass_conditions"]["confidence_event_failures"]),
    }
    payload: dict[str, object] = {
        "name": "MOSAIC paired baseline deterministic replay audit v1",
        "status": "development_replay_not_independent_human_review",
        "baseline_preregistration_sha256": prereg_hash,
        "audit_lock_sha256": audit_lock_hash,
        "baseline_report_sha256": sha256(args.report),
        "tables_replayed": len(payloads),
        "replicate_rows_replayed": len(replayed_rows),
        "diagnostic_fields_checked": diagnostic_checks,
        "aggregate_cells_checked": len(expected_cells),
        "maximum_false_acceptance_rate": maximum_false_acceptance_rate,
        "confidence_event_failures": confidence_event_failures,
        "pass_conditions": pass_conditions,
        "pass": all(pass_conditions.values()),
    }
    atomic_json_dump(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
