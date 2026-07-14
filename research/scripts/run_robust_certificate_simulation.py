"""Run the preregistered synthetic coverage and abstention experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

from vera_robust_certificate import dkw_epsilon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "prereg.json"
DEFAULT_JSON = ROOT / "artifacts" / "vera_robust_synthetic_report.json"
DEFAULT_CSV = ROOT / "artifacts" / "vera_robust_synthetic_cells.csv"
DEFAULT_HASH = ROOT / "prereg.sha256"


@dataclass(frozen=True)
class Cell:
    n: int
    delta: float
    replicates: int
    false_acceptances: int
    abstentions: int
    empirical_false_acceptance: float
    false_acceptance_cp_lower: float
    false_acceptance_cp_upper: float
    empirical_abstention: float
    predicted_abstention: float
    predicted_abstention_band_lower: float
    predicted_abstention_band_upper: float
    coverage_pass: bool
    overlay_pass: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def enforce_preregistration_lock(prereg_path: Path) -> tuple[str, str]:
    prereg_path = prereg_path.resolve()
    repository = ROOT.parent.resolve()
    try:
        relative_path = prereg_path.relative_to(repository)
    except ValueError as error:
        raise RuntimeError("claim-grade preregistration must live in the repository") from error

    observed_hash = sha256(prereg_path)
    if not DEFAULT_HASH.exists():
        raise RuntimeError(f"missing preregistration hash sidecar: {DEFAULT_HASH}")
    expected_hash = DEFAULT_HASH.read_text(encoding="utf-8").split()[0]
    if observed_hash != expected_hash:
        raise RuntimeError("preregistration hash does not match prereg.sha256")

    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative_path.as_posix()}"],
        cwd=repository,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != observed_hash:
        raise RuntimeError("claim-grade preregistration is not committed at HEAD")
    return observed_hash, git_head()


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial count")
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def true_robust_bernoulli_risk(probability: float, gamma: float) -> float:
    return min(1.0, gamma * probability)


def predicted_abstention_probability(config: dict[str, object], n: int, delta: float) -> float:
    m = int(config["candidate_count"])
    attacker_offsets = [float(value) for value in config["attacker_probability_offsets"]]
    gamma = float(config["gamma"])
    target_threshold = float(config["target_threshold"])
    leakage_threshold = float(config["leakage_threshold"])
    target_probabilities = [float(value) for value in config["target_harm_probabilities"]]
    leakage_probabilities = [float(value) for value in config["leakage_probabilities"]]
    risk_count = m * (1 + len(attacker_offsets))
    epsilon = dkw_epsilon(n, delta / risk_count)

    pass_probabilities: list[float] = []
    target_limit = target_threshold / gamma - 2.0 * epsilon
    target_cutoff = min(n, int(np.floor(n * target_limit + 1e-12)))
    for target_p, leakage_p in zip(target_probabilities, leakage_probabilities):
        if target_cutoff < 0:
            target_pass = 0.0
        else:
            target_pass = float(binom.cdf(target_cutoff, n, target_p))
        leakage_pass = 1.0
        leakage_limit = leakage_threshold / gamma - epsilon
        leakage_cutoff = min(n, int(np.floor(n * leakage_limit + 1e-12)))
        for offset in attacker_offsets:
            attacker_p = float(np.clip(leakage_p + offset, 0.0, 1.0))
            if leakage_cutoff < 0:
                leakage_pass = 0.0
                break
            leakage_pass *= float(binom.cdf(leakage_cutoff, n, attacker_p))
        pass_probabilities.append(target_pass * leakage_pass)

    return float(np.prod([1.0 - value for value in pass_probabilities]))


def run_cell(payload: tuple[dict[str, object], int, float, int]) -> Cell:
    config, n, delta, cell_seed = payload
    rng = np.random.default_rng(cell_seed)
    replicates = int(config["replicates"])
    candidate_count = int(config["candidate_count"])
    attacker_offsets = [float(value) for value in config["attacker_probability_offsets"]]
    target_probabilities = np.asarray(
        config["target_harm_probabilities"], dtype=np.float64
    )
    leakage_probabilities = np.asarray(
        config["leakage_probabilities"], dtype=np.float64
    )
    gamma = float(config["gamma"])
    target_threshold = float(config["target_threshold"])
    leakage_threshold = float(config["leakage_threshold"])

    attacker_probabilities = np.clip(
        leakage_probabilities[:, None] + np.asarray(attacker_offsets)[None, :],
        0.0,
        1.0,
    )
    target_counts = rng.binomial(
        n, target_probabilities, size=(replicates, candidate_count)
    )
    leakage_counts = rng.binomial(
        n,
        attacker_probabilities,
        size=(replicates, candidate_count, len(attacker_offsets)),
    )

    risk_count = candidate_count * (1 + len(attacker_offsets))
    epsilon = dkw_epsilon(n, delta / risk_count)
    target_ucb = np.minimum(1.0, gamma * target_counts / n) + 2.0 * gamma * epsilon
    target_ucb = np.minimum(1.0, target_ucb)
    leakage_ucb = np.minimum(1.0, gamma * leakage_counts / n) + gamma * epsilon
    leakage_ucb = np.minimum(1.0, leakage_ucb)
    accepted = (target_ucb <= target_threshold) & np.all(
        leakage_ucb <= leakage_threshold, axis=2
    )
    any_accepted = np.any(accepted, axis=1)
    abstentions = int(np.count_nonzero(~any_accepted))

    # The protocol selects the strongest accepted edit, i.e. the last True.
    selected_indices = candidate_count - 1 - np.argmax(accepted[:, ::-1], axis=1)
    true_target = np.minimum(1.0, gamma * target_probabilities)
    true_leakage = np.max(
        np.minimum(1.0, gamma * attacker_probabilities), axis=1
    )
    unsafe = (true_target > target_threshold) | (
        true_leakage > leakage_threshold
    )
    false_acceptances = int(
        np.count_nonzero(any_accepted & unsafe[selected_indices])
    )

    false_rate = false_acceptances / replicates
    abstention_rate = abstentions / replicates
    family_size = len(config["validation_sizes"]) * len(config["delta_levels"])
    family_alpha = 0.05 / family_size
    false_lower, false_upper = clopper_pearson(
        false_acceptances, replicates, alpha=family_alpha
    )
    predicted = predicted_abstention_probability(config, n, delta)
    band_lower = float(
        binom.ppf(family_alpha / 2.0, replicates, predicted) / replicates
    )
    band_upper = float(
        binom.ppf(1.0 - family_alpha / 2.0, replicates, predicted) / replicates
    )
    return Cell(
        n=n,
        delta=delta,
        replicates=replicates,
        false_acceptances=false_acceptances,
        abstentions=abstentions,
        empirical_false_acceptance=false_rate,
        false_acceptance_cp_lower=false_lower,
        false_acceptance_cp_upper=false_upper,
        empirical_abstention=abstention_rate,
        predicted_abstention=predicted,
        predicted_abstention_band_lower=band_lower,
        predicted_abstention_band_upper=band_upper,
        coverage_pass=false_upper <= delta,
        overlay_pass=band_lower <= abstention_rate <= band_upper,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pilot", action="store_true", help="Run 100 replicates per cell and mark output non-claim-grade.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    config = dict(prereg["synthetic_study"])
    claim_grade = not args.pilot
    if args.pilot:
        config["replicates"] = 100
        prereg_hash = sha256(args.prereg)
        commit = git_head()
    else:
        if prereg.get("status") != "locked_before_claim_grade_run":
            raise RuntimeError("claim-grade run requires a locked preregistration")
        prereg_hash, commit = enforce_preregistration_lock(args.prereg)
    sizes = [int(value) for value in config["validation_sizes"]]
    deltas = [float(value) for value in config["delta_levels"]]
    seed = int(config["seed"])
    payloads = [
        (config, n, delta, seed + 1009 * index)
        for index, (n, delta) in enumerate((n, d) for n in sizes for d in deltas)
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        cells = list(executor.map(run_cell, payloads))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(asdict(cells[0]))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(cell) for cell in cells)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_grade": claim_grade,
        "prereg_path": str(args.prereg),
        "prereg_sha256": prereg_hash,
        "git_commit": commit,
        "interval_family": "simultaneous 95% via Bonferroni over all design cells",
        "config": config,
        "cells": [asdict(cell) for cell in cells],
        "coverage_pass": all(cell.coverage_pass for cell in cells),
        "overlay_pass": all(cell.overlay_pass for cell in cells),
        "all_cells_pass": all(cell.coverage_pass and cell.overlay_pass for cell in cells),
    }
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "claim_grade": claim_grade,
        "cell_count": len(cells),
        "coverage_pass": report["coverage_pass"],
        "overlay_pass": report["overlay_pass"],
        "all_cells_pass": report["all_cells_pass"],
        "output_json": str(args.output_json),
        "output_csv": str(args.output_csv),
    }, indent=2))


if __name__ == "__main__":
    main()
