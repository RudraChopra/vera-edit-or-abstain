"""Validate the exact discrete VERA certificate on a preregistered 18-cell grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREG = ROOT / "prereg_real.json"
DEFAULT_HASH = ROOT / "prereg_real.sha256"
DEFAULT_OUTPUT = ROOT / "artifacts" / "vera_exact_synthetic_report.json"


@dataclass(frozen=True)
class Cell:
    n: int
    delta: float
    replicates: int
    false_acceptances: int
    abstentions: int
    false_acceptance_rate: float
    false_acceptance_cp95_upper_simultaneous: float
    observed_abstention: float
    predicted_abstention: float
    prediction_band_lower: float
    prediction_band_upper: float
    coverage_pass: bool
    overlay_pass: bool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cp_upper(successes: np.ndarray | int, n: int, alpha: float) -> np.ndarray:
    successes = np.asarray(successes)
    output = np.ones(successes.shape, dtype=np.float64)
    mask = successes < n
    output[mask] = beta.ppf(1.0 - alpha, successes[mask] + 1, n - successes[mask])
    return output


def cp_lower(successes: np.ndarray | int, n: int, alpha: float) -> np.ndarray:
    successes = np.asarray(successes)
    output = np.zeros(successes.shape, dtype=np.float64)
    mask = successes > 0
    output[mask] = beta.ppf(alpha, successes[mask], n - successes[mask] + 1)
    return output


def robust_paired(positive: np.ndarray, negative: np.ndarray, gamma: float) -> np.ndarray:
    positive = np.minimum(positive, 1.0 - negative)
    zero = np.maximum(0.0, 1.0 - positive - negative)
    positive_mass = np.minimum(1.0, gamma * positive)
    remaining = 1.0 - positive_mass
    zero_mass = np.minimum(remaining, gamma * zero)
    return positive_mass - np.maximum(0.0, remaining - zero_mass)


def exact_target_pass_probability(
    n: int,
    p_positive: float,
    p_negative: float,
    *,
    gamma: float,
    threshold: float,
    alpha: float,
) -> float:
    positive_counts = np.arange(n + 1)
    positive_upper = cp_upper(positive_counts, n, alpha / 2.0)
    negative_lower = cp_lower(np.arange(n + 1), n, alpha / 2.0)
    conditional_negative = p_negative / max(1e-15, 1.0 - p_positive)
    pass_probability = 0.0
    positive_probability = binom.pmf(positive_counts, n, p_positive)
    for positive_count in range(n + 1):
        remaining = n - positive_count
        low, high = 0, remaining + 1
        while low < high:
            midpoint = (low + high) // 2
            ucb = float(
                robust_paired(
                    np.asarray([positive_upper[positive_count]]),
                    np.asarray([negative_lower[midpoint]]),
                    gamma,
                )[0]
            )
            if ucb <= threshold:
                high = midpoint
            else:
                low = midpoint + 1
        if low <= remaining:
            minimum_negative = low
            conditional_pass = float(
                binom.sf(minimum_negative - 1, remaining, conditional_negative)
            )
            pass_probability += float(positive_probability[positive_count]) * conditional_pass
    return min(1.0, max(0.0, pass_probability))


def exact_leakage_pass_probability(
    n: int,
    probability: float,
    *,
    gamma: float,
    threshold: float,
    alpha: float,
) -> float:
    counts = np.arange(n + 1)
    passing = np.flatnonzero(np.minimum(1.0, gamma * cp_upper(counts, n, alpha)) <= threshold)
    return 0.0 if passing.size == 0 else float(binom.cdf(int(passing[-1]), n, probability))


def predicted_abstention(config: dict[str, object], n: int, delta: float) -> float:
    positives = [float(value) for value in config["target_positive_probabilities"]]
    negatives = [float(value) for value in config["target_negative_probabilities"]]
    leakage = [float(value) for value in config["leakage_probabilities"]]
    offsets = [float(value) for value in config["attacker_probability_offsets"]]
    gamma = float(config["gamma"])
    target_threshold = float(config["target_threshold"])
    leakage_threshold = float(config["leakage_threshold"])
    family_size = len(positives) * (1 + len(offsets))
    alpha = delta / family_size
    candidate_pass: list[float] = []
    for p_positive, p_negative, p_leakage in zip(positives, negatives, leakage):
        target_pass = exact_target_pass_probability(
            n,
            p_positive,
            p_negative,
            gamma=gamma,
            threshold=target_threshold,
            alpha=alpha,
        )
        leakage_pass = np.prod([
            exact_leakage_pass_probability(
                n,
                float(np.clip(p_leakage + offset, 0.0, 1.0)),
                gamma=gamma,
                threshold=leakage_threshold,
                alpha=alpha,
            )
            for offset in offsets
        ])
        candidate_pass.append(target_pass * float(leakage_pass))
    return float(np.prod([1.0 - probability for probability in candidate_pass]))


def run_cell(payload: tuple[dict[str, object], int, float, int]) -> Cell:
    config, n, delta, seed = payload
    rng = np.random.default_rng(seed)
    replicates = int(config["replicates"])
    positives = np.asarray(config["target_positive_probabilities"], dtype=float)
    negatives = np.asarray(config["target_negative_probabilities"], dtype=float)
    leakage = np.asarray(config["leakage_probabilities"], dtype=float)
    offsets = np.asarray(config["attacker_probability_offsets"], dtype=float)
    candidate_count = len(positives)
    attacker_count = len(offsets)
    gamma = float(config["gamma"])
    target_threshold = float(config["target_threshold"])
    leakage_threshold = float(config["leakage_threshold"])
    family_size = candidate_count * (1 + attacker_count)
    alpha = delta / family_size

    target_counts = np.empty((replicates, candidate_count, 3), dtype=np.int64)
    for candidate in range(candidate_count):
        target_counts[:, candidate] = rng.multinomial(
            n,
            [negatives[candidate], 1.0 - positives[candidate] - negatives[candidate], positives[candidate]],
            size=replicates,
        )
    positive_upper = cp_upper(target_counts[:, :, 2], n, alpha / 2.0)
    negative_lower = cp_lower(target_counts[:, :, 0], n, alpha / 2.0)
    target_ucb = robust_paired(positive_upper, negative_lower, gamma)
    attacker_probabilities = np.clip(leakage[:, None] + offsets[None, :], 0.0, 1.0)
    leakage_counts = rng.binomial(
        n,
        attacker_probabilities,
        size=(replicates, candidate_count, attacker_count),
    )
    leakage_ucb = np.minimum(1.0, gamma * cp_upper(leakage_counts, n, alpha))
    accepted = (target_ucb <= target_threshold) & np.all(
        leakage_ucb <= leakage_threshold, axis=2
    )
    any_accepted = np.any(accepted, axis=1)
    selected = candidate_count - 1 - np.argmax(accepted[:, ::-1], axis=1)
    true_target = robust_paired(positives, negatives, gamma)
    true_leakage = np.max(np.minimum(1.0, gamma * attacker_probabilities), axis=1)
    unsafe = (true_target > target_threshold) | (true_leakage > leakage_threshold)
    false_acceptances = int(np.count_nonzero(any_accepted & unsafe[selected]))
    abstentions = int(np.count_nonzero(~any_accepted))

    design_cells = len(config["validation_sizes"]) * len(config["delta_levels"])
    family_alpha = 0.05 / design_cells
    false_upper = float(cp_upper(false_acceptances, replicates, family_alpha / 2.0))
    predicted = predicted_abstention(config, n, delta)
    band_lower = float(binom.ppf(family_alpha / 2.0, replicates, predicted) / replicates)
    band_upper = float(binom.ppf(1.0 - family_alpha / 2.0, replicates, predicted) / replicates)
    observed = abstentions / replicates
    return Cell(
        n=n,
        delta=delta,
        replicates=replicates,
        false_acceptances=false_acceptances,
        abstentions=abstentions,
        false_acceptance_rate=false_acceptances / replicates,
        false_acceptance_cp95_upper_simultaneous=false_upper,
        observed_abstention=observed,
        predicted_abstention=predicted,
        prediction_band_lower=band_lower,
        prediction_band_upper=band_upper,
        coverage_pass=false_upper <= delta,
        overlay_pass=band_lower <= observed <= band_upper,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--hash-file", type=Path, default=DEFAULT_HASH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    prereg_hash = sha256(args.prereg)
    expected_hash = args.hash_file.read_text(encoding="utf-8").split()[0]
    if prereg_hash != expected_hash or prereg.get("status") != "locked_before_claim_grade_runs":
        raise RuntimeError("exact synthetic study requires the committed real-study lock")
    relative = args.prereg.resolve().relative_to(ROOT.parent.resolve()).as_posix()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT.parent,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != prereg_hash:
        raise RuntimeError("preregistration is not committed at HEAD")
    config = prereg["exact_synthetic_study"]
    sizes = [int(value) for value in config["validation_sizes"]]
    deltas = [float(value) for value in config["delta_levels"]]
    payloads = [
        (config, n, delta, int(config["seed"]) + 1009 * index)
        for index, (n, delta) in enumerate((n, delta) for n in sizes for delta in deltas)
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        cells = list(executor.map(run_cell, payloads))
    report = {
        "name": "VERA exact-discrete synthetic validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_grade": True,
        "prereg_sha256": prereg_hash,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "config": config,
        "cells": [asdict(cell) for cell in cells],
        "cell_count": len(cells),
        "coverage_pass": all(cell.coverage_pass for cell in cells),
        "overlay_pass": all(cell.overlay_pass for cell in cells),
        "all_cells_pass": all(cell.coverage_pass and cell.overlay_pass for cell in cells),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "cell_count": len(cells),
        "coverage_pass": report["coverage_pass"],
        "overlay_pass": report["overlay_pass"],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
