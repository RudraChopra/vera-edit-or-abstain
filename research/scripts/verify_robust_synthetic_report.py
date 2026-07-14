"""Independently verify the claim-grade robust-certificate simulation receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "artifacts" / "vera_robust_synthetic_report.json"
DEFAULT_PREREG = ROOT / "prereg.json"
DEFAULT_HASH = ROOT / "prereg.sha256"
DEFAULT_OUTPUT = ROOT / "artifacts" / "vera_robust_synthetic_verification.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cp_interval(successes: int, trials: int, alpha: float) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2.0, successes, trials - successes + 1)
    )
    upper = 1.0 if successes == trials else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes)
    )
    return lower, upper


def dkw_radius(n: int, failure_probability: float) -> float:
    return float(np.sqrt(np.log(2.0 / failure_probability) / (2.0 * n)))


def predicted_abstention(config: dict[str, object], n: int, delta: float) -> float:
    candidate_count = int(config["candidate_count"])
    offsets = np.asarray(config["attacker_probability_offsets"], dtype=float)
    target_p = np.asarray(config["target_harm_probabilities"], dtype=float)
    leakage_p = np.asarray(config["leakage_probabilities"], dtype=float)
    gamma = float(config["gamma"])
    risk_count = candidate_count * (1 + offsets.size)
    epsilon = dkw_radius(n, delta / risk_count)

    target_limit = float(config["target_threshold"]) / gamma - 2.0 * epsilon
    leakage_limit = float(config["leakage_threshold"]) / gamma - epsilon
    target_cutoff = int(np.floor(n * target_limit + 1e-12))
    leakage_cutoff = int(np.floor(n * leakage_limit + 1e-12))
    target_pass = np.zeros(candidate_count) if target_cutoff < 0 else binom.cdf(
        min(n, target_cutoff), n, target_p
    )
    if leakage_cutoff < 0:
        leakage_pass = np.zeros(candidate_count)
    else:
        attacker_p = np.clip(leakage_p[:, None] + offsets[None, :], 0.0, 1.0)
        leakage_pass = np.prod(
            binom.cdf(min(n, leakage_cutoff), n, attacker_p), axis=1
        )
    return float(np.prod(1.0 - target_pass * leakage_pass))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    prereg = json.loads(args.prereg.read_text(encoding="utf-8"))
    config = prereg["synthetic_study"]
    expected_hash = DEFAULT_HASH.read_text(encoding="utf-8").split()[0]
    observed_hash = sha256(args.prereg)
    family_size = len(config["validation_sizes"]) * len(config["delta_levels"])
    family_alpha = 0.05 / family_size

    failures: list[str] = []
    if report.get("claim_grade") is not True:
        failures.append("report is not claim-grade")
    if expected_hash != observed_hash or report.get("prereg_sha256") != observed_hash:
        failures.append("preregistration hash mismatch")
    expected_cells = {
        (int(n), float(delta))
        for n in config["validation_sizes"]
        for delta in config["delta_levels"]
    }
    observed_cells = {(int(cell["n"]), float(cell["delta"])) for cell in report["cells"]}
    if observed_cells != expected_cells or len(report["cells"]) != family_size:
        failures.append("design cells do not match preregistration")

    commit = str(report.get("git_commit", ""))
    commit_check = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT.parent,
        capture_output=True,
    )
    if commit_check.returncode != 0:
        failures.append("recorded Git commit does not exist")

    verified_cells: list[dict[str, object]] = []
    for cell in report["cells"]:
        n = int(cell["n"])
        delta = float(cell["delta"])
        replicates = int(cell["replicates"])
        false_acceptances = int(cell["false_acceptances"])
        abstentions = int(cell["abstentions"])
        lower, upper = cp_interval(false_acceptances, replicates, family_alpha)
        predicted = predicted_abstention(config, n, delta)
        band_lower = float(
            binom.ppf(family_alpha / 2.0, replicates, predicted) / replicates
        )
        band_upper = float(
            binom.ppf(1.0 - family_alpha / 2.0, replicates, predicted) / replicates
        )
        observed_abstention = abstentions / replicates
        checks = {
            "cp_matches": bool(np.isclose(lower, cell["false_acceptance_cp_lower"]) and np.isclose(
                upper, cell["false_acceptance_cp_upper"]
            )),
            "prediction_matches": bool(
                np.isclose(predicted, cell["predicted_abstention"])
            ),
            "band_matches": bool(np.isclose(band_lower, cell["predicted_abstention_band_lower"]) and np.isclose(
                band_upper, cell["predicted_abstention_band_upper"]
            )),
            "coverage_pass": bool(upper <= delta),
            "overlay_pass": bool(
                band_lower <= observed_abstention <= band_upper
            ),
        }
        if not all(checks.values()):
            failures.append(f"cell n={n}, delta={delta} failed: {checks}")
        verified_cells.append({"n": n, "delta": delta, **checks})

    verification = {
        "verified": not failures,
        "report": str(args.report),
        "prereg_sha256": observed_hash,
        "git_commit": commit,
        "cell_count": len(verified_cells),
        "failures": failures,
        "cells": verified_cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"verified": not failures, "failures": failures}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
