"""Enumerate tiny exact-discrete cases and compare mathematics with code."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom, multinomial

from vera_robust_certificate import (
    certify_balanced_iut_fixed_profile,
    certify_balanced_shift_envelope,
    certify_balanced_shift_radius,
    certify_discrete_iut_fixed_profile,
    certify_discrete_shift_radius,
    exact_balanced_leakage_certificate,
    exact_discrete_risk_certificate,
)


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
DEFAULT_OUTPUT = ROOT / "artifacts" / "vera_exact_small_case_audit.json"


@dataclass
class Scenario:
    name: str
    checks: int = 0
    maximum_absolute_error: float = 0.0
    failures: list[str] = field(default_factory=list)
    maximum_exact_false_acceptance_probability: float | None = None

    def compare(self, observed: float, expected: float, label: str, tolerance: float = 1e-10) -> None:
        self.checks += 1
        error = abs(float(observed) - float(expected))
        self.maximum_absolute_error = max(self.maximum_absolute_error, error)
        if error > tolerance:
            self.failures.append(f"{label}: observed={observed}, expected={expected}")

    def require(self, condition: bool, label: str) -> None:
        self.checks += 1
        if not condition:
            self.failures.append(label)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": not self.failures,
            "checks": self.checks,
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_exact_false_acceptance_probability": (
                self.maximum_exact_false_acceptance_probability
            ),
            "failures": self.failures,
        }


def cp_upper(successes: int, n: int, alpha: float) -> float:
    return 1.0 if successes == n else float(beta.ppf(1.0 - alpha, successes + 1, n - successes))


def cp_lower(successes: int, n: int, alpha: float) -> float:
    return 0.0 if successes == 0 else float(beta.ppf(alpha, successes, n - successes + 1))


def paired_risk(positive: float, negative: float, gamma: float) -> float:
    bounded_positive = min(positive, 1.0 - negative)
    zero = max(0.0, 1.0 - bounded_positive - negative)
    positive_mass = min(1.0, gamma * bounded_positive)
    remaining = 1.0 - positive_mass
    zero_mass = min(remaining, gamma * zero)
    return positive_mass - max(0.0, remaining - zero_mass)


def bernoulli_leakage_scenario() -> Scenario:
    scenario = Scenario("bernoulli_leakage")
    n, alpha, threshold = 4, 0.1, 0.6
    for gamma in (1.0, 1.25, 2.0):
        passing_counts = []
        for successes in range(n + 1):
            values = np.asarray([1] * successes + [0] * (n - successes))
            certificate = exact_discrete_risk_certificate(
                "leakage",
                values,
                gamma=gamma,
                failure_probability=alpha,
                support=(0, 1),
            )
            expected = min(1.0, gamma * cp_upper(successes, n, alpha))
            scenario.compare(certificate.upper_confidence_bound, expected, f"gamma={gamma}, k={successes}")
            if expected <= threshold:
                passing_counts.append(successes)
        unsafe_probabilities = [p for p in np.linspace(0.05, 0.95, 19) if min(1.0, gamma * p) > threshold]
        for probability in unsafe_probabilities:
            false_acceptance = sum(
                float(binom.pmf(successes, n, probability))
                for successes in passing_counts
            )
            current = scenario.maximum_exact_false_acceptance_probability or 0.0
            scenario.maximum_exact_false_acceptance_probability = max(current, false_acceptance)
            scenario.require(false_acceptance <= alpha + 1e-12, f"false acceptance exceeds alpha at gamma={gamma}, p={probability}")
    return scenario


def paired_harm_scenario() -> Scenario:
    scenario = Scenario("paired_harm_minus_one_zero_one")
    n, alpha = 4, 0.1
    for gamma in (1.0, 1.25, 2.0):
        for positive in range(n + 1):
            for negative in range(n - positive + 1):
                values = np.asarray([1] * positive + [-1] * negative + [0] * (n - positive - negative))
                certificate = exact_discrete_risk_certificate(
                    "target",
                    values,
                    gamma=gamma,
                    failure_probability=alpha,
                    support=(-1, 0, 1),
                )
                positive_upper = cp_upper(positive, n, alpha / 2.0)
                negative_lower = cp_lower(negative, n, alpha / 2.0)
                expected = paired_risk(
                    min(positive_upper, 1.0 - negative_lower),
                    negative_lower,
                    gamma,
                )
                scenario.compare(
                    certificate.upper_confidence_bound,
                    expected,
                    f"gamma={gamma}, positive={positive}, negative={negative}",
                )
    threshold, gamma = 0.1, 1.25
    for p_negative, p_positive in ((0.05, 0.20), (0.10, 0.30)):
        true_risk = paired_risk(p_positive, p_negative, gamma)
        scenario.require(true_risk > threshold, "chosen paired law is not unsafe")
        false_acceptance = 0.0
        for positive in range(n + 1):
            for negative in range(n - positive + 1):
                values = np.asarray([1] * positive + [-1] * negative + [0] * (n - positive - negative))
                certificate = exact_discrete_risk_certificate(
                    "target",
                    values,
                    gamma=gamma,
                    failure_probability=alpha,
                    support=(-1, 0, 1),
                )
                if certificate.upper_confidence_bound <= threshold:
                    false_acceptance += float(
                        multinomial.pmf(
                            [negative, n - positive - negative, positive],
                            n=n,
                            p=[p_negative, 1.0 - p_negative - p_positive, p_positive],
                        )
                    )
        current = scenario.maximum_exact_false_acceptance_probability or 0.0
        scenario.maximum_exact_false_acceptance_probability = max(current, false_acceptance)
        scenario.require(false_acceptance <= alpha + 1e-12, "paired false acceptance exceeds alpha")
    return scenario


def iut_scenario(candidate_count: int, name: str) -> Scenario:
    scenario = Scenario(name)
    delta = 0.12
    result = certify_discrete_iut_fixed_profile(
        {
            "target": np.asarray([-1] * 2 + [0] * 38),
            "leakage": np.asarray([0] * 38 + [1] * 2),
        },
        gamma=1.1,
        delta=delta,
        candidate_count=candidate_count,
        supports={"target": (-1, 0, 1), "leakage": (0, 1)},
        thresholds={"target": 0.2, "leakage": 0.4},
    )
    scenario.compare(result.candidate_failure_probability, delta / candidate_count, "candidate alpha")
    expected = all(
        certificate.upper_confidence_bound
        <= (0.2 if certificate.key == "target" else 0.4)
        for certificate in result.certificates
    )
    scenario.require((result.decision == "EDIT") == expected, "IUT decision differs from component intersection")
    return scenario


def one_environment_scenario() -> Scenario:
    scenario = Scenario("one_environment")
    source = np.asarray([0] * 100 + [1] * 100)
    target = {"target::environment=0": np.zeros(200, dtype=int)}
    leakage = {"linear": np.asarray([0] * 180 + [1] * 20)}
    envelope = certify_balanced_shift_envelope(
        target,
        leakage,
        source,
        delta=0.05,
        family_size=2,
        target_threshold=0.2,
        leakage_threshold=0.4,
        registered_target_environments=[0],
        gamma_cap=3.0,
    )
    radius = certify_balanced_shift_radius(
        target,
        leakage,
        source,
        delta=0.05,
        family_size=2,
        target_threshold=0.2,
        leakage_threshold=0.4,
        gamma_cap=3.0,
    )
    scenario.compare(envelope.observed_common_radius, radius.certified_radius, "one-environment radius", tolerance=1e-5)
    scenario.require(not envelope.unsupported_target_environments, "observed environment marked unsupported")
    return scenario


def multiple_environment_scenario() -> Scenario:
    scenario = Scenario("multiple_environments")
    source = np.asarray([0] * 100 + [1] * 100)
    target = {
        "target::environment=0": np.zeros(200, dtype=int),
        "target::environment=1": np.asarray([0] * 195 + [1] * 5),
    }
    leakage = {"linear": np.asarray([0] * 180 + [1] * 20)}
    envelope = certify_balanced_shift_envelope(
        target,
        leakage,
        source,
        delta=0.05,
        family_size=3,
        target_threshold=0.2,
        leakage_threshold=0.4,
        registered_target_environments=[0, 1],
        gamma_cap=3.0,
    )
    radius = certify_balanced_shift_radius(
        target,
        leakage,
        source,
        delta=0.05,
        family_size=3,
        target_threshold=0.2,
        leakage_threshold=0.4,
        gamma_cap=3.0,
    )
    scenario.compare(envelope.observed_common_radius, radius.certified_radius, "multi-environment radius", tolerance=1e-5)
    scenario.require(set(envelope.target_environment_radii) == {"0", "1"}, "environment coordinates differ")
    return scenario


def missing_source_scenario() -> Scenario:
    scenario = Scenario("missing_source_class")
    try:
        exact_balanced_leakage_certificate(
            "missing",
            np.zeros(10),
            np.zeros(10, dtype=int),
            gamma=1.0,
            failure_probability=0.05,
        )
    except ValueError:
        scenario.require(True, "missing source class rejected")
    else:
        scenario.require(False, "missing source class was accepted")
    return scenario


def missing_environment_scenario() -> Scenario:
    scenario = Scenario("missing_deployment_environment")
    source = np.asarray([0] * 50 + [1] * 50)
    envelope = certify_balanced_shift_envelope(
        {"target::environment=0": np.zeros(100, dtype=int)},
        {"linear": np.zeros(100, dtype=int)},
        source,
        delta=0.05,
        family_size=4,
        target_threshold=0.2,
        leakage_threshold=0.4,
        registered_target_environments=[0, 1],
        gamma_cap=3.0,
    )
    scenario.require(envelope.target_environment_radii["1"] == 0.0, "missing environment radius is nonzero")
    scenario.require(envelope.deployment_common_radius == 0.0, "missing environment did not force zero deployment radius")
    scenario.require(envelope.decision == "ABSTAIN", "missing environment did not abstain")
    return scenario


def identity_scenario() -> Scenario:
    scenario = Scenario("identity_included_as_candidate")
    delta = 0.1
    result = certify_balanced_iut_fixed_profile(
        {"target::environment=0": np.zeros(100, dtype=int)},
        {"linear": np.zeros(100, dtype=int)},
        np.asarray([0] * 50 + [1] * 50),
        gamma=1.0,
        delta=delta,
        candidate_count=2,
        target_threshold=0.2,
        leakage_threshold=0.4,
    )
    scenario.require(result.candidate_count == 2, "identity was not counted in candidate family")
    scenario.compare(result.candidate_failure_probability, delta / 2.0, "identity-adjusted alpha")
    return scenario


def gamma_and_radius_scenario() -> Scenario:
    scenario = Scenario("several_gamma_values_and_reported_radius")
    values = np.asarray([0] * 18 + [1] * 2)
    alpha, threshold, gamma_cap = 0.05, 0.6, 4.0
    upper_probability = cp_upper(2, len(values), alpha)
    observed_bounds = []
    for gamma in (1.0, 1.1, 1.25, 1.5, 2.0, 4.0):
        certificate = exact_discrete_risk_certificate(
            "leakage",
            values,
            gamma=gamma,
            failure_probability=alpha,
            support=(0, 1),
        )
        expected = min(1.0, gamma * upper_probability)
        scenario.compare(certificate.upper_confidence_bound, expected, f"gamma={gamma}")
        observed_bounds.append(certificate.upper_confidence_bound)
    scenario.require(all(left <= right for left, right in zip(observed_bounds, observed_bounds[1:])), "upper curve is not monotone")
    radius = certify_discrete_shift_radius(
        {"leakage": values},
        delta=alpha,
        supports={"leakage": (0, 1)},
        thresholds={"leakage": threshold},
        family_size=1,
        gamma_cap=gamma_cap,
    )
    if upper_probability > threshold:
        expected_radius = 0.0
    elif gamma_cap * upper_probability <= threshold:
        expected_radius = gamma_cap
    else:
        lower, upper = 1.0, gamma_cap
        while upper - lower > 1e-4:
            midpoint = (lower + upper) / 2.0
            if midpoint * upper_probability <= threshold:
                lower = midpoint
            else:
                upper = midpoint
        expected_radius = lower
    scenario.compare(radius.certified_radius, expected_radius, "inverted Bernoulli radius")
    return scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = [
        bernoulli_leakage_scenario(),
        paired_harm_scenario(),
        iut_scenario(1, "one_candidate"),
        iut_scenario(3, "multiple_candidates"),
        one_environment_scenario(),
        multiple_environment_scenario(),
        missing_source_scenario(),
        missing_environment_scenario(),
        identity_scenario(),
        gamma_and_radius_scenario(),
    ]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "name": "VERA exact tiny-case enumeration audit",
        "passed": all(not scenario.failures for scenario in scenarios),
        "git_commit": head,
        "scenario_count": len(scenarios),
        "total_checks": sum(scenario.checks for scenario in scenarios),
        "exact_agreement_every_case": all(not scenario.failures for scenario in scenarios),
        "scenarios": [scenario.to_dict() for scenario in scenarios],
        "formal_proof_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
