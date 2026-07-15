"""Finite-sample certificates for paired representation-edit risks.

The certificate is distributionally robust over all deployment distributions
whose density ratio with respect to validation is bounded by ``gamma``.  Every
attacker and candidate must be fixed independently of the certification fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import log, sqrt
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RiskCertificate:
    key: str
    n: int
    gamma: float
    lower: float
    upper: float
    empirical_robust_risk: float
    dkw_epsilon: float
    simultaneous_failure_probability: float
    upper_confidence_bound: float
    method: str = "dkw_cvar"
    confidence_details: Mapping[str, float | int | str] | None = None

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class EditDecision:
    decision: str
    selected_edit: str | None
    accepted_edits: tuple[str, ...]
    target_threshold: float
    leakage_threshold: float
    delta: float
    gamma: float
    certificates: tuple[RiskCertificate, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["certificates"] = [cert.to_dict() for cert in self.certificates]
        return payload


@dataclass(frozen=True)
class ShiftRadiusCertificate:
    decision: str
    certified_radius: float
    gamma_cap: float
    right_censored: bool
    delta: float
    limiting_contracts: tuple[str, ...]
    certificates_at_radius: tuple[RiskCertificate, ...]
    method: str = "dkw_cvar"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["certificates_at_radius"] = [
            cert.to_dict() for cert in self.certificates_at_radius
        ]
        return payload


@dataclass(frozen=True)
class BalancedShiftEnvelopeCertificate:
    decision: str
    target_environment_radii: Mapping[str, float]
    source_class_radii: Mapping[str, float]
    observed_common_radius: float
    deployment_common_radius: float
    unsupported_target_environments: tuple[str, ...]
    gamma_cap: float
    delta: float
    family_size: int
    right_censored_coordinates: tuple[str, ...]
    simultaneous_curve_parameters: Mapping[str, Mapping[str, float | int | str]]
    method: str = "exact_balanced_support_aware_envelope"

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "target_environment_radii": dict(self.target_environment_radii),
            "source_class_radii": dict(self.source_class_radii),
            "observed_common_radius": self.observed_common_radius,
            "deployment_common_radius": self.deployment_common_radius,
            "unsupported_target_environments": list(
                self.unsupported_target_environments
            ),
            "gamma_cap": self.gamma_cap,
            "delta": self.delta,
            "family_size": self.family_size,
            "right_censored_coordinates": list(self.right_censored_coordinates),
            "simultaneous_curve_parameters": {
                key: dict(value)
                for key, value in self.simultaneous_curve_parameters.items()
            },
            "method": self.method,
        }


@dataclass(frozen=True)
class GroupShiftEnvelopeCertificate:
    decision: str
    group_radii: Mapping[str, float]
    observed_common_radius: float
    deployment_common_radius: float
    unsupported_groups: tuple[str, ...]
    registered_groups: tuple[str, ...]
    gamma_cap: float
    delta: float
    family_size: int
    group_certificates: Mapping[str, ShiftRadiusCertificate]
    method: str = "exact_discrete_reweighting"

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "group_radii": dict(self.group_radii),
            "observed_common_radius": self.observed_common_radius,
            "deployment_common_radius": self.deployment_common_radius,
            "unsupported_groups": list(self.unsupported_groups),
            "registered_groups": list(self.registered_groups),
            "gamma_cap": self.gamma_cap,
            "delta": self.delta,
            "family_size": self.family_size,
            "group_certificates": {
                key: value.to_dict() for key, value in self.group_certificates.items()
            },
            "method": self.method,
        }


@dataclass(frozen=True)
class IUTFixedProfileCertificate:
    decision: str
    gamma: float
    delta: float
    candidate_count: int
    candidate_failure_probability: float
    limiting_contracts: tuple[str, ...]
    certificates: tuple[RiskCertificate, ...]
    method: str = "candidate_iut_exact_discrete"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["certificates"] = [
            certificate.to_dict() for certificate in self.certificates
        ]
        return payload


@dataclass(frozen=True)
class BalancedProfileCertificate:
    decision: str
    target_profile: Mapping[str, float]
    source_profile: Mapping[str, float]
    delta: float
    candidate_count: int
    candidate_failure_probability: float
    limiting_contracts: tuple[str, ...]
    certificates: tuple[RiskCertificate, ...]
    method: str = "candidate_iut_exact_balanced_vector_profile"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["target_profile"] = dict(self.target_profile)
        payload["source_profile"] = dict(self.source_profile)
        payload["certificates"] = [
            certificate.to_dict() for certificate in self.certificates
        ]
        return payload


def _as_bounded_array(
    values: Iterable[float], *, lower: float, upper: float
) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    if lower >= upper:
        raise ValueError("lower must be strictly smaller than upper")
    tolerance = 1e-12
    if array.min() < lower - tolerance or array.max() > upper + tolerance:
        raise ValueError(f"values must lie in [{lower}, {upper}]")
    return np.clip(array, lower, upper)


def empirical_reweighting_risk(values: Iterable[float], gamma: float) -> float:
    """Return the exact empirical worst-case mean under ``0 <= dQ/dP <= gamma``.

    For an empirical distribution with mass ``1/n`` per observation, the
    adversary assigns at most ``gamma/n`` mass to each observation.  Allocating
    that mass greedily from the largest value downward solves the linear
    program exactly and equals empirical upper-tail CVaR.
    """

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.isfinite(array).all():
        raise ValueError("values must be finite")
    if gamma < 1.0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and at least one")

    capacity = gamma / array.size
    remaining = 1.0
    robust_mean = 0.0
    for value in np.sort(array)[::-1]:
        mass = min(capacity, remaining)
        robust_mean += mass * float(value)
        remaining -= mass
        if remaining <= 1e-15:
            break
    if remaining > 1e-10:
        raise RuntimeError("failed to allocate a unit deployment mass")
    return robust_mean


def dkw_epsilon(n: int, failure_probability: float) -> float:
    """Two-sided DKW radius with failure probability ``failure_probability``."""

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    return sqrt(log(2.0 / failure_probability) / (2.0 * n))


def robust_risk_certificate(
    key: str,
    values: Iterable[float],
    *,
    gamma: float,
    failure_probability: float,
    lower: float,
    upper: float,
) -> RiskCertificate:
    """Upper-certify population robust risk using DKW and CVaR duality."""

    array = _as_bounded_array(values, lower=lower, upper=upper)
    empirical = empirical_reweighting_risk(array, gamma)
    epsilon = dkw_epsilon(array.size, failure_probability)
    radius = gamma * (upper - lower) * epsilon
    ucb = min(upper, empirical + radius)
    return RiskCertificate(
        key=key,
        n=int(array.size),
        gamma=float(gamma),
        lower=float(lower),
        upper=float(upper),
        empirical_robust_risk=float(empirical),
        dkw_epsilon=float(epsilon),
        simultaneous_failure_probability=float(failure_probability),
        upper_confidence_bound=float(ucb),
    )


@lru_cache(maxsize=None)
def _clopper_pearson_upper(successes: int, n: int, alpha: float) -> float:
    from scipy.stats import beta

    if successes >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, n - successes))


@lru_cache(maxsize=None)
def _clopper_pearson_lower(successes: int, n: int, alpha: float) -> float:
    from scipy.stats import beta

    if successes <= 0:
        return 0.0
    return float(beta.ppf(alpha, successes, n - successes + 1))


def exact_discrete_risk_certificate(
    key: str,
    values: Iterable[float],
    *,
    gamma: float,
    failure_probability: float,
    support: tuple[int, ...],
) -> RiskCertificate:
    """Certify Bernoulli or paired {-1,0,1} risk without a range-only bound."""

    if support not in {(0, 1), (-1, 0, 1)}:
        raise ValueError("exact discrete support must be (0, 1) or (-1, 0, 1)")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    if gamma < 1.0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and at least one")
    array = _as_bounded_array(values, lower=float(support[0]), upper=float(support[-1]))
    if not np.isin(array, support).all():
        raise ValueError(f"values must lie on the declared support {support}")
    n = int(array.size)
    empirical = empirical_reweighting_risk(array, gamma)

    if support == (0, 1):
        successes = int(np.sum(array == 1))
        probability_upper = _clopper_pearson_upper(successes, n, failure_probability)
        ucb = min(1.0, gamma * probability_upper)
        details: dict[str, float | int | str] = {
            "successes": successes,
            "probability_upper": probability_upper,
            "interval": "one-sided Clopper-Pearson",
        }
    else:
        positive = int(np.sum(array == 1))
        negative = int(np.sum(array == -1))
        tail_alpha = failure_probability / 2.0
        positive_upper = _clopper_pearson_upper(positive, n, tail_alpha)
        negative_lower = _clopper_pearson_lower(negative, n, tail_alpha)
        positive_probability = min(positive_upper, 1.0 - negative_lower)
        zero_probability = max(0.0, 1.0 - negative_lower - positive_probability)
        positive_mass = min(1.0, gamma * positive_probability)
        remaining = 1.0 - positive_mass
        zero_mass = min(remaining, gamma * zero_probability)
        negative_mass = max(0.0, remaining - zero_mass)
        ucb = min(1.0, max(-1.0, positive_mass - negative_mass))
        details = {
            "positive_count": positive,
            "negative_count": negative,
            "positive_probability_upper": positive_upper,
            "negative_probability_lower": negative_lower,
            "interval": "two one-sided Clopper-Pearson bounds with Bonferroni",
        }
    return RiskCertificate(
        key=key,
        n=n,
        gamma=float(gamma),
        lower=float(support[0]),
        upper=float(support[-1]),
        empirical_robust_risk=float(empirical),
        dkw_epsilon=0.0,
        simultaneous_failure_probability=float(failure_probability),
        upper_confidence_bound=float(ucb),
        method="exact_discrete_reweighting",
        confidence_details=details,
    )


def exact_balanced_leakage_certificate(
    key: str,
    correct: Iterable[float],
    source: Iterable[int],
    *,
    gamma: float,
    failure_probability: float,
) -> RiskCertificate:
    """Upper-certify binary-source balanced attacker accuracy under shift."""

    correctness = _as_bounded_array(correct, lower=0.0, upper=1.0)
    source_array = np.asarray(list(source), dtype=np.int64)
    if source_array.ndim != 1 or len(source_array) != len(correctness):
        raise ValueError("source must be one-dimensional and match correctness")
    if set(map(int, np.unique(source_array))) != {0, 1}:
        raise ValueError("balanced leakage requires represented source classes {0, 1}")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    if gamma < 1.0 or not np.isfinite(gamma):
        raise ValueError("gamma must be finite and at least one")
    class_alpha = failure_probability / 2.0
    empirical_components: list[float] = []
    upper_components: list[float] = []
    details: dict[str, float | int | str] = {
        "interval": "two class-conditional one-sided Clopper-Pearson bounds",
        "aggregation": "equal-weight robust class recall",
    }
    for source_class in (0, 1):
        values = correctness[source_array == source_class]
        successes = int(np.sum(values == 1))
        n_class = int(len(values))
        probability_upper = _clopper_pearson_upper(
            successes, n_class, class_alpha
        )
        empirical_components.append(empirical_reweighting_risk(values, gamma))
        upper_components.append(min(1.0, gamma * probability_upper))
        details[f"class_{source_class}_n"] = n_class
        details[f"class_{source_class}_successes"] = successes
        details[f"class_{source_class}_probability_upper"] = probability_upper
    return RiskCertificate(
        key=key,
        n=int(len(correctness)),
        gamma=float(gamma),
        lower=0.0,
        upper=1.0,
        empirical_robust_risk=float(np.mean(empirical_components)),
        dkw_epsilon=0.0,
        simultaneous_failure_probability=float(failure_probability),
        upper_confidence_bound=float(np.mean(upper_components)),
        method="exact_balanced_leakage",
        confidence_details=details,
    )


def exact_balanced_leakage_profile_certificate(
    key: str,
    correct: Iterable[float],
    source: Iterable[int],
    *,
    source_profile: Mapping[str | int, float],
    failure_probability: float,
) -> RiskCertificate:
    """Certify balanced leakage under separate source-conditional budgets."""

    profile = {str(key): float(value) for key, value in source_profile.items()}
    if set(profile) != {"0", "1"}:
        raise ValueError("source_profile must provide source classes 0 and 1")
    if any(value < 1.0 or not np.isfinite(value) for value in profile.values()):
        raise ValueError("source-profile budgets must be finite and at least one")
    correctness = _as_bounded_array(correct, lower=0.0, upper=1.0)
    source_array = np.asarray(list(source), dtype=np.int64)
    if source_array.ndim != 1 or len(source_array) != len(correctness):
        raise ValueError("source must be one-dimensional and match correctness")
    if set(map(int, np.unique(source_array))) != {0, 1}:
        raise ValueError("balanced leakage requires represented source classes {0, 1}")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")

    class_alpha = failure_probability / 2.0
    empirical_components: list[float] = []
    upper_components: list[float] = []
    details: dict[str, float | int | str] = {
        "interval": "two class-conditional one-sided Clopper-Pearson bounds",
        "aggregation": "equal-weight robust class recall",
    }
    for source_class in (0, 1):
        values = correctness[source_array == source_class]
        successes = int(np.sum(values == 1))
        n_class = int(len(values))
        gamma = profile[str(source_class)]
        probability_upper = _clopper_pearson_upper(
            successes, n_class, class_alpha
        )
        empirical_components.append(empirical_reweighting_risk(values, gamma))
        upper_components.append(min(1.0, gamma * probability_upper))
        details[f"class_{source_class}_n"] = n_class
        details[f"class_{source_class}_successes"] = successes
        details[f"class_{source_class}_probability_upper"] = probability_upper
        details[f"class_{source_class}_gamma"] = gamma
    return RiskCertificate(
        key=key,
        n=int(len(correctness)),
        gamma=max(profile.values()),
        lower=0.0,
        upper=1.0,
        empirical_robust_risk=float(np.mean(empirical_components)),
        dkw_epsilon=0.0,
        simultaneous_failure_probability=float(failure_probability),
        upper_confidence_bound=float(np.mean(upper_components)),
        method="exact_balanced_leakage_vector_profile",
        confidence_details=details,
    )


def balanced_profile_contract_certificates(
    target_samples: Mapping[str, Iterable[float]],
    leakage_correct: Mapping[str, Iterable[float]],
    source: Iterable[int],
    *,
    target_profile: Mapping[str, float],
    source_profile: Mapping[str | int, float],
    local_failure_probability: float,
) -> dict[str, RiskCertificate]:
    """Evaluate every contract at one anisotropic deployment profile."""

    normalized_target = {str(key): float(value) for key, value in target_profile.items()}
    if set(normalized_target) != set(target_samples):
        raise ValueError("target_profile and target_samples must have identical keys")
    if any(
        value < 1.0 or not np.isfinite(value)
        for value in normalized_target.values()
    ):
        raise ValueError("target-profile budgets must be finite and at least one")
    certificates = {
        key: exact_discrete_risk_certificate(
            key,
            values,
            gamma=normalized_target[key],
            failure_probability=local_failure_probability,
            support=(-1, 0, 1),
        )
        for key, values in target_samples.items()
    }
    for attacker, values in leakage_correct.items():
        key = f"balanced_leakage::{attacker}"
        certificates[key] = exact_balanced_leakage_profile_certificate(
            key,
            values,
            source,
            source_profile=source_profile,
            failure_probability=local_failure_probability,
        )
    return certificates


def certify_balanced_iut_profile(
    target_samples: Mapping[str, Iterable[float]],
    leakage_correct: Mapping[str, Iterable[float]],
    source: Iterable[int],
    *,
    target_profile: Mapping[str, float],
    source_profile: Mapping[str | int, float],
    delta: float,
    candidate_count: int,
    target_threshold: float,
    leakage_threshold: float,
) -> BalancedProfileCertificate:
    """Candidate-wise IUT at a fixed anisotropic shift profile."""

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    candidate_alpha = delta / candidate_count
    certificates = balanced_profile_contract_certificates(
        target_samples,
        leakage_correct,
        source,
        target_profile=target_profile,
        source_profile=source_profile,
        local_failure_probability=candidate_alpha,
    )
    thresholds = {
        key: target_threshold if key.startswith("target::") else leakage_threshold
        for key in certificates
    }
    margins = {
        key: thresholds[key] - certificate.upper_confidence_bound
        for key, certificate in certificates.items()
    }
    worst = min(margins.values())
    return BalancedProfileCertificate(
        decision="EDIT" if all(margin >= 0.0 for margin in margins.values()) else "ABSTAIN",
        target_profile={str(key): float(value) for key, value in target_profile.items()},
        source_profile={str(key): float(value) for key, value in source_profile.items()},
        delta=float(delta),
        candidate_count=int(candidate_count),
        candidate_failure_probability=float(candidate_alpha),
        limiting_contracts=tuple(
            sorted(key for key, margin in margins.items() if margin <= worst + 1e-12)
        ),
        certificates=tuple(certificates[key] for key in sorted(certificates)),
    )


def balanced_profile_in_envelope(
    envelope: BalancedShiftEnvelopeCertificate,
    *,
    target_profile: Mapping[str | int, float],
    source_profile: Mapping[str | int, float],
    tolerance: float = 1e-12,
) -> bool:
    """Return whether a full profile satisfies the reported joint envelope."""

    if envelope.unsupported_target_environments:
        return False
    target = {str(key): float(value) for key, value in target_profile.items()}
    source = {str(key): float(value) for key, value in source_profile.items()}
    if set(target) != set(envelope.target_environment_radii):
        raise ValueError("target profile must provide every registered environment")
    if set(source) != {"0", "1"}:
        raise ValueError("source profile must provide source classes 0 and 1")
    if any(
        value < 1.0 or not np.isfinite(value)
        for value in (*target.values(), *source.values())
    ):
        raise ValueError("profile budgets must be finite and at least one")
    if any(
        target[group] > float(radius) + tolerance
        for group, radius in envelope.target_environment_radii.items()
    ):
        return False

    leakage_curves = {
        key: details
        for key, details in envelope.simultaneous_curve_parameters.items()
        if key.startswith("balanced_leakage::")
    }
    if not leakage_curves:
        return False
    for details in leakage_curves.values():
        upper = 0.5 * sum(
            min(
                1.0,
                source[str(source_class)]
                * float(details[f"class_{source_class}_probability_upper"]),
            )
            for source_class in (0, 1)
        )
        if upper > float(details["threshold"]) + tolerance:
            return False
    return True


def balanced_contract_certificates(
    target_samples: Mapping[str, Iterable[float]],
    leakage_correct: Mapping[str, Iterable[float]],
    source: Iterable[int],
    *,
    gamma: float,
    local_failure_probability: float,
) -> dict[str, RiskCertificate]:
    """Evaluate target and balanced-leakage contracts at one local alpha."""

    if not target_samples or not leakage_correct:
        raise ValueError("target and leakage contract families must be nonempty")
    certificates = {
        key: exact_discrete_risk_certificate(
            key,
            values,
            gamma=gamma,
            failure_probability=local_failure_probability,
            support=(-1, 0, 1),
        )
        for key, values in target_samples.items()
    }
    for attacker, values in leakage_correct.items():
        key = f"balanced_leakage::{attacker}"
        if key in certificates:
            raise ValueError(f"duplicate balanced contract key: {key}")
        certificates[key] = exact_balanced_leakage_certificate(
            key,
            values,
            source,
            gamma=gamma,
            failure_probability=local_failure_probability,
        )
    return certificates


def certify_balanced_iut_fixed_profile(
    target_samples: Mapping[str, Iterable[float]],
    leakage_correct: Mapping[str, Iterable[float]],
    source: Iterable[int],
    *,
    gamma: float,
    delta: float,
    candidate_count: int,
    target_threshold: float,
    leakage_threshold: float,
) -> IUTFixedProfileCertificate:
    """Candidate-wise IUT for target and balanced-leakage contracts."""

    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    candidate_alpha = delta / candidate_count
    certificates = balanced_contract_certificates(
        target_samples,
        leakage_correct,
        source,
        gamma=gamma,
        local_failure_probability=candidate_alpha,
    )
    thresholds = {
        key: target_threshold if key.startswith("target::") else leakage_threshold
        for key in certificates
    }
    margins = {
        key: thresholds[key] - certificate.upper_confidence_bound
        for key, certificate in certificates.items()
    }
    accepted = all(margin >= 0.0 for margin in margins.values())
    worst = min(margins.values())
    return IUTFixedProfileCertificate(
        decision="EDIT" if accepted else "ABSTAIN",
        gamma=float(gamma),
        delta=float(delta),
        candidate_count=int(candidate_count),
        candidate_failure_probability=float(candidate_alpha),
        limiting_contracts=tuple(
            sorted(key for key, margin in margins.items() if margin <= worst + 1e-12)
        ),
        certificates=tuple(certificates[key] for key in sorted(certificates)),
        method="candidate_iut_exact_balanced_leakage",
    )


def certify_balanced_shift_radius(
    target_samples: Mapping[str, Iterable[float]],
    leakage_correct: Mapping[str, Iterable[float]],
    source: Iterable[int],
    *,
    delta: float,
    family_size: int,
    target_threshold: float,
    leakage_threshold: float,
    gamma_cap: float = 32.0,
    tolerance: float = 1e-4,
) -> ShiftRadiusCertificate:
    """Simultaneously lower-certify the balanced-contract common radius."""

    contract_count = len(target_samples) + len(leakage_correct)
    if family_size < contract_count:
        raise ValueError("family_size cannot be smaller than the supplied contracts")
    if gamma_cap < 1.0 or not np.isfinite(gamma_cap):
        raise ValueError("gamma_cap must be finite and at least one")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    local_alpha = delta / family_size

    def evaluate(gamma: float) -> tuple[bool, dict[str, RiskCertificate]]:
        certificates = balanced_contract_certificates(
            target_samples,
            leakage_correct,
            source,
            gamma=gamma,
            local_failure_probability=local_alpha,
        )
        passed = all(
            certificate.upper_confidence_bound
            <= (
                target_threshold
                if key.startswith("target::")
                else leakage_threshold
            )
            for key, certificate in certificates.items()
        )
        return passed, certificates

    iid_passes, iid_certificates = evaluate(1.0)
    thresholds = {
        key: target_threshold if key.startswith("target::") else leakage_threshold
        for key in iid_certificates
    }
    if not iid_passes:
        margins = {
            key: thresholds[key] - certificate.upper_confidence_bound
            for key, certificate in iid_certificates.items()
        }
        worst = min(margins.values())
        return ShiftRadiusCertificate(
            decision="ABSTAIN",
            certified_radius=0.0,
            gamma_cap=float(gamma_cap),
            right_censored=False,
            delta=float(delta),
            limiting_contracts=tuple(
                sorted(key for key, margin in margins.items() if margin <= worst + 1e-12)
            ),
            certificates_at_radius=tuple(
                iid_certificates[key] for key in sorted(iid_certificates)
            ),
            method="exact_balanced_leakage",
        )
    cap_passes, cap_certificates = evaluate(gamma_cap)
    if cap_passes:
        margins = {
            key: thresholds[key] - certificate.upper_confidence_bound
            for key, certificate in cap_certificates.items()
        }
        worst = min(margins.values())
        return ShiftRadiusCertificate(
            decision="EDIT",
            certified_radius=float(gamma_cap),
            gamma_cap=float(gamma_cap),
            right_censored=True,
            delta=float(delta),
            limiting_contracts=tuple(
                sorted(key for key, margin in margins.items() if margin <= worst + 1e-12)
            ),
            certificates_at_radius=tuple(
                cap_certificates[key] for key in sorted(cap_certificates)
            ),
            method="exact_balanced_leakage",
        )
    lower_gamma, upper_gamma = 1.0, float(gamma_cap)
    lower_certificates = iid_certificates
    while upper_gamma - lower_gamma > tolerance:
        midpoint = (lower_gamma + upper_gamma) / 2.0
        passed, certificates = evaluate(midpoint)
        if passed:
            lower_gamma, lower_certificates = midpoint, certificates
        else:
            upper_gamma = midpoint
    margins = {
        key: thresholds[key] - certificate.upper_confidence_bound
        for key, certificate in lower_certificates.items()
    }
    worst = min(margins.values())
    return ShiftRadiusCertificate(
        decision="EDIT",
        certified_radius=float(lower_gamma),
        gamma_cap=float(gamma_cap),
        right_censored=False,
        delta=float(delta),
        limiting_contracts=tuple(
            sorted(key for key, margin in margins.items() if margin <= worst + tolerance)
        ),
        certificates_at_radius=tuple(
            lower_certificates[key] for key in sorted(lower_certificates)
        ),
        method="exact_balanced_leakage",
    )


def certify_balanced_shift_envelope(
    target_samples: Mapping[str, Iterable[float]],
    leakage_correct: Mapping[str, Iterable[float]],
    source: Iterable[int],
    *,
    delta: float,
    family_size: int,
    target_threshold: float,
    leakage_threshold: float,
    registered_target_environments: Sequence[str | int] | None = None,
    gamma_cap: float = 32.0,
    tolerance: float = 1e-4,
) -> BalancedShiftEnvelopeCertificate:
    """Certify coordinate intercepts and curves for the balanced shift envelope.

    Each target coordinate varies one registered environment budget while all
    other coordinates remain at one. Each source coordinate varies one
    source-conditional budget while the other remains at one; every registered
    attacker must continue to pass. The complete envelope is represented by
    the simultaneous curve parameters and inequalities, while these intercepts
    make its geometry directly inspectable.
    """

    if not target_samples or not leakage_correct:
        raise ValueError("target and leakage contract families must be nonempty")
    if family_size < len(target_samples) + len(leakage_correct):
        raise ValueError("family_size is smaller than the supplied contract family")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if gamma_cap < 1.0 or not np.isfinite(gamma_cap):
        raise ValueError("gamma_cap must be finite and at least one")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")

    materialized_target = {
        str(key): _as_bounded_array(values, lower=-1.0, upper=1.0)
        for key, values in target_samples.items()
    }
    materialized_leakage = {
        str(key): _as_bounded_array(values, lower=0.0, upper=1.0)
        for key, values in leakage_correct.items()
    }
    source_array = np.asarray(list(source), dtype=np.int64)
    if set(map(int, np.unique(source_array))) != {0, 1}:
        raise ValueError("balanced leakage requires represented source classes {0, 1}")
    if any(len(values) != len(source_array) for values in materialized_leakage.values()):
        raise ValueError("source labels must align with every leakage array")
    local_alpha = delta / family_size

    iid = balanced_contract_certificates(
        materialized_target,
        materialized_leakage,
        source_array,
        gamma=1.0,
        local_failure_probability=local_alpha,
    )
    iid_passes = all(
        certificate.upper_confidence_bound
        <= (
            target_threshold
            if key.startswith("target::")
            else leakage_threshold
        )
        for key, certificate in iid.items()
    )
    curve_parameters: dict[str, Mapping[str, float | int | str]] = {}
    for key, certificate in iid.items():
        details = dict(certificate.confidence_details or {})
        details["threshold"] = (
            float(target_threshold)
            if key.startswith("target::")
            else float(leakage_threshold)
        )
        curve_parameters[key] = details

    observed_group_by_key = {
        key: key.split("target::environment=", 1)[1]
        if key.startswith("target::environment=")
        else key
        for key in materialized_target
    }
    observed_groups = set(observed_group_by_key.values())
    registered_groups = (
        observed_groups
        if registered_target_environments is None
        else set(map(str, registered_target_environments))
    )
    unsupported = tuple(sorted(registered_groups - observed_groups))

    def radius(predicate: Any) -> tuple[float, bool]:
        if not predicate(1.0):
            return 0.0, False
        if predicate(float(gamma_cap)):
            return float(gamma_cap), True
        lower, upper = 1.0, float(gamma_cap)
        while upper - lower > tolerance:
            midpoint = (lower + upper) / 2.0
            if predicate(midpoint):
                lower = midpoint
            else:
                upper = midpoint
        return float(lower), False

    target_radii: dict[str, float] = {
        group: 0.0 for group in sorted(registered_groups)
    }
    source_radii = {"0": 0.0, "1": 0.0}
    right_censored: list[str] = []
    if iid_passes:
        for key, values in materialized_target.items():
            group = observed_group_by_key[key]

            def target_passes(gamma: float, *, values: np.ndarray = values, key: str = key) -> bool:
                certificate = exact_discrete_risk_certificate(
                    key,
                    values,
                    gamma=gamma,
                    failure_probability=local_alpha,
                    support=(-1, 0, 1),
                )
                return certificate.upper_confidence_bound <= target_threshold

            value, censored = radius(target_passes)
            target_radii[group] = value
            if censored:
                right_censored.append(f"target::environment={group}")

        leakage_probabilities = {
            attacker: {
                source_class: float(
                    iid[f"balanced_leakage::{attacker}"].confidence_details[
                        f"class_{source_class}_probability_upper"
                    ]
                )
                for source_class in (0, 1)
            }
            for attacker in materialized_leakage
        }
        for source_class in (0, 1):
            other = 1 - source_class

            def source_passes(gamma: float, *, source_class: int = source_class, other: int = other) -> bool:
                return all(
                    0.5
                    * (
                        min(1.0, gamma * probabilities[source_class])
                        + probabilities[other]
                    )
                    <= leakage_threshold
                    for probabilities in leakage_probabilities.values()
                )

            value, censored = radius(source_passes)
            source_radii[str(source_class)] = value
            if censored:
                right_censored.append(f"source_class={source_class}")

    common = certify_balanced_shift_radius(
        materialized_target,
        materialized_leakage,
        source_array,
        delta=delta,
        family_size=family_size,
        target_threshold=target_threshold,
        leakage_threshold=leakage_threshold,
        gamma_cap=gamma_cap,
        tolerance=tolerance,
    )
    observed_common = float(common.certified_radius)
    deployment_common = 0.0 if unsupported else observed_common
    return BalancedShiftEnvelopeCertificate(
        decision="EDIT" if deployment_common >= 1.0 else "ABSTAIN",
        target_environment_radii=target_radii,
        source_class_radii=source_radii,
        observed_common_radius=observed_common,
        deployment_common_radius=deployment_common,
        unsupported_target_environments=unsupported,
        gamma_cap=float(gamma_cap),
        delta=float(delta),
        family_size=int(family_size),
        right_censored_coordinates=tuple(sorted(right_censored)),
        simultaneous_curve_parameters=curve_parameters,
    )


def simultaneous_exact_discrete_certificates(
    samples: Mapping[str, Iterable[float]],
    *,
    gamma: float,
    delta: float,
    supports: Mapping[str, tuple[int, ...]],
    family_size: int | None = None,
) -> dict[str, RiskCertificate]:
    """Certify a registered family, optionally spending alpha over a larger family."""

    if not samples or set(samples) != set(supports):
        raise ValueError("nonempty samples and supports must have identical keys")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    denominator = len(samples) if family_size is None else int(family_size)
    if denominator < len(samples):
        raise ValueError("family_size cannot be smaller than the supplied sample family")
    per_risk_delta = delta / denominator
    return {
        key: exact_discrete_risk_certificate(
            key,
            values,
            gamma=gamma,
            failure_probability=per_risk_delta,
            support=supports[key],
        )
        for key, values in samples.items()
    }


def certify_discrete_iut_fixed_profile(
    samples: Mapping[str, Iterable[float]],
    *,
    gamma: float,
    delta: float,
    candidate_count: int,
    supports: Mapping[str, tuple[int, ...]],
    thresholds: Mapping[str, float],
) -> IUTFixedProfileCertificate:
    """Certify one fixed candidate/profile by an intersection-union test.

    For an unsafe candidate, at least one component contract is a true null.
    Requiring every component to pass at level ``delta / candidate_count`` is
    therefore a level-``delta / candidate_count`` test of the candidate union
    null. A union bound across candidates controls selection of any unsafe
    candidate. This rule does not provide a simultaneous post-hoc envelope.
    """

    if not samples or set(samples) != set(supports) or set(samples) != set(thresholds):
        raise ValueError("nonempty samples, supports, and thresholds need identical keys")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    candidate_alpha = delta / candidate_count
    certificates = {
        key: exact_discrete_risk_certificate(
            key,
            values,
            gamma=gamma,
            failure_probability=candidate_alpha,
            support=supports[key],
        )
        for key, values in samples.items()
    }
    margins = {
        key: thresholds[key] - certificate.upper_confidence_bound
        for key, certificate in certificates.items()
    }
    accepted = all(margin >= 0.0 for margin in margins.values())
    worst = min(margins.values())
    limiting = tuple(
        sorted(key for key, margin in margins.items() if margin <= worst + 1e-12)
    )
    return IUTFixedProfileCertificate(
        decision="EDIT" if accepted else "ABSTAIN",
        gamma=float(gamma),
        delta=float(delta),
        candidate_count=int(candidate_count),
        candidate_failure_probability=float(candidate_alpha),
        limiting_contracts=limiting,
        certificates=tuple(certificates[key] for key in sorted(certificates)),
    )


def certify_discrete_shift_radius(
    samples: Mapping[str, Iterable[float]],
    *,
    delta: float,
    supports: Mapping[str, tuple[int, ...]],
    thresholds: Mapping[str, float],
    family_size: int | None = None,
    gamma_cap: float = 32.0,
    tolerance: float = 1e-4,
) -> ShiftRadiusCertificate:
    """Lower-certify a common shift radius using exact discrete intervals."""

    if set(samples) != set(supports) or set(samples) != set(thresholds):
        raise ValueError("samples, supports, and thresholds must have identical keys")
    if gamma_cap < 1.0 or not np.isfinite(gamma_cap):
        raise ValueError("gamma_cap must be finite and at least one")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")
    materialized = {
        key: _as_bounded_array(
            values, lower=float(supports[key][0]), upper=float(supports[key][-1])
        )
        for key, values in samples.items()
    }

    def evaluate(gamma: float) -> tuple[bool, dict[str, RiskCertificate]]:
        certificates = simultaneous_exact_discrete_certificates(
            materialized,
            gamma=gamma,
            delta=delta,
            supports=supports,
            family_size=family_size,
        )
        return (
            all(certificates[key].upper_confidence_bound <= thresholds[key] for key in certificates),
            certificates,
        )

    iid_passes, iid_certificates = evaluate(1.0)
    if not iid_passes:
        margins = {
            key: thresholds[key] - certificate.upper_confidence_bound
            for key, certificate in iid_certificates.items()
        }
        worst = min(margins.values())
        return ShiftRadiusCertificate(
            decision="ABSTAIN",
            certified_radius=0.0,
            gamma_cap=float(gamma_cap),
            right_censored=False,
            delta=float(delta),
            limiting_contracts=tuple(sorted(key for key, value in margins.items() if value <= worst + 1e-12)),
            certificates_at_radius=tuple(iid_certificates[key] for key in sorted(iid_certificates)),
            method="exact_discrete_reweighting",
        )

    cap_passes, cap_certificates = evaluate(gamma_cap)
    if cap_passes:
        margins = {
            key: thresholds[key] - certificate.upper_confidence_bound
            for key, certificate in cap_certificates.items()
        }
        worst = min(margins.values())
        return ShiftRadiusCertificate(
            decision="EDIT",
            certified_radius=float(gamma_cap),
            gamma_cap=float(gamma_cap),
            right_censored=True,
            delta=float(delta),
            limiting_contracts=tuple(sorted(key for key, value in margins.items() if value <= worst + 1e-12)),
            certificates_at_radius=tuple(cap_certificates[key] for key in sorted(cap_certificates)),
            method="exact_discrete_reweighting",
        )

    lower_gamma, upper_gamma = 1.0, float(gamma_cap)
    lower_certificates = iid_certificates
    while upper_gamma - lower_gamma > tolerance:
        midpoint = (lower_gamma + upper_gamma) / 2.0
        passes, certificates = evaluate(midpoint)
        if passes:
            lower_gamma, lower_certificates = midpoint, certificates
        else:
            upper_gamma = midpoint
    margins = {
        key: thresholds[key] - certificate.upper_confidence_bound
        for key, certificate in lower_certificates.items()
    }
    worst = min(margins.values())
    return ShiftRadiusCertificate(
        decision="EDIT",
        certified_radius=float(lower_gamma),
        gamma_cap=float(gamma_cap),
        right_censored=False,
        delta=float(delta),
        limiting_contracts=tuple(sorted(key for key, value in margins.items() if value <= worst + tolerance)),
        certificates_at_radius=tuple(lower_certificates[key] for key in sorted(lower_certificates)),
        method="exact_discrete_reweighting",
    )


def certify_discrete_group_shift_envelope(
    grouped_samples: Mapping[str, Mapping[str, Iterable[float]]],
    *,
    delta: float,
    grouped_supports: Mapping[str, Mapping[str, tuple[int, ...]]],
    grouped_thresholds: Mapping[str, Mapping[str, float]],
    family_size: int,
    registered_groups: Sequence[str] | None = None,
    gamma_cap: float = 32.0,
    tolerance: float = 1e-4,
) -> GroupShiftEnvelopeCertificate:
    """Certify the support-aware vector of groupwise reweighting radii.

    ``family_size`` is the multiplicity denominator for the complete registered
    candidate-contract family, not just the groups supplied to this call. A
    registered group with no certification samples receives radius zero.
    """

    observed_groups = tuple(sorted(map(str, grouped_samples)))
    if set(grouped_samples) != set(grouped_supports):
        raise ValueError("grouped_samples and grouped_supports must have identical groups")
    if set(grouped_samples) != set(grouped_thresholds):
        raise ValueError("grouped_samples and grouped_thresholds must have identical groups")
    if any(not samples for samples in grouped_samples.values()):
        raise ValueError("every observed group must contain at least one contract")
    supplied_contracts = sum(len(samples) for samples in grouped_samples.values())
    if family_size < supplied_contracts:
        raise ValueError("family_size cannot be smaller than the supplied contract family")
    if registered_groups is None:
        all_groups = observed_groups
    else:
        all_groups = tuple(sorted({str(group) for group in registered_groups}))
        if not set(observed_groups).issubset(all_groups):
            raise ValueError("registered_groups must contain every observed group")

    certificates: dict[str, ShiftRadiusCertificate] = {}
    group_radii: dict[str, float] = {}
    for group in observed_groups:
        certificate = certify_discrete_shift_radius(
            grouped_samples[group],
            delta=delta,
            supports=grouped_supports[group],
            thresholds=grouped_thresholds[group],
            family_size=family_size,
            gamma_cap=gamma_cap,
            tolerance=tolerance,
        )
        certificates[group] = certificate
        group_radii[group] = certificate.certified_radius

    unsupported = tuple(sorted(set(all_groups) - set(observed_groups)))
    for group in unsupported:
        group_radii[group] = 0.0
    observed_common = min(
        (group_radii[group] for group in observed_groups), default=0.0
    )
    deployment_common = min(
        (group_radii[group] for group in all_groups), default=0.0
    )
    return GroupShiftEnvelopeCertificate(
        decision="EDIT" if deployment_common >= 1.0 else "ABSTAIN",
        group_radii={group: group_radii[group] for group in all_groups},
        observed_common_radius=float(observed_common),
        deployment_common_radius=float(deployment_common),
        unsupported_groups=unsupported,
        registered_groups=all_groups,
        gamma_cap=float(gamma_cap),
        delta=float(delta),
        family_size=int(family_size),
        group_certificates=certificates,
    )


def simultaneous_certificates(
    samples: Mapping[str, Iterable[float]],
    *,
    gamma: float,
    delta: float,
    bounds: Mapping[str, tuple[float, float]],
) -> dict[str, RiskCertificate]:
    """Certify all supplied risks with Bonferroni simultaneous coverage."""

    if not samples:
        raise ValueError("at least one risk sample is required")
    if set(samples) != set(bounds):
        raise ValueError("samples and bounds must have identical keys")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    per_risk_delta = delta / len(samples)
    return {
        key: robust_risk_certificate(
            key,
            values,
            gamma=gamma,
            failure_probability=per_risk_delta,
            lower=bounds[key][0],
            upper=bounds[key][1],
        )
        for key, values in samples.items()
    }


def certify_shift_radius(
    samples: Mapping[str, Iterable[float]],
    *,
    delta: float,
    bounds: Mapping[str, tuple[float, float]],
    thresholds: Mapping[str, float],
    gamma_cap: float = 32.0,
    tolerance: float = 1e-4,
) -> ShiftRadiusCertificate:
    """Lower-certify the largest common density-ratio shift budget.

    The returned radius is simultaneously valid for every supplied contract.
    A radius of zero means that even the IID case ``gamma=1`` does not certify.
    ``right_censored`` means every contract still certifies at ``gamma_cap``.
    """

    if set(samples) != set(bounds) or set(samples) != set(thresholds):
        raise ValueError("samples, bounds, and thresholds must have identical keys")
    if gamma_cap < 1.0 or not np.isfinite(gamma_cap):
        raise ValueError("gamma_cap must be finite and at least one")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and positive")

    materialized = {
        key: _as_bounded_array(values, lower=bounds[key][0], upper=bounds[key][1])
        for key, values in samples.items()
    }

    def evaluate(gamma: float) -> tuple[bool, dict[str, RiskCertificate]]:
        certificates = simultaneous_certificates(
            materialized, gamma=gamma, delta=delta, bounds=bounds
        )
        passes = all(
            certificates[key].upper_confidence_bound <= thresholds[key]
            for key in certificates
        )
        return passes, certificates

    iid_passes, iid_certificates = evaluate(1.0)
    if not iid_passes:
        margins = {
            key: thresholds[key] - cert.upper_confidence_bound
            for key, cert in iid_certificates.items()
        }
        worst_margin = min(margins.values())
        limiting = tuple(
            sorted(key for key, margin in margins.items() if margin <= worst_margin + 1e-12)
        )
        return ShiftRadiusCertificate(
            decision="ABSTAIN",
            certified_radius=0.0,
            gamma_cap=float(gamma_cap),
            right_censored=False,
            delta=float(delta),
            limiting_contracts=limiting,
            certificates_at_radius=tuple(
                iid_certificates[key] for key in sorted(iid_certificates)
            ),
        )

    cap_passes, cap_certificates = evaluate(gamma_cap)
    if cap_passes:
        margins = {
            key: thresholds[key] - cert.upper_confidence_bound
            for key, cert in cap_certificates.items()
        }
        best_margin = min(margins.values())
        limiting = tuple(
            sorted(key for key, margin in margins.items() if margin <= best_margin + 1e-12)
        )
        return ShiftRadiusCertificate(
            decision="EDIT",
            certified_radius=float(gamma_cap),
            gamma_cap=float(gamma_cap),
            right_censored=True,
            delta=float(delta),
            limiting_contracts=limiting,
            certificates_at_radius=tuple(
                cap_certificates[key] for key in sorted(cap_certificates)
            ),
        )

    lower_gamma = 1.0
    upper_gamma = float(gamma_cap)
    lower_certificates = iid_certificates
    while upper_gamma - lower_gamma > tolerance:
        midpoint = (lower_gamma + upper_gamma) / 2.0
        midpoint_passes, midpoint_certificates = evaluate(midpoint)
        if midpoint_passes:
            lower_gamma = midpoint
            lower_certificates = midpoint_certificates
        else:
            upper_gamma = midpoint

    margins = {
        key: thresholds[key] - cert.upper_confidence_bound
        for key, cert in lower_certificates.items()
    }
    worst_margin = min(margins.values())
    limiting = tuple(
        sorted(key for key, margin in margins.items() if margin <= worst_margin + tolerance)
    )
    return ShiftRadiusCertificate(
        decision="EDIT",
        certified_radius=float(lower_gamma),
        gamma_cap=float(gamma_cap),
        right_censored=False,
        delta=float(delta),
        limiting_contracts=limiting,
        certificates_at_radius=tuple(
            lower_certificates[key] for key in sorted(lower_certificates)
        ),
    )


def certify_edits(
    target_harm: Mapping[str, Iterable[float]],
    leakage: Mapping[tuple[str, str], Iterable[float]],
    *,
    edit_order: Sequence[str],
    gamma: float,
    delta: float,
    target_threshold: float,
    leakage_threshold: float,
) -> EditDecision:
    """Return the strongest edit whose target and attacker contracts certify.

    ``target_harm`` values must lie in ``[-1, 1]`` and compare each edit to the
    identity intervention on the same example.  ``leakage`` values must lie in
    ``[0, 1]`` and are keyed by ``(edit, attacker)``.  The final element of
    ``edit_order`` is treated as the strongest edit.
    """

    edits = tuple(edit_order)
    if not edits or len(set(edits)) != len(edits):
        raise ValueError("edit_order must contain unique edit names")
    if set(target_harm) != set(edits):
        raise ValueError("target_harm must provide every edit exactly once")
    attackers_by_edit = {
        edit: {attacker for candidate, attacker in leakage if candidate == edit}
        for edit in edits
    }
    if any(not attackers for attackers in attackers_by_edit.values()):
        raise ValueError("every edit needs at least one leakage attacker")
    attacker_sets = list(attackers_by_edit.values())
    if any(attackers != attacker_sets[0] for attackers in attacker_sets[1:]):
        raise ValueError("every edit must use the same attacker set")

    samples: dict[str, Iterable[float]] = {}
    bounds: dict[str, tuple[float, float]] = {}
    for edit in edits:
        target_key = f"target::{edit}"
        samples[target_key] = target_harm[edit]
        bounds[target_key] = (-1.0, 1.0)
        for attacker in sorted(attackers_by_edit[edit]):
            leakage_key = f"leakage::{edit}::{attacker}"
            samples[leakage_key] = leakage[(edit, attacker)]
            bounds[leakage_key] = (0.0, 1.0)

    certificates = simultaneous_certificates(
        samples, gamma=gamma, delta=delta, bounds=bounds
    )
    accepted: list[str] = []
    for edit in edits:
        target_ok = (
            certificates[f"target::{edit}"].upper_confidence_bound
            <= target_threshold
        )
        leakage_ok = all(
            certificates[f"leakage::{edit}::{attacker}"].upper_confidence_bound
            <= leakage_threshold
            for attacker in attackers_by_edit[edit]
        )
        if target_ok and leakage_ok:
            accepted.append(edit)

    selected = accepted[-1] if accepted else None
    return EditDecision(
        decision="EDIT" if selected is not None else "ABSTAIN",
        selected_edit=selected,
        accepted_edits=tuple(accepted),
        target_threshold=float(target_threshold),
        leakage_threshold=float(leakage_threshold),
        delta=float(delta),
        gamma=float(gamma),
        certificates=tuple(certificates[key] for key in sorted(certificates)),
    )
