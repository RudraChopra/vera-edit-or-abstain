"""Finite-sample certificates for paired representation-edit risks.

The certificate is distributionally robust over all deployment distributions
whose density ratio with respect to validation is bounded by ``gamma``.  Every
attacker and candidate must be fixed independently of the certification fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


def _clopper_pearson_upper(successes: int, n: int, alpha: float) -> float:
    from scipy.stats import beta

    if successes >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, successes + 1, n - successes))


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
