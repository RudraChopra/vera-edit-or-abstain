"""Finite-sample certificates for paired representation-edit risks.

The certificate is distributionally robust over all deployment distributions
whose density ratio with respect to validation is bounded by ``gamma``.  Every
attacker and candidate must be fixed independently of the certification fold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import log, sqrt
from typing import Iterable, Mapping, Sequence

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
