"""Finite lower and upper sample-size calculations for release certification."""

from __future__ import annotations

from math import ceil, log


def bernoulli_kl(first: float, second: float) -> float:
    """KL(Ber(first) || Ber(second)), with exact endpoint conventions."""

    if not 0.0 <= first <= 1.0 or not 0.0 <= second <= 1.0:
        raise ValueError("Bernoulli probabilities must lie in [0, 1]")
    if first == second:
        return 0.0
    if second in (0.0, 1.0):
        return float("inf")
    terms = 0.0
    if first > 0.0:
        terms += first * log(first / second)
    if first < 1.0:
        terms += (1.0 - first) * log((1.0 - first) / (1.0 - second))
    return terms


def binary_certification_sample_lower_bound(
    *,
    contract: float,
    margin: float,
    soundness_error: float,
    power_error: float,
) -> int:
    """Bretagnolle-Huber lower bound for the embedded binary experiment."""

    if not 0.0 < margin < min(contract, 0.5 - contract):
        raise ValueError("margin must keep both hypotheses inside (0, 1/2)")
    if not 0.0 < soundness_error + power_error < 0.5:
        raise ValueError("soundness_error + power_error must lie in (0, 1/2)")
    safe = 0.5 + contract - margin
    unsafe = 0.5 + contract + margin
    divergence = bernoulli_kl(safe, unsafe)
    return int(
        ceil(log(1.0 / (2.0 * (soundness_error + power_error))) / divergence)
    )


def weissman_sample_upper_bound(
    *,
    alphabet_size: int,
    margin: float,
    failure_probability: float,
) -> int:
    """Sufficient sample size for a Weissman L1 radius at most ``margin``."""

    if alphabet_size < 2:
        raise ValueError("alphabet_size must be at least two")
    if margin <= 0.0:
        raise ValueError("margin must be positive")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    return int(
        ceil(
            2.0
            * log((2.0**alphabet_size - 2.0) / failure_probability)
            / margin**2
        )
    )
