"""Exact finite-token envelopes for MOSAIC.

MOSAIC (Minimax-Optimized Source-Agnostic Invariant Channels) is a new,
separate research direction from VERA. Its central design choice is to release
a finite token rather than an unconstrained continuous embedding.  On a finite
alphabet, total variation is the exact advantage of the best *arbitrary*
binary downstream attacker; no attacker portfolio is required.

This module implements the deterministic part of the proposed theorem.  It
does not estimate a real-world shift: callers must explicitly declare a
conditional likelihood-ratio budget ``gamma`` and only receive a guarantee for
external distributions in that ambiguity set.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import log, log2, sqrt
from typing import Iterable, Sequence

import numpy as np


MAX_EXACT_TOKENS = 20
MAX_EXACT_ASSIGNMENTS = 5_000_000


@dataclass(frozen=True)
class TVEnvelope:
    """An exact or high-confidence upper envelope for robust total variation."""

    value: float
    gamma: float
    maximizing_mask: int
    token_count: int
    method: str
    l1_radius_group_0: float | None = None
    l1_radius_group_1: float | None = None

    @property
    def universal_balanced_attacker_accuracy(self) -> float:
        """Accuracy of the strongest binary attacker covered by this envelope."""

        return 0.5 * (1.0 + self.value)


@dataclass(frozen=True)
class MulticlassAttackerEnvelope:
    """Worst balanced accuracy over every multiclass token attacker."""

    balanced_accuracy: float
    gamma: float
    maximizing_assignment: tuple[int, ...]
    source_count: int
    token_count: int
    method: str
    l1_radii: tuple[float, ...] | None = None

    @property
    def normalized_advantage(self) -> float:
        """Map chance accuracy 1/G to zero and perfect recovery to one."""

        chance = 1.0 / self.source_count
        return (self.balanced_accuracy - chance) / (1.0 - chance)


def _probability_vector(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("a probability vector with at least two tokens is required")
    if array.size > MAX_EXACT_TOKENS:
        raise ValueError(
            f"exact subset enumeration supports at most {MAX_EXACT_TOKENS} tokens"
        )
    if not np.isfinite(array).all() or np.any(array < -1e-12):
        raise ValueError("probabilities must be finite and non-negative")
    total = float(array.sum())
    if not np.isclose(total, 1.0, atol=1e-10):
        raise ValueError(f"probabilities must sum to one, received {total}")
    return np.clip(array, 0.0, 1.0)


def _probability_matrix(values: Sequence[Sequence[float]]) -> np.ndarray:
    rows = tuple(_probability_vector(row) for row in values)
    if len(rows) < 2:
        raise ValueError("at least two source distributions are required")
    token_count = rows[0].size
    if any(row.size != token_count for row in rows):
        raise ValueError("all source distributions must share a token alphabet")
    return np.stack(rows)


def _validate_gamma(gamma: float) -> float:
    if not np.isfinite(gamma) or gamma < 1.0:
        raise ValueError("gamma must be finite and at least one")
    return float(gamma)


def _subset_masses(probabilities: np.ndarray) -> np.ndarray:
    """Return P(A) for every subset A of a small finite token alphabet."""

    masses = np.zeros(1 << probabilities.size, dtype=np.float64)
    for mask in range(1, masses.size):
        least_significant_bit = mask & -mask
        token = least_significant_bit.bit_length() - 1
        masses[mask] = masses[mask ^ least_significant_bit] + probabilities[token]
    return masses


def upper_event_mass(probability: float | np.ndarray, gamma: float) -> float | np.ndarray:
    """Largest Q(A) when gamma^-1 <= dQ/dP <= gamma and P(A) is known.

    The two branches respectively come from the upper density-ratio cap on A
    and the lower cap on its complement.  Their minimum is attainable by a
    two-level reweighting, so this is not a relaxation.
    """

    gamma = _validate_gamma(gamma)
    probability = np.asarray(probability, dtype=np.float64)
    return np.minimum(gamma * probability, 1.0 - (1.0 - probability) / gamma)


def lower_event_mass(probability: float | np.ndarray, gamma: float) -> float | np.ndarray:
    """Smallest Q(A) when gamma^-1 <= dQ/dP <= gamma and P(A) is known."""

    gamma = _validate_gamma(gamma)
    probability = np.asarray(probability, dtype=np.float64)
    return np.maximum(probability / gamma, 1.0 - gamma * (1.0 - probability))


def _envelope_from_subset_masses(
    group_0_masses: np.ndarray,
    group_1_masses: np.ndarray,
    *,
    gamma: float,
    method: str,
    radius_group_0: float | None = None,
    radius_group_1: float | None = None,
) -> TVEnvelope:
    scores = upper_event_mass(group_1_masses, gamma) - lower_event_mass(
        group_0_masses, gamma
    )
    mask = int(np.argmax(scores))
    value = float(np.clip(scores[mask], 0.0, 1.0))
    token_count = int(round(log2(group_0_masses.size)))
    return TVEnvelope(
        value=value,
        gamma=float(gamma),
        maximizing_mask=mask,
        token_count=token_count,
        method=method,
        l1_radius_group_0=radius_group_0,
        l1_radius_group_1=radius_group_1,
    )


def _l1_event_upper(
    empirical_mass: float, l1_radius: float, event_size: int, token_count: int
) -> float:
    if event_size == 0:
        return 0.0
    if event_size == token_count:
        return 1.0
    return min(1.0, empirical_mass + l1_radius / 2.0)


def _l1_event_lower(
    empirical_mass: float, l1_radius: float, event_size: int, token_count: int
) -> float:
    if event_size == 0:
        return 0.0
    if event_size == token_count:
        return 1.0
    return max(0.0, empirical_mass - l1_radius / 2.0)


def robust_total_variation(
    group_0: Iterable[float], group_1: Iterable[float], *, gamma: float
) -> TVEnvelope:
    """Exactly maximize TV over independently likelihood-ratio-shifted groups.

    Let ``p_s`` be a reference token law for group ``s`` and let

        W_gamma(p_s) = {q_s: gamma^-1 p_s(c) <= q_s(c) <= gamma p_s(c)}.

    This returns ``sup_{q_0 in W(p_0), q_1 in W(p_1)} TV(q_0, q_1)``.  The
    enumeration is exact for at most 20 tokens and has complexity O(2^K).
    """

    p0 = _probability_vector(group_0)
    p1 = _probability_vector(group_1)
    if p0.size != p1.size:
        raise ValueError("group distributions must share the same token alphabet")
    return _envelope_from_subset_masses(
        _subset_masses(p0),
        _subset_masses(p1),
        gamma=gamma,
        method="exact_finite_token_likelihood_ratio_envelope",
    )


def weissman_l1_radius(
    n: int, token_count: int, failure_probability: float
) -> float:
    """Distribution-free multinomial L1 radius from the Weissman inequality.

    For K >= 2, P(||p - p_hat||_1 > eps) <= (2^K - 2) exp(-n eps^2 / 2).
    The returned radius is clipped at two, the diameter of the simplex.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if token_count < 2 or token_count > MAX_EXACT_TOKENS:
        raise ValueError(f"token_count must lie in [2, {MAX_EXACT_TOKENS}]")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    log_prefactor = log((1 << token_count) - 2)
    return min(2.0, sqrt(2.0 * (log_prefactor - log(failure_probability)) / n))


def robust_total_variation_confidence_bound(
    group_0_empirical: Iterable[float],
    group_1_empirical: Iterable[float],
    *,
    gamma: float,
    l1_radius_group_0: float,
    l1_radius_group_1: float,
) -> TVEnvelope:
    """Upper-bound the robust TV envelope for all laws in two L1 confidence sets.

    This function is deterministic.  If the supplied radii cover the two true
    token laws, its result covers the true robust TV at every gamma >= 1.  It
    is the exact supremum over the two stated L1 balls, not a plug-in estimate.
    """

    p0 = _probability_vector(group_0_empirical)
    p1 = _probability_vector(group_1_empirical)
    if p0.size != p1.size:
        raise ValueError("group distributions must share the same token alphabet")
    if l1_radius_group_0 < 0.0 or l1_radius_group_1 < 0.0:
        raise ValueError("L1 radii must be non-negative")
    masses_0 = np.maximum(0.0, _subset_masses(p0) - l1_radius_group_0 / 2.0)
    masses_1 = np.minimum(1.0, _subset_masses(p1) + l1_radius_group_1 / 2.0)
    # Empty and full events have known probability zero and one.  Treating
    # them as ordinary L1-ball events is valid but needlessly conservative and
    # would make the conditional-set exactness statement false at the edges.
    masses_0[0] = 0.0
    masses_1[0] = 0.0
    masses_0[-1] = 1.0
    masses_1[-1] = 1.0
    return _envelope_from_subset_masses(
        masses_0,
        masses_1,
        gamma=gamma,
        method="exact_l1_confidence_likelihood_ratio_envelope",
        radius_group_0=float(l1_radius_group_0),
        radius_group_1=float(l1_radius_group_1),
    )


def robust_multiclass_attacker_accuracy(
    source_distributions: Sequence[Sequence[float]], *, gamma: float
) -> MulticlassAttackerEnvelope:
    """Exactly cover every attacker predicting one of G sources from K tokens.

    Balanced source accuracy gives each source prior mass 1/G.  For a fixed
    attacker assignment ``a: [K] -> [G]``, source ``s`` is correctly predicted
    on the event ``{c: a(c)=s}``.  Independent conditional likelihood-ratio
    shifts make the exact robust score the sum of the attainable upper event
    masses.  Enumerating all G^K assignments is exact and intentionally capped.
    """

    probabilities = _probability_matrix(source_distributions)
    source_count, token_count = probabilities.shape
    assignment_count = source_count**token_count
    if assignment_count > MAX_EXACT_ASSIGNMENTS:
        raise ValueError(
            f"exact multiclass enumeration requires {assignment_count:,} assignments; "
            f"cap is {MAX_EXACT_ASSIGNMENTS:,}"
        )
    best_score = -1.0
    best_assignment: tuple[int, ...] = ()
    for assignment in product(range(source_count), repeat=token_count):
        score = 0.0
        assignment_array = np.asarray(assignment)
        for source in range(source_count):
            event = assignment_array == source
            mass = float(probabilities[source, event].sum())
            score += float(upper_event_mass(mass, gamma))
        score /= source_count
        if score > best_score:
            best_score = score
            best_assignment = tuple(int(value) for value in assignment)
    return MulticlassAttackerEnvelope(
        balanced_accuracy=float(np.clip(best_score, 1.0 / source_count, 1.0)),
        gamma=float(gamma),
        maximizing_assignment=best_assignment,
        source_count=source_count,
        token_count=token_count,
        method="exact_multiclass_finite_token_likelihood_ratio_envelope",
    )


def robust_multiclass_attacker_confidence_bound(
    source_empirical_distributions: Sequence[Sequence[float]],
    *,
    gamma: float,
    l1_radii: Sequence[float],
) -> MulticlassAttackerEnvelope:
    """Exact robust multiclass envelope over stated multinomial L1 balls."""

    probabilities = _probability_matrix(source_empirical_distributions)
    source_count, token_count = probabilities.shape
    radii = tuple(float(radius) for radius in l1_radii)
    if len(radii) != source_count or any(radius < 0.0 for radius in radii):
        raise ValueError("provide one non-negative L1 radius per source")
    assignment_count = source_count**token_count
    if assignment_count > MAX_EXACT_ASSIGNMENTS:
        raise ValueError(
            f"exact multiclass enumeration requires {assignment_count:,} assignments; "
            f"cap is {MAX_EXACT_ASSIGNMENTS:,}"
        )
    best_score = -1.0
    best_assignment: tuple[int, ...] = ()
    for assignment in product(range(source_count), repeat=token_count):
        score = 0.0
        assignment_array = np.asarray(assignment)
        for source in range(source_count):
            event = assignment_array == source
            event_size = int(event.sum())
            empirical_mass = float(probabilities[source, event].sum())
            population_upper = _l1_event_upper(
                empirical_mass, radii[source], event_size, token_count
            )
            score += float(upper_event_mass(population_upper, gamma))
        score /= source_count
        if score > best_score:
            best_score = score
            best_assignment = tuple(int(value) for value in assignment)
    return MulticlassAttackerEnvelope(
        balanced_accuracy=float(np.clip(best_score, 1.0 / source_count, 1.0)),
        gamma=float(gamma),
        maximizing_assignment=best_assignment,
        source_count=source_count,
        token_count=token_count,
        method="exact_multiclass_l1_confidence_likelihood_ratio_envelope",
        l1_radii=radii,
    )


def multinomial_robust_tv_certificate(
    group_0_counts: Sequence[int],
    group_1_counts: Sequence[int],
    *,
    gamma: float,
    failure_probability: float,
) -> TVEnvelope:
    """Certify robust universal-attacker advantage from two token histograms.

    Failure probability is split evenly between the two group-conditional
    multinomial confidence events.  A caller handling many source/label strata
    must allocate that familywise error budget before calling this function.
    """

    counts_0 = np.asarray(tuple(group_0_counts), dtype=np.int64)
    counts_1 = np.asarray(tuple(group_1_counts), dtype=np.int64)
    if counts_0.ndim != 1 or counts_1.ndim != 1 or counts_0.size != counts_1.size:
        raise ValueError("count vectors must be one-dimensional and equally sized")
    if counts_0.size < 2 or np.any(counts_0 < 0) or np.any(counts_1 < 0):
        raise ValueError("counts must be non-negative over at least two tokens")
    n0, n1 = int(counts_0.sum()), int(counts_1.sum())
    if n0 == 0 or n1 == 0:
        raise ValueError("both group-conditional samples require positive support")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    token_count = int(counts_0.size)
    radius_0 = weissman_l1_radius(n0, token_count, failure_probability / 2.0)
    radius_1 = weissman_l1_radius(n1, token_count, failure_probability / 2.0)
    return robust_total_variation_confidence_bound(
        counts_0 / n0,
        counts_1 / n1,
        gamma=gamma,
        l1_radius_group_0=radius_0,
        l1_radius_group_1=radius_1,
    )


def coarsen_distribution(
    distribution: Iterable[float], mapping: Sequence[int]
) -> np.ndarray:
    """Push a fine-token distribution through a deterministic token coarsening."""

    p = _probability_vector(distribution)
    mapping_array = np.asarray(tuple(mapping), dtype=np.int64)
    if mapping_array.shape != p.shape or np.any(mapping_array < 0):
        raise ValueError("mapping must provide one non-negative coarse token per fine token")
    coarse_count = int(mapping_array.max()) + 1
    if coarse_count < 2:
        raise ValueError("a coarsening must retain at least two coarse tokens")
    if coarse_count > MAX_EXACT_TOKENS:
        raise ValueError("too many coarse tokens for exact enumeration")
    return np.bincount(mapping_array, weights=p, minlength=coarse_count)


def coarsened_confidence_certificate(
    fine_group_0_empirical: Iterable[float],
    fine_group_1_empirical: Iterable[float],
    mapping: Sequence[int],
    *,
    gamma: float,
    fine_l1_radius_group_0: float,
    fine_l1_radius_group_1: float,
) -> TVEnvelope:
    """Certify a coarsening with the *fine* confidence event and no new alpha.

    Deterministic coarsening contracts L1 distance.  Consequently one event
    controlling the fine histogram controls every coarsening, including a
    coarsening selected after inspecting the certificate histogram.  This is
    MOSAIC's post-selection closure; it is deliberately not a claim about
    arbitrary re-training on the certification fold.
    """

    return robust_total_variation_confidence_bound(
        coarsen_distribution(fine_group_0_empirical, mapping),
        coarsen_distribution(fine_group_1_empirical, mapping),
        gamma=gamma,
        l1_radius_group_0=fine_l1_radius_group_0,
        l1_radius_group_1=fine_l1_radius_group_1,
    )


def coarsened_multiclass_confidence_certificate(
    fine_source_empirical_distributions: Sequence[Sequence[float]],
    mapping: Sequence[int],
    *,
    gamma: float,
    fine_l1_radii: Sequence[float],
) -> MulticlassAttackerEnvelope:
    """Certify all-source recovery after a data-selected token coarsening."""

    coarse = tuple(
        coarsen_distribution(distribution, mapping)
        for distribution in fine_source_empirical_distributions
    )
    return robust_multiclass_attacker_confidence_bound(
        coarse,
        gamma=gamma,
        l1_radii=fine_l1_radii,
    )


def robust_selected_rule_error_bound(
    empirical_distribution: Sequence[float],
    error_tokens: Sequence[bool | int],
    *,
    gamma: float,
    l1_radius: float,
) -> float:
    """Uniform robust error bound for any token rule selected from one L1 event.

    ``error_tokens[c]`` says whether the selected token classifier errs on fine
    token ``c`` in the current source/label stratum.  Since the L1 confidence
    event controls every token subset at once, this remains valid even when the
    coarsening and token rule are selected after inspecting the histogram.
    """

    distribution = _probability_vector(empirical_distribution)
    errors = np.asarray(tuple(error_tokens), dtype=bool)
    if errors.shape != distribution.shape:
        raise ValueError("error_tokens must provide one indicator per fine token")
    if l1_radius < 0.0:
        raise ValueError("l1_radius must be non-negative")
    event_size = int(errors.sum())
    empirical_mass = float(distribution[errors].sum())
    population_upper = _l1_event_upper(
        empirical_mass, l1_radius, event_size, distribution.size
    )
    return float(upper_event_mass(population_upper, gamma))


def robust_binary_event_upper_bound(
    successes: int,
    n: int,
    *,
    gamma: float,
    failure_probability: float,
) -> float:
    """Distribution-free robust upper bound for a frozen binary event.

    This is intended for a fixed diagnostic error event.  It is separate from
    the universal-attacker theorem, which needs a token histogram because the
    attacker can choose any token subset.
    """

    if n <= 0 or successes < 0 or successes > n:
        raise ValueError("successes must lie in [0, n] with positive n")
    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    empirical_upper = min(
        1.0, successes / n + sqrt(log(1.0 / failure_probability) / (2.0 * n))
    )
    return float(upper_event_mass(empirical_upper, gamma))
