"""Global optimizer for transform-exact MOSAIC certificates.

For a fixed finite decoder, every term in the exact structured-shift envelope
has a linear epigraph.  Enumerating the finite decoder family and solving one
linear program per decoder therefore gives a global solution for the stated
finite alphabets.  Returned solutions are independently re-evaluated by the
non-optimization certificate implementation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np
from scipy.optimize import milp

from mosaic_channel import MAX_EXACT_ASSIGNMENTS
from mosaic_optimizer import (
    GLOBAL_MIP_GAP_TOLERANCE,
    MAX_DECODER_ASSIGNMENTS,
    MIP_FEASIBILITY_TOLERANCE,
    POSTHOC_TOLERANCE,
    _MILPBuilder,
    _add_term,
    _constraint_violation,
    _validated_problem,
)
from mosaic_transform_exact import (
    TransformExactAttackerCertificate,
    TransformExactUtilityCertificate,
    transform_exact_attacker_confidence_bound,
    transform_exact_utility_confidence_bound,
)


@dataclass(frozen=True)
class TransformExactChannelSolution:
    """Globally optimized channel with independently recomputed certificates."""

    release_channel: np.ndarray
    decoder: tuple[int, ...]
    certified_worst_conditional_error: float
    solver_objective: float
    privacy_certificates: tuple[TransformExactAttackerCertificate, ...]
    utility_certificates: tuple[tuple[TransformExactUtilityCertificate, ...], ...]
    solved_decoder_assignments: int
    source_count: int
    label_count: int
    fine_token_count: int
    released_token_count: int
    solver_status: str
    solver_mip_gap: float
    solver_dual_bound: float
    max_constraint_violation: float
    active_attacker_assignments: int
    constraint_generation_iterations: int
    method: str


def _add_transformed_score(
    expression: dict[int, float],
    channel_indices: np.ndarray,
    transform: np.ndarray,
    fine_token: int,
    output_weights: np.ndarray,
    coefficient: float = 1.0,
) -> None:
    for transformed_token, transform_weight in enumerate(transform[fine_token]):
        if abs(float(transform_weight)) <= 1e-15:
            continue
        for output, output_weight in enumerate(output_weights):
            _add_term(
                expression,
                int(channel_indices[transformed_token, output]),
                coefficient * float(transform_weight) * float(output_weight),
            )


def _add_untransformed_score(
    expression: dict[int, float],
    channel_indices: np.ndarray,
    fine_token: int,
    output_weights: np.ndarray,
    coefficient: float = 1.0,
) -> None:
    for output, output_weight in enumerate(output_weights):
        _add_term(
            expression,
            int(channel_indices[fine_token, output]),
            coefficient * float(output_weight),
        )


def _add_transformed_l1_support_epigraph(
    builder: _MILPBuilder,
    channel_indices: np.ndarray,
    empirical: np.ndarray,
    radius: float,
    transform: np.ndarray,
    output_weights: np.ndarray,
) -> int:
    """Add the exact support dual for scores T M w."""

    fine_count = empirical.size
    phi = builder.add_variable(lower=0.0, upper=1.0)
    nu_positive = builder.add_variable(lower=0.0)
    nu_negative = builder.add_variable(lower=0.0)
    lam = builder.add_variable(lower=0.0)
    theta = np.asarray(
        [builder.add_variable(lower=-np.inf, upper=np.inf) for _ in range(fine_count)]
    )

    for token in range(fine_count):
        expression: dict[int, float] = {
            nu_positive: -1.0,
            nu_negative: 1.0,
            int(theta[token]): -1.0,
        }
        _add_transformed_score(
            expression,
            channel_indices,
            transform,
            token,
            output_weights,
        )
        builder.add_upper(expression, 0.0)
        builder.add_upper({int(theta[token]): 1.0, lam: -1.0}, 0.0)
        builder.add_upper({int(theta[token]): -1.0, lam: -1.0}, 0.0)

    expression = {
        nu_positive: 1.0,
        nu_negative: -1.0,
        lam: float(radius),
        phi: -1.0,
    }
    for token, probability in enumerate(empirical):
        _add_term(expression, int(theta[token]), float(probability))
    builder.add_upper(expression, 0.0)
    return phi


def _add_row_max_epigraph(
    builder: _MILPBuilder,
    channel_indices: np.ndarray,
    output_weights: np.ndarray,
) -> int:
    maximum = builder.add_variable(lower=0.0, upper=1.0)
    for token in range(channel_indices.shape[0]):
        expression = {maximum: -1.0}
        _add_untransformed_score(
            expression, channel_indices, token, output_weights
        )
        builder.add_upper(expression, 0.0)
    return maximum


def _build_exact_decoder_problem(
    empirical: np.ndarray,
    radii: np.ndarray,
    libraries: tuple[tuple[np.ndarray, ...], ...],
    eta: np.ndarray,
    thresholds: np.ndarray,
    *,
    released_count: int,
    decoder: tuple[int, ...],
    attacker_assignments: Sequence[Sequence[int]] | None = None,
) -> tuple[_MILPBuilder, np.ndarray]:
    label_count, source_count, fine_count = empirical.shape
    builder = _MILPBuilder()
    channel_indices = np.asarray(
        [
            [builder.add_variable(lower=0.0, upper=1.0) for _ in range(released_count)]
            for _ in range(fine_count)
        ]
    )
    for token in range(fine_count):
        builder.add_equality(
            {int(channel_indices[token, output]): 1.0 for output in range(released_count)},
            1.0,
        )

    if attacker_assignments is None:
        assignments = tuple(product(range(source_count), repeat=released_count))
    else:
        assignments = tuple(
            dict.fromkeys(
                tuple(int(value) for value in assignment)
                for assignment in attacker_assignments
            )
        )
        if not assignments:
            raise ValueError("attacker_assignments cannot be empty")
        for assignment in assignments:
            if len(assignment) != released_count:
                raise ValueError("each attacker assignment needs one source per output")
            if any(value < 0 or value >= source_count for value in assignment):
                raise ValueError("attacker source labels are out of range")
    chance = 1.0 / source_count
    privacy_ba_thresholds = chance + (1.0 - chance) * thresholds
    for label in range(label_count):
        for assignment in assignments:
            assignment_array = np.asarray(assignment, dtype=np.int64)
            residual_bounds = []
            output_weights_by_source = []
            for source in range(source_count):
                output_weights = (assignment_array == source).astype(np.float64)
                output_weights_by_source.append(output_weights)
                residual_bounds.append(
                    _add_row_max_epigraph(builder, channel_indices, output_weights)
                )
            for transform in libraries[label]:
                expression: dict[int, float] = {}
                for source in range(source_count):
                    support = _add_transformed_l1_support_epigraph(
                        builder,
                        channel_indices,
                        empirical[label, source],
                        float(radii[label, source]),
                        transform,
                        output_weights_by_source[source],
                    )
                    _add_term(
                        expression,
                        support,
                        (1.0 - float(eta[label])) / source_count,
                    )
                    _add_term(
                        expression,
                        residual_bounds[source],
                        float(eta[label]) / source_count,
                    )
                builder.add_upper(expression, float(privacy_ba_thresholds[label]))

    worst_error = builder.add_variable(lower=0.0, upper=1.0, objective=1.0)
    decoder_array = np.asarray(decoder, dtype=np.int64)
    for label in range(label_count):
        loss = (decoder_array != label).astype(np.float64)
        residual_capacity = _add_row_max_epigraph(builder, channel_indices, loss)
        for source in range(source_count):
            for transform in libraries[label]:
                support = _add_transformed_l1_support_epigraph(
                    builder,
                    channel_indices,
                    empirical[label, source],
                    float(radii[label, source]),
                    transform,
                    loss,
                )
                expression = {
                    support: 1.0 - float(eta[label]),
                    residual_capacity: float(eta[label]),
                    worst_error: -1.0,
                }
                builder.add_upper(expression, 0.0)
    return builder, channel_indices


def optimize_transform_exact_channel(
    empirical_distributions: Sequence[Sequence[Sequence[float]]],
    *,
    l1_radii: Sequence[Sequence[float]],
    common_channels_by_label: Sequence[Sequence[Sequence[Sequence[float]]]],
    contaminations: Sequence[float],
    privacy_advantage_thresholds: Sequence[float],
    released_token_count: int,
    decoder_candidates: Sequence[Sequence[int]] | None = None,
    maximum_worst_conditional_error: float | None = None,
    solver_time_limit_seconds: float | None = None,
    attacker_constraint_generation: bool = False,
) -> TransformExactChannelSolution:
    """Globally optimize the exact finite structured-shift certificate."""

    empirical, radii, libraries, eta, thresholds = _validated_problem(
        empirical_distributions,
        l1_radii,
        common_channels_by_label,
        contaminations,
        privacy_advantage_thresholds,
    )
    label_count, source_count, fine_count = empirical.shape
    if released_token_count < 2:
        raise ValueError("released_token_count must be at least two")
    if source_count**released_token_count > MAX_EXACT_ASSIGNMENTS:
        raise ValueError("attacker-assignment family exceeds the exact cap")

    if decoder_candidates is None:
        decoder_count = label_count**released_token_count
        if decoder_count > MAX_DECODER_ASSIGNMENTS:
            raise ValueError("decoder family exceeds the exact enumeration cap")
        decoders = tuple(product(range(label_count), repeat=released_token_count))
    else:
        decoders = tuple(tuple(int(value) for value in decoder) for decoder in decoder_candidates)
        if not decoders:
            raise ValueError("decoder_candidates cannot be empty")
        for decoder in decoders:
            if len(decoder) != released_token_count:
                raise ValueError("each decoder needs one label per released token")
            if any(value < 0 or value >= label_count for value in decoder):
                raise ValueError("decoder labels are out of range")

    if maximum_worst_conditional_error is not None and not (
        np.isfinite(maximum_worst_conditional_error)
        and 0.0 <= maximum_worst_conditional_error <= 1.0
    ):
        raise ValueError("maximum_worst_conditional_error must lie in [0, 1]")
    if solver_time_limit_seconds is not None and not (
        np.isfinite(solver_time_limit_seconds) and solver_time_limit_seconds > 0.0
    ):
        raise ValueError("solver time limit must be positive")

    best: dict[str, object] | None = None
    messages = []
    for decoder in decoders:
        active_assignments: set[tuple[int, ...]] | None
        if attacker_constraint_generation:
            active_assignments = {
                tuple([source] * released_token_count)
                for source in range(source_count)
            }
        else:
            active_assignments = None
        generation_iteration = 0
        while True:
            generation_iteration += 1
            builder, channel_indices = _build_exact_decoder_problem(
                empirical,
                radii,
                libraries,
                eta,
                thresholds,
                released_count=released_token_count,
                decoder=decoder,
                attacker_assignments=(
                    sorted(active_assignments)
                    if active_assignments is not None
                    else None
                ),
            )
            objective, bounds, constraints = builder.matrices()
            options: dict[str, float] = {
                "mip_rel_gap": 0.0,
                "mip_feasibility_tolerance": MIP_FEASIBILITY_TOLERANCE,
            }
            if solver_time_limit_seconds is not None:
                options["time_limit"] = float(solver_time_limit_seconds)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Unrecognized options detected.*mip_feasibility_tolerance",
                    category=RuntimeWarning,
                )
                result = milp(
                    objective,
                    integrality=np.zeros(len(objective), dtype=np.int32),
                    bounds=bounds,
                    constraints=constraints,
                    options=options,
                )
            messages.append(str(result.message))
            if not result.success or result.x is None or result.fun is None:
                break
            values = np.asarray(result.x, dtype=np.float64)
            channel = np.clip(values[channel_indices], 0.0, 1.0)
            channel /= channel.sum(axis=1, keepdims=True)
            privacy = tuple(
                transform_exact_attacker_confidence_bound(
                    empirical[label],
                    channel,
                    l1_radii=radii[label],
                    common_fine_token_channels=libraries[label],
                    contamination=float(eta[label]),
                )
                for label in range(label_count)
            )
            violating_assignments = {
                certificate.maximizing_assignment
                for label, certificate in enumerate(privacy)
                if certificate.normalized_advantage
                > float(thresholds[label]) + POSTHOC_TOLERANCE
            }
            if violating_assignments and active_assignments is not None:
                new_assignments = violating_assignments - active_assignments
                if not new_assignments:
                    raise RuntimeError("POSTHOC_EXACT_PRIVACY_MISMATCH")
                active_assignments.update(new_assignments)
                continue
            if violating_assignments:
                raise RuntimeError("POSTHOC_EXACT_PRIVACY_MISMATCH")
            break
        if not result.success or result.x is None or result.fun is None:
            continue
        raw_gap = getattr(result, "mip_gap", None)
        raw_dual_bound = getattr(result, "mip_dual_bound", None)
        # HiGHS omits MIP-only fields when every integrality flag is zero.  A
        # successful status-zero LP solve is already a global optimum.
        gap = 0.0 if raw_gap is None else float(raw_gap)
        dual_bound = float(result.fun) if raw_dual_bound is None else float(raw_dual_bound)
        if not np.isfinite(gap) or gap > GLOBAL_MIP_GAP_TOLERANCE:
            messages.append(f"rejected non-global solution with gap {gap}")
            continue

        utility = tuple(
            tuple(
                transform_exact_utility_confidence_bound(
                    empirical[label, source],
                    channel,
                    decoder,
                    true_label=label,
                    l1_radius=float(radii[label, source]),
                    common_fine_token_channels=libraries[label],
                    contamination=float(eta[label]),
                )
                for source in range(source_count)
            )
            for label in range(label_count)
        )
        certified_error = max(
            certificate.error_probability
            for label_certificates in utility
            for certificate in label_certificates
        )
        violation = _constraint_violation(values, constraints, bounds)
        if violation > POSTHOC_TOLERANCE:
            raise RuntimeError("POSTHOC_EXACT_CONSTRAINT_MISMATCH")
        if abs(float(result.fun) - certified_error) > POSTHOC_TOLERANCE:
            raise RuntimeError("POSTHOC_EXACT_OBJECTIVE_MISMATCH")

        candidate: dict[str, object] = {
            "channel": channel,
            "decoder": decoder,
            "objective": float(result.fun),
            "certified_error": certified_error,
            "privacy": privacy,
            "utility": utility,
            "message": str(result.message),
            "gap": gap,
            "dual_bound": dual_bound,
            "violation": violation,
            "active_assignments": (
                source_count**released_token_count
                if active_assignments is None
                else len(active_assignments)
            ),
            "generation_iterations": generation_iteration,
        }
        if best is None or float(candidate["objective"]) < float(best["objective"]) - 1e-10:
            best = candidate

    if best is None:
        raise RuntimeError(
            "ABSTAIN_NO_FEASIBLE_EXACT_CHANNEL: " + "; ".join(dict.fromkeys(messages))
        )
    if (
        maximum_worst_conditional_error is not None
        and float(best["certified_error"])
        > maximum_worst_conditional_error + POSTHOC_TOLERANCE
    ):
        raise RuntimeError("ABSTAIN_EXACT_UTILITY_CONTRACT")

    return TransformExactChannelSolution(
        release_channel=np.asarray(best["channel"]),
        decoder=tuple(best["decoder"]),
        certified_worst_conditional_error=float(best["certified_error"]),
        solver_objective=float(best["objective"]),
        privacy_certificates=tuple(best["privacy"]),
        utility_certificates=tuple(best["utility"]),
        solved_decoder_assignments=len(decoders),
        source_count=source_count,
        label_count=label_count,
        fine_token_count=fine_count,
        released_token_count=released_token_count,
        solver_status=str(best["message"]),
        solver_mip_gap=float(best["gap"]),
        solver_dual_bound=float(best["dual_bound"]),
        max_constraint_violation=float(best["violation"]),
        active_attacker_assignments=int(best["active_assignments"]),
        constraint_generation_iterations=int(best["generation_iterations"]),
        method=(
            "global_decoder_enumeration_constraint_generated_transform_exact_lp"
            if attacker_constraint_generation
            else "global_decoder_enumeration_transform_exact_lp"
        ),
    )
