"""Exact and scalable certificates for bounded multi-item release transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import product
from operator import mul
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TranscriptCapacityCertificate:
    item_dobrushin_coefficients: tuple[float, ...]
    exact_joint_dobrushin_coefficient: float
    multiplicative_upper_bound: float
    fine_tuple_count: int
    released_tuple_count: int


def _channel(value: Sequence[Sequence[float]]) -> np.ndarray:
    channel = np.asarray(value, dtype=np.float64)
    if channel.ndim != 2 or min(channel.shape) < 2:
        raise ValueError("each channel must be a matrix with at least two rows and columns")
    if np.any(channel < -1e-12) or not np.allclose(channel.sum(axis=1), 1.0):
        raise ValueError("channels must be row-stochastic")
    return channel


def product_release_channel(
    channels: Sequence[Sequence[Sequence[float]]],
) -> np.ndarray:
    """Compile independent item randomizers into one exact transcript channel."""

    matrices = tuple(_channel(channel) for channel in channels)
    if not matrices:
        raise ValueError("at least one item channel is required")
    fine_shapes = tuple(matrix.shape[0] for matrix in matrices)
    output_shapes = tuple(matrix.shape[1] for matrix in matrices)
    joint = np.empty(
        (reduce(mul, fine_shapes, 1), reduce(mul, output_shapes, 1)),
        dtype=np.float64,
    )
    for row_index, fine_tuple in enumerate(product(*map(range, fine_shapes))):
        for column_index, output_tuple in enumerate(
            product(*map(range, output_shapes))
        ):
            joint[row_index, column_index] = float(
                np.prod(
                    [
                        matrix[fine, output]
                        for matrix, fine, output in zip(
                            matrices, fine_tuple, output_tuple, strict=True
                        )
                    ]
                )
            )
    return joint


def dobrushin_coefficient(
    channel: Sequence[Sequence[float]],
) -> float:
    """Maximum total-variation distance between two channel rows."""

    matrix = _channel(channel)
    largest = 0.0
    for first in range(matrix.shape[0]):
        for second in range(first + 1, matrix.shape[0]):
            largest = max(
                largest,
                0.5 * float(np.abs(matrix[first] - matrix[second]).sum()),
            )
    return largest


def multiplicative_session_capacity(
    channels: Sequence[Sequence[Sequence[float]]],
) -> float:
    """Upper bound ``1-product_i(1-alpha_i)`` for a bounded transcript."""

    alphas = [dobrushin_coefficient(channel) for channel in channels]
    return float(1.0 - np.prod([1.0 - value for value in alphas]))


def certify_product_transcript(
    channels: Sequence[Sequence[Sequence[float]]],
) -> TranscriptCapacityCertificate:
    matrices = tuple(_channel(channel) for channel in channels)
    joint = product_release_channel(matrices)
    alphas = tuple(dobrushin_coefficient(channel) for channel in matrices)
    return TranscriptCapacityCertificate(
        item_dobrushin_coefficients=alphas,
        exact_joint_dobrushin_coefficient=dobrushin_coefficient(joint),
        multiplicative_upper_bound=float(
            1.0 - np.prod([1.0 - value for value in alphas])
        ),
        fine_tuple_count=joint.shape[0],
        released_tuple_count=joint.shape[1],
    )
