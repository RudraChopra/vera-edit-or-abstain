import numpy as np

from mosaic_transcript import (
    certify_product_transcript,
    product_release_channel,
)


def test_product_channel_is_row_stochastic() -> None:
    channel = np.asarray([[0.8, 0.2], [0.3, 0.7]])
    joint = product_release_channel([channel, channel])
    assert joint.shape == (4, 4)
    assert np.allclose(joint.sum(axis=1), 1.0)


def test_multiplicative_capacity_dominates_exact_capacity() -> None:
    first = np.asarray([[0.9, 0.1], [0.4, 0.6]])
    second = np.asarray([[0.7, 0.3], [0.5, 0.5]])
    result = certify_product_transcript([first, second])
    assert (
        result.exact_joint_dobrushin_coefficient
        <= result.multiplicative_upper_bound + 1e-12
    )
