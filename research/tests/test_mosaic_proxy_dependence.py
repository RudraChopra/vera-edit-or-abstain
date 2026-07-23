import numpy as np

from mosaic_proxy_bridge import (
    certify_proxy_label_conditionals,
    certify_proxy_token_dependence,
)


def test_proxy_dependence_bound_is_finite_and_charged() -> None:
    counts = np.zeros((2, 2, 2, 2), dtype=int)
    counts[..., 0] = 80
    counts[..., 1] = 20
    counts[1, 1, 1] = [60, 40]
    dependence = certify_proxy_token_dependence(
        counts, family_failure_probability=0.02
    )
    assert not dependence.empty_calibration_cells
    assert 0.0 < max(dependence.per_label_l1_slack) < 2.0

    proxy_counts = np.full((2, 2, 2), 250, dtype=int)
    certificate = certify_proxy_label_conditionals(
        proxy_counts,
        family_failure_probability=0.02,
        calibration_confusion_counts=dependence.pooled_confusion_counts,
        observation_model_l1_slack=dependence.per_label_l1_slack,
    )
    for label, slack in zip(
        certificate.labels, dependence.per_label_l1_slack, strict=True
    ):
        assert label.observation_model_l1_slack == slack
        assert label.effective_proxy_l1_radius >= slack


def test_empty_proxy_dependence_cell_fails_closed() -> None:
    counts = np.ones((2, 2, 2, 2), dtype=int) * 10
    counts[0, 0, 1] = 0
    result = certify_proxy_token_dependence(
        counts, family_failure_probability=0.05
    )
    assert result.per_label_l1_slack[0] == 2.0
    assert (0, 0, 1) in result.empty_calibration_cells


def test_token_dependent_confusion_tensor_is_supported() -> None:
    proxy_counts = np.full((2, 2, 2), 300, dtype=int)
    calibration = np.zeros((2, 2, 2, 2), dtype=int)
    calibration[..., 0] = 180
    calibration[..., 1] = 20
    calibration[:, 1, :, 0] = 30
    calibration[:, 1, :, 1] = 170
    calibration[1, 0, 1] = [150, 50]
    result = certify_proxy_label_conditionals(
        proxy_counts,
        family_failure_probability=0.05,
        calibration_confusion_counts=calibration,
    )
    assert (
        result.calibration_mode
        == "label_and_token_specific_confusion_calibration"
    )
    assert result.labels[0].confusion_matrix.shape == (2, 2, 2)
    assert np.isfinite(result.conditional_l1_radii).all()
