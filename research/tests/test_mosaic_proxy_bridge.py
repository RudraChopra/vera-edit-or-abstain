from __future__ import annotations

import numpy as np

from mosaic_proxy_bridge import certify_proxy_label_conditionals


def _counts() -> np.ndarray:
    return np.asarray(
        [
            [[1800, 900, 250, 50], [150, 350, 900, 1600]],
            [[1500, 1000, 400, 100], [100, 300, 1100, 1500]],
        ],
        dtype=np.int64,
    )


def _proxy_counts(true_counts: np.ndarray, confusion: np.ndarray) -> np.ndarray:
    labels, sources, tokens = true_counts.shape
    proxy = np.zeros((labels, sources, tokens), dtype=np.int64)
    for label in range(labels):
        for source in range(sources):
            for token in range(tokens):
                count = int(true_counts[label, source, token])
                first = int(round(count * confusion[source, 0]))
                proxy[label, 0, token] += first
                proxy[label, 1, token] += count - first
    return proxy


def test_known_proxy_certificate_covers_constructed_true_conditionals() -> None:
    true_counts = _counts()
    confusion = np.asarray([[0.95, 0.05], [0.05, 0.95]])
    proxy = _proxy_counts(true_counts, confusion)
    certificate = certify_proxy_label_conditionals(
        proxy,
        family_failure_probability=0.05,
        known_confusion_matrix=confusion,
    )
    true_conditionals = true_counts / true_counts.sum(axis=2, keepdims=True)
    distances = np.abs(
        certificate.conditional_empirical_distributions - true_conditionals
    ).sum(axis=2)
    assert np.all(distances <= certificate.conditional_l1_radii + 2e-8)
    assert certificate.calibration_sample_size == 0
    assert certificate.calibration_mode == "known_confusion_matrix"
    assert np.all(certificate.conditional_l1_radii < 1.0)


def test_pooled_calibration_adds_row_uncertainty_and_counts_labels() -> None:
    true_counts = _counts()
    confusion = np.asarray([[0.95, 0.05], [0.05, 0.95]])
    proxy = _proxy_counts(true_counts, confusion)
    calibration = np.asarray([[1900, 100], [100, 1900]], dtype=np.int64)
    known = certify_proxy_label_conditionals(
        proxy,
        family_failure_probability=0.05,
        known_confusion_matrix=confusion,
    )
    estimated = certify_proxy_label_conditionals(
        proxy,
        family_failure_probability=0.05,
        calibration_confusion_counts=calibration,
    )
    assert estimated.calibration_sample_size == 4000
    assert estimated.calibration_mode == "pooled_confusion_calibration"
    assert np.all(
        estimated.conditional_l1_radii >= known.conditional_l1_radii - 2e-8
    )


def test_uninformative_proxy_fails_closed_with_radius_two() -> None:
    true_counts = _counts()
    confusion = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    proxy = _proxy_counts(true_counts, confusion)
    certificate = certify_proxy_label_conditionals(
        proxy,
        family_failure_probability=0.05,
        known_confusion_matrix=confusion,
    )
    assert np.allclose(certificate.conditional_l1_radii, 2.0)


def test_label_specific_calibration_is_supported() -> None:
    true_counts = _counts()
    confusion = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    proxy = _proxy_counts(true_counts, confusion)
    calibration = np.asarray(
        [
            [[900, 100], [100, 900]],
            [[900, 100], [100, 900]],
        ],
        dtype=np.int64,
    )
    certificate = certify_proxy_label_conditionals(
        proxy,
        family_failure_probability=0.05,
        calibration_confusion_counts=calibration,
    )
    assert certificate.calibration_mode == "label_specific_confusion_calibration"
    assert certificate.calibration_sample_size == 4000
    assert certificate.conditional_l1_radii.shape == (2, 2)


def test_binary_symmetric_calibration_uses_exact_pooled_binomial_interval() -> None:
    true_counts = _counts()
    confusion = np.asarray([[0.95, 0.05], [0.05, 0.95]])
    proxy = _proxy_counts(true_counts, confusion)
    calibration = np.asarray([[7600, 400], [400, 7600]], dtype=np.int64)
    generic = certify_proxy_label_conditionals(
        proxy,
        family_failure_probability=0.01,
        calibration_confusion_counts=calibration,
    )
    structured = certify_proxy_label_conditionals(
        proxy,
        family_failure_probability=0.01,
        calibration_confusion_counts=calibration,
        binary_symmetric_calibration=True,
    )
    assert structured.calibration_mode == (
        "pooled_binary_symmetric_exact_binomial_calibration"
    )
    assert structured.calibration_sample_size == 16000
    assert np.all(
        structured.conditional_l1_radii < generic.conditional_l1_radii
    )


def test_coordinate_region_is_valid_and_distinct_from_weissman() -> None:
    true_counts = _counts()
    confusion = np.asarray([[0.95, 0.05], [0.05, 0.95]])
    proxy = _proxy_counts(true_counts, confusion)
    coordinate = certify_proxy_label_conditionals(
        proxy,
        family_failure_probability=0.01,
        known_confusion_matrix=confusion,
        confidence_region="coordinate_clopper_pearson",
    )
    weissman = certify_proxy_label_conditionals(
        proxy,
        family_failure_probability=0.01,
        known_confusion_matrix=confusion,
        confidence_region="l1_weissman",
    )
    true_conditionals = true_counts / true_counts.sum(axis=2, keepdims=True)
    coordinate_distances = np.abs(
        coordinate.conditional_empirical_distributions - true_conditionals
    ).sum(axis=2)
    assert np.all(
        coordinate_distances <= coordinate.conditional_l1_radii + 2e-8
    )
    assert not np.allclose(
        coordinate.conditional_l1_radii, weissman.conditional_l1_radii
    )
    assert all(
        label.proxy_coordinate_lowers is not None for label in coordinate.labels
    )


def test_known_source_masses_tighten_the_proxy_conditional_envelope() -> None:
    true_counts = _counts()
    confusion = np.asarray([[0.95, 0.05], [0.05, 0.95]])
    proxy = _proxy_counts(true_counts, confusion)
    masses = true_counts.sum(axis=2)
    masses = masses / masses.sum(axis=1, keepdims=True)
    unknown_mass = certify_proxy_label_conditionals(
        proxy,
        family_failure_probability=0.01,
        known_confusion_matrix=confusion,
    )
    fixed_mass = certify_proxy_label_conditionals(
        proxy,
        family_failure_probability=0.01,
        known_confusion_matrix=confusion,
        known_source_masses=masses,
    )
    true_conditionals = true_counts / true_counts.sum(axis=2, keepdims=True)
    distances = np.abs(
        fixed_mass.conditional_empirical_distributions - true_conditionals
    ).sum(axis=2)
    assert np.all(distances <= fixed_mass.conditional_l1_radii + 2e-8)
    assert np.all(
        fixed_mass.conditional_l1_radii < unknown_mass.conditional_l1_radii
    )
