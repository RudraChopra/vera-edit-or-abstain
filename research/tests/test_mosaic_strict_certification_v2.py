from __future__ import annotations

import numpy as np

from mosaic_bridge import BridgeLabelCertificate
from mosaic_strict_certification_v2 import _strict_label


def _label(transform: np.ndarray, *, retained_mass: float = 0.8) -> BridgeLabelCertificate:
    return BridgeLabelCertificate(
        transform=transform,
        retained_mass=retained_mass,
        contamination=1.0 - retained_mass,
        optimal_retained_mass_upper=retained_mass,
        reference_l1_radii=(0.0, 0.0),
        bridge_l1_radii=(0.0, 0.0),
        bridge_coordinate_lowers=(),
        minimum_membership_slack=0.0,
        transform_trace=float(np.trace(transform)),
        solver_status="test",
        solver_iterations=0,
        method="test",
    )


def test_structurally_zero_output_does_not_destroy_retained_mass() -> None:
    transform = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.50, 0.25, 0.25, 0.0],
        ]
    )
    reference = np.asarray(
        [[0.40, 0.30, 0.20, 0.10], [0.10, 0.20, 0.30, 0.40]]
    )
    bridge = reference @ transform
    radii = np.zeros(2)
    repaired = _strict_label(
        _label(transform),
        reference,
        radii,
        bridge,
        radii,
        feasibility_guard=1e-9,
    )
    assert repaired.retained_mass > 0.79
    assert repaired.minimum_membership_slack >= 0.0
    assert np.all(repaired.transform[:, 3] == 0.0)


def test_active_zero_support_is_not_treated_as_structural_zero() -> None:
    transform = np.eye(2)
    reference = np.asarray([[1.0, 0.0], [0.5, 0.5]])
    bridge = reference.copy()
    radii = np.zeros(2)
    repaired = _strict_label(
        _label(transform),
        reference,
        radii,
        bridge,
        radii,
        feasibility_guard=1e-9,
    )
    assert repaired.retained_mass == 0.0
    assert repaired.minimum_membership_slack >= 0.0
