from mosaic_lower_bound import (
    bernoulli_kl,
    binary_certification_sample_lower_bound,
    weissman_sample_upper_bound,
)


def test_lower_bound_increases_as_margin_shrinks() -> None:
    wide = binary_certification_sample_lower_bound(
        contract=0.2,
        margin=0.1,
        soundness_error=0.025,
        power_error=0.025,
    )
    narrow = binary_certification_sample_lower_bound(
        contract=0.2,
        margin=0.05,
        soundness_error=0.025,
        power_error=0.025,
    )
    assert narrow > wide > 0


def test_weissman_upper_is_no_smaller_in_registered_cells() -> None:
    lower = binary_certification_sample_lower_bound(
        contract=0.2,
        margin=0.05,
        soundness_error=0.025,
        power_error=0.025,
    )
    upper = weissman_sample_upper_bound(
        alphabet_size=4,
        margin=0.05,
        failure_probability=0.025,
    )
    assert upper >= lower
    assert bernoulli_kl(0.6, 0.7) > 0.0
