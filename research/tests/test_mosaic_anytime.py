import numpy as np

from mosaic_anytime import (
    anytime_multinomial_region,
    dirichlet_mixture_log_e_value,
    simultaneous_anytime_regions,
)


def test_log_e_value_is_smallest_at_empirical_law() -> None:
    counts = [25, 50, 25]
    empirical = [0.25, 0.5, 0.25]
    assert dirichlet_mixture_log_e_value(
        counts, empirical
    ) < dirichlet_mixture_log_e_value(counts, [0.1, 0.8, 0.1])


def test_anytime_region_shrinks_with_balanced_repeated_data() -> None:
    first = anytime_multinomial_region(
        [50, 50], failure_probability=0.01
    )
    second = anytime_multinomial_region(
        [500, 500], failure_probability=0.01
    )
    assert second.l1_radius < first.l1_radius


def test_simultaneous_regions_allocate_familywise_error() -> None:
    regions = simultaneous_anytime_regions(
        np.asarray([[20, 30], [10, 40]]),
        failure_probability=0.04,
    )
    assert len(regions) == 2
    assert all(region.failure_probability == 0.02 for region in regions)
