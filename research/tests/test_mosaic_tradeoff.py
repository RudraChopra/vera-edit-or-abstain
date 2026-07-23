from mosaic_tradeoff import privacy_utility_error_lower_bound


def test_perfect_source_task_conflict_has_sharp_bound() -> None:
    assert privacy_utility_error_lower_bound(0.35) == 0.325


def test_disagreement_relaxes_the_bound() -> None:
    assert privacy_utility_error_lower_bound(0.35, 0.10) == 0.225
    assert privacy_utility_error_lower_bound(1.0, 0.10) == 0.0
