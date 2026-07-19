from __future__ import annotations

import pytest

from summarize_mosaic_baseline_extension import aggregate_cell, paired_contrast


def row(seed: int, method: str, deployed: bool, safe: bool = True):
    return {
        "seed": seed,
        "method": method,
        "deployed": deployed,
        "exact_safe": safe,
        "false_acceptance": deployed and not safe,
    }


def test_aggregate_cell_counts_safe_and_false_deployments():
    summary = aggregate_cell(
        [row(1, "m", True), row(2, "m", True, False), row(3, "m", False)]
    )
    assert summary["trials"] == 3
    assert summary["deployments"] == 2
    assert summary["safe_deployments"] == 1
    assert summary["false_acceptances"] == 1


def test_paired_contrast_uses_matched_seeds():
    rows = [
        row(1, "mosaic_continuum", True),
        row(1, "holm_ltt_grid", False),
        row(2, "mosaic_continuum", True),
        row(2, "holm_ltt_grid", True),
        row(3, "mosaic_continuum", False),
        row(3, "holm_ltt_grid", True),
        row(4, "mosaic_continuum", False),
        row(4, "holm_ltt_grid", False),
    ]
    summary = paired_contrast(rows, comparator="holm_ltt_grid")
    assert summary["mosaic_only_deployments"] == 1
    assert summary["comparator_only_deployments"] == 1
    assert summary["both_deploy"] == 1
    assert summary["neither_deploys"] == 1
    assert summary["paired_deployment_difference"] == 0.0
    assert summary["exact_two_sided_mcnemar_p"] == pytest.approx(1.0)
