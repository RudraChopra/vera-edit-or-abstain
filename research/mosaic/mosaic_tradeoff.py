"""Sharp privacy-utility lower bounds for source-task conflict."""

from __future__ import annotations


def privacy_utility_error_lower_bound(
    privacy_advantage_threshold: float,
    source_task_disagreement: float = 0.0,
) -> float:
    """Lower-bound task error under a binary source-inference contract.

    Under the balanced-source audit law, let ``kappa=P(Y != S)``. Any task
    decoder is also a source decoder after identifying the two binary labels.
    Its source error is at most its task error plus ``kappa``. Since normalized
    binary Bayes advantage is ``1-2*BayesError``, a source-advantage contract
    ``A <= tau`` forces task error at least ``(1-tau)/2-kappa``.
    """

    if not 0.0 <= privacy_advantage_threshold <= 1.0:
        raise ValueError("privacy_advantage_threshold must lie in [0, 1]")
    if not 0.0 <= source_task_disagreement <= 1.0:
        raise ValueError("source_task_disagreement must lie in [0, 1]")
    return max(
        0.0,
        (1.0 - privacy_advantage_threshold) / 2.0
        - source_task_disagreement,
    )
