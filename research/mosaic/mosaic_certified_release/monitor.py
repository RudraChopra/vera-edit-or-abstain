"""Anytime-valid monitoring for continuously updated categorical audits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from mosaic_anytime import (
    AnytimeMultinomialRegion,
    simultaneous_anytime_regions,
)


@dataclass(frozen=True)
class MonitorSnapshot:
    """One optional-stopping-safe audit snapshot."""

    update_count: int
    regions: tuple[AnytimeMultinomialRegion, ...]
    maximum_l1_radius: float
    method: str = "simultaneous_dirichlet_mixture_e_process"


@dataclass
class AnytimeAuditMonitor:
    """Accumulate registered stratum counts without spending alpha over time."""

    stratum_count: int
    category_count: int
    failure_probability: float = 0.05
    prior: float = 0.5
    counts_: np.ndarray = field(init=False, repr=False)
    update_count_: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.stratum_count < 1 or self.category_count < 2:
            raise ValueError("monitor dimensions are invalid")
        if not 0.0 < self.failure_probability < 1.0:
            raise ValueError("failure_probability must lie in (0, 1)")
        self.counts_ = np.zeros(
            (self.stratum_count, self.category_count), dtype=np.int64
        )

    def update(
        self, stratum: int, categories: int | Iterable[int]
    ) -> MonitorSnapshot:
        """Add observations and return a valid snapshot at this stopping time."""

        if not 0 <= int(stratum) < self.stratum_count:
            raise ValueError("stratum is outside the registered monitor")
        values = (
            np.asarray([categories], dtype=np.int64)
            if np.isscalar(categories)
            else np.asarray(tuple(categories), dtype=np.int64)
        )
        if values.ndim != 1 or np.any(values < 0) or np.any(
            values >= self.category_count
        ):
            raise ValueError("category is outside the registered alphabet")
        np.add.at(self.counts_[int(stratum)], values, 1)
        self.update_count_ += int(values.size)
        return self.snapshot()

    def snapshot(self) -> MonitorSnapshot:
        regions = simultaneous_anytime_regions(
            self.counts_,
            failure_probability=self.failure_probability,
            prior=self.prior,
        )
        return MonitorSnapshot(
            update_count=self.update_count_,
            regions=regions,
            maximum_l1_radius=max(region.l1_radius for region in regions),
        )
