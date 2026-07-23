"""A small deployment API around MOSAIC's exact finite certificate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Hashable

import joblib
import numpy as np

from mosaic_bridge import BridgeMembershipCertificate, certify_bridge_membership
from mosaic_real import ScoreTokenizer, build_token_table, fit_score_tokenizer
from mosaic_release import PersistentReleaseMechanism
from mosaic_transform_exact_optimizer import (
    TransformExactChannelSolution,
    optimize_transform_exact_channel,
)


@dataclass(frozen=True)
class MosaicConfig:
    """Registered finite certificate and release configuration."""

    fine_token_count: int = 4
    released_token_count: int = 2
    privacy_advantage_threshold: float = 0.35
    utility_error_threshold: float = 0.45
    familywise_delta: float = 0.05
    seed: int = 0
    solver_time_limit_seconds: float | None = 300.0
    attacker_constraint_generation: bool = False

    def __post_init__(self) -> None:
        if self.fine_token_count < 2 or self.released_token_count < 2:
            raise ValueError("token counts must be at least two")
        if not 0.0 <= self.privacy_advantage_threshold <= 1.0:
            raise ValueError("privacy threshold must lie in [0, 1]")
        if not 0.0 <= self.utility_error_threshold <= 1.0:
            raise ValueError("utility threshold must lie in [0, 1]")
        if not 0.0 < self.familywise_delta < 1.0:
            raise ValueError("familywise delta must lie in (0, 1)")


@dataclass(frozen=True)
class CertificationResult:
    """Decision and auditable bounds from one certification attempt."""

    status: str
    reason: str
    certified_source_advantage_upper: tuple[float, ...] = ()
    certified_worst_conditional_error_upper: float | None = None
    retained_masses: tuple[float, ...] = ()
    reference_stratum_counts: tuple[tuple[int, ...], ...] = ()
    bridge_stratum_counts: tuple[tuple[int, ...], ...] = ()

    @property
    def certified(self) -> bool:
        return self.status == "CERTIFIED"


@dataclass(frozen=True)
class ReleaseResult:
    """Public response for one protected item."""

    status: str
    released_token: int | None = None
    predicted_label: int | None = None
    reason: str | None = None


@dataclass
class Mosaic:
    """Fit a finite interface, certify it, then release or abstain.

    Construction, reference, and bridge rows must be disjoint. The protected
    source is used only during certification and must not be included in the
    release-time feature vector. One instance covers one registered tokenizer
    and configuration; adaptive selection across instances requires an external
    familywise allocation.
    """

    config: MosaicConfig = field(default_factory=MosaicConfig)
    tokenizer_: ScoreTokenizer | None = field(default=None, init=False, repr=False)
    bridge_certificate_: BridgeMembershipCertificate | None = field(
        default=None, init=False, repr=False
    )
    channel_solution_: TransformExactChannelSolution | None = field(
        default=None, init=False, repr=False
    )
    certification_: CertificationResult | None = field(
        default=None, init=False
    )
    _release_mechanism: PersistentReleaseMechanism | None = field(
        default=None, init=False, repr=False
    )

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "Mosaic":
        """Fit the task score and registered finite tokenizer."""

        self._clear_certificate()
        self.tokenizer_ = None
        tokenizer = fit_score_tokenizer(
            features,
            targets,
            token_count=self.config.fine_token_count,
            seed=self.config.seed,
        )
        self.tokenizer_ = tokenizer
        return self

    def certify(
        self,
        reference_features: np.ndarray,
        reference_targets: np.ndarray,
        reference_sources: np.ndarray,
        bridge_features: np.ndarray,
        bridge_targets: np.ndarray,
        bridge_sources: np.ndarray,
    ) -> CertificationResult:
        """Certify a release channel or return an explicit abstention."""

        if self.tokenizer_ is None:
            raise RuntimeError("fit must be called before certify")
        self._clear_certificate()
        reference_targets = _binary_vector(reference_targets, "reference targets")
        reference_sources = _binary_vector(reference_sources, "reference sources")
        bridge_targets = _binary_vector(bridge_targets, "bridge targets")
        bridge_sources = _binary_vector(bridge_sources, "bridge sources")
        table_delta = self.config.familywise_delta / 2.0
        reference = build_token_table(
            self.tokenizer_.encode(reference_features),
            reference_targets,
            reference_sources,
            token_count=self.config.fine_token_count,
            familywise_delta=table_delta,
        )
        bridge = build_token_table(
            self.tokenizer_.encode(bridge_features),
            bridge_targets,
            bridge_sources,
            token_count=self.config.fine_token_count,
            familywise_delta=table_delta,
        )
        reference_counts = _counts(reference.counts)
        bridge_counts = _counts(bridge.counts)
        if np.any(reference.counts.sum(axis=2) == 0) or np.any(
            bridge.counts.sum(axis=2) == 0
        ):
            return self._abstain(
                "MISSING_SOURCE_LABEL_SUPPORT",
                reference_counts,
                bridge_counts,
            )

        bridge_certificate: BridgeMembershipCertificate | None = None
        try:
            bridge_certificate = certify_bridge_membership(
                reference.probabilities,
                reference_l1_radii=reference.l1_radii,
                bridge_empirical_distributions=bridge.probabilities,
                bridge_l1_radii=bridge.l1_radii,
            )
            solution = optimize_transform_exact_channel(
                reference.probabilities,
                l1_radii=reference.l1_radii,
                common_channels_by_label=bridge_certificate.transforms_by_label,
                contaminations=bridge_certificate.contaminations,
                privacy_advantage_thresholds=(
                    self.config.privacy_advantage_threshold,
                    self.config.privacy_advantage_threshold,
                ),
                released_token_count=self.config.released_token_count,
                maximum_worst_conditional_error=self.config.utility_error_threshold,
                solver_time_limit_seconds=self.config.solver_time_limit_seconds,
                attacker_constraint_generation=self.config.attacker_constraint_generation,
            )
        except RuntimeError as error:
            return self._abstain(
                str(error).split(":", 1)[0],
                reference_counts,
                bridge_counts,
                retained_masses=(
                    bridge_certificate.retained_masses
                    if bridge_certificate is not None
                    else ()
                ),
            )

        source_bounds = tuple(
            float(value.normalized_advantage)
            for value in solution.privacy_certificates
        )
        result = CertificationResult(
            status="CERTIFIED",
            reason="ALL_REGISTERED_CONTRACTS_CERTIFIED",
            certified_source_advantage_upper=source_bounds,
            certified_worst_conditional_error_upper=float(
                solution.certified_worst_conditional_error
            ),
            retained_masses=tuple(bridge_certificate.retained_masses),
            reference_stratum_counts=reference_counts,
            bridge_stratum_counts=bridge_counts,
        )
        self.bridge_certificate_ = bridge_certificate
        self.channel_solution_ = solution
        self.certification_ = result
        self._release_mechanism = PersistentReleaseMechanism(
            solution.release_channel,
            np.random.default_rng(self.config.seed),
        )
        return result

    def release_or_abstain(
        self, item_identifier: Hashable, features: np.ndarray
    ) -> ReleaseResult:
        """Release one persistent public token, or return ``ABSTAIN``."""

        if not self.certification_ or not self.certification_.certified:
            reason = (
                self.certification_.reason
                if self.certification_ is not None
                else "NO_CERTIFICATION"
            )
            return ReleaseResult(status="ABSTAIN", reason=reason)
        if self.tokenizer_ is None or self.channel_solution_ is None:
            raise RuntimeError("certified model state is incomplete")
        values = np.asarray(features)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or len(values) != 1:
            raise ValueError("release_or_abstain accepts exactly one feature row")
        fine_token = int(self.tokenizer_.encode(values)[0])
        released = int(self._release_mechanism.release(item_identifier, fine_token))
        return ReleaseResult(
            status="RELEASED",
            released_token=released,
            predicted_label=int(self.channel_solution_.decoder[released]),
        )

    def save(self, path: str | Path) -> None:
        """Persist the fitted certificate and private release state."""

        joblib.dump(self, Path(path))

    def write_report(
        self,
        directory: str | Path,
        *,
        title: str = "MOSAIC Certification Report",
    ) -> tuple[Path, Path]:
        """Write machine-readable and human-readable audit reports."""

        from .report import write_certification_report

        return write_certification_report(self, directory, title=title)

    @classmethod
    def load(cls, path: str | Path) -> "Mosaic":
        """Load a trusted local MOSAIC artifact."""

        value = joblib.load(Path(path))
        if not isinstance(value, cls):
            raise TypeError("artifact does not contain a Mosaic model")
        return value

    def _clear_certificate(self) -> None:
        self.bridge_certificate_ = None
        self.channel_solution_ = None
        self.certification_ = None
        self._release_mechanism = None

    def _abstain(
        self,
        reason: str,
        reference_counts: tuple[tuple[int, ...], ...],
        bridge_counts: tuple[tuple[int, ...], ...],
        *,
        retained_masses: tuple[float, ...] = (),
    ) -> CertificationResult:
        result = CertificationResult(
            status="ABSTAIN",
            reason=reason,
            retained_masses=tuple(float(value) for value in retained_masses),
            reference_stratum_counts=reference_counts,
            bridge_stratum_counts=bridge_counts,
        )
        self.certification_ = result
        return result


def _counts(values: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(value) for value in row)
        for row in np.asarray(values).sum(axis=2)
    )


def _binary_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 1 or set(np.unique(vector)) - {0, 1}:
        raise ValueError(f"{name} must be a binary vector")
    return vector.astype(np.int16, copy=False)
