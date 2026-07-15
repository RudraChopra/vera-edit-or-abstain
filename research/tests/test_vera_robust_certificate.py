from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from vera_robust_certificate import (  # noqa: E402
    balanced_profile_in_envelope,
    certify_balanced_iut_fixed_profile,
    certify_balanced_iut_profile,
    certify_balanced_shift_envelope,
    certify_balanced_shift_radius,
    certify_discrete_iut_fixed_profile,
    certify_discrete_group_shift_envelope,
    certify_discrete_shift_radius,
    certify_shift_radius,
    certify_edits,
    empirical_reweighting_risk,
    exact_balanced_leakage_certificate,
    exact_discrete_risk_certificate,
    robust_risk_certificate,
)
from vera_controlled_shift import (  # noqa: E402
    allocate_integer_budget,
    bernoulli_testing_lower_bound,
    conditional_density_ratio_profile,
    design_controlled_shift,
    design_controlled_shift_from_fold,
    dkw_sufficient_sample_size,
)


class EmpiricalReweightingRiskTests(unittest.TestCase):
    def test_gamma_one_is_sample_mean(self) -> None:
        values = np.array([0.0, 0.25, 0.75, 1.0])
        self.assertAlmostEqual(empirical_reweighting_risk(values, 1.0), values.mean())

    def test_gamma_two_averages_upper_half(self) -> None:
        values = np.array([0.0, 0.25, 0.75, 1.0])
        self.assertAlmostEqual(empirical_reweighting_risk(values, 2.0), 0.875)

    def test_large_gamma_reaches_maximum(self) -> None:
        values = np.array([-1.0, -0.5, 0.4])
        self.assertAlmostEqual(empirical_reweighting_risk(values, 10.0), 0.4)


class CertificateTests(unittest.TestCase):
    def test_balanced_leakage_scores_constant_attacker_at_chance(self) -> None:
        source = np.asarray([0] * 500 + [1] * 500)
        correct = np.asarray([1] * 500 + [0] * 500)
        certificate = exact_balanced_leakage_certificate(
            "constant",
            correct,
            source,
            gamma=1.0,
            failure_probability=0.05,
        )
        self.assertAlmostEqual(certificate.empirical_robust_risk, 0.5)
        self.assertGreaterEqual(certificate.upper_confidence_bound, 0.5)
        self.assertLess(certificate.upper_confidence_bound, 0.51)

    def test_balanced_leakage_rejects_missing_source_class(self) -> None:
        with self.assertRaises(ValueError):
            exact_balanced_leakage_certificate(
                "missing",
                np.ones(20),
                np.zeros(20, dtype=int),
                gamma=1.0,
                failure_probability=0.05,
            )

    def test_balanced_iut_and_envelope_use_distinct_multiplicity(self) -> None:
        source = np.asarray([0] * 500 + [1] * 500)
        leakage = {"linear": np.asarray([1] * 500 + [0] * 500)}
        target = {"target::environment=0": np.zeros(1000, dtype=int)}
        iut = certify_balanced_iut_fixed_profile(
            target,
            leakage,
            source,
            gamma=1.25,
            delta=0.05,
            candidate_count=12,
            target_threshold=0.1,
            leakage_threshold=0.7,
        )
        envelope = certify_balanced_shift_radius(
            target,
            leakage,
            source,
            delta=0.05,
            family_size=24,
            target_threshold=0.1,
            leakage_threshold=0.7,
            gamma_cap=2.0,
        )
        self.assertEqual(iut.decision, "EDIT")
        self.assertGreaterEqual(envelope.certified_radius, 1.25)

    def test_balanced_envelope_exposes_coordinates_and_unsupported_zero(self) -> None:
        source = np.asarray([0] * 1000 + [1] * 1000)
        leakage = {
            "asymmetric": np.asarray(
                [1] * 600 + [0] * 400 + [1] * 400 + [0] * 600
            )
        }
        target = {
            "target::environment=0": np.zeros(1000, dtype=int),
            "target::environment=1": np.zeros(1000, dtype=int),
        }
        envelope = certify_balanced_shift_envelope(
            target,
            leakage,
            source,
            delta=0.05,
            family_size=36,
            target_threshold=0.1,
            leakage_threshold=0.55,
            registered_target_environments=[0, 1, 2],
            gamma_cap=4.0,
        )
        self.assertEqual(envelope.target_environment_radii["2"], 0.0)
        self.assertEqual(envelope.unsupported_target_environments, ("2",))
        self.assertEqual(envelope.deployment_common_radius, 0.0)
        self.assertEqual(envelope.decision, "ABSTAIN")
        self.assertGreaterEqual(envelope.observed_common_radius, 1.0)
        self.assertGreater(
            envelope.source_class_radii["1"],
            envelope.source_class_radii["0"],
        )
        self.assertIn(
            "balanced_leakage::asymmetric",
            envelope.simultaneous_curve_parameters,
        )

    def test_balanced_envelope_is_empty_when_iid_contract_fails(self) -> None:
        source = np.asarray([0] * 500 + [1] * 500)
        envelope = certify_balanced_shift_envelope(
            {"target::environment=0": np.zeros(1000, dtype=int)},
            {"perfect": np.ones(1000, dtype=int)},
            source,
            delta=0.05,
            family_size=24,
            target_threshold=0.1,
            leakage_threshold=0.7,
            gamma_cap=4.0,
        )
        self.assertEqual(envelope.observed_common_radius, 0.0)
        self.assertEqual(envelope.target_environment_radii["0"], 0.0)
        self.assertEqual(envelope.source_class_radii, {"0": 0.0, "1": 0.0})
        self.assertEqual(envelope.decision, "ABSTAIN")

    def test_anisotropic_profile_can_pass_when_common_budget_fails(self) -> None:
        source = np.asarray([0] * 2000 + [1] * 2000)
        leakage = {
            "asymmetric": np.asarray(
                [1] * 880 + [0] * 1120 + [1] * 200 + [0] * 1800
            )
        }
        target = {
            "target::environment=0": np.zeros(2000, dtype=int),
            "target::environment=1": np.zeros(2000, dtype=int),
        }
        envelope = certify_balanced_shift_envelope(
            target,
            leakage,
            source,
            delta=0.05,
            family_size=36,
            target_threshold=0.1,
            leakage_threshold=0.55,
            registered_target_environments=[0, 1],
            gamma_cap=3.0,
        )
        vector_passes = balanced_profile_in_envelope(
            envelope,
            target_profile={"0": 1.0, "1": 1.0},
            source_profile={"0": 1.0, "1": 2.0},
        )
        common_passes = balanced_profile_in_envelope(
            envelope,
            target_profile={"0": 2.0, "1": 2.0},
            source_profile={"0": 2.0, "1": 2.0},
        )
        self.assertTrue(vector_passes)
        self.assertFalse(common_passes)

    def test_fixed_vector_profile_uses_candidate_iut_budget(self) -> None:
        source = np.asarray([0] * 2000 + [1] * 2000)
        leakage = {
            "asymmetric": np.asarray(
                [1] * 800 + [0] * 1200 + [1] * 200 + [0] * 1800
            )
        }
        target = {
            "target::environment=0": np.zeros(2000, dtype=int),
            "target::environment=1": np.zeros(2000, dtype=int),
        }
        result = certify_balanced_iut_profile(
            target,
            leakage,
            source,
            target_profile={
                "target::environment=0": 1.25,
                "target::environment=1": 1.0,
            },
            source_profile={0: 1.0, 1: 2.0},
            delta=0.05,
            candidate_count=12,
            target_threshold=0.1,
            leakage_threshold=0.55,
        )
        self.assertEqual(result.decision, "EDIT")
        self.assertAlmostEqual(result.candidate_failure_probability, 0.05 / 12)
        leakage_certificate = next(
            certificate
            for certificate in result.certificates
            if certificate.key == "balanced_leakage::asymmetric"
        )
        self.assertEqual(
            leakage_certificate.method,
            "exact_balanced_leakage_vector_profile",
        )

    def test_profile_membership_rejects_unsupported_environment(self) -> None:
        source = np.asarray([0] * 500 + [1] * 500)
        envelope = certify_balanced_shift_envelope(
            {"target::environment=0": np.zeros(1000, dtype=int)},
            {"linear": np.zeros(1000, dtype=int)},
            source,
            delta=0.05,
            family_size=24,
            target_threshold=0.1,
            leakage_threshold=0.55,
            registered_target_environments=[0, 1],
            gamma_cap=2.0,
        )
        self.assertFalse(
            balanced_profile_in_envelope(
                envelope,
                target_profile={0: 1.0, 1: 1.0},
                source_profile={0: 1.0, 1: 1.0},
            )
        )

    def test_iut_fixed_profile_spends_alpha_over_candidates_only(self) -> None:
        samples = {
            "target": np.asarray([-1] * 20 + [0] * 470 + [1] * 10),
            "leakage_a": np.asarray([0] * 350 + [1] * 150),
            "leakage_b": np.asarray([0] * 340 + [1] * 160),
        }
        supports = {
            "target": (-1, 0, 1),
            "leakage_a": (0, 1),
            "leakage_b": (0, 1),
        }
        thresholds = {"target": 0.1, "leakage_a": 0.6, "leakage_b": 0.6}
        result = certify_discrete_iut_fixed_profile(
            samples,
            gamma=1.25,
            delta=0.05,
            candidate_count=12,
            supports=supports,
            thresholds=thresholds,
        )
        self.assertAlmostEqual(result.candidate_failure_probability, 0.05 / 12)
        self.assertTrue(
            all(
                certificate.simultaneous_failure_probability == 0.05 / 12
                for certificate in result.certificates
            )
        )

    def test_empirical_paired_formula_matches_greedy_linear_program(self) -> None:
        for positive in range(6):
            for negative in range(6 - positive):
                zero = 5 - positive - negative
                values = np.asarray([1] * positive + [0] * zero + [-1] * negative)
                p_positive = positive / 5
                p_negative = negative / 5
                for gamma in (1.0, 1.25, 2.0, 5.0):
                    if gamma * p_positive >= 1.0:
                        expected = 1.0
                    elif gamma * (1.0 - p_negative) >= 1.0:
                        expected = gamma * p_positive
                    else:
                        expected = gamma * p_positive - (
                            1.0 - gamma * (1.0 - p_negative)
                        )
                    self.assertAlmostEqual(
                        empirical_reweighting_risk(values, gamma), expected
                    )

    def test_exact_bernoulli_certificate_rejects_high_leakage(self) -> None:
        certificate = exact_discrete_risk_certificate(
            "leakage",
            [1] * 80 + [0] * 20,
            gamma=1.0,
            failure_probability=0.05,
            support=(0, 1),
        )
        self.assertGreater(certificate.upper_confidence_bound, 0.8)

    def test_exact_paired_certificate_uses_beneficial_negative_harm(self) -> None:
        certificate = exact_discrete_risk_certificate(
            "harm",
            [-1] * 100 + [0] * 900,
            gamma=1.0,
            failure_probability=0.05,
            support=(-1, 0, 1),
        )
        self.assertLess(certificate.upper_confidence_bound, 0.0)

    def test_exact_paired_probability_bounds_respect_simplex(self) -> None:
        certificate = exact_discrete_risk_certificate(
            "harm",
            [1] * 2 + [-1] * 98,
            gamma=1.25,
            failure_probability=0.05,
            support=(-1, 0, 1),
        )
        details = certificate.confidence_details
        assert details is not None
        positive = min(
            float(details["positive_probability_upper"]),
            1.0 - float(details["negative_probability_lower"]),
        )
        self.assertLessEqual(
            positive + float(details["negative_probability_lower"]), 1.0
        )

    def test_exact_shift_radius_spends_alpha_over_full_family(self) -> None:
        report = certify_discrete_shift_radius(
            {"harm": np.zeros(1000), "leakage": np.zeros(1000)},
            delta=0.05,
            supports={"harm": (-1, 0, 1), "leakage": (0, 1)},
            thresholds={"harm": 0.1, "leakage": 0.1},
            family_size=20,
            gamma_cap=4.0,
        )
        self.assertEqual(report.decision, "EDIT")
        self.assertGreaterEqual(report.certified_radius, 1.0)
        self.assertTrue(
            all(
                certificate.simultaneous_failure_probability == 0.05 / 20
                for certificate in report.certificates_at_radius
            )
        )

    def test_group_envelope_minimum_matches_joint_radius(self) -> None:
        grouped_samples = {
            "0": {
                "target::0": np.r_[np.ones(2), np.zeros(998)],
                "leakage::0": np.r_[np.ones(300), np.zeros(700)],
            },
            "1": {
                "target::1": np.r_[np.ones(5), np.zeros(995)],
                "leakage::1": np.r_[np.ones(350), np.zeros(650)],
            },
        }
        grouped_supports = {
            group: {"target::" + group: (-1, 0, 1), "leakage::" + group: (0, 1)}
            for group in grouped_samples
        }
        grouped_thresholds = {
            group: {"target::" + group: 0.1, "leakage::" + group: 0.7}
            for group in grouped_samples
        }
        envelope = certify_discrete_group_shift_envelope(
            grouped_samples,
            delta=0.05,
            grouped_supports=grouped_supports,
            grouped_thresholds=grouped_thresholds,
            family_size=20,
            gamma_cap=4.0,
        )
        flat_samples = {
            key: values for samples in grouped_samples.values() for key, values in samples.items()
        }
        flat_supports = {
            key: value for supports in grouped_supports.values() for key, value in supports.items()
        }
        flat_thresholds = {
            key: value
            for thresholds in grouped_thresholds.values()
            for key, value in thresholds.items()
        }
        joint = certify_discrete_shift_radius(
            flat_samples,
            delta=0.05,
            supports=flat_supports,
            thresholds=flat_thresholds,
            family_size=20,
            gamma_cap=4.0,
        )
        self.assertAlmostEqual(
            envelope.observed_common_radius, joint.certified_radius, delta=1e-4
        )

    def test_group_envelope_assigns_zero_to_unsupported_group(self) -> None:
        envelope = certify_discrete_group_shift_envelope(
            {"seen": {"target": np.zeros(1000)}},
            delta=0.05,
            grouped_supports={"seen": {"target": (-1, 0, 1)}},
            grouped_thresholds={"seen": {"target": 0.1}},
            family_size=4,
            registered_groups=["seen", "unseen"],
            gamma_cap=4.0,
        )
        self.assertEqual(envelope.group_radii["unseen"], 0.0)
        self.assertEqual(envelope.deployment_common_radius, 0.0)
        self.assertEqual(envelope.decision, "ABSTAIN")
        self.assertEqual(envelope.unsupported_groups, ("unseen",))

    def test_ucb_is_at_least_empirical_robust_risk(self) -> None:
        cert = robust_risk_certificate(
            "risk",
            [0.0] * 90 + [1.0] * 10,
            gamma=2.0,
            failure_probability=0.05,
            lower=0.0,
            upper=1.0,
        )
        self.assertGreaterEqual(cert.upper_confidence_bound, cert.empirical_robust_risk)
        self.assertLessEqual(cert.upper_confidence_bound, 1.0)

    def test_edit_rule_abstains_when_target_cannot_certify(self) -> None:
        n = 200
        target = {"weak": np.ones(n), "strong": np.ones(n)}
        leakage = {
            ("weak", "linear"): np.zeros(n),
            ("strong", "linear"): np.zeros(n),
        }
        decision = certify_edits(
            target,
            leakage,
            edit_order=["weak", "strong"],
            gamma=1.0,
            delta=0.05,
            target_threshold=0.1,
            leakage_threshold=0.6,
        )
        self.assertEqual(decision.decision, "ABSTAIN")
        self.assertIsNone(decision.selected_edit)

    def test_edit_rule_selects_strongest_certified_edit(self) -> None:
        n = 10_000
        target = {"weak": np.zeros(n), "strong": np.zeros(n)}
        leakage = {
            ("weak", "linear"): np.zeros(n),
            ("strong", "linear"): np.zeros(n),
        }
        decision = certify_edits(
            target,
            leakage,
            edit_order=["weak", "strong"],
            gamma=1.0,
            delta=0.05,
            target_threshold=0.1,
            leakage_threshold=0.1,
        )
        self.assertEqual(decision.decision, "EDIT")
        self.assertEqual(decision.selected_edit, "strong")

    def test_shift_radius_abstains_when_iid_contract_fails(self) -> None:
        report = certify_shift_radius(
            {"harm": np.ones(500)},
            delta=0.05,
            bounds={"harm": (0.0, 1.0)},
            thresholds={"harm": 0.2},
            gamma_cap=4.0,
        )
        self.assertEqual(report.decision, "ABSTAIN")
        self.assertEqual(report.certified_radius, 0.0)

    def test_shift_radius_is_positive_and_limiting_contract_is_reported(self) -> None:
        samples = {
            "harm": np.r_[np.ones(20), np.zeros(9980)],
            "leakage": np.r_[np.ones(3000), np.zeros(7000)],
        }
        report = certify_shift_radius(
            samples,
            delta=0.05,
            bounds={"harm": (0.0, 1.0), "leakage": (0.0, 1.0)},
            thresholds={"harm": 0.1, "leakage": 0.55},
            gamma_cap=8.0,
        )
        self.assertEqual(report.decision, "EDIT")
        self.assertGreaterEqual(report.certified_radius, 1.0)
        self.assertLess(report.certified_radius, 8.0)
        self.assertIn("leakage", report.limiting_contracts)


class AssumptionBoundaryTests(unittest.TestCase):
    def test_shift_outside_declared_density_ratio_budget_is_not_covered(self) -> None:
        validation = np.r_[np.ones(4), np.zeros(96)]
        robust_bound_at_gamma_two = empirical_reweighting_risk(validation, 2.0)
        deployment_positive_rate = 0.10
        required_positive_density_ratio = deployment_positive_rate / 0.04

        self.assertAlmostEqual(robust_bound_at_gamma_two, 0.08)
        self.assertGreater(required_positive_density_ratio, 2.0)
        self.assertGreater(deployment_positive_rate, robust_bound_at_gamma_two)

    def test_unseen_group_safe_and_unsafe_worlds_are_observationally_identical(self) -> None:
        observed_safe_world = np.zeros(1000)
        observed_unsafe_world = np.zeros(1000)
        unseen_safe_risk = 0.0
        unseen_unsafe_risk = 1.0

        np.testing.assert_array_equal(observed_safe_world, observed_unsafe_world)
        self.assertNotEqual(unseen_safe_risk, unseen_unsafe_risk)
        envelope = certify_discrete_group_shift_envelope(
            {"seen": {"target": observed_safe_world}},
            delta=0.05,
            grouped_supports={"seen": {"target": (-1, 0, 1)}},
            grouped_thresholds={"seen": {"target": 0.1}},
            family_size=1,
            registered_groups=["seen", "unseen"],
            gamma_cap=4.0,
        )
        self.assertEqual(envelope.decision, "ABSTAIN")
        self.assertEqual(envelope.group_radii["unseen"], 0.0)

    def test_reusing_single_candidate_alpha_after_adaptive_search_is_invalid(self) -> None:
        alpha = 0.05
        candidate_count = 20
        false_accept_probability = 1.0 - (1.0 - alpha) ** candidate_count

        self.assertGreater(false_accept_probability, alpha)
        self.assertGreater(false_accept_probability, 0.60)
        bonferroni_bound = candidate_count * (alpha / candidate_count)
        self.assertAlmostEqual(bonferroni_bound, alpha)


class ControlledShiftTests(unittest.TestCase):
    def test_density_ratio_profile_preserves_unit_lower_bound(self) -> None:
        weights = np.asarray([1.0 + 1e-15, 1.0, 1.0])
        profile = conditional_density_ratio_profile(weights, [0, 1, 1])
        self.assertGreaterEqual(profile["0"], 1.0)
        self.assertGreaterEqual(profile["1"], 1.0)

    def test_controlled_shift_has_exact_known_membership(self) -> None:
        environment = np.repeat([0, 1], 200)
        source = np.tile(np.repeat([0, 1], 100), 2)
        target = np.tile([0, 1], 200)
        design = np.arange(len(environment))
        probabilities, shift = design_controlled_shift(
            environment,
            source,
            target,
            design,
            requested_gamma=1.5,
            minimum_design_cell_count=20,
        )
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertTrue(np.all(probabilities >= 0.0))
        density_ratio = probabilities * len(probabilities)
        self.assertLessEqual(float(density_ratio.max()), 1.5 + 1e-12)
        self.assertAlmostEqual(shift.global_density_ratio_cap, 1.5)
        self.assertTrue(
            all(value >= 1.0 for value in shift.target_profile.values())
        )
        self.assertTrue(
            all(value >= 1.0 for value in shift.source_profile.values())
        )

    def test_separate_design_fold_cannot_introduce_reference_support(self) -> None:
        environment = np.repeat([0, 1], 100)
        source = np.tile([0, 1], 100)
        target = np.tile(np.repeat([0, 1], 50), 2)
        design_environment = np.repeat([0, 1, 2], 80)
        design_source = np.tile([0, 1], 120)
        design_target = np.tile([0, 1], 120)
        probabilities, shift = design_controlled_shift_from_fold(
            environment,
            source,
            target,
            design_environment,
            design_source,
            design_target,
            requested_gamma=1.25,
            minimum_design_cell_count=10,
        )
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertIn(shift.focus_environment, {0, 1})
        self.assertNotEqual(shift.focus_environment, 2)

    def test_integer_allocation_preserves_total_and_minimum(self) -> None:
        allocation = allocate_integer_budget(
            {"easy": 1.0, "hard": 9.0, "middle": 3.0},
            total_budget=101,
            minimum_per_cell=10,
        )
        self.assertEqual(sum(allocation.values()), 101)
        self.assertGreater(allocation["hard"], allocation["middle"])
        self.assertGreater(allocation["middle"], allocation["easy"])
        self.assertGreaterEqual(min(allocation.values()), 10)

    def test_sample_complexity_tracks_margin_and_shift(self) -> None:
        easy = dkw_sufficient_sample_size(
            robust_range=1.0,
            margin=0.2,
            coverage_error=0.01,
            power_error=0.1,
        )
        hard = dkw_sufficient_sample_size(
            robust_range=1.5,
            margin=0.1,
            coverage_error=0.01,
            power_error=0.1,
        )
        self.assertGreater(hard, easy)
        lower = bernoulli_testing_lower_bound(
            safe_probability=0.4,
            unsafe_probability=0.5,
            type_one_error=0.05,
            type_two_error=0.1,
        )
        self.assertGreater(lower, 0)
        self.assertLess(lower, hard)


if __name__ == "__main__":
    unittest.main()
