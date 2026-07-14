from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import binom


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from analyze_real_theory_match import leakage_ucb, paired_ucb  # noqa: E402
from analyze_vera_attacker_ablation import filter_portfolio  # noqa: E402
from analyze_vera_confirmatory_balanced import (  # noqa: E402
    build_abstract_record,
    exact_one_sided_signflip,
    exact_one_sided_sign_test,
    seed_cluster_ratio_interval,
    summarize,
)
from analyze_vera_independent_stress_replication import (  # noqa: E402
    exact_one_sided_mcnemar,
    make_abstract_record as make_replication_abstract_record,
    summarize as summarize_replication,
)
from audit_vera_confirmatory_analysis import (  # noqa: E402
    independent_balanced_leakage_ucb,
    independent_target_ucb,
)
from analyze_vera_confirmatory_ablations import make_row  # noqa: E402
from analyze_vera_learning_curve_diagnostic import candidate_bounds  # noqa: E402
from analyze_vera_real_study import (  # noqa: E402
    exact_cluster_signflip,
    group_contract_family,
    holm_adjust,
)
from analyze_vera_secondary_ablations import (  # noqa: E402
    cluster_summary,
    select_candidate,
)
from audit_vera_independent_stress_replication import (  # noqa: E402
    exact_one_sided_mcnemar as audit_one_sided_mcnemar,
)
from run_exact_balanced_simulation import (  # noqa: E402
    cp_upper,
    exact_balanced_pass_probability,
)
from vera_robust_certificate import (  # noqa: E402
    balanced_contract_certificates,
    exact_balanced_leakage_certificate,
    exact_discrete_risk_certificate,
)


class SeedBlockedInferenceTests(unittest.TestCase):
    def test_independent_mcnemar_matches_preregistered_holm_boundary(self) -> None:
        raw = {
            dataset: exact_one_sided_mcnemar(7, 0)
            for dataset in ("bios", "civil", "gait", "waterbirds")
        }
        self.assertEqual(set(raw.values()), {0.0078125})
        self.assertTrue(all(value == 0.03125 for value in holm_adjust(raw).values()))

    def test_independent_mcnemar_is_directional(self) -> None:
        self.assertEqual(exact_one_sided_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_one_sided_mcnemar(0, 7), 1.0)
        self.assertLess(exact_one_sided_mcnemar(8, 1), 0.05)

    def test_independent_audit_reimplements_paired_test_exactly(self) -> None:
        for point_only in range(10):
            for vera_only in range(10):
                self.assertEqual(
                    audit_one_sided_mcnemar(point_only, vera_only),
                    exact_one_sided_mcnemar(point_only, vera_only),
                )

    def test_independent_abstract_falls_back_when_any_gate_fails(self) -> None:
        record = make_replication_abstract_record(
            prereg_hash="abc",
            supported_count=128,
            point_rate=0.25,
            vera_rate=0.0,
            retention=0.5,
            passed=False,
            camelyon_forced_count=64,
        )
        self.assertFalse(record["registered_pass_conditions_met"])
        self.assertEqual(record["headline_mode"], "theory_and_support_impossibility")
        self.assertIn("did not satisfy", record["sentence"])

    def test_independent_summary_counts_abstention_in_false_accept_denominator(self) -> None:
        rows = [
            {
                "external_contract_estimable": True,
                "deployed": True,
                "external_contract_satisfied": False,
                "measured_external_contract_violation": True,
                "procedurally_unsupported_deployment": False,
            },
            {
                "external_contract_estimable": True,
                "deployed": False,
                "external_contract_satisfied": "NA",
                "measured_external_contract_violation": False,
                "procedurally_unsupported_deployment": False,
            },
        ]
        observed = summarize_replication(rows)
        self.assertEqual(observed["measured_external_violation_rate"], 0.5)
        self.assertEqual(observed["violation_rate_conditional_on_estimable_deployment"], 1.0)

    def test_abstract_uses_empirical_headline_only_when_gap_passes(self) -> None:
        record = build_abstract_record(
            prereg_hash="abc",
            stress_configuration_count=32,
            point_rate=0.25,
            vera_rate=0.05,
            retention=0.4,
            empirical_pass=True,
            gap_pass=True,
            camelyon_abstention_pass=True,
            camelyon_forced_count=144,
        )
        self.assertTrue(record["verified"])
        self.assertEqual(record["headline_mode"], "empirical_gap")
        self.assertIn("25.0%", record["sentence"])

    def test_abstract_falls_back_without_suppressing_verified_numbers(self) -> None:
        record = build_abstract_record(
            prereg_hash="abc",
            stress_configuration_count=32,
            point_rate=0.10,
            vera_rate=0.05,
            retention=0.4,
            empirical_pass=False,
            gap_pass=False,
            camelyon_abstention_pass=True,
            camelyon_forced_count=144,
        )
        self.assertTrue(record["verified"])
        self.assertFalse(record["registered_pass_conditions_met"])
        self.assertEqual(record["headline_mode"], "theory_forced_abstention")
        self.assertIn("all 144", record["sentence"])

    def test_abstract_falls_back_when_gap_passes_but_other_gates_fail(self) -> None:
        record = build_abstract_record(
            prereg_hash="abc",
            stress_configuration_count=32,
            point_rate=0.25,
            vera_rate=0.05,
            retention=0.1,
            empirical_pass=False,
            gap_pass=True,
            camelyon_abstention_pass=True,
            camelyon_forced_count=144,
        )
        self.assertTrue(record["headline_gap_condition_met"])
        self.assertEqual(record["headline_mode"], "theory_forced_abstention")

    def test_five_concordant_seed_blocks_have_minimum_two_sided_p(self) -> None:
        self.assertEqual(exact_cluster_signflip([1.0] * 5), 0.0625)

    def test_zero_seed_block_differences_have_unit_p(self) -> None:
        self.assertEqual(exact_cluster_signflip([0.0] * 5), 1.0)

    def test_holm_adjustment_is_monotone_in_rank(self) -> None:
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})
        self.assertEqual(adjusted, {"a": 0.03, "b": 0.06, "c": 0.2})

    def test_eight_positive_blocks_have_exact_one_sided_p(self) -> None:
        self.assertEqual(exact_one_sided_signflip([1.0] * 8), 1 / 256)

    def test_nonpositive_direction_has_unit_one_sided_p(self) -> None:
        self.assertEqual(exact_one_sided_signflip([-1.0] * 8), 1.0)

    def test_sign_test_discards_zero_seed_blocks(self) -> None:
        self.assertEqual(exact_one_sided_sign_test([1.0] * 5 + [0.0] * 3), 1 / 32)

    def test_seed_cluster_ratio_resamples_whole_clusters(self) -> None:
        lower, upper = seed_cluster_ratio_interval(
            {5: 1, 6: 3}, {5: 2, 6: 4}, replicates=2_000
        )
        self.assertGreaterEqual(lower, 0.5)
        self.assertLessEqual(upper, 0.75)

    def test_confirmatory_summary_uses_estimable_configuration_denominator(self) -> None:
        rows = [
            {
                "external_contract_estimable": True,
                "deployed": True,
                "external_contract_satisfied": False,
                "measured_external_contract_violation": True,
                "procedurally_unsupported_deployment": False,
            },
            {
                "external_contract_estimable": True,
                "deployed": False,
                "external_contract_satisfied": "NA",
                "measured_external_contract_violation": False,
                "procedurally_unsupported_deployment": False,
            },
            {
                "external_contract_estimable": False,
                "deployed": True,
                "external_contract_satisfied": "NA",
                "measured_external_contract_violation": False,
                "procedurally_unsupported_deployment": True,
            },
        ]
        observed = summarize(rows)
        self.assertEqual(observed["estimable_configuration_count"], 2)
        self.assertEqual(observed["procedurally_unsupported_deployment_count"], 1)
        self.assertEqual(observed["measured_external_violation_rate"], 0.5)


class VectorizedCertificateTests(unittest.TestCase):
    def test_independent_raw_audit_target_bound_matches_production(self) -> None:
        values = np.asarray([1] * 7 + [0] * 80 + [-1] * 13)
        expected = exact_discrete_risk_certificate(
            "harm",
            values,
            gamma=1.25,
            failure_probability=0.001,
            support=(-1, 0, 1),
        )
        observed = independent_target_ucb(
            values, gamma=1.25, alpha=0.001
        )
        self.assertAlmostEqual(observed, expected.upper_confidence_bound)

    def test_independent_raw_audit_balanced_bound_matches_production(self) -> None:
        source = np.asarray([0] * 50 + [1] * 50)
        correct = np.asarray([1] * 17 + [0] * 33 + [1] * 29 + [0] * 21)
        expected = exact_balanced_leakage_certificate(
            "leakage",
            correct,
            source,
            gamma=1.25,
            failure_probability=0.001,
        )
        observed = independent_balanced_leakage_ucb(
            correct, source, gamma=1.25, alpha=0.001
        )
        self.assertAlmostEqual(observed, expected.upper_confidence_bound)

    def test_paired_vectorized_bound_matches_production_certificate(self) -> None:
        values = np.asarray([1] * 7 + [0] * 80 + [-1] * 13)
        expected = exact_discrete_risk_certificate(
            "harm",
            values,
            gamma=1.25,
            failure_probability=0.001,
            support=(-1, 0, 1),
        )
        observed = paired_ucb(
            np.asarray([7]), np.asarray([13]), 100, 0.001, 1.25
        )[0]
        self.assertAlmostEqual(observed, expected.upper_confidence_bound)

    def test_leakage_vectorized_bound_matches_production_certificate(self) -> None:
        values = np.asarray([1] * 17 + [0] * 83)
        expected = exact_discrete_risk_certificate(
            "leakage",
            values,
            gamma=1.25,
            failure_probability=0.001,
            support=(0, 1),
        )
        observed = leakage_ucb(np.asarray([17]), 100, 0.001, 1.25)[0]
        self.assertAlmostEqual(observed, expected.upper_confidence_bound)

    def test_exact_balanced_probability_matches_brute_force_sum(self) -> None:
        n = 8
        p0, p1 = 0.35, 0.55
        gamma = 1.01
        threshold = 0.7
        candidate_alpha = 0.05
        upper = np.minimum(
            1.0,
            gamma
            * cp_upper(np.arange(n + 1), n, candidate_alpha / 2.0),
        )
        brute = sum(
            float(binom.pmf(first, n, p0))
            * float(binom.pmf(second, n, p1))
            for first in range(n + 1)
            for second in range(n + 1)
            if 0.5 * (upper[first] + upper[second]) <= threshold
        )
        exact = exact_balanced_pass_probability(
            n,
            (p0, p1),
            gamma=gamma,
            threshold=threshold,
            candidate_alpha=candidate_alpha,
        )
        self.assertAlmostEqual(exact, brute)

    def test_learning_curve_bounds_match_production_balanced_certificate(self) -> None:
        source = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        environment = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
        target_harm = np.asarray(
            [
                [1, 0, -1, 0, 0, 1, 0, -1],
                [0, 0, -1, -1, 1, 0, 0, 0],
            ]
        )
        leakage = np.asarray(
            [
                [[1, 1, 0, 1, 1, 0, 1, 0], [1, 0, 1, 0, 1, 1, 1, 0]],
                [[0, 1, 0, 0, 1, 0, 0, 1], [1, 1, 0, 0, 0, 1, 0, 0]],
            ]
        )
        indices = np.arange(len(source))
        alpha = 0.01
        target_bound, leakage_bound = candidate_bounds(
            target_harm,
            leakage,
            source,
            environment,
            indices,
            alpha,
        )
        for candidate in range(2):
            certificates = balanced_contract_certificates(
                {
                    f"target::environment={group}": target_harm[
                        candidate, environment == group
                    ]
                    for group in (0, 1)
                },
                {
                    f"attacker-{attacker}": leakage[candidate, attacker]
                    for attacker in range(2)
                },
                source,
                gamma=1.0,
                local_failure_probability=alpha,
            )
            expected_target = max(
                certificate.upper_confidence_bound
                for key, certificate in certificates.items()
                if key.startswith("target::")
            )
            expected_leakage = max(
                certificate.upper_confidence_bound
                for key, certificate in certificates.items()
                if key.startswith("balanced_leakage::")
            )
            self.assertAlmostEqual(target_bound[candidate], expected_target)
            self.assertAlmostEqual(leakage_bound[candidate], expected_leakage)


class GroupContractFamilyTests(unittest.TestCase):
    def test_contracts_are_partitioned_by_environment(self) -> None:
        samples = {
            "target::environment=0": np.zeros(10),
            "leakage::linear::environment=0::source=1": np.zeros(5),
            "target::environment=2": np.zeros(8),
        }
        supports = {
            "target::environment=0": (-1, 0, 1),
            "leakage::linear::environment=0::source=1": (0, 1),
            "target::environment=2": (-1, 0, 1),
        }
        thresholds = {key: 0.1 for key in samples}
        grouped_samples, grouped_supports, grouped_thresholds = group_contract_family(
            samples, supports, thresholds
        )
        self.assertEqual(set(grouped_samples), {"0", "2"})
        self.assertEqual(len(grouped_samples["0"]), 2)
        self.assertEqual(grouped_supports["2"]["target::environment=2"], (-1, 0, 1))
        self.assertEqual(grouped_thresholds["0"]["target::environment=0"], 0.1)


class SecondaryAblationTests(unittest.TestCase):
    def test_reduced_attacker_ablation_keeps_full_external_safety(self) -> None:
        candidates = [
            {
                "candidate": "INLP::rank=1",
                "method": "INLP",
                "validation_max_balanced_leakage": 0.4,
                "validation_max_target_harm": 0.01,
                "target_ucb": 0.02,
                "attacker_ucbs": {"linear": 0.5, "mlp": 0.9},
                "external_max_target_harm": 0.02,
                "external_max_balanced_leakage": 0.9,
            }
        ]
        row = make_row(
            dimension="attacker_portfolio",
            condition="linear_only",
            dataset="D",
            seed=5,
            target_threshold=0.1,
            leakage_threshold=0.7,
            gamma=1.0,
            support_mismatch=False,
            candidates=candidates,
            attackers={"linear"},
        )
        self.assertTrue(row["deployed"])
        self.assertTrue(row["measured_external_contract_violation"])

    def test_support_mismatch_forces_ablation_abstention(self) -> None:
        candidates = [
            {
                "candidate": "LEACE::closed_form",
                "method": "LEACE",
                "validation_max_balanced_leakage": 0.4,
                "validation_max_target_harm": 0.0,
                "target_ucb": 0.01,
                "attacker_ucbs": {"linear": 0.5},
                "external_max_target_harm": 0.0,
                "external_max_balanced_leakage": 0.5,
            }
        ]
        row = make_row(
            dimension="eraser_frontier",
            condition="all",
            dataset="Camelyon17-WILDS",
            seed=5,
            target_threshold=0.1,
            leakage_threshold=0.7,
            gamma=1.0,
            support_mismatch=True,
            candidates=candidates,
            attackers={"linear"},
        )
        self.assertFalse(row["deployed"])
        self.assertFalse(row["external_contract_estimable"])

    def test_attacker_portfolio_keeps_target_and_named_attackers(self) -> None:
        samples = {
            "target::environment=0": np.zeros(4),
            "leakage::linear::environment=0::source=0": np.zeros(2),
            "leakage::forest::environment=0::source=0": np.zeros(2),
        }
        supports = {
            "target::environment=0": (-1, 0, 1),
            "leakage::linear::environment=0::source=0": (0, 1),
            "leakage::forest::environment=0::source=0": (0, 1),
        }
        filtered_samples, filtered_supports = filter_portfolio(
            samples, supports, {"linear"}
        )
        self.assertEqual(
            set(filtered_samples),
            {
                "target::environment=0",
                "leakage::linear::environment=0::source=0",
            },
        )
        self.assertEqual(set(filtered_samples), set(filtered_supports))

    def test_selection_uses_radius_support_and_registered_tiebreak(self) -> None:
        candidates = [
            {
                "candidate": "A",
                "method": "INLP",
                "support_mismatch": "False",
                "certified_radius": "1.5",
                "validation_max_leakage": "0.4",
                "validation_max_target_harm": "0.1",
            },
            {
                "candidate": "B",
                "method": "LEACE",
                "support_mismatch": "False",
                "certified_radius": "1.2",
                "validation_max_leakage": "0.2",
                "validation_max_target_harm": "0.0",
            },
            {
                "candidate": "C",
                "method": "TaCo",
                "support_mismatch": "True",
                "certified_radius": "8.0",
                "validation_max_leakage": "0.1",
                "validation_max_target_harm": "0.0",
            },
        ]
        selected = select_candidate(candidates, gamma=1.25)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["candidate"], "A")
        self.assertIsNone(
            select_candidate(candidates, gamma=1.25, excluded_method="INLP")
        )

    def test_cluster_summary_keeps_configuration_denominator(self) -> None:
        records = [
            {
                "dataset": "D",
                "seed": seed,
                "deployed": deployed,
                "violation": violation,
                "safe": deployed and not violation,
                "oracle_deployed": True,
            }
            for seed, deployed, violation in [
                (0, True, True),
                (0, False, False),
                (1, True, False),
                (1, True, False),
            ]
        ]
        summary = cluster_summary(records, bootstrap_seed=3, replicates=100)
        self.assertEqual(summary["seed_cluster_count"], 2)
        self.assertAlmostEqual(summary["deployment_rate"], 0.75)
        self.assertAlmostEqual(summary["measured_external_violation_rate"], 0.25)
        self.assertAlmostEqual(summary["violation_rate_conditional_on_deployment"], 1 / 3)
        self.assertAlmostEqual(summary["safe_deployment_retention"], 0.5)

if __name__ == "__main__":
    unittest.main()
