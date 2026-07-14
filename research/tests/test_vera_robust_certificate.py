from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from vera_robust_certificate import (  # noqa: E402
    certify_shift_radius,
    certify_edits,
    empirical_reweighting_risk,
    robust_risk_certificate,
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


if __name__ == "__main__":
    unittest.main()
