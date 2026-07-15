from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from audit_theory_code_mapping import (  # noqa: E402
    REQUIRED_COLUMNS,
    audit_mapping,
)


class TheoryCodeMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mapping = json.loads(
            (ROOT / "theory_code_mapping.json").read_text(encoding="utf-8")
        )
        cls.theory = (ROOT / "maintrack" / "appendix_shift_robust_theory.tex").read_text(
            encoding="utf-8"
        )
        cls.implementation = (ROOT / "scripts" / "vera_robust_certificate.py").read_text(
            encoding="utf-8"
        )
        cls.analysis = (
            ROOT / "scripts" / "design_vera_controlled_shift_study.py"
        ).read_text(encoding="utf-8") + "\n" + (
            ROOT / "scripts" / "vera_controlled_shift.py"
        ).read_text(encoding="utf-8")

    def audit(self, mapping: dict[str, object]) -> list[str]:
        return audit_mapping(
            mapping,
            theory_text=self.theory,
            implementation_text=self.implementation,
            analysis_text=self.analysis,
        )

    def test_checked_in_mapping_passes(self) -> None:
        self.assertEqual(self.audit(self.mapping), [])

    def test_every_required_column_fails_closed_when_removed(self) -> None:
        for column in REQUIRED_COLUMNS:
            with self.subTest(column=column):
                mutated = copy.deepcopy(self.mapping)
                del mutated["rows"][0][column]
                self.assertTrue(self.audit(mutated))

    def test_component_duplication_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.mapping)
        mutated["rows"][0]["component"] = mutated["rows"][1]["component"]
        self.assertTrue(self.audit(mutated))

    def test_missing_code_symbol_fails_closed(self) -> None:
        self.assertTrue(
            audit_mapping(
                self.mapping,
                theory_text=self.theory,
                implementation_text="",
                analysis_text="",
            )
        )

    def test_missing_theorem_label_fails_closed(self) -> None:
        self.assertTrue(
            audit_mapping(
                self.mapping,
                theory_text="",
                implementation_text=self.implementation,
                analysis_text=self.analysis,
            )
        )


if __name__ == "__main__":
    unittest.main()
