#!/usr/bin/env python3
"""Run the locked ACS proxy-label confirmation with source-mass calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import run_mosaic_real_proxy_powered as powered
from mosaic_proxy_bridge import certify_proxy_label_conditionals


ROOT = Path(__file__).resolve().parents[2]
LOCK = (
    ROOT / "research/mosaic/prereg_mosaic_real_proxy_mass_confirmation_v1.json"
)
OUTPUT = (
    ROOT / "research/artifacts/mosaic_real_proxy_mass_confirmation_v1.json"
)
SEED = 20270725
BASE_EXPECTED_PROTOCOL = powered.expected_protocol


def expected_protocol() -> dict[str, Any]:
    protocol = BASE_EXPECTED_PROTOCOL()
    protocol.update(
        {
            "design": (
                "locked source-mass confirmation after the powered proxy "
                "study abstained; no new diagnostic outcome selects the design"
            ),
            "seed": SEED,
            "calibration_model": (
                "task-label, true-source, and two-token-specific source-to-"
                "proxy confusion tensor plus simultaneous task-label-specific "
                "true-source mass intervals"
            ),
            "source_mass_family_allocation": (
                "5 percent of the .05 familywise budget, split across four "
                "label-source binomial intervals"
            ),
            "conditional_center": (
                "exact L1 Chebyshev center over each calibrated latent-law "
                "polytope"
            ),
            "primary_gate": (
                "full-calibration release, proxy balanced accuracy at least "
                ".60, and zero diagnostic contract violations"
            ),
        }
    )
    return protocol


def mass_calibrated_certificate(
    *args: Any,
    calibration_confusion_counts: Any,
    **kwargs: Any,
):
    confusion = np.asarray(calibration_confusion_counts)
    if confusion.ndim != 4:
        raise ValueError("mass confirmation requires JxGxKxG calibration")
    source_counts = confusion.sum(axis=(2, 3))
    return certify_proxy_label_conditionals(
        *args,
        calibration_confusion_counts=confusion,
        calibration_source_counts=source_counts,
        **kwargs,
    )


def configure() -> None:
    powered.LOCK = LOCK
    powered.OUTPUT = OUTPUT
    powered.SEED = SEED
    powered.expected_protocol = expected_protocol
    powered.certify_proxy_label_conditionals = mass_calibrated_certificate


def main() -> None:
    configure()
    powered.main()
    output_path = OUTPUT
    if "--output" in sys.argv:
        output_path = Path(sys.argv[sys.argv.index("--output") + 1])
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    payload["name"] = "MOSAIC real ACS proxy source-mass confirmation v1"
    payload["status"] = "complete_locked_source_mass_confirmation"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
