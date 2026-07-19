#!/usr/bin/env python3
"""Lock the post-outcome released-interface utility analysis before running it."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from mosaic_release_utility_common import selected_jobs
from mosaic_real import sha256


ROOT = Path(__file__).resolve().parent
REPOSITORY = ROOT.parents[1]
DEFAULT_OUTPUT = ROOT / "prereg_mosaic_release_utility_v1.json"
STRICT_DIR = REPOSITORY / "research/artifacts/mosaic_bridge_strict_v2_receipts_v1"
UTILITY_OUTPUT = REPOSITORY / "research/artifacts/mosaic_release_utility_v1.json"
SLICES = {"BiasBios-Clinical": ["0.40"], "Waterbirds": ["0.49"]}
CODE_FILES = (
    "research/mosaic/mosaic_release_utility_common.py",
    "research/mosaic/run_mosaic_release_utility.py",
    "research/mosaic/lock_mosaic_release_utility.py",
    "research/mosaic/audit_mosaic_release_utility.py",
    "research/mosaic/run_mosaic_bridge_frontier.py",
    "research/mosaic/mosaic_real.py",
    "research/scripts/run_official_eraser_frontier.py",
    "research/scripts/official_eraser_adapters.py",
    "research/tests/test_mosaic_release_utility.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("refusing to overwrite an existing utility lock")
    if UTILITY_OUTPUT.exists():
        raise FileExistsError("utility results already exist")
    jobs = selected_jobs(STRICT_DIR, SLICES)
    payload = {
        "project": "MOSAIC released-interface task utility analysis",
        "status": "locked_post_outcome_utility_analysis_before_execution",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "slices": SLICES,
        "expected_deployed_release_count": len(jobs),
        "selection_source": "strict v2 MOSAIC receipts, retained exactly as already recorded",
        "timing_disclosure": (
            "This is a reviewer-requested post-outcome measurement of already selected "
            "releases. It is not a preregistered efficacy result and cannot alter any "
            "MOSAIC deployment decision."
        ),
        "metrics": {
            "released_interface": "Exact one-draw expected diagnostic accuracy and balanced accuracy of the serialized (M,g).",
            "tokenizer": "Diagnostic accuracy and balanced accuracy of a four-bin tokenizer decoder fit on construction labels only.",
            "selected_edit": "Diagnostic accuracy and balanced accuracy of a full-feature logistic task classifier trained on eraser-training labels.",
            "unedited": "Diagnostic accuracy and balanced accuracy of the same full-feature logistic task classifier on the unedited preprocessed representation.",
        },
        "claim_boundary": (
            "The comparison describes task information retained by the finite released "
            "interface on untouched diagnostics. It does not certify unrestricted "
            "downstream classifiers or establish application-level utility."
        ),
        "code_sha256": {relative: sha256(REPOSITORY / relative) for relative in CODE_FILES},
    }
    data = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.write_text(data, encoding="utf-8")
    sidecar.write_text(hashlib.sha256(data.encode("utf-8")).hexdigest() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
