#!/usr/bin/env python3
"""Replay every multistate ACS bridge and release with exact arithmetic."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_mosaic_acs_natural_shift import main


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--prereg",
        str(ROOT / "artifacts/natural_shift/preregistration.json"),
        "--data-lock",
        str(ROOT / "data/real/acs_natural_shift_data_lock.json"),
        "--receipts",
        str(ROOT / "artifacts/natural_shift/receipts"),
        "--output",
        str(ROOT / "artifacts/reproduced/acs_natural_shift_audit.json"),
    ]
    main()
