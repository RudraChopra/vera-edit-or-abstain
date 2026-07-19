#!/usr/bin/env python3
"""Run a clearly labeled post-review ACS alphabet-scale extension."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_mosaic_bridge_frontier as runner


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "research/artifacts/mosaic_acs_k8_exploratory_seed1309.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine-token-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1309)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    if arguments.fine_token_count < 4:
        raise ValueError("fine-token-count must be at least four")
    if arguments.output.exists():
        raise FileExistsError(f"refusing to overwrite {arguments.output}")
    runner.FINE_TOKEN_COUNT = arguments.fine_token_count
    previous_argv = sys.argv
    sys.argv = [
        str(Path(runner.__file__)),
        "--dataset",
        "ACSIncome-CA-TX",
        "--seed",
        str(arguments.seed),
        "--output",
        str(arguments.output),
    ]
    try:
        runner.main()
    finally:
        sys.argv = previous_argv
    payload = json.loads(arguments.output.read_text(encoding="utf-8"))
    payload["analysis_status"] = "post-review exploratory alphabet extension"
    payload["claim_boundary"] = (
        "This K>4 point reuses a previously studied seed and was not preregistered. "
        "It measures real-table feasibility and runtime, not confirmatory error control."
    )
    runner.atomic_json_dump(payload, arguments.output)


if __name__ == "__main__":
    main()
