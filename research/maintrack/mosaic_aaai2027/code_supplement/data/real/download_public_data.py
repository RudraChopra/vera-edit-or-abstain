#!/usr/bin/env python3
"""Acquire package-managed public data without embedding external addresses."""

from __future__ import annotations

import argparse
from pathlib import Path


def download_wilds(name: str, destination: Path) -> None:
    try:
        from wilds import get_dataset
    except ImportError as error:
        raise SystemExit("Install the optional WILDS package to download this dataset.") from error
    get_dataset(dataset=name, download=True, root_dir=str(destination))


def download_acs(destination: Path, states: list[str]) -> None:
    try:
        from folktables import ACSDataSource
    except ImportError as error:
        raise SystemExit("Install the optional Folktables package to download ACS data.") from error
    source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person", root_dir=str(destination))
    source.get_data(states=states, download=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "camelyon17_wilds",
            "civilcomments_wilds",
            "acs_income_ca_tx",
            "acs_multistate",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dataset == "camelyon17_wilds":
        download_wilds("camelyon17", args.output)
    elif args.dataset == "civilcomments_wilds":
        download_wilds("civilcomments", args.output)
    elif args.dataset == "acs_income_ca_tx":
        download_acs(args.output, ["CA", "TX"])
    else:
        download_acs(args.output, ["CA", "WA", "IL", "NY", "FL"])
    print(f"public data prepared under {args.output}")


if __name__ == "__main__":
    main()
