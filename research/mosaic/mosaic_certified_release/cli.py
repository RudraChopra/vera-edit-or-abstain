"""Command-line utilities for MOSAIC auditors."""

from __future__ import annotations

import argparse
import json
import platform
from importlib import metadata

from .report import verify_report


def runtime() -> dict[str, str]:
    values = {"python": platform.python_version()}
    for name in ("mosaic-certified-release", "numpy", "scipy", "scikit-learn"):
        try:
            values[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            values[name] = "missing"
    return values


def main() -> None:
    parser = argparse.ArgumentParser(prog="mosaic-audit")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="print the versioned runtime")
    verify = subparsers.add_parser(
        "verify-report", help="validate a JSON certification report"
    )
    verify.add_argument("path")
    args = parser.parse_args()
    result = (
        runtime()
        if args.command == "doctor"
        else verify_report(args.path)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
