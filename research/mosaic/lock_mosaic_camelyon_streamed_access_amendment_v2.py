#!/usr/bin/env python3
"""Lock the Camelyon serial-access recovery before features or outcomes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = (
    ROOT
    / "research/mosaic/"
    "prereg_mosaic_camelyon_streamed_confirmation_v1.json"
)
OUTPUT = (
    ROOT
    / "research/mosaic/"
    "prereg_mosaic_camelyon_streamed_access_amendment_v2.json"
)
RECOVERY = (
    ROOT
    / "research/mosaic/"
    "prepare_camelyon_streamed_confirmation_store_v2.py"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    payload = {
        "name": "MOSAIC Camelyon streamed data-access amendment v2",
        "status": "locked_before_recovery_features_and_outcomes",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "trigger": {
            "failed_script": (
                "research/mosaic/"
                "prepare_camelyon_streamed_confirmation_store.py"
            ),
            "failure": "HTTP 429 while opening train shard 2 of 14",
            "outcomes_available": False,
            "completed_manifest_available": False,
        },
        "scope": (
            "Data access only: serialize DuckDB HTTP reads, add bounded "
            "exponential retry, and checkpoint completed shards. Selected IDs, "
            "remote revision, images, preprocessing, model, feature layer, "
            "seeds, thresholds, bridge, optimizer, and gates are unchanged."
        ),
        "base_preregistration_sha256": sha256(BASE),
        "code_sha256": {
            (
                "research/mosaic/"
                "prepare_camelyon_streamed_confirmation_store_v2.py"
            ): sha256(RECOVERY),
        },
        "access_controls": {
            "duckdb_threads": 1,
            "duckdb_http_retries": 12,
            "duckdb_http_retry_wait_ms": 2000,
            "duckdb_http_timeout_seconds": 120,
            "max_attempts_per_shard": 8,
            "base_backoff_seconds": 30,
            "inter_shard_wait_seconds": 15,
            "checkpoint_unit": "completed parquet shard",
        },
    }
    OUTPUT.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    sidecar = OUTPUT.with_suffix(OUTPUT.suffix + ".sha256")
    sidecar.write_text(f"{sha256(OUTPUT)}  {OUTPUT.name}\n", encoding="utf-8")
    print(json.dumps({"lock": str(OUTPUT), "sha256": sha256(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
