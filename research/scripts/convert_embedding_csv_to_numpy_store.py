"""Convert a FARO embedding CSV table into the NumPy store format."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SPLIT_CODES = {
    "train": 0,
    "validation": 1,
    "external": 2,
    "counterfactual": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        embedding_fields = [field for field in reader.fieldnames if field.startswith("embedding_")]
        required = {"id", "split", "y", "s"}
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required fields: {missing}")
        if not embedding_fields:
            raise ValueError("CSV has no embedding_* fields")

        ids: list[str] = []
        y_values: list[int] = []
        s_values: list[int] = []
        split_values: list[int] = []
        embeddings: list[list[float]] = []
        split_counts: Counter[str] = Counter()
        for row in reader:
            split_name = row["split"]
            if split_name not in SPLIT_CODES:
                raise ValueError(f"unknown split {split_name!r}")
            ids.append(row["id"])
            y_values.append(int(row["y"]))
            s_values.append(int(row["s"]))
            split_values.append(SPLIT_CODES[split_name])
            split_counts[split_name] += 1
            embeddings.append([float(row[field]) for field in embedding_fields])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    z = np.asarray(embeddings, dtype=np.float32)
    y = np.asarray(y_values, dtype=np.int64)
    s = np.asarray(s_values, dtype=np.int64)
    split = np.asarray(split_values, dtype=np.int64)

    np.save(args.output_dir / "z.npy", z)
    np.save(args.output_dir / "y.npy", y)
    np.save(args.output_dir / "s.npy", s)
    np.save(args.output_dir / "split.npy", split)
    with (args.output_dir / "ids.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "id"])
        writer.writerows((idx, value) for idx, value in enumerate(ids))

    manifest = {
        "format": "trace_embedding_store_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": args.name,
        "source_csv": str(args.csv_path),
        "n_examples": int(z.shape[0]),
        "feature_count": int(z.shape[1]),
        "dtype": "float32",
        "split_codes": SPLIT_CODES,
        "split_counts": dict(sorted(split_counts.items())),
        "arrays": {
            "z": "z.npy",
            "y": "y.npy",
            "s": "s.npy",
            "split": "split.npy",
        },
        "ids": "ids.csv",
        "claim_grade_embedding_store": True,
        "claim_boundary": "Converted from an existing claim-grade FARO embedding CSV table.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "n_examples": int(z.shape[0])}))


if __name__ == "__main__":
    main()
