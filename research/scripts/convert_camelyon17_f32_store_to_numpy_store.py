"""Convert the Camelyon17 raw float32 embedding store into a NumPy store."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = json.loads(args.source_report.read_text(encoding="utf-8"))
    manifest_csv = Path(str(source["manifest_path"]))
    embeddings_path = Path(str(source["embeddings_path"]))
    n_examples = int(source["sample_count"])
    feature_count = int(source["feature_count"])
    expected_bytes = n_examples * feature_count * np.dtype("float32").itemsize
    if embeddings_path.stat().st_size != expected_bytes:
        raise ValueError(
            f"embedding byte size mismatch: got {embeddings_path.stat().st_size}, "
            f"expected {expected_bytes}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    z = np.memmap(
        embeddings_path,
        dtype=np.float32,
        mode="r",
        shape=(n_examples, feature_count),
    )
    np.save(args.out_dir / "z.npy", np.asarray(z))

    y = np.empty(n_examples, dtype=np.int64)
    s = np.empty(n_examples, dtype=np.int64)
    split = np.empty(n_examples, dtype=np.int64)
    ids_path = args.out_dir / "ids.csv"
    split_name_to_code = {
        "train": 0,
        "validation": 1,
        "external": 2,
        "counterfactual": 3,
    }
    split_counts: dict[str, int] = {}
    with manifest_csv.open(newline="", encoding="utf-8") as src, ids_path.open(
        "w", newline="", encoding="utf-8"
    ) as dst:
        reader = csv.DictReader(src)
        writer = csv.writer(dst)
        writer.writerow(["row", "id"])
        for row_idx, row in enumerate(reader):
            split_name = row.get("split", "")
            if split_name not in split_name_to_code:
                raise ValueError(f"unknown split {split_name!r} at row {row_idx}")
            y[row_idx] = int(row["y"])
            s[row_idx] = int(row["s"])
            split[row_idx] = split_name_to_code[split_name]
            split_counts[split_name] = split_counts.get(split_name, 0) + 1
            writer.writerow([row_idx, row.get("id", str(row_idx))])
    if row_idx + 1 != n_examples:
        raise ValueError(f"manifest row count mismatch: got {row_idx + 1}, expected {n_examples}")

    np.save(args.out_dir / "y.npy", y)
    np.save(args.out_dir / "s.npy", s)
    np.save(args.out_dir / "split.npy", split)
    manifest = {
        "format": "trace_embedding_store_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": "FARO Camelyon17 ResNet18 NumPy embedding store",
        "source_report": str(args.source_report),
        "source_manifest_csv": str(manifest_csv),
        "source_embeddings_f32": str(embeddings_path),
        "n_examples": n_examples,
        "feature_count": feature_count,
        "dtype": "float32",
        "split_codes": split_name_to_code,
        "split_counts": dict(sorted(split_counts.items())),
        "arrays": {
            "z": "z.npy",
            "y": "y.npy",
            "s": "s.npy",
            "split": "split.npy",
        },
        "ids": "ids.csv",
        "claim_grade_embedding_store": True,
        "claim_grade_benchmark_row": False,
        "claim_boundary": "Frozen embedding store only; benchmark claims require receipts and paired statistics.",
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = {
        "name": "FARO Camelyon17 NumPy store conversion report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str(args.source_report),
        "out_dir": str(args.out_dir),
        "n_examples": n_examples,
        "feature_count": feature_count,
        "claim_grade_embedding_store": True,
        "claim_grade_benchmark_row": False,
        "manifest_json": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "z_sha256": sha256_file(args.out_dir / "z.npy"),
        "y_sha256": sha256_file(args.out_dir / "y.npy"),
        "s_sha256": sha256_file(args.out_dir / "s.npy"),
        "split_sha256": sha256_file(args.out_dir / "split.npy"),
        "ids_sha256": sha256_file(ids_path),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "n_examples": n_examples}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
