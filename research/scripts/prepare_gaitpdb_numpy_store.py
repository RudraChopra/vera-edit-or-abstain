"""Build a FARO NumPy store from PhysioNet Gait in Parkinson's Disease files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SPLIT_CODES = {"train": 0, "validation": 1, "external": 2}


def stable_unit(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def parse_name(name: str) -> tuple[str, int, int, str]:
    stem = Path(name).stem
    site = stem[:2]
    cohort = stem[2:4]
    if site not in {"Ga", "Ju", "Si"} or cohort not in {"Pt", "Co"}:
        raise ValueError(f"not a gait data recording: {name}")
    subject = stem.split("_", 1)[0]
    y = 1 if cohort == "Pt" else 0
    source = 1 if site == "Ju" else 0
    return subject, y, source, site


def assign_subject_splits(subject_rows: dict[str, list[str]]) -> dict[str, str]:
    grouped: dict[tuple[int, int], list[str]] = defaultdict(list)
    for subject, files in subject_rows.items():
        _, y, s, _ = parse_name(files[0])
        grouped[(y, s)].append(subject)
    split_by_subject: dict[str, str] = {}
    for key, subjects in grouped.items():
        subjects = sorted(subjects, key=lambda item: stable_unit(f"{key}:{item}"))
        n = len(subjects)
        n_train = max(1, round(0.6 * n))
        n_val = max(1, round(0.2 * n)) if n >= 3 else 0
        for idx, subject in enumerate(subjects):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "validation"
            else:
                split = "external"
            split_by_subject[subject] = split
    return split_by_subject


def signal_features(array: np.ndarray) -> np.ndarray:
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError("expected a 2D gait array with time plus sensor columns")
    x = np.asarray(array[:, 1:], dtype=np.float64)
    x = x[np.isfinite(x).all(axis=1)]
    if x.shape[0] < 4:
        raise ValueError("too few finite gait rows")
    dx = np.diff(x, axis=0)
    total = x.sum(axis=1, keepdims=True)
    blocks = [x, dx, total]
    features: list[float] = [float(x.shape[0])]
    for block in blocks:
        q25, q75 = np.percentile(block, [25, 75], axis=0)
        stats = [
            block.mean(axis=0),
            block.std(axis=0),
            block.min(axis=0),
            block.max(axis=0),
            np.median(block, axis=0),
            q25,
            q75,
            np.sqrt(np.mean(block**2, axis=0)),
            np.mean(np.abs(block), axis=0),
        ]
        for stat in stats:
            features.extend(np.asarray(stat, dtype=np.float64).ravel().tolist())
    return np.asarray(features, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/Volumes/Backups/FARO/data/gaitpdb"))
    parser.add_argument("--out-dir", type=Path, default=Path("/Volumes/Backups/FARO/artifacts/gaitpdb_numpy_store"))
    parser.add_argument("--report", type=Path, default=Path("research/artifacts/gaitpdb_numpy_store_report.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sha_path = args.data_dir / "SHA256SUMS.txt"
    lines = sha_path.read_text(encoding="utf-8").splitlines()
    names = []
    for line in lines:
        if not line.strip():
            continue
        name = line.split(maxsplit=1)[1]
        if name.endswith(".txt") and name[:2] in {"Ga", "Ju", "Si"}:
            names.append(name)
    missing = [name for name in names if not (args.data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} gait files; first={missing[:5]}")

    subject_rows: dict[str, list[str]] = defaultdict(list)
    for name in names:
        subject, _, _, _ = parse_name(name)
        subject_rows[subject].append(name)
    split_by_subject = assign_subject_splits(subject_rows)

    ids: list[str] = []
    features: list[np.ndarray] = []
    y_values: list[int] = []
    s_values: list[int] = []
    split_values: list[int] = []
    site_values: list[str] = []
    for name in sorted(names):
        subject, y, s, site = parse_name(name)
        arr = np.loadtxt(args.data_dir / name)
        features.append(signal_features(arr))
        ids.append(Path(name).stem)
        y_values.append(y)
        s_values.append(s)
        site_values.append(site)
        split_values.append(SPLIT_CODES[split_by_subject[subject]])

    z = np.vstack(features).astype(np.float32)
    y = np.asarray(y_values, dtype=np.int64)
    s = np.asarray(s_values, dtype=np.int64)
    split = np.asarray(split_values, dtype=np.int64)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "z.npy", z)
    np.save(args.out_dir / "y.npy", y)
    np.save(args.out_dir / "s.npy", s)
    np.save(args.out_dir / "split.npy", split)
    with (args.out_dir / "ids.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "id", "site"])
        writer.writerows((idx, item, site_values[idx]) for idx, item in enumerate(ids))

    split_counts = Counter(int(item) for item in split)
    group_counts = Counter((int(a), int(b), int(c)) for a, b, c in zip(y, s, split))
    manifest = {
        "format": "trace_embedding_store_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": "FARO PhysioNet gaitpdb NumPy feature store",
        "source_dataset": "PhysioNet Gait in Parkinson's Disease v1.0.0",
        "source_url": "https://physionet.org/content/gaitpdb/1.0.0/",
        "n_examples": int(z.shape[0]),
        "feature_count": int(z.shape[1]),
        "target_label": "Parkinson patient vs control inferred from Pt/Co filename code",
        "source_label": "site_is_Ju binary source inferred from Ga/Ju/Si filename prefix",
        "split_protocol": "deterministic subject-level stratified locked split",
        "split_codes": SPLIT_CODES,
        "split_counts": {name: int((split == code).sum()) for name, code in SPLIT_CODES.items()},
        "arrays": {"z": "z.npy", "y": "y.npy", "s": "s.npy", "split": "split.npy"},
        "ids": "ids.csv",
        "claim_grade_embedding_store": True,
        "claim_boundary": "Public gait feature benchmark; not an official challenge split.",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {
        "name": "FARO gaitpdb NumPy store report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "store_dir": str(args.out_dir),
        "n_examples": int(z.shape[0]),
        "feature_count": int(z.shape[1]),
        "subject_count": len(subject_rows),
        "split_counts": manifest["split_counts"],
        "target_counts": {str(k): int(v) for k, v in Counter(y_values).items()},
        "source_counts": {str(k): int(v) for k, v in Counter(s_values).items()},
        "site_counts": {str(k): int(v) for k, v in Counter(site_values).items()},
        "target_source_split_counts": {str(k): int(v) for k, v in group_counts.items()},
        "claim_grade_embedding_store": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"store_dir": str(args.out_dir), "n_examples": int(z.shape[0]), "feature_count": int(z.shape[1])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
