#!/usr/bin/env python3
"""Freeze multi-task ACS stores for held-out geographic-shift evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


DEFAULT_RAW = Path("/Volumes/Backups/FARO/artifacts/acs_folktables_raw")
DEFAULT_OUTPUT = Path("/Volumes/Backups/FARO/artifacts/acs_natural_shift_stores")
REFERENCE_STATE = "CA"
TARGET_STATES = ("WA", "IL", "NY", "FL")
TASK_NAMES = ("income", "employment", "public_coverage")
SOURCE_COLUMN = "SEX"
ENVIRONMENT_COLUMN = "PUMA"
REFERENCE_FRACTION = 0.20
SPLIT_TRAIN = 0
SPLIT_REFERENCE = 1
SPLIT_EXTERNAL = 2
STATE_CODES = {"CA": "06", "FL": "12", "IL": "17", "NY": "36", "WA": "53"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def task_registry() -> dict[str, object]:
    from folktables import ACSIncome, ACSEmployment, ACSPublicCoverage

    return {
        "income": ACSIncome,
        "employment": ACSEmployment,
        "public_coverage": ACSPublicCoverage,
    }


def extract_task(data: object, problem: object) -> tuple[np.ndarray, ...]:
    """Apply the official task filter while excluding sex from released features."""

    filtered = problem._preprocess(data).copy()
    feature_columns = tuple(value for value in problem.features if value != SOURCE_COLUMN)
    features = problem._postprocess(filtered.loc[:, feature_columns].to_numpy())
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(
        problem.target_transform(filtered[problem.target]).to_numpy(), dtype=np.int8
    )
    source_values = np.asarray(filtered[SOURCE_COLUMN].to_numpy(), dtype=np.int16)
    sources = (source_values == 2).astype(np.int8)
    environments = np.asarray(filtered[ENVIRONMENT_COLUMN].to_numpy(), dtype=np.int32)
    if set(int(value) for value in np.unique(source_values)) != {1, 2}:
        raise ValueError("unexpected ACS sex coding")
    if set(int(value) for value in np.unique(labels)) != {0, 1}:
        raise ValueError("registered ACS tasks must remain binary")
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError("ACS task features must be a finite matrix")
    if not all(
        len(value) == len(features) for value in (labels, sources, environments)
    ):
        raise ValueError("ACS task arrays disagree")
    return features, labels, sources, environments, feature_columns


def reference_split(size: int, *, seed: int) -> np.ndarray:
    if size < 2:
        raise ValueError("reference state requires at least two records")
    count = min(max(1, int(np.floor(REFERENCE_FRACTION * size))), size - 1)
    order = np.random.default_rng(seed).permutation(size)
    split = np.full(size, SPLIT_TRAIN, dtype=np.int8)
    split[order[:count]] = SPLIT_REFERENCE
    return split


def value_counts(values: np.ndarray) -> dict[str, int]:
    keys, counts = np.unique(values, return_counts=True)
    return {str(int(key)): int(count) for key, count in zip(keys, counts, strict=True)}


def raw_asset(raw_dir: Path, *, year: str, state: str) -> Path:
    return raw_dir / year / "1-Year" / f"psam_p{STATE_CODES[state]}.csv"


def write_store(
    *,
    output_root: Path,
    raw_dir: Path,
    survey_year: str,
    task_name: str,
    problem: object,
    target_state: str,
    reference: tuple[np.ndarray, ...],
    target: tuple[np.ndarray, ...],
    seed: int,
) -> Path:
    output = output_root / f"acs_{task_name}_ca_{target_state.lower()}_natural_store"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite populated store: {output}")
    x_ref, y_ref, s_ref, env_ref, feature_columns = reference
    x_target, y_target, s_target, env_target, target_columns = target
    if feature_columns != target_columns:
        raise RuntimeError("reference and target feature columns differ")
    split_ref = reference_split(len(x_ref), seed=seed)
    split_target = np.full(len(x_target), SPLIT_EXTERNAL, dtype=np.int8)
    z = np.concatenate((x_ref, x_target), axis=0)
    y = np.concatenate((y_ref, y_target), axis=0)
    s = np.concatenate((s_ref, s_target), axis=0)
    environment = np.concatenate((env_ref, env_target), axis=0)
    split = np.concatenate((split_ref, split_target), axis=0)
    state = np.concatenate(
        (np.zeros(len(x_ref), dtype=np.int8), np.ones(len(x_target), dtype=np.int8))
    )
    if not all(len(value) == len(z) for value in (y, s, environment, split, state)):
        raise RuntimeError("ACS natural-shift arrays are incomplete")

    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "z": z,
        "y": y,
        "s": s,
        "environment": environment,
        "split": split,
        "g": state,
    }
    for name, values in arrays.items():
        np.save(output / f"{name}.npy", values)
    raw_paths = {
        value: raw_asset(raw_dir, year=survey_year, state=value)
        for value in (REFERENCE_STATE, target_state)
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": f"Folktables {problem.target} {REFERENCE_STATE}-to-{target_state}",
        "source": "US Census ACS PUMS via Folktables 0.0.12",
        "survey_year": survey_year,
        "horizon": "1-Year",
        "task": task_name,
        "official_target_column": problem.target,
        "states": {"reference": REFERENCE_STATE, "external": target_state},
        "n_examples": int(len(z)),
        "dimension": int(z.shape[1]),
        "arrays": {name: f"{name}.npy" for name in arrays},
        "split_codes": {
            "eraser_train": SPLIT_TRAIN,
            "reference": SPLIT_REFERENCE,
            "external_target": SPLIT_EXTERNAL,
        },
        "source_concept": "binary sex; excluded from the released feature matrix",
        "environment": "target-state Public Use Microdata Area (PUMA)",
        "feature_columns": list(feature_columns),
        "reference_fraction": REFERENCE_FRACTION,
        "split_seed": seed,
        "target_counts": value_counts(y),
        "source_counts": value_counts(s),
        "split_counts": value_counts(split),
        "external_environment_count": int(len(np.unique(env_target))),
        "state_target_counts": {
            REFERENCE_STATE: value_counts(y_ref),
            target_state: value_counts(y_target),
        },
        "state_source_counts": {
            REFERENCE_STATE: value_counts(s_ref),
            target_state: value_counts(s_target),
        },
        "raw_assets": {
            str(path.relative_to(raw_dir)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in raw_paths.values()
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-state", default=REFERENCE_STATE, choices=(REFERENCE_STATE,))
    parser.add_argument("--target-states", nargs="+", default=list(TARGET_STATES))
    parser.add_argument("--tasks", nargs="+", choices=TASK_NAMES, default=list(TASK_NAMES))
    parser.add_argument("--survey-year", default="2018")
    parser.add_argument("--seed", type=int, default=20_270_721)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    unknown_states = set(args.target_states) - set(TARGET_STATES)
    if unknown_states:
        raise ValueError(f"states outside the registered design: {sorted(unknown_states)}")

    from folktables import ACSDataSource

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    source = ACSDataSource(
        survey_year=args.survey_year,
        horizon="1-Year",
        survey="person",
        root_dir=str(args.raw_dir),
    )
    problems = task_registry()
    reference_frame = source.get_data(states=[REFERENCE_STATE], download=args.download)
    reference_tasks = {
        name: extract_task(reference_frame, problems[name]) for name in args.tasks
    }
    outputs: list[str] = []
    for state_index, state in enumerate(args.target_states):
        target_frame = source.get_data(states=[state], download=args.download)
        for task_index, task_name in enumerate(args.tasks):
            target_task = extract_task(target_frame, problems[task_name])
            output = write_store(
                output_root=args.output_root,
                raw_dir=args.raw_dir,
                survey_year=args.survey_year,
                task_name=task_name,
                problem=problems[task_name],
                target_state=state,
                reference=reference_tasks[task_name],
                target=target_task,
                seed=args.seed + 100 * state_index + task_index,
            )
            outputs.append(str(output))
        del target_frame
    print(json.dumps({"stores": outputs}, indent=2))


if __name__ == "__main__":
    main()
