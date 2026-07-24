#!/usr/bin/env python3
"""Recover the locked Camelyon store with serial, checkpointed HTTP access."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from prepare_camelyon_streamed_confirmation_store import (
    DEFAULT_METADATA,
    DEFAULT_OUTPUT,
    DEFAULT_PREREG,
    sha256,
    sha256_array,
    validate_lock,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AMENDMENT = (
    ROOT
    / "research/mosaic/"
    "prereg_mosaic_camelyon_streamed_access_amendment_v2.json"
)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_amendment(path: Path, prereg_path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(path):
        raise ValueError("Camelyon access-amendment sidecar mismatch")
    amendment = load_json(path)
    if amendment.get("status") != "locked_before_recovery_features_and_outcomes":
        raise RuntimeError("Camelyon access amendment is not locked")
    if amendment["base_preregistration_sha256"] != sha256(prereg_path):
        raise RuntimeError("Camelyon access amendment names another base lock")
    for relative, expected in amendment["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"access-amendment source mismatch: {relative}")
    for local in (path, sidecar):
        relative = local.relative_to(ROOT)
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if committed != local.read_bytes():
            raise RuntimeError(f"{relative} is not the committed amendment")
    return amendment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prereg, selected = validate_lock(args.prereg, args.metadata)
    amendment = validate_amendment(args.amendment, args.prereg)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"completed store already exists: {args.output}")

    import duckdb
    import torch
    from PIL import Image
    from torchvision.models import ResNet18_Weights, resnet18

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    transform = weights.transforms()

    args.output.mkdir(parents=True, exist_ok=True)
    ids = selected["image_id"].to_numpy(dtype=np.uint32)
    target = selected["y"].to_numpy(dtype=np.int8)
    centers = selected["center"].to_numpy(dtype=np.int8)
    source = selected["source"].to_numpy(dtype=np.int8)
    split = np.where(
        selected["split"].to_numpy() == "train", 0, 1
    ).astype(np.int8)
    positions = {int(identifier): index for index, identifier in enumerate(ids)}
    feature_path = args.output / "z.npy"
    completed_path = args.output / "completed.npy"
    progress_path = args.output / "progress.json"
    if progress_path.exists():
        progress = load_json(progress_path)
        if progress["base_preregistration_sha256"] != sha256(args.prereg):
            raise RuntimeError("partial Camelyon store belongs to another lock")
        if progress["access_amendment_sha256"] != sha256(args.amendment):
            raise RuntimeError("partial Camelyon store used another amendment")
        if progress["selected_image_ids_sha256"] != sha256_array(ids):
            raise RuntimeError("partial Camelyon selected IDs differ")
        features = np.lib.format.open_memmap(feature_path, mode="r+")
        completed = np.load(completed_path)
        completed_shards = set(progress["completed_shards"])
    else:
        features = np.lib.format.open_memmap(
            feature_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(selected), 512),
        )
        completed = np.zeros(len(selected), dtype=bool)
        completed_shards: set[str] = set()

    repository = prereg["remote_dataset"]
    revision = repository["revision"]
    base = (
        "https://huggingface.co/datasets/"
        f"{repository['repository']}/resolve/{revision}/"
    )

    def make_connection() -> Any:
        connection = duckdb.connect(config={"threads": "1"})
        connection.execute("INSTALL httpfs")
        connection.execute("LOAD httpfs")
        connection.execute("SET http_retries=12")
        connection.execute("SET http_retry_wait_ms=2000")
        connection.execute("SET http_timeout=120")
        connection.execute("CREATE TEMP TABLE wanted(image_id UINTEGER)")
        connection.executemany(
            "INSERT INTO wanted VALUES (?)",
            [(int(identifier),) for identifier in ids],
        )
        return connection

    connection = make_connection()
    images: list[torch.Tensor] = []
    output_positions: list[int] = []

    def flush() -> None:
        if not images:
            return
        batch = torch.stack(images).to(device)
        with torch.inference_mode():
            encoded = model(batch).float().cpu().numpy()
        positions_array = np.asarray(output_positions, dtype=np.int64)
        features[positions_array] = encoded
        completed[positions_array] = True
        images.clear()
        output_positions.clear()

    def checkpoint(relative: str) -> None:
        features.flush()
        np.save(completed_path, completed)
        atomic_json(
            progress_path,
            {
                "base_preregistration_sha256": sha256(args.prereg),
                "access_amendment_sha256": sha256(args.amendment),
                "selected_image_ids_sha256": sha256_array(ids),
                "completed_rows": int(np.sum(completed)),
                "completed_shards": sorted(completed_shards | {relative}),
            },
        )

    fields = (
        "p.image.bytes, p.label, p.center, p.image_id, p.patient, "
        "p.node, p.x_coord, p.y_coord"
    )
    max_attempts = int(amendment["access_controls"]["max_attempts_per_shard"])
    base_wait = float(amendment["access_controls"]["base_backoff_seconds"])
    inter_shard_wait = float(
        amendment["access_controls"]["inter_shard_wait_seconds"]
    )
    for relative in repository["parquet_files"]:
        if relative in completed_shards:
            continue
        url = base + relative
        for attempt in range(max_attempts):
            images.clear()
            output_positions.clear()
            try:
                cursor = connection.execute(
                    f"SELECT {fields} FROM read_parquet(?) p "
                    "SEMI JOIN wanted w USING (image_id)",
                    [url],
                )
                while rows := cursor.fetchmany(args.batch_size):
                    for (
                        image_bytes,
                        label,
                        center,
                        image_id,
                        patient,
                        node,
                        x_coord,
                        y_coord,
                    ) in rows:
                        position = positions[int(image_id)]
                        expected = selected.iloc[position]
                        observed = (
                            int(label),
                            int(center),
                            int(patient),
                            int(node),
                            int(x_coord),
                            int(y_coord),
                        )
                        locked = (
                            int(expected["y"]),
                            int(expected["center"]),
                            int(expected["patient"]),
                            int(expected["node"]),
                            int(expected["x_coord"]),
                            int(expected["y_coord"]),
                        )
                        if observed != locked:
                            raise RuntimeError(
                                f"remote metadata mismatch for image {image_id}: "
                                f"{observed} != {locked}"
                            )
                        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        images.append(transform(image))
                        output_positions.append(position)
                        if len(images) >= args.batch_size:
                            flush()
                flush()
                completed_shards.add(relative)
                checkpoint(relative)
                print(
                    json.dumps(
                        {
                            "completed_shard": relative,
                            "completed_rows": int(np.sum(completed)),
                            "completed_shards": len(completed_shards),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                time.sleep(inter_shard_wait)
                break
            except duckdb.HTTPException as error:
                if attempt + 1 >= max_attempts:
                    raise
                connection.close()
                wait = min(base_wait * (2**attempt), 300.0)
                print(
                    json.dumps(
                        {
                            "retry_shard": relative,
                            "attempt": attempt + 1,
                            "wait_seconds": wait,
                            "error": str(error),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                time.sleep(wait)
                connection = make_connection()

    features.flush()
    del features
    if not np.all(completed):
        missing = ids[~completed][:10].tolist()
        raise RuntimeError(f"remote mirror omitted selected IDs: {missing}")

    np.save(args.output / "y.npy", target)
    np.save(args.output / "s.npy", source)
    np.save(args.output / "split.npy", split)
    np.save(args.output / "g.npy", centers)
    np.save(args.output / "ids.npy", ids)
    model_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(
        weights.url
    ).name
    manifest = {
        "name": "MOSAIC streamed Camelyon17 ResNet18 confirmation store",
        "n_examples": len(selected),
        "dimension": 512,
        "arrays": {
            "z": "z.npy",
            "y": "y.npy",
            "s": "s.npy",
            "split": "split.npy",
            "g": "g.npy",
        },
        "ids": "ids.npy",
        "preregistration_sha256": sha256(args.prereg),
        "data_access_amendment_sha256": sha256(args.amendment),
        "metadata_sha256": prereg["metadata"]["sha256"],
        "selected_image_ids_sha256": sha256_array(ids),
        "remote_dataset": repository,
        "model": "torchvision ResNet18 ImageNet-1K V1 penultimate",
        "model_weights_url": weights.url,
        "model_weights_sha256": sha256(model_path),
        "preprocessing": str(transform),
        "device": device.type,
        "split_counts": {
            "train": int(np.sum(split == 0)),
            "validation": int(np.sum(split == 1)),
        },
        "source_label_counts": {
            f"split={role},y={label},s={group}": int(
                np.sum(
                    (split == role)
                    & (target == label)
                    & (source == group)
                )
            )
            for role in (0, 1)
            for label in (0, 1)
            for group in (0, 1)
        },
        "claim_boundary": prereg["claim_boundary"],
        "data_access_amendment": amendment["scope"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
