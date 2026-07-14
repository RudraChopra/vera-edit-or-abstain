"""Stream Camelyon17-WILDS tar PNGs into a compact ResNet-18 embedding store.

The normal WILDS extraction path creates 455,954 small PNG files. That is slow
and brittle on external drives and cloud-managed folders. This exporter treats
the official WILDS archive itself as the image source, preserving the same
manual no-PIL ResNet-18 encoder and the same f32+manifest store contract used
by `convert_camelyon17_f32_store_to_numpy_store.py`.
"""

from __future__ import annotations

import argparse
import array
import csv
import hashlib
import json
import math
import shutil
import struct
import sys
import tarfile
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts"
DEFAULT_DATA_DIR = Path("/Volumes/Backups/FARO/data/wilds/camelyon17_v1.0")
DEFAULT_ARCHIVE = DEFAULT_DATA_DIR / "archive.tar.gz"
DEFAULT_RAW_METADATA = DEFAULT_DATA_DIR / "metadata.csv"
DEFAULT_STORE_DIR = Path("/Volumes/Backups/FARO/artifacts/camelyon17_resnet18_torch_full_store")
DEFAULT_REPORT = ARTIFACT_DIR / "camelyon17_resnet18_torch_full_store_report.json"
DEFAULT_DRYRUN_REPORT = ARTIFACT_DIR / "camelyon17_resnet18_torch_full_store_dryrun_report.json"
DEFAULT_WEIGHTS = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
EXPECTED_CAMELYON17_EXAMPLES = 455_954
FEATURE_COUNT = 512
FLOAT_BYTES = 4
ROW_BYTES = FEATURE_COUNT * FLOAT_BYTES
EXPECTED_RESNET18_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
TEST_CENTER = 2
OOD_VAL_CENTER = 1
SOURCE_CENTERS = {0, 3, 4}


def import_no_pil_encoder():
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import export_camelyon17_resnet18_torch_no_pil as encoder

    return encoder


encoder = import_no_pil_encoder()
torch = encoder.torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def disk_report(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def parse_centers(value: str) -> set[int]:
    return {int(item) for item in value.split(",") if item.strip()}


def trace_split(center: int, raw_split: int) -> str:
    if center == TEST_CENTER:
        return "external"
    if center == OOD_VAL_CENTER:
        return "validation"
    if center in SOURCE_CENTERS and raw_split == 1:
        return "validation"
    return "train"


def archive_member(row: dict[str, str]) -> str:
    patient = row["patient"]
    node = row["node"]
    x_coord = row["x_coord"]
    y_coord = row["y_coord"]
    dirname = f"patient_{patient}_node_{node}"
    filename = f"patch_patient_{patient}_node_{node}_x_{x_coord}_y_{y_coord}.png"
    return f"patches/{dirname}/{filename}"


def normalized_member_name(name: str) -> str:
    return name.removeprefix("./")


def load_metadata(raw_metadata: Path, source_positive_centers: set[int]) -> dict[str, dict[str, str]]:
    with raw_metadata.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"patient", "node", "x_coord", "y_coord", "tumor", "center", "split"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{raw_metadata} missing required columns: {missing}")
        rows = list(reader)

    out: dict[str, dict[str, str]] = {}
    for idx, row in enumerate(rows):
        center = int(row["center"])
        raw_split = int(row["split"])
        member = archive_member(row)
        out[member] = {
            "row_index": str(idx),
            "id": f"camelyon17_{idx:06d}",
            "split": trace_split(center, raw_split),
            "y": str(int(row["tumor"])),
            "s": "1" if center in source_positive_centers else "0",
            "image_path": f"tar://{member}",
            "archive_member": member,
            "center": str(center),
            "raw_split": str(raw_split),
        }
    if len(out) != len(rows):
        raise ValueError("metadata rows did not map to unique archive members")
    return out


def completed_ids(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["id"] for row in reader if row.get("id")}


def summarize_manifest(manifest_path: Path) -> tuple[int, dict[str, int]]:
    if not manifest_path.exists():
        return 0, {}
    splits: Counter[str] = Counter()
    total = 0
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total += 1
            splits[row.get("split", "")] += 1
    return total, dict(sorted(splits.items()))


def repair_resume_binary(embeddings_path: Path, manifest_rows: int) -> dict[str, int | str]:
    expected_bytes = manifest_rows * ROW_BYTES
    if not embeddings_path.exists():
        return {
            "status": "missing_binary",
            "actual_bytes": 0,
            "expected_bytes": expected_bytes,
            "truncated_bytes": 0,
        }

    actual_bytes = embeddings_path.stat().st_size
    if actual_bytes == expected_bytes:
        return {
            "status": "already_consistent",
            "actual_bytes": actual_bytes,
            "expected_bytes": expected_bytes,
            "truncated_bytes": 0,
        }
    if actual_bytes < expected_bytes:
        raise ValueError(
            f"{embeddings_path} has {actual_bytes} bytes; "
            f"expected at least {expected_bytes} for {manifest_rows} manifest rows"
        )

    # A signal can land after writing embeddings but before manifest rows.
    # The manifest is the authoritative completed-row ledger for resume.
    with embeddings_path.open("r+b") as handle:
        handle.truncate(expected_bytes)
    return {
        "status": "truncated_binary_tail",
        "actual_bytes": actual_bytes,
        "expected_bytes": expected_bytes,
        "truncated_bytes": actual_bytes - expected_bytes,
        "truncated_complete_rows": (actual_bytes - expected_bytes) // ROW_BYTES,
        "truncated_partial_bytes": (actual_bytes - expected_bytes) % ROW_BYTES,
    }


def png_chunks(data: bytes):
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    offset = len(PNG_SIGNATURE)
    while offset + 8 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        yield kind, chunk
        offset += length + 12
        if kind == b"IEND":
            break


def read_png_rgb_bytes(data: bytes, label: str) -> tuple[bytes, int, int]:
    ihdr: tuple[int, int, int, int, int, int, int] | None = None
    idat_parts: list[bytes] = []
    for kind, chunk in png_chunks(data):
        if kind == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", chunk)
        elif kind == b"IDAT":
            idat_parts.append(chunk)
    if ihdr is None:
        raise ValueError(f"{label} has no IHDR chunk")

    width, height, bit_depth, color_type, compression, filter_method, interlace = ihdr
    if bit_depth != 8 or compression != 0 or filter_method != 0 or interlace != 0:
        raise ValueError(
            f"{label} must be non-interlaced 8-bit PNG; got bit_depth={bit_depth}, "
            f"compression={compression}, filter={filter_method}, interlace={interlace}"
        )

    channels_by_color = {0: 1, 2: 3, 4: 2, 6: 4}
    if color_type not in channels_by_color:
        raise ValueError(f"{label} has unsupported PNG color type: {color_type}")
    channels = channels_by_color[color_type]
    decoded = encoder.unfilter_png(zlib.decompress(b"".join(idat_parts)), width, height, channels)
    if color_type == 2:
        rgb = decoded
    elif color_type == 6:
        rgb = bytes(value for idx, value in enumerate(decoded) if idx % 4 != 3)
    elif color_type == 0:
        rgb = bytes(channel for value in decoded for channel in (value, value, value))
    else:
        rgb = bytes(
            channel
            for idx in range(0, len(decoded), 2)
            for channel in (decoded[idx], decoded[idx], decoded[idx])
        )
    return rgb, height, width


def append_float32(handle, values: list[float]) -> None:  # type: ignore[no-untyped-def]
    if len(values) != FEATURE_COUNT:
        raise ValueError(f"expected {FEATURE_COUNT} features, got {len(values)}")
    packed = array.array("f", values)
    if packed.itemsize != FLOAT_BYTES:
        raise RuntimeError("array('f') is not 32-bit on this Python runtime")
    if sys.byteorder != "little":
        packed.byteswap()
    handle.write(packed.tobytes())


def encode_png_batch(model, batch: list[tuple[dict[str, str], bytes]], device, short_size: int, crop_size: int) -> list[list[float]]:  # type: ignore[no-untyped-def]
    tensors = []
    for row, data in batch:
        rgb, height, width = read_png_rgb_bytes(data, row["archive_member"])
        tensors.append(encoder.preprocess_image(rgb, height, width, short_size, crop_size))
    batch_tensor = torch.cat(tensors, dim=0).to(device)
    with torch.no_grad():
        embeddings = model(batch_tensor).detach().cpu()
    return [[float(value) for value in embedding] for embedding in embeddings]


def write_store(
    archive_path: Path,
    raw_metadata: Path,
    store_dir: Path,
    weights_path: Path,
    device_name: str,
    batch_size: int,
    stop_after_new_rows: int,
    resume: bool,
    progress_every: int,
    short_size: int,
    crop_size: int,
    source_positive_centers: set[int],
) -> dict[str, object]:
    device = encoder.resolve_device(device_name)
    model, missing_keys, unexpected_keys = encoder.load_encoder(weights_path, device)
    metadata_by_member = load_metadata(raw_metadata, source_positive_centers)

    store_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = store_dir / "manifest.csv"
    embeddings_path = store_dir / "embeddings.f32"
    existing_ids = completed_ids(manifest_path) if resume else set()
    existing_count, _ = summarize_manifest(manifest_path) if resume else (0, {})
    resume_binary_repair = (
        repair_resume_binary(embeddings_path, existing_count)
        if resume
        else {"status": "not_resume", "truncated_bytes": 0}
    )

    manifest_mode = "a" if resume and manifest_path.exists() else "w"
    binary_mode = "ab" if resume and embeddings_path.exists() else "wb"
    manifest_fields = ["row_index", "id", "split", "y", "s", "image_path", "archive_member"]
    archive_png_count = 0
    matched_metadata_count = 0
    skipped_completed = 0
    rows_written = 0
    row_index = existing_count
    written_splits: Counter[str] = Counter()
    started = time.monotonic()

    with manifest_path.open(manifest_mode, newline="", encoding="utf-8") as manifest_handle:
        writer = csv.DictWriter(manifest_handle, fieldnames=manifest_fields)
        if manifest_mode == "w":
            writer.writeheader()

        with embeddings_path.open(binary_mode) as binary_handle:
            batch: list[tuple[dict[str, str], bytes]] = []
            with tarfile.open(archive_path, "r:gz") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".png"):
                        continue
                    archive_png_count += 1
                    member_name = normalized_member_name(member.name)
                    row = metadata_by_member.get(member_name)
                    if row is None:
                        continue
                    matched_metadata_count += 1
                    if row["id"] in existing_ids:
                        skipped_completed += 1
                        continue
                    if stop_after_new_rows > 0 and rows_written + len(batch) >= stop_after_new_rows:
                        break
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        raise ValueError(f"could not read {member.name}")
                    batch.append((row, extracted.read()))
                    should_flush = len(batch) >= batch_size or (
                        stop_after_new_rows > 0
                        and rows_written + len(batch) >= stop_after_new_rows
                    )
                    if should_flush:
                        embeddings = encode_png_batch(model, batch, device, short_size, crop_size)
                        for source_row, embedding in zip([item[0] for item in batch], embeddings):
                            append_float32(binary_handle, embedding)
                            writer.writerow(
                                {
                                    "row_index": row_index,
                                    "id": source_row["id"],
                                    "split": source_row["split"],
                                    "y": source_row["y"],
                                    "s": source_row["s"],
                                    "image_path": source_row["image_path"],
                                    "archive_member": source_row["archive_member"],
                                }
                            )
                            row_index += 1
                            rows_written += 1
                            written_splits[source_row["split"]] += 1
                        batch.clear()
                        if progress_every > 0 and rows_written and rows_written % progress_every == 0:
                            elapsed = time.monotonic() - started
                            print(f"wrote {rows_written} tar-stream rows in {elapsed:.1f}s", flush=True)
                        if stop_after_new_rows > 0 and rows_written >= stop_after_new_rows:
                            break

            if batch:
                embeddings = encode_png_batch(model, batch, device, short_size, crop_size)
                for source_row, embedding in zip([item[0] for item in batch], embeddings):
                    append_float32(binary_handle, embedding)
                    writer.writerow(
                        {
                            "row_index": row_index,
                            "id": source_row["id"],
                            "split": source_row["split"],
                            "y": source_row["y"],
                            "s": source_row["s"],
                            "image_path": source_row["image_path"],
                            "archive_member": source_row["archive_member"],
                        }
                    )
                    row_index += 1
                    rows_written += 1
                    written_splits[source_row["split"]] += 1

    return {
        "device": str(device),
        "manifest_path": str(manifest_path),
        "embeddings_path": str(embeddings_path),
        "metadata_rows": len(metadata_by_member),
        "archive_png_count_seen": archive_png_count,
        "matched_metadata_count_seen": matched_metadata_count,
        "skipped_completed": skipped_completed,
        "resume_binary_repair": resume_binary_repair,
        "rows_written_this_run": rows_written,
        "written_splits_this_run": dict(sorted(written_splits.items())),
        "missing_keys": list(missing_keys),
        "unexpected_keys": list(unexpected_keys),
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--raw-metadata", type=Path, default=DEFAULT_RAW_METADATA)
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after-new-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=4096)
    parser.add_argument("--short-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--source-positive-centers", default="1,2,3,4")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.stop_after_new_rows < 0:
        raise ValueError("--stop-after-new-rows must be non-negative")
    if not args.archive.exists():
        raise FileNotFoundError(args.archive)
    if not args.raw_metadata.exists():
        raise FileNotFoundError(args.raw_metadata)
    if not args.dry_run and not args.weights.exists():
        raise FileNotFoundError(args.weights)
    if args.dry_run and args.report == DEFAULT_REPORT:
        args.report = DEFAULT_DRYRUN_REPORT

    source_positive_centers = parse_centers(args.source_positive_centers)
    metadata_by_member = load_metadata(args.raw_metadata, source_positive_centers)
    args.store_dir.parent.mkdir(parents=True, exist_ok=True)
    disk_at_start = disk_report(args.store_dir.parent)
    estimated_full_embedding_binary_bytes = EXPECTED_CAMELYON17_EXAMPLES * FEATURE_COUNT * FLOAT_BYTES

    if args.dry_run:
        report = {
            "name": "FARO Camelyon17 WILDS tar-stream store dry run",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "dry_run": True,
            "archive_path": str(args.archive),
            "raw_metadata": str(args.raw_metadata),
            "store_dir": str(args.store_dir),
            "input_mode": "official_wilds_archive_tar_stream",
            "metadata_rows": len(metadata_by_member),
            "sample_count": len(metadata_by_member),
            "expected_full_dataset_examples": EXPECTED_CAMELYON17_EXAMPLES,
            "feature_count": FEATURE_COUNT,
            "embedding_dtype": "float32_little_endian",
            "estimated_full_embedding_binary_bytes": estimated_full_embedding_binary_bytes,
            "estimated_embedding_binary_bytes": estimated_full_embedding_binary_bytes,
            "disk_at_start": disk_at_start,
            "claim_grade_embedding_store": False,
            "claim_grade_benchmark_row": False,
        }
        write_report(args.report, report)
        print("FARO Camelyon17 tar-stream store dry run complete")
        print(f"metadata_rows={len(metadata_by_member)}")
        print(f"report={args.report}")
        return 0

    weights_sha = sha256_file(args.weights)
    started = time.monotonic()
    export_report = write_store(
        archive_path=args.archive,
        raw_metadata=args.raw_metadata,
        store_dir=args.store_dir,
        weights_path=args.weights,
        device_name=args.device,
        batch_size=args.batch_size,
        stop_after_new_rows=args.stop_after_new_rows,
        resume=args.resume,
        progress_every=args.progress_every,
        short_size=args.short_size,
        crop_size=args.crop_size,
        source_positive_centers=source_positive_centers,
    )

    manifest_path = Path(str(export_report["manifest_path"]))
    embeddings_path = Path(str(export_report["embeddings_path"]))
    total_rows, split_summary = summarize_manifest(manifest_path)
    expected_embedding_bytes = total_rows * FEATURE_COUNT * FLOAT_BYTES
    actual_embedding_bytes = embeddings_path.stat().st_size
    binary_size_matches = actual_embedding_bytes == expected_embedding_bytes
    full_export = total_rows == EXPECTED_CAMELYON17_EXAMPLES
    claim_grade_embedding_store = (
        total_rows > 0
        and binary_size_matches
        and weights_sha == EXPECTED_RESNET18_SHA256
        and not export_report["missing_keys"]
        and not export_report["unexpected_keys"]
    )
    report = {
        "name": "FARO Camelyon17 WILDS tar-stream ResNet18 embedding store",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": False,
        "input_mode": "official_wilds_archive_tar_stream",
        "encoder_family": "manual_resnet18_imagenet_frozen_no_pil",
        "archive_path": str(args.archive),
        "raw_metadata": str(args.raw_metadata),
        "store_dir": str(args.store_dir),
        "manifest_path": str(manifest_path),
        "embeddings_path": str(embeddings_path),
        "weights_path": str(args.weights),
        "weights_sha256": weights_sha,
        "device": export_report["device"],
        "batch_size": args.batch_size,
        "stop_after_new_rows": args.stop_after_new_rows,
        "short_size": args.short_size,
        "crop_size": args.crop_size,
        "sample_count": total_rows,
        "splits": split_summary,
        "metadata_rows": export_report["metadata_rows"],
        "archive_png_count_seen": export_report["archive_png_count_seen"],
        "matched_metadata_count_seen": export_report["matched_metadata_count_seen"],
        "feature_count": FEATURE_COUNT,
        "embedding_dtype": "float32_little_endian",
        "embedding_layout": "row-major contiguous [sample_count, feature_count]",
        "embedding_binary_bytes": actual_embedding_bytes,
        "expected_embedding_binary_bytes": expected_embedding_bytes,
        "binary_size_matches": binary_size_matches,
        "estimated_full_embedding_binary_bytes": estimated_full_embedding_binary_bytes,
        "full_dataset_embedding_export": full_export,
        "expected_full_dataset_examples": EXPECTED_CAMELYON17_EXAMPLES,
        "disk_at_start": disk_at_start,
        "disk_at_end": disk_report(args.store_dir.parent),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "claim_grade_embedding_store": claim_grade_embedding_store,
        "claim_grade_benchmark_row": False,
        "claim_boundary": (
            "Claim-grade frozen embedding store only. A benchmark row becomes "
            "claim-ready only after the full tar-stream store is consumed by "
            "the benchmark runner, receipts are written, and paired statistics pass."
        ),
        "export_report": export_report,
    }
    report["manifest_sha256"] = sha256_file(manifest_path)
    report["embeddings_sha256"] = sha256_file(embeddings_path)
    write_report(args.report, report)

    print("FARO Camelyon17 tar-stream ResNet-18 store export complete")
    print(f"store_dir={args.store_dir}")
    print(f"report={args.report}")
    print(f"sample_count={total_rows}")
    print(f"splits={split_summary}")
    print(f"embedding_binary_bytes={actual_embedding_bytes}")
    print(f"claim_grade_embedding_store={str(claim_grade_embedding_store).lower()}")
    print("claim_grade_benchmark_row=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
