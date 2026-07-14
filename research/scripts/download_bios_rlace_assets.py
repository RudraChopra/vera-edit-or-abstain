"""Download the official R-LACE BiasBios assets with resumable byte ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = "https://nlp.biu.ac.il/~ravfogs/rlace-cr/bios/bios_data"
DEFAULT_OUTPUT = Path("/Volumes/Backups/FARO/artifacts/bios_rlace_upstream/raw")
ASSETS = (
    "train_cls.npy",
    "dev_cls.npy",
    "test_cls.npy",
    "train.pickle",
    "dev.pickle",
    "test.pickle",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ranges(size: int, count: int) -> list[tuple[int, int]]:
    width = (size + count - 1) // count
    return [(start, min(size - 1, start + width - 1)) for start in range(0, size, width)]


def download_part(url: str, path: Path, start: int, end: int, retries: int) -> None:
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(retries):
        try:
            downloaded = temporary.stat().st_size if temporary.exists() else 0
            if downloaded > expected:
                temporary.unlink()
                downloaded = 0
            if downloaded == expected:
                os.replace(temporary, path)
                return
            request_start = start + downloaded
            request = urllib.request.Request(
                url,
                headers={
                    "Range": f"bytes={request_start}-{end}",
                    "User-Agent": "VERA-research-artifact-downloader/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(f"range request returned HTTP {response.status}")
                with temporary.open("ab" if downloaded else "wb") as handle:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
            if temporary.stat().st_size != expected:
                raise RuntimeError(
                    f"range {start}-{end} has {temporary.stat().st_size} bytes, expected {expected}"
                )
            os.replace(temporary, path)
            return
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(30, 2 ** attempt))


def remote_metadata(name: str) -> dict[str, object]:
    request = urllib.request.Request(
        f"{BASE_URL}/{name}",
        method="HEAD",
        headers={"User-Agent": "VERA-research-artifact-downloader/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        size = int(response.headers["Content-Length"])
        return {
            "bytes": size,
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "accept_ranges": response.headers.get("Accept-Ranges"),
        }


def download_asset(
    output_dir: Path,
    name: str,
    metadata: dict[str, object],
    workers: int,
    retries: int,
) -> dict[str, object]:
    size = int(metadata["bytes"])
    destination = output_dir / name
    if destination.exists() and destination.stat().st_size == size:
        return {
            "name": name,
            "url": f"{BASE_URL}/{name}",
            "bytes": size,
            "sha256": sha256(destination),
            "reused": True,
            "remote_metadata": metadata,
        }

    part_dir = output_dir / ".parts" / name
    part_dir.mkdir(parents=True, exist_ok=True)
    jobs = ranges(size, workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                download_part,
                f"{BASE_URL}/{name}",
                part_dir / f"part-{index:03d}",
                start,
                end,
                retries,
            )
            for index, (start, end) in enumerate(jobs)
        ]
        for future in futures:
            future.result()

    temporary = output_dir / f".{name}.assembling"
    with temporary.open("wb") as output:
        for index in range(len(jobs)):
            with (part_dir / f"part-{index:03d}").open("rb") as source:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
    if temporary.stat().st_size != size:
        raise RuntimeError(f"assembled {name} has the wrong size")
    os.replace(temporary, destination)
    return {
        "name": name,
        "url": f"{BASE_URL}/{name}",
        "bytes": size,
        "sha256": sha256(destination),
        "reused": False,
        "remote_metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--assets", default=",".join(ASSETS))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = [item.strip() for item in args.assets.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(ASSETS))
    if unknown:
        raise ValueError(f"unknown assets: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {name: remote_metadata(name) for name in selected}
    records = [
        download_asset(
            args.output_dir,
            name,
            metadata[name],
            max(1, args.workers),
            max(1, args.retries),
        )
        for name in selected
    ]
    receipt = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "official R-LACE author artifact server",
        "base_url": BASE_URL,
        "assets": records,
    }
    receipt_path = args.output_dir.parent / "download_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(receipt_path)


if __name__ == "__main__":
    main()
