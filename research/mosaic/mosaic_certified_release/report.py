"""Auditor-facing certification report generation."""

from __future__ import annotations

import hashlib
import html
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .api import Mosaic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime() -> dict[str, str]:
    packages = {}
    for name in ("mosaic-certified-release", "numpy", "scipy", "scikit-learn"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed-as-distribution"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **packages,
    }


def _payload(model: "Mosaic", title: str) -> dict[str, Any]:
    certification = (
        None
        if model.certification_ is None
        else asdict(model.certification_)
    )
    solution = model.channel_solution_
    channel = None
    decoder = None
    if solution is not None:
        channel = np.asarray(solution.release_channel).tolist()
        decoder = list(solution.decoder)
    return {
        "schema": "mosaic_certification_report_v1",
        "title": title,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime(),
        "configuration": asdict(model.config),
        "certification": certification,
        "release_channel": channel,
        "decoder": decoder,
        "audit_notes": [
            "A CERTIFIED result applies only to the registered data splits, "
            "shift class, source concept, tokenizer, and release mechanism.",
            "ABSTAIN is an auditable outcome and must not be overridden by "
            "this runtime.",
        ],
    }


def _render_html(payload: dict[str, Any]) -> str:
    certification = payload["certification"]
    status = "NOT RUN" if certification is None else certification["status"]
    reason = "" if certification is None else certification["reason"]
    details = html.escape(json.dumps(payload, indent=2, sort_keys=True))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(payload["title"])}</title>
<style>
body {{ font: 15px/1.45 system-ui, sans-serif; margin: 40px auto; max-width: 900px; color: #17202a; }}
h1 {{ font-size: 28px; }} .status {{ border-left: 5px solid #1d6f42; padding: 12px 16px; background: #f3f7f5; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f6f7f8; padding: 16px; }}
</style>
</head>
<body>
<h1>{html.escape(payload["title"])}</h1>
<div class="status"><strong>{html.escape(status)}</strong><br>{html.escape(reason)}</div>
<h2>Audit Record</h2>
<pre>{details}</pre>
</body>
</html>
"""


def write_certification_report(
    model: "Mosaic",
    directory: str | Path,
    *,
    title: str = "MOSAIC Certification Report",
) -> tuple[Path, Path]:
    """Write a JSON receipt and self-contained HTML report."""

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    payload = _payload(model, title)
    json_path = output / "certification_report.json"
    html_path = output / "certification_report.html"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["json_sha256"] = _sha256(json_path)
    html_path.write_text(_render_html(payload), encoding="utf-8")
    return json_path, html_path


def verify_report(path: str | Path) -> dict[str, Any]:
    """Validate the report schema and return its central decision."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != "mosaic_certification_report_v1":
        raise ValueError("unknown MOSAIC report schema")
    certification = value.get("certification")
    return {
        "valid_schema": True,
        "status": None if certification is None else certification["status"],
        "reason": None if certification is None else certification["reason"],
        "sha256": _sha256(Path(path)),
    }
