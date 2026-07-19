"""Shared helpers for deterministic claim replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts" / "frozen"
REPRODUCED = ROOT / "artifacts" / "reproduced"


def load_json(relative: str) -> Any:
    return json.loads((FROZEN / relative).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parser(description: str, *, default_seed: int, allowed_seeds: Iterable[int]) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    allowed = tuple(int(seed) for seed in allowed_seeds)
    result.add_argument("--seed", type=int, default=default_seed, choices=allowed)
    result.add_argument("--output", type=Path)
    return result


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def require_equal(label: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise AssertionError(f"{label}: expected {expected!r}, observed {observed!r}")


def require_close(label: str, observed: float, expected: float, tolerance: float = 1e-12) -> None:
    if abs(float(observed) - float(expected)) > tolerance:
        raise AssertionError(
            f"{label}: expected {expected:.16g}, observed {observed:.16g}, tolerance {tolerance:.3g}"
        )


def finish(name: str, seed: int, claims: dict[str, Any], output: Path | None) -> Path:
    destination = output or (REPRODUCED / f"{name}__seed{seed}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "seed": seed,
        "status": "pass",
        "claims": claims,
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"receipt: {destination.relative_to(ROOT)}")
    return destination
