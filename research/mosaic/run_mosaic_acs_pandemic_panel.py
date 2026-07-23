#!/usr/bin/env python3
"""Evaluate frozen 2018 ACS interfaces on the first post-2020 ACS 1-year panel."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mosaic_channel import l1_ball_expectation_lower, l1_ball_expectation_upper
from mosaic_real import balanced_stratum_sample, build_token_table, fit_score_tokenizer
from prepare_acs_natural_shift_stores import extract_task, reference_split, task_registry
from run_mosaic_acs_natural_shift import puma_bridge_diagnostic_split
from run_mosaic_official_frontier_exact_confirmation import identity_candidate, materialize
from run_official_eraser_frontier import (
    dispatch_candidates,
    preprocess,
    random_cap,
    split_eraser_train_construction,
)


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "research/mosaic/prereg_mosaic_acs_pandemic_panel_v1.json"
OUTPUT = ROOT / "research/artifacts/mosaic_acs_pandemic_panel_v1.json"
RECEIPTS = ROOT / "research/artifacts/mosaic_acs_natural_shift_v1_receipts"
DATA_LOCK = ROOT / "research/mosaic/prereg_mosaic_acs_natural_shift_data_v1.json"
REFERENCE_YEAR = "2018"
FUTURE_YEAR = "2021"
TARGET_STATES = ("FL", "IL", "NY", "WA")
STATE_FIPS = {"CA": "06", "FL": "12", "IL": "17", "NY": "36", "WA": "53"}
TASKS = ("employment", "income", "public_coverage")
SEEDS = (1400, 1401, 1402, 1403, 1404)
RULES = ("mosaic", "direct")
SOURCE_THRESHOLD = 0.35
UTILITY_THRESHOLD = 0.40
MAXIMUM_FUTURE_ROWS = 64_000
FUTURE_TABLE_DELTA = 0.05 / (
    len(TARGET_STATES) * len(TASKS) * len(SEEDS) * len(RULES) * 2
)
METHOD_KEYS = {
    "INLP": "inlp",
    "LEACE": "leace",
    "R-LACE": "rlace",
    "TaCo": "taco",
    "MANCE++": "mance",
}
RELSHIPP_TO_RELP = {
    20: 0,
    21: 1,
    22: 13,
    23: 1,
    24: 13,
    25: 2,
    26: 3,
    27: 4,
    28: 5,
    29: 6,
    30: 7,
    31: 8,
    32: 9,
    33: 10,
    34: 12,
    35: 14,
    36: 15,
    37: 16,
    38: 17,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def input_set_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def census_url(state: str) -> str:
    return (
        "https://www2.census.gov/programs-surveys/acs/data/pums/"
        f"{FUTURE_YEAR}/1-Year/csv_p{state.lower()}.zip"
    )


def expected_protocol() -> dict[str, Any]:
    return {
        "reference_year": REFERENCE_YEAR,
        "future_year": FUTURE_YEAR,
        "future_year_interpretation": (
            "first standard ACS 1-year PUMS release after the unavailable 2020 "
            "standard 1-year product"
        ),
        "target_states": list(TARGET_STATES),
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "rules": list(RULES),
        "population_jobs": len(TARGET_STATES) * len(TASKS) * len(SEEDS),
        "source_advantage_threshold": SOURCE_THRESHOLD,
        "utility_threshold": UTILITY_THRESHOLD,
        "future_table_delta": FUTURE_TABLE_DELTA,
        "familywise_delta": 0.05,
        "future_tables_in_family": (
            len(TARGET_STATES) * len(TASKS) * len(SEEDS) * len(RULES) * 2
        ),
        "maximum_future_rows_per_role": MAXIMUM_FUTURE_ROWS,
        "membership_fraction_by_puma": 2.0 / 3.0,
        "selection": (
            "reuse every rule's frozen 2018 primary decision and interface "
            "without 2021 tuning"
        ),
        "natural_failure_witness": (
            "a frozen direct deployment has a familywise lower confidence bound "
            "above either contract on the 2021 diagnostic fold, while MOSAIC "
            "abstains at the 2018 decision, rejects 2021 bridge membership, or "
            "has familywise upper confidence bounds below both contracts"
        ),
        "relationship_crosswalk": {
            str(key): value for key, value in RELSHIPP_TO_RELP.items()
        },
        "future_asset_urls": {state: census_url(state) for state in TARGET_STATES},
    }


def validate_lock(path: Path, reference_csv: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not sidecar.exists():
        raise ValueError("pandemic-panel lock or sidecar is missing")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(path):
        raise ValueError("pandemic-panel lock sidecar mismatch")
    lock = load(path)
    if lock.get("status") != "locked_before_2021_download":
        raise ValueError("pandemic-panel lock has the wrong status")
    if lock.get("protocol") != expected_protocol():
        raise ValueError("pandemic-panel protocol differs from its lock")
    for relative, expected in lock["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"locked code mismatch: {relative}")
    for relative, expected in lock["input_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"locked input mismatch: {relative}")
    receipts = sorted(RECEIPTS.glob("ACS-*-CA-*__seed*.json"))
    if input_set_sha256(receipts) != lock["receipt_set_sha256"]:
        raise ValueError("frozen 2018 receipt set differs from the lock")
    expected_reference = lock["reference_raw_asset"]
    if reference_csv.stat().st_size != expected_reference["bytes"]:
        raise ValueError("2018 California raw file size differs from the lock")
    if sha256(reference_csv) != expected_reference["sha256"]:
        raise ValueError("2018 California raw file hash differs from the lock")
    for local in (path, sidecar):
        relative = local.resolve().relative_to(ROOT.resolve())
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative.as_posix()}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if committed != local.read_bytes():
            raise ValueError(f"{relative} is not the committed lock")
    return lock


def required_columns() -> set[str]:
    columns = {"PUMA", "SEX", "PWGTP"}
    for problem in task_registry().values():
        columns.update(problem.features)
        columns.add(problem.target)
        if getattr(problem, "_group", None):
            columns.add(problem._group)
    return columns


def read_selected_csv(source: Any, columns: set[str]):
    import pandas as pd

    header = pd.read_csv(source, nrows=0)
    available = set(header.columns)
    selected = set(columns)
    if "RELP" not in available and "RELSHIPP" in available:
        selected.remove("RELP")
        selected.add("RELSHIPP")
    missing = selected - available
    if missing:
        raise ValueError(f"ACS file lacks required columns: {sorted(missing)}")
    if hasattr(source, "seek"):
        source.seek(0)
    return pd.read_csv(source, usecols=sorted(selected), low_memory=False)


def apply_relationship_crosswalk(frame: Any, *, state: str) -> Any:
    if "RELP" in frame.columns:
        return frame
    if "RELSHIPP" not in frame.columns:
        raise ValueError(f"{FUTURE_YEAR} {state} has neither RELP nor RELSHIPP")
    mapped = frame["RELSHIPP"].map(RELSHIPP_TO_RELP)
    if mapped.isna().any():
        unknown = sorted(
            int(value) for value in frame.loc[mapped.isna(), "RELSHIPP"].unique()
        )
        raise ValueError(f"unmapped {FUTURE_YEAR} {state} RELSHIPP values: {unknown}")
    frame["RELP"] = mapped.astype("int16")
    return frame


def load_reference_frame(path: Path):
    frame = read_selected_csv(path, required_columns())
    return apply_relationship_crosswalk(frame, state="CA")


def load_future_frame(
    state: str,
    *,
    raw_root: Path,
    allow_download: bool,
) -> tuple[Any, dict[str, Any]]:
    local = raw_root / FUTURE_YEAR / "1-Year" / f"psam_p{STATE_FIPS[state]}.csv"
    if local.exists():
        frame = read_selected_csv(local, required_columns())
        metadata = {
            "state": state,
            "source": "local_uncompressed_csv",
            "path": str(local),
            "bytes": local.stat().st_size,
            "sha256": sha256(local),
            "url": census_url(state),
        }
        return apply_relationship_crosswalk(frame, state=state), metadata
    if not allow_download:
        raise FileNotFoundError(
            f"{local} is absent; pass --download to stream the locked Census asset"
        )
    request = urllib.request.Request(
        census_url(state), headers={"User-Agent": "MOSAIC-research/1.0"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        archive = response.read()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        members = [
            name for name in bundle.namelist()
            if name.lower().endswith(".csv") and "psam_p" in name.lower()
        ]
        if len(members) != 1:
            raise ValueError(f"unexpected Census archive members: {members}")
        with bundle.open(members[0]) as handle:
            frame = read_selected_csv(handle, required_columns())
    metadata = {
        "state": state,
        "source": "locked_url_streamed_in_memory",
        "archive_member": members[0],
        "compressed_bytes": len(archive),
        "compressed_sha256": sha256_bytes(archive),
        "url": census_url(state),
    }
    return apply_relationship_crosswalk(frame, state=state), metadata


def normalized_advantage(counts: np.ndarray, channel: np.ndarray) -> float:
    probabilities = counts / counts.sum(axis=2, keepdims=True)
    released = probabilities @ channel
    values = []
    for label in range(released.shape[0]):
        accuracy = float(np.max(released[label], axis=0).sum() / 2.0)
        values.append(2.0 * accuracy - 1.0)
    return float(max(values))


def worst_error(
    counts: np.ndarray, channel: np.ndarray, decoder: Sequence[int]
) -> float:
    probabilities = counts / counts.sum(axis=2, keepdims=True)
    released = probabilities @ channel
    decoder_array = np.asarray(decoder, dtype=np.int64)
    return float(
        max(
            released[label, source, decoder_array != label].sum()
            for label in range(released.shape[0])
            for source in range(released.shape[1])
        )
    )


def diagnostic_bounds(
    counts: np.ndarray,
    radii: np.ndarray,
    channel: np.ndarray,
    decoder: Sequence[int],
) -> dict[str, float]:
    probabilities = counts / counts.sum(axis=2, keepdims=True)
    released = probabilities @ channel
    source_empirical = []
    source_lower = []
    source_upper = []
    for label in range(released.shape[0]):
        advantage = float(2.0 * (np.max(released[label], axis=0).sum() / 2.0) - 1.0)
        perturbation = 0.5 * float(radii[label, 0] + radii[label, 1])
        source_empirical.append(advantage)
        source_lower.append(max(0.0, advantage - perturbation))
        source_upper.append(min(1.0, advantage + perturbation))

    decoder_array = np.asarray(decoder, dtype=np.int64)
    error_empirical = []
    error_lower = []
    error_upper = []
    for label in range(probabilities.shape[0]):
        loss = channel[:, decoder_array != label].sum(axis=1)
        for source in range(probabilities.shape[1]):
            distribution = probabilities[label, source]
            error_empirical.append(float(distribution @ loss))
            error_lower.append(
                float(
                    l1_ball_expectation_lower(
                        distribution, loss, l1_radius=float(radii[label, source])
                    )
                )
            )
            error_upper.append(
                float(
                    l1_ball_expectation_upper(
                        distribution, loss, l1_radius=float(radii[label, source])
                    )
                )
            )
    return {
        "source_advantage_empirical": max(source_empirical),
        "source_advantage_lower": max(source_lower),
        "source_advantage_upper": max(source_upper),
        "worst_conditional_error_empirical": max(error_empirical),
        "worst_conditional_error_lower": max(error_lower),
        "worst_conditional_error_upper": max(error_upper),
    }


def fixed_class_membership(
    reference_counts: np.ndarray,
    reference_radii: np.ndarray,
    future_counts: np.ndarray,
    future_radii: np.ndarray,
    bridge: dict[str, Any],
) -> dict[str, Any]:
    reference = reference_counts / reference_counts.sum(axis=2, keepdims=True)
    future = future_counts / future_counts.sum(axis=2, keepdims=True)
    indicators = np.eye(reference_counts.shape[2])
    rows = []
    for label, certificate in enumerate(bridge["labels"]):
        transform = np.asarray(certificate["transform"], dtype=np.float64)
        retained = float(certificate["retained_mass"])
        for source in range(reference.shape[1]):
            for output in range(reference.shape[2]):
                lower = l1_ball_expectation_lower(
                    future[label, source],
                    indicators[output],
                    l1_radius=float(future_radii[label, source]),
                )
                upper = l1_ball_expectation_upper(
                    reference[label, source],
                    transform[:, output],
                    l1_radius=float(reference_radii[label, source]),
                )
                rows.append(
                    {
                        "label": label,
                        "source": source,
                        "output": output,
                        "lower_future": float(lower),
                        "retained_reference_upper": float(retained * upper),
                        "slack": float(lower - retained * upper),
                    }
                )
    return {
        "accepted": all(row["slack"] >= 0.0 for row in rows),
        "minimum_slack": min(row["slack"] for row in rows),
        "constraints": rows,
    }


def selected_row(
    payload: dict[str, Any], rule: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    alphabet = payload["alphabets"]["4"]
    selection = alphabet["primary_selection"][rule]
    if selection["decision"] != "deploy":
        return None
    row = next(
        item for item in alphabet["rows"]
        if item["candidate"] == selection["candidate"]
    )
    return selection, row


def prepare_reference(
    frame: Any,
    *,
    task: str,
    state: str,
    data_lock: dict[str, Any],
) -> tuple[np.ndarray, ...]:
    extracted = extract_task(frame, task_registry()[task])
    x, y, s, environment, feature_columns = extracted
    manifest = data_lock["stores"][f"{task}:CA->{state}"]["manifest"]
    if list(feature_columns) != manifest["feature_columns"]:
        raise ValueError(f"{task} {state} feature columns differ from the frozen store")
    split = reference_split(len(x), seed=int(manifest["split_seed"]))
    return x, y, s, environment, split, feature_columns


def reconstruct_job(
    *,
    state: str,
    task: str,
    seed: int,
    reference: tuple[np.ndarray, ...],
    future: tuple[np.ndarray, ...],
    receipt: dict[str, Any],
) -> list[dict[str, Any]]:
    x, y, s, _, split, feature_columns = reference
    x_future, y_future, s_future, env_future, future_columns = future
    if tuple(feature_columns) != tuple(future_columns):
        raise ValueError(f"{task} {state} reference and future columns differ")

    rng = np.random.default_rng(100_003 * seed + 2027)
    train_idx, construction_idx = split_eraser_train_construction(
        np.flatnonzero(split == 0), y, s, s, rng
    )
    train_idx = random_cap(train_idx, 8000, rng)
    construction_idx = random_cap(construction_idx, 4000, rng)
    reference_idx = balanced_stratum_sample(
        np.flatnonzero(split == 1),
        y,
        s,
        maximum_total=24_000,
        seed=seed * 100 + 2,
    )

    future_all = np.arange(len(x_future), dtype=np.int64)
    member_raw, diagnostic_raw, member_pumas, diagnostic_pumas = (
        puma_bridge_diagnostic_split(
            future_all, env_future, seed=seed * 100 + 73
        )
    )
    member_idx = balanced_stratum_sample(
        member_raw,
        y_future,
        s_future,
        maximum_total=MAXIMUM_FUTURE_ROWS,
        seed=seed * 100 + 74,
    )
    diagnostic_idx = balanced_stratum_sample(
        diagnostic_raw,
        y_future,
        s_future,
        maximum_total=MAXIMUM_FUTURE_ROWS,
        seed=seed * 100 + 75,
    )

    train = np.asarray(x[train_idx], dtype=np.float32).copy()
    construction = np.asarray(x[construction_idx], dtype=np.float32).copy()
    reference_values = np.asarray(x[reference_idx], dtype=np.float32).copy()
    future_order = np.concatenate((member_idx, diagnostic_idx))
    future_values = np.asarray(x_future[future_order], dtype=np.float32).copy()
    (train, construction, reference_values, future_values), _ = preprocess(
        train,
        construction,
        reference_values,
        future_values,
        dimension=128,
        seed=seed,
    )
    deployment = np.concatenate((reference_values, future_values), axis=0)
    y_deployment = np.concatenate(
        (y[reference_idx], y_future[future_order]), axis=0
    )
    s_deployment = np.concatenate(
        (s[reference_idx], s_future[future_order]), axis=0
    )

    needed: dict[str, str] = {}
    for rule in RULES:
        selected = selected_row(receipt, rule)
        if selected is not None:
            needed[selected[0]["candidate"]] = selected[0]["method"]
    candidates = [identity_candidate(train, construction, deployment)]
    for method in sorted(set(needed.values())):
        if method == "Identity":
            continue
        candidates.extend(
            dispatch_candidates(
                METHOD_KEYS[method],
                train,
                construction,
                deployment,
                y[train_idx],
                y[construction_idx],
                y_deployment,
                s[train_idx],
                s[construction_idx],
                s_deployment,
                seed=seed,
                smoke=False,
            )
        )
    indexed = {candidate.key: candidate for candidate in candidates}
    member_offset = len(reference_idx)
    diagnostic_offset = member_offset + len(member_idx)
    results = []
    for rule in RULES:
        selected = selected_row(receipt, rule)
        if selected is None:
            results.append(
                {
                    "rule": rule,
                    "decision_2018": "abstain",
                    "runtime_action_2021": "abstain_at_2018_certification",
                }
            )
            continue
        selection, row = selected
        candidate = indexed[selection["candidate"]]
        tokenizer = fit_score_tokenizer(
            candidate.validation,
            y[construction_idx],
            token_count=4,
            seed=seed,
        )
        reference_tokens = tokenizer.encode(
            candidate.external[: len(reference_idx)]
        )
        observed_reference = build_token_table(
            reference_tokens,
            y[reference_idx],
            s[reference_idx],
            token_count=4,
            familywise_delta=0.5,
        ).counts.tolist()
        if observed_reference != row["reference_table"]["token_counts"]:
            raise ValueError(
                f"{state} {task} {seed} {rule} reference reconstruction differs"
            )

        member_tokens = tokenizer.encode(
            candidate.external[member_offset:diagnostic_offset]
        )
        diagnostic_tokens = tokenizer.encode(
            candidate.external[diagnostic_offset:]
        )
        member_table = build_token_table(
            member_tokens,
            y_future[member_idx],
            s_future[member_idx],
            token_count=4,
            familywise_delta=FUTURE_TABLE_DELTA,
        )
        diagnostic_table = build_token_table(
            diagnostic_tokens,
            y_future[diagnostic_idx],
            s_future[diagnostic_idx],
            token_count=4,
            familywise_delta=FUTURE_TABLE_DELTA,
        )
        release = row["mosaic_release" if rule == "mosaic" else "direct_release"]
        channel = np.asarray(release["release_channel"], dtype=np.float64)
        decoder = release["decoder"]
        bounds = diagnostic_bounds(
            diagnostic_table.counts,
            diagnostic_table.l1_radii,
            channel,
            decoder,
        )
        membership = (
            fixed_class_membership(
                np.asarray(row["reference_table"]["token_counts"]),
                np.asarray(row["reference_table"]["l1_radii"]),
                member_table.counts,
                member_table.l1_radii,
                row["bridge_membership"],
            )
            if rule == "mosaic"
            else None
        )
        runtime_action = (
            "release"
            if rule == "direct" or bool(membership["accepted"])
            else "abstain_out_of_bridge_class"
        )
        results.append(
            {
                "rule": rule,
                "decision_2018": "deploy",
                "candidate": selection["candidate"],
                "runtime_action_2021": runtime_action,
                "future_bridge_class_membership": membership,
                "future_diagnostic": {
                    **bounds,
                    "source_contract_violation_empirical": (
                        bounds["source_advantage_empirical"] > SOURCE_THRESHOLD
                    ),
                    "utility_contract_violation_empirical": (
                        bounds["worst_conditional_error_empirical"]
                        > UTILITY_THRESHOLD
                    ),
                    "source_contract_violation_confirmed": (
                        bounds["source_advantage_lower"] > SOURCE_THRESHOLD
                    ),
                    "utility_contract_violation_confirmed": (
                        bounds["worst_conditional_error_lower"]
                        > UTILITY_THRESHOLD
                    ),
                    "both_contracts_safe_confirmed": (
                        bounds["source_advantage_upper"] <= SOURCE_THRESHOLD
                        and bounds["worst_conditional_error_upper"]
                        <= UTILITY_THRESHOLD
                    ),
                },
                "future_membership_pumas": list(member_pumas),
                "future_diagnostic_pumas": list(diagnostic_pumas),
                "future_membership_rows": int(len(member_idx)),
                "future_diagnostic_rows": int(len(diagnostic_idx)),
                "reference_reconstruction_match": True,
            }
        )
    return results


def paired_witnesses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["target_state"]), str(row["task"]), int(row["seed"]))
        grouped.setdefault(key, {})[str(row["rule"])] = row
    witnesses = []
    for (state, task, seed), pair in sorted(grouped.items()):
        direct = pair["direct"]
        mosaic = pair["mosaic"]
        direct_diagnostic = direct.get("future_diagnostic", {})
        direct_confirmed_failure = bool(
            direct_diagnostic.get("source_contract_violation_confirmed", False)
            or direct_diagnostic.get("utility_contract_violation_confirmed", False)
        )
        direct_empirical_failure = bool(
            direct_diagnostic.get("source_contract_violation_empirical", False)
            or direct_diagnostic.get("utility_contract_violation_empirical", False)
        )
        mosaic_abstains = mosaic.get("runtime_action_2021", "").startswith("abstain")
        mosaic_safe = bool(
            mosaic.get("future_diagnostic", {}).get(
                "both_contracts_safe_confirmed", False
            )
        )
        witnesses.append(
            {
                "target_state": state,
                "task": task,
                "seed": seed,
                "direct_deployed_2018": direct.get("decision_2018") == "deploy",
                "direct_empirical_failure_2021": direct_empirical_failure,
                "direct_confirmed_failure_2021": direct_confirmed_failure,
                "mosaic_runtime_abstains_2021": mosaic_abstains,
                "mosaic_confirmed_safe_2021": mosaic_safe,
                "empirical_natural_failure_witness": bool(
                    direct.get("decision_2018") == "deploy"
                    and direct_empirical_failure
                    and (mosaic_abstains or mosaic_safe)
                ),
                "confirmed_natural_failure_witness": bool(
                    direct.get("decision_2018") == "deploy"
                    and direct_confirmed_failure
                    and (mosaic_abstains or mosaic_safe)
                ),
            }
        )
    return witnesses


def summarize(
    rows: list[dict[str, Any]], witnesses: list[dict[str, Any]]
) -> dict[str, Any]:
    direct = [
        row for row in rows
        if row["rule"] == "direct" and row["decision_2018"] == "deploy"
    ]
    mosaic = [row for row in rows if row["rule"] == "mosaic"]
    mosaic_releases = [
        row for row in mosaic if row["runtime_action_2021"] == "release"
    ]
    return {
        "population_jobs": len(witnesses),
        "reported_rule_rows": len(rows),
        "direct_frozen_deployments": len(direct),
        "direct_empirical_contract_violations_2021": sum(
            row["future_diagnostic"]["source_contract_violation_empirical"]
            or row["future_diagnostic"]["utility_contract_violation_empirical"]
            for row in direct
        ),
        "direct_confirmed_contract_violations_2021": sum(
            row["future_diagnostic"]["source_contract_violation_confirmed"]
            or row["future_diagnostic"]["utility_contract_violation_confirmed"]
            for row in direct
        ),
        "mosaic_runtime_releases_2021": len(mosaic_releases),
        "mosaic_runtime_abstentions_2021": len(mosaic) - len(mosaic_releases),
        "mosaic_confirmed_safe_releases_2021": sum(
            row["future_diagnostic"]["both_contracts_safe_confirmed"]
            for row in mosaic_releases
        ),
        "empirical_natural_failure_witnesses": sum(
            row["empirical_natural_failure_witness"] for row in witnesses
        ),
        "confirmed_natural_failure_witnesses": sum(
            row["confirmed_natural_failure_witness"] for row in witnesses
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=LOCK)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("data/acs_pums/2018/1-Year/psam_p06.csv"),
    )
    parser.add_argument("--future-raw-root", type=Path, default=Path("data/acs_pums"))
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock = validate_lock(args.lock, args.reference_csv)
    data_lock = load(DATA_LOCK)
    reference_frame = load_reference_frame(args.reference_csv)
    rows: list[dict[str, Any]] = []
    assets: list[dict[str, Any]] = []
    for state in TARGET_STATES:
        future_frame, asset = load_future_frame(
            state,
            raw_root=args.future_raw_root,
            allow_download=args.download,
        )
        assets.append(asset)
        for task in TASKS:
            reference = prepare_reference(
                reference_frame,
                task=task,
                state=state,
                data_lock=data_lock,
            )
            future = extract_task(future_frame, task_registry()[task])
            for seed in SEEDS:
                receipt = load(
                    RECEIPTS / f"ACS-{task}-CA-{state}__seed{seed}.json"
                )
                for row in reconstruct_job(
                    state=state,
                    task=task,
                    seed=seed,
                    reference=reference,
                    future=future,
                    receipt=receipt,
                ):
                    rows.append(
                        {
                            "target_state": state,
                            "task": task,
                            "seed": seed,
                            **row,
                        }
                    )
        del future_frame

    witnesses = paired_witnesses(rows)
    summary = summarize(rows, witnesses)
    payload = {
        "name": "MOSAIC locked ACS pandemic-discontinuity panel v1",
        "status": "complete_locked_pandemic_panel",
        "lock_sha256": sha256(args.lock),
        "future_raw_assets": assets,
        "rows": rows,
        "paired_witnesses": witnesses,
        "summary": summary,
        "claim_boundary": lock["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
