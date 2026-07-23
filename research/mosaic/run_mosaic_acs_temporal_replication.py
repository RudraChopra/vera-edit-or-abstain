#!/usr/bin/env python3
"""Confirm the discovered 2021 ACS failures on an untouched 2022 population."""

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

import official_eraser_adapters as eraser_adapters
from mosaic_real import balanced_stratum_sample, build_token_table, fit_score_tokenizer
from prepare_acs_natural_shift_stores import extract_task, task_registry
from run_mosaic_acs_pandemic_panel import (
    METHOD_KEYS,
    OFFICIAL_REPOSITORIES,
    RELSHIPP_TO_RELP,
    diagnostic_bounds,
    identity_candidate,
    load_reference_frame,
    prepare_reference,
    read_selected_csv,
    required_columns,
    selected_row,
)
from run_official_eraser_frontier import (
    dispatch_candidates,
    preprocess,
    random_cap,
    split_eraser_train_construction,
)


ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "research/mosaic/prereg_mosaic_acs_temporal_replication_v1.json"
OUTPUT = ROOT / "research/artifacts/mosaic_acs_temporal_replication_v1.json"
RECEIPTS = ROOT / "research/artifacts/mosaic_acs_natural_shift_v1_receipts"
DISCOVERY = ROOT / "research/artifacts/mosaic_acs_pandemic_panel_v1.json"
PANDEMIC_LOCK = ROOT / "research/mosaic/prereg_mosaic_acs_pandemic_panel_v1.json"
FUTURE_YEAR = "2022"
STATE_FIPS = {"FL": "12", "IL": "17"}
SOURCE_THRESHOLD = 0.35
UTILITY_THRESHOLD = 0.40
FAMILYWISE_DELTA = 0.05
MAXIMUM_FUTURE_ROWS = 128_000
WITNESSES = (
    {
        "target_state": "FL",
        "task": "public_coverage",
        "seed": 1401,
        "candidate": "TaCo::components_removed=1",
    },
    {
        "target_state": "FL",
        "task": "public_coverage",
        "seed": 1402,
        "candidate": "R-LACE::rank=4",
    },
    {
        "target_state": "IL",
        "task": "public_coverage",
        "seed": 1402,
        "candidate": "Identity::unedited",
    },
)
TABLE_DELTA = FAMILYWISE_DELTA / len(WITNESSES)


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


def census_url(state: str) -> str:
    return (
        "https://www2.census.gov/programs-surveys/acs/data/pums/"
        f"{FUTURE_YEAR}/1-Year/csv_p{state.lower()}.zip"
    )


def expected_protocol() -> dict[str, Any]:
    return {
        "design": (
            "discovery-confirmation temporal replication: the locked 2021 "
            "diagnostic fold nominated exactly three direct-rule utility "
            "failures; this protocol freezes those interfaces before any 2022 "
            "ACS asset is accessed"
        ),
        "discovery_year": "2021",
        "confirmation_year": FUTURE_YEAR,
        "reference_year": "2018",
        "witnesses": [dict(row) for row in WITNESSES],
        "hypotheses": len(WITNESSES),
        "familywise_delta": FAMILYWISE_DELTA,
        "table_delta": TABLE_DELTA,
        "source_advantage_threshold": SOURCE_THRESHOLD,
        "utility_threshold": UTILITY_THRESHOLD,
        "maximum_confirmation_rows_per_interface": MAXIMUM_FUTURE_ROWS,
        "sampling": (
            "one deterministic balanced sample over all 2022 rows in each "
            "source-task stratum; no 2022 model, channel, tokenizer, candidate, "
            "threshold, or hypothesis selection"
        ),
        "confirmation": (
            "a witness is confirmed only when the simultaneous lower bound on "
            "the frozen direct interface's worst conditional task error exceeds "
            "0.40; its paired MOSAIC decision must be the frozen 2018 abstention"
        ),
        "future_asset_urls": {
            state: census_url(state) for state in sorted(STATE_FIPS)
        },
    }


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


def load_future_frame(
    state: str,
    *,
    raw_root: Path,
    allow_download: bool,
) -> tuple[Any, dict[str, Any]]:
    local = raw_root / FUTURE_YEAR / "1-Year" / f"psam_p{STATE_FIPS[state]}.csv"
    if local.exists():
        frame = read_selected_csv(local, required_columns())
        return apply_relationship_crosswalk(frame, state=state), {
            "state": state,
            "source": "local_uncompressed_csv",
            "path": str(local),
            "bytes": local.stat().st_size,
            "sha256": sha256(local),
            "url": census_url(state),
        }
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
            name
            for name in bundle.namelist()
            if name.lower().endswith(".csv") and "psam_p" in name.lower()
        ]
        if len(members) != 1:
            raise ValueError(f"unexpected Census archive members: {members}")
        with bundle.open(members[0]) as handle:
            frame = read_selected_csv(handle, required_columns())
    return apply_relationship_crosswalk(frame, state=state), {
        "state": state,
        "source": "locked_url_streamed_in_memory",
        "archive_member": members[0],
        "compressed_bytes": len(archive),
        "compressed_sha256": sha256_bytes(archive),
        "url": census_url(state),
    }


def receipt_path(witness: dict[str, Any]) -> Path:
    return RECEIPTS / (
        f"ACS-{witness['task']}-CA-{witness['target_state']}"
        f"__seed{witness['seed']}.json"
    )


def validate_discovery() -> None:
    report = load(DISCOVERY)
    observed = []
    for row in report["paired_witnesses"]:
        if row["empirical_natural_failure_witness"]:
            direct = next(
                item
                for item in report["rows"]
                if item["target_state"] == row["target_state"]
                and item["task"] == row["task"]
                and item["seed"] == row["seed"]
                and item["rule"] == "direct"
            )
            observed.append(
                {
                    "target_state": row["target_state"],
                    "task": row["task"],
                    "seed": row["seed"],
                    "candidate": direct["candidate"],
                }
            )
    if tuple(sorted(observed, key=json.dumps)) != tuple(
        sorted((dict(row) for row in WITNESSES), key=json.dumps)
    ):
        raise ValueError("locked witness family differs from the discovery report")


def validate_lock(path: Path, reference_csv: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if sidecar.read_text(encoding="utf-8").split()[0] != sha256(path):
        raise ValueError("temporal-replication lock sidecar mismatch")
    lock = load(path)
    if lock.get("status") != "locked_before_2022_download":
        raise ValueError("temporal-replication lock has the wrong status")
    if lock.get("protocol") != expected_protocol():
        raise ValueError("temporal-replication protocol differs from its lock")
    for relative, expected in lock["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"locked code mismatch: {relative}")
    for relative, expected in lock["input_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"locked input mismatch: {relative}")
    expected_reference = lock["reference_raw_asset"]
    if reference_csv.stat().st_size != expected_reference["bytes"]:
        raise ValueError("2018 California raw file size differs from the lock")
    if expected_reference != load(PANDEMIC_LOCK)["reference_raw_asset"]:
        raise ValueError("2018 reference asset differs from the prior committed lock")
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
    validate_discovery()
    return lock


def configure_official_repositories(root: Path) -> None:
    for attribute, directory in OFFICIAL_REPOSITORIES.items():
        setattr(eraser_adapters, attribute, root / directory)


def reconstruct_interface(
    witness: dict[str, Any],
    *,
    reference_frame: Any,
    future_frame: Any,
    data_lock: dict[str, Any],
) -> dict[str, Any]:
    state = str(witness["target_state"])
    task = str(witness["task"])
    seed = int(witness["seed"])
    receipt = load(receipt_path(witness))
    selected = selected_row(receipt, "direct")
    if selected is None:
        raise ValueError(f"{state} {task} {seed}: direct rule is not deployed")
    selection, receipt_row = selected
    if selection["candidate"] != witness["candidate"]:
        raise ValueError(f"{state} {task} {seed}: frozen candidate differs")
    mosaic_selection = receipt["alphabets"]["4"]["primary_selection"]["mosaic"]
    if mosaic_selection["decision"] != "abstain":
        raise ValueError(f"{state} {task} {seed}: paired MOSAIC decision is not abstain")

    x, y, s, _, split, feature_columns = prepare_reference(
        reference_frame,
        task=task,
        state=state,
        data_lock=data_lock,
    )
    x_future, y_future, s_future, _, future_columns = extract_task(
        future_frame, task_registry()[task]
    )
    if tuple(feature_columns) != tuple(future_columns):
        raise ValueError(f"{task} {state}: reference and future columns differ")

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
    future_idx = balanced_stratum_sample(
        np.arange(len(x_future), dtype=np.int64),
        y_future,
        s_future,
        maximum_total=MAXIMUM_FUTURE_ROWS,
        seed=seed * 100 + 87,
    )

    train = np.asarray(x[train_idx], dtype=np.float32).copy()
    construction = np.asarray(x[construction_idx], dtype=np.float32).copy()
    reference_values = np.asarray(x[reference_idx], dtype=np.float32).copy()
    future_values = np.asarray(x_future[future_idx], dtype=np.float32).copy()
    (train, construction, reference_values, future_values), _ = preprocess(
        train,
        construction,
        reference_values,
        future_values,
        dimension=128,
        seed=seed,
    )
    deployment = np.concatenate((reference_values, future_values), axis=0)
    y_deployment = np.concatenate((y[reference_idx], y_future[future_idx]))
    s_deployment = np.concatenate((s[reference_idx], s_future[future_idx]))
    candidates = [identity_candidate(train, construction, deployment)]
    method = str(selection["method"])
    if method != "Identity":
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
    candidate = indexed[selection["candidate"]]
    tokenizer = fit_score_tokenizer(
        candidate.validation,
        y[construction_idx],
        token_count=4,
        seed=seed,
    )
    reference_tokens = tokenizer.encode(candidate.external[: len(reference_idx)])
    observed_reference = build_token_table(
        reference_tokens,
        y[reference_idx],
        s[reference_idx],
        token_count=4,
        familywise_delta=0.5,
    ).counts.tolist()
    if observed_reference != receipt_row["reference_table"]["token_counts"]:
        raise ValueError(f"{state} {task} {seed}: reference reconstruction differs")

    future_tokens = tokenizer.encode(candidate.external[len(reference_idx) :])
    future_table = build_token_table(
        future_tokens,
        y_future[future_idx],
        s_future[future_idx],
        token_count=4,
        familywise_delta=TABLE_DELTA,
    )
    release = receipt_row["direct_release"]
    bounds = diagnostic_bounds(
        future_table.counts,
        future_table.l1_radii,
        np.asarray(release["release_channel"], dtype=np.float64),
        release["decoder"],
    )
    return {
        **witness,
        "direct_decision_2018": "deploy",
        "mosaic_decision_2018": "abstain",
        "reference_reconstruction_match": True,
        "confirmation_rows": int(len(future_idx)),
        "confirmation_stratum_counts": future_table.counts.sum(axis=2).tolist(),
        "confirmation_table_delta": TABLE_DELTA,
        "future_diagnostic": {
            **bounds,
            "utility_contract_violation_empirical": (
                bounds["worst_conditional_error_empirical"] > UTILITY_THRESHOLD
            ),
            "utility_contract_violation_confirmed": (
                bounds["worst_conditional_error_lower"] > UTILITY_THRESHOLD
            ),
            "source_contract_violation_empirical": (
                bounds["source_advantage_empirical"] > SOURCE_THRESHOLD
            ),
            "source_contract_violation_confirmed": (
                bounds["source_advantage_lower"] > SOURCE_THRESHOLD
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--future-raw-root", type=Path, required=True)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock = validate_lock(LOCK, args.reference_csv)
    configure_official_repositories(args.official_root)
    data_lock = load(
        ROOT / "research/mosaic/prereg_mosaic_acs_natural_shift_data_v1.json"
    )
    reference_frame = load_reference_frame(args.reference_csv)
    future_frames: dict[str, Any] = {}
    assets = []
    for state in sorted({row["target_state"] for row in WITNESSES}):
        future_frames[state], asset = load_future_frame(
            state,
            raw_root=args.future_raw_root,
            allow_download=args.download,
        )
        assets.append(asset)

    rows = [
        reconstruct_interface(
            dict(witness),
            reference_frame=reference_frame,
            future_frame=future_frames[str(witness["target_state"])],
            data_lock=data_lock,
        )
        for witness in WITNESSES
    ]
    summary = {
        "registered_interfaces": len(rows),
        "empirical_2022_utility_violations": sum(
            row["future_diagnostic"]["utility_contract_violation_empirical"]
            for row in rows
        ),
        "familywise_confirmed_2022_utility_violations": sum(
            row["future_diagnostic"]["utility_contract_violation_confirmed"]
            for row in rows
        ),
        "paired_mosaic_2018_abstentions": sum(
            row["mosaic_decision_2018"] == "abstain" for row in rows
        ),
    }
    payload = {
        "name": "MOSAIC ACS 2022 temporal failure replication v1",
        "status": "complete_locked_confirmation",
        "claim_boundary": lock["claim_boundary"],
        "lock_sha256": sha256(LOCK),
        "discovery_report_sha256": sha256(DISCOVERY),
        "future_raw_assets": assets,
        "rows": rows,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
