"""Shared reconstruction and evaluation helpers for released-interface utility."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mosaic_real import (
    SPLIT_EXTERNAL,
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    balanced_stratum_sample,
    build_token_table,
    fit_score_tokenizer,
    load_frozen_store,
    sha256,
)
from run_mosaic_bridge_frontier import stratified_bridge_diagnostic_split
from run_mosaic_real_pilot import DATASETS
from run_official_eraser_frontier import (
    dispatch_candidates,
    preprocess,
    random_cap,
    split_eraser_train_construction,
)


METHOD_KEYS = {
    "INLP": "inlp",
    "LEACE": "leace",
    "R-LACE": "rlace",
    "TaCo": "taco",
    "MANCE++": "mance",
}
FINE_TOKEN_COUNT = 4


@dataclass(frozen=True)
class UtilityJob:
    dataset: str
    seed: int
    threshold: str
    strict_path: Path
    selection: dict[str, object]
    release: dict[str, object]


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def selected_jobs(
    strict_dir: Path, slices: dict[str, list[str]]
) -> list[UtilityJob]:
    jobs: list[UtilityJob] = []
    seen: set[tuple[str, int, str]] = set()
    for path in sorted(strict_dir.glob("*.json")):
        payload = _load(path)
        dataset = str(payload["dataset"])
        for threshold in slices.get(dataset, []):
            selection = payload["selection_by_utility_threshold"][threshold]
            if selection.get("decision") != "deploy":
                continue
            candidate = str(selection["candidate"])
            matches = [row for row in payload["results"] if row["candidate"] == candidate]
            if len(matches) != 1 or not isinstance(matches[0].get("release_l2"), dict):
                raise ValueError(f"{path}: selected candidate has no release")
            key = (dataset, int(payload["seed"]), threshold)
            if key in seen:
                raise ValueError(f"duplicate selected utility job {key}")
            seen.add(key)
            jobs.append(
                UtilityJob(
                    dataset=dataset,
                    seed=int(payload["seed"]),
                    threshold=threshold,
                    strict_path=path,
                    selection=selection,
                    release=matches[0]["release_l2"],
                )
            )
    return jobs


def task_classifier(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=int(seed) + 101,
        ),
    )


def token_decoder(tokens: np.ndarray, labels: np.ndarray, token_count: int) -> np.ndarray:
    decoder = np.zeros(token_count, dtype=np.int16)
    for token in range(token_count):
        current = labels[tokens == token]
        if current.size:
            counts = np.bincount(current, minlength=2)
            decoder[token] = int(np.argmax(counts))
    return decoder


def metric_payload(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(np.mean(prediction == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
    }


def expected_release_metrics(
    tokens: np.ndarray,
    labels: np.ndarray,
    channel: np.ndarray,
    decoder: np.ndarray,
) -> dict[str, float]:
    probabilities = channel[tokens]
    correct = (probabilities * (decoder[None, :] == labels[:, None])).sum(axis=1)
    expected_prediction_one = probabilities[:, decoder == 1].sum(axis=1)
    positive = labels == 1
    negative = ~positive
    tpr = float(correct[positive].mean()) if positive.any() else 0.0
    tnr = float(correct[negative].mean()) if negative.any() else 0.0
    return {
        "expected_accuracy": float(correct.mean()),
        "expected_balanced_accuracy": (tpr + tnr) / 2.0,
        "expected_positive_prediction_rate": float(expected_prediction_one.mean()),
    }


def evaluate_job(job: UtilityJob) -> dict[str, object]:
    config = DATASETS[job.dataset]
    store_path = Path(config["path"])
    store = load_frozen_store(store_path, target_mode=str(config["target_mode"]))
    y = store.target
    s = store.source
    rng = np.random.default_rng(100_003 * job.seed + 2027)
    train_indices, construction_indices = split_eraser_train_construction(
        np.flatnonzero(store.split == SPLIT_TRAIN), y, s, s, rng
    )
    train_indices = random_cap(train_indices, 8000, rng)
    construction_indices = random_cap(construction_indices, 2000, rng)
    reference_indices = balanced_stratum_sample(
        np.flatnonzero(store.split == SPLIT_VALIDATION),
        y,
        s,
        maximum_total=8000,
        seed=job.seed * 100 + 2,
    )
    try:
        external_indices = balanced_stratum_sample(
            np.flatnonzero(store.split == SPLIT_EXTERNAL),
            y,
            s,
            maximum_total=12000,
            seed=job.seed * 100 + 3,
        )
    except ValueError:
        external_indices = np.flatnonzero(store.split == SPLIT_EXTERNAL).astype(np.int64)
        external_indices = random_cap(external_indices, 12000, rng)
    bridge_indices, diagnostic_indices = stratified_bridge_diagnostic_split(
        external_indices, y, s, seed=job.seed * 100 + 4
    )
    train = np.asarray(store.features[train_indices], dtype=np.float32).copy()
    construction = np.asarray(store.features[construction_indices], dtype=np.float32).copy()
    reference = np.asarray(store.features[reference_indices], dtype=np.float32).copy()
    external_order = np.concatenate((bridge_indices, diagnostic_indices))
    external = np.asarray(store.features[external_order], dtype=np.float32).copy()
    (train, construction, reference, external), preprocessing = preprocess(
        train, construction, reference, external, dimension=128, seed=job.seed
    )
    deployment = np.concatenate((reference, external), axis=0)
    y_deployment = np.concatenate((y[reference_indices], y[external_order]))
    s_deployment = np.concatenate((s[reference_indices], s[external_order]))
    method_key = METHOD_KEYS.get(str(job.selection["method"]))
    if method_key is None:
        raise ValueError(f"unsupported selected method {job.selection['method']}")
    candidates = dispatch_candidates(
        method_key,
        train,
        construction,
        deployment,
        y[train_indices],
        y[construction_indices],
        y_deployment,
        s[train_indices],
        s[construction_indices],
        s_deployment,
        seed=job.seed,
        smoke=False,
    )
    selected = [candidate for candidate in candidates if candidate.key == job.selection["candidate"]]
    if len(selected) != 1:
        raise ValueError(f"could not reconstruct {job.selection['candidate']}")
    candidate = selected[0]
    tokenizer = fit_score_tokenizer(
        candidate.validation,
        y[construction_indices],
        token_count=FINE_TOKEN_COUNT,
        seed=job.seed,
    )
    reference_tokens = tokenizer.encode(reference)
    bridge_tokens = tokenizer.encode(external[: len(bridge_indices)])
    diagnostic_features = candidate.external[len(reference_indices) + len(bridge_indices) :]
    diagnostic_tokens = tokenizer.encode(diagnostic_features)
    original_receipt = _load(Path(str(_load(job.strict_path)["original_receipt"])))
    raw_candidate = next(
        row for row in original_receipt["results"] if row["candidate"] == job.selection["candidate"]
    )
    for observed, expected, name in (
        (build_token_table(reference_tokens, y[reference_indices], s[reference_indices], token_count=4, familywise_delta=0.5).counts.tolist(), raw_candidate["reference_token_counts"], "reference"),
        (build_token_table(bridge_tokens, y[bridge_indices], s[bridge_indices], token_count=4, familywise_delta=0.5).counts.tolist(), raw_candidate["bridge_token_counts"], "bridge"),
        (build_token_table(diagnostic_tokens, y[diagnostic_indices], s[diagnostic_indices], token_count=4, familywise_delta=0.5).counts.tolist(), raw_candidate["diagnostic_token_counts"], "diagnostic"),
    ):
        if observed != expected:
            raise ValueError(f"reconstructed {name} token counts differ from locked receipt")
    diagnostic_labels = np.asarray(y[diagnostic_indices], dtype=np.int16)
    original_diagnostic = deployment[len(reference_indices) + len(bridge_indices) :]
    original_classifier = task_classifier(job.seed)
    original_classifier.fit(train, y[train_indices])
    edited_classifier = task_classifier(job.seed)
    edited_classifier.fit(candidate.train, y[train_indices])
    construction_tokens = tokenizer.encode(candidate.validation)
    before_channel_decoder = token_decoder(
        construction_tokens, np.asarray(y[construction_indices], dtype=np.int16), FINE_TOKEN_COUNT
    )
    channel = np.asarray(job.release["release_channel"], dtype=np.float64)
    decoder = np.asarray(job.release["decoder"], dtype=np.int16)
    if channel.shape != (FINE_TOKEN_COUNT, 2) or decoder.shape != (2,):
        raise ValueError("selected release has unexpected finite-token shape")
    if not np.allclose(channel.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("selected release channel is not stochastic")
    return {
        "dataset": job.dataset,
        "seed": job.seed,
        "utility_threshold": job.threshold,
        "strict_receipt": str(job.strict_path),
        "strict_receipt_sha256": sha256(job.strict_path),
        "original_receipt": str(_load(job.strict_path)["original_receipt"]),
        "selected_candidate": str(job.selection["candidate"]),
        "selected_method": str(job.selection["method"]),
        "selected_strength": str(job.selection["strength"]),
        "diagnostic_examples": int(len(diagnostic_indices)),
        "reconstruction": {
            "store_manifest_sha256": sha256(store_path / "manifest.json"),
            "preprocessing": preprocessing,
            "reference_examples": int(len(reference_indices)),
            "bridge_examples": int(len(bridge_indices)),
            "tokenizer_thresholds": list(tokenizer.thresholds),
            "token_count_receipt_match": True,
        },
        "released_interface": expected_release_metrics(
            diagnostic_tokens, diagnostic_labels, channel, decoder
        ),
        "four_bin_tokenizer_before_channel": metric_payload(
            diagnostic_labels, before_channel_decoder[diagnostic_tokens]
        ),
        "full_feature_classifier_on_selected_edit": metric_payload(
            diagnostic_labels, edited_classifier.predict(diagnostic_features)
        ),
        "full_feature_classifier_on_unedited_representation": metric_payload(
            diagnostic_labels, original_classifier.predict(original_diagnostic)
        ),
        "claim_boundary": (
            "All task metrics use the untouched diagnostic fold. The released "
            "interface score is the exact expectation over one persistent token draw; "
            "the two full-feature scores are downstream logistic task classifiers and "
            "are not MOSAIC certificates."
        ),
    }
