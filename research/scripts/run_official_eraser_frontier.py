"""Run one pinned official eraser family under VERA's shared real-data protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from official_eraser_adapters import (
    EditedCandidate,
    inlp_candidates,
    leace_candidate,
    mance_candidate,
    rlace_candidate,
    taco_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
DEFAULT_EXTERNAL = Path("/Volumes/Backups/FARO/artifacts/vera_real_study")
DEFAULT_RECEIPTS = ROOT / "artifacts" / "real_study_receipts"
DEFAULT_PREREG = ROOT / "prereg_real.json"
DEFAULT_HASH = ROOT / "prereg_real.sha256"
SPLIT_TRAIN = 0
SPLIT_VALIDATION = 1
SPLIT_EXTERNAL = 2


@dataclass(frozen=True)
class Store:
    manifest: dict[str, object]
    z: np.ndarray
    y: np.ndarray
    s: np.ndarray
    g: np.ndarray
    split: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


METHOD_NAMES = {
    "inlp": "INLP",
    "leace": "LEACE",
    "rlace": "RLACE",
    "taco": "TaCo",
    "mance": "MANCE++",
}


def enforce_preregistration(path: Path) -> tuple[dict[str, object], str, str]:
    if path.resolve() != DEFAULT_PREREG.resolve():
        raise RuntimeError(f"claim-grade runs require {DEFAULT_PREREG}")
    if not path.is_file() or not DEFAULT_HASH.is_file():
        raise RuntimeError("real-study preregistration or hash sidecar is missing")
    observed = sha256(path)
    expected = DEFAULT_HASH.read_text(encoding="utf-8").split()[0]
    if observed != expected:
        raise RuntimeError("real-study preregistration hash mismatch")
    relative = path.resolve().relative_to(REPOSITORY.resolve()).as_posix()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != observed:
        raise RuntimeError("real-study preregistration is not committed at HEAD")
    prereg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(prereg, dict) or prereg.get("status") != "locked_before_claim_grade_runs":
        raise RuntimeError("real-study preregistration is not marked as locked")
    return prereg, observed, git_head()


def load_store(path: Path) -> Store:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    arrays = manifest["arrays"]
    source_path = path / arrays.get("s", "s.npy")
    environment_path = path / arrays.get("g", arrays.get("s", "s.npy"))
    store = Store(
        manifest=manifest,
        z=np.load(path / arrays.get("z", "z.npy"), mmap_mode="r"),
        y=np.load(path / arrays.get("y", "y.npy"), mmap_mode="r"),
        s=np.load(source_path, mmap_mode="r"),
        g=np.load(environment_path, mmap_mode="r"),
        split=np.load(path / arrays.get("split", "split.npy"), mmap_mode="r"),
    )
    n = int(manifest["n_examples"])
    if not all(len(values) == n for values in (store.z, store.y, store.s, store.g, store.split)):
        raise ValueError("store arrays do not match manifest")
    return store


def validate_claim_configuration(
    args: argparse.Namespace,
    prereg: dict[str, object],
    store: Store,
) -> dict[str, object]:
    study = prereg.get("real_study")
    if not isinstance(study, dict):
        raise RuntimeError("preregistration has no real_study object")
    datasets = study.get("datasets")
    if not isinstance(datasets, dict) or args.dataset not in datasets:
        raise RuntimeError(f"dataset {args.dataset!r} is not preregistered")
    dataset = datasets[args.dataset]
    if not isinstance(dataset, dict):
        raise RuntimeError("invalid preregistered dataset record")
    expected_store = Path(str(dataset.get("store_dir", ""))).resolve()
    if args.store_dir.resolve() != expected_store:
        raise RuntimeError(
            f"store path differs from preregistration: {args.store_dir.resolve()} != {expected_store}"
        )
    manifest_hash = sha256(args.store_dir / "manifest.json")
    if manifest_hash != dataset.get("manifest_sha256"):
        raise RuntimeError("store manifest differs from preregistration")

    expected_scalars = {
        "seed": set(int(value) for value in study.get("seeds", [])),
        "max_train": int(study.get("max_train", -1)),
        "max_construction": int(study.get("max_construction", -1)),
        "max_certification": int(study.get("max_certification", -1)),
        "max_external": int(study.get("max_external", -1)),
        "dimension": int(study.get("pca_dimension", -1)),
    }
    if args.seed not in expected_scalars.pop("seed"):
        raise RuntimeError(f"seed {args.seed} is not preregistered")
    for argument, expected in expected_scalars.items():
        observed = int(getattr(args, argument))
        if observed != expected:
            raise RuntimeError(f"{argument}={observed} differs from preregistered {expected}")

    methods = study.get("methods")
    if not isinstance(methods, dict) or args.method not in methods:
        raise RuntimeError(f"method {args.method!r} is not preregistered")
    method = methods[args.method]
    if not isinstance(method, dict) or method.get("display_name") != METHOD_NAMES[args.method]:
        raise RuntimeError("invalid preregistered method configuration")
    expected_method_configs = {
        "inlp": {
            "ranks": [1, 2, 4, 8],
            "classifier": "SGDClassifier",
            "loss": "log_loss",
            "fit_intercept": True,
            "max_iter": 5000,
            "tol": 0.0001,
            "n_iter_no_change": 20,
            "alpha": 0.0001,
            "by_class": False,
            "dropout_rate": 0,
        },
        "leace": {"candidate": "closed_form", "method": "leace"},
        "rlace": {
            "ranks": [1, 4],
            "iterations": 10_000,
            "epsilon": 0.002,
            "batch_size": 256,
            "evaluate_every": 1000,
            "optimizer": "SGD",
            "projection_optimizer": {"lr": 0.005, "weight_decay": 0.0001, "momentum": 0.0},
            "predictor_optimizer": {"lr": 0.005, "weight_decay": 0.00001, "momentum": 0.9},
        },
        "taco": {
            "removals": [1, 2, 3, 5],
            "components": 20,
            "sobol_sampled": 500,
            "sobol_design": 8,
            "head_steps": 250,
            "pca_fit": "eraser_train_only",
            "head_optimizer": "AdamW",
            "head_learning_rate": 0.002,
            "head_weight_decay": 0.0001,
        },
        "mance": {
            "variant": "mance++",
            "epsilon": 0.05,
            "steps": 3,
            "n_neighbors": 8,
            "scorer_hidden": 128,
            "scorer_steps": 120,
            "scorer_refit_every": 3,
            "eval_hidden": 64,
            "eval_steps": 80,
            "stop_at_floor": False,
        },
    }
    if method.get("candidate_configuration") != expected_method_configs[args.method]:
        raise RuntimeError("method candidate configuration differs from the locked runner")
    expected_target_probe = {
        "class": "sklearn.linear_model.LogisticRegression",
        "C": 1.0,
        "class_weight": "balanced",
        "max_iter": 2000,
        "solver": "lbfgs",
        "seed_offset": 101,
    }
    expected_attackers = {
        "linear": {
            "class": "sklearn.linear_model.LogisticRegression",
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 2000,
            "solver": "lbfgs",
            "seed_offset": 11,
        },
        "rbf": {
            "class": "Nystroem-RBF plus LogisticRegression",
            "gamma": "1 / representation_dimension",
            "n_components": "min(256, max(32, 2 * representation_dimension))",
            "standardize_after_features": True,
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1500,
            "nystroem_seed_offset": 13,
            "logistic_seed_offset": 17,
        },
        "forest": {
            "class": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": 200,
            "min_samples_leaf": 3,
            "class_weight": "balanced_subsample",
            "n_jobs": 8,
            "seed_offset": 19,
        },
        "mlp": {
            "class": "sklearn.neural_network.MLPClassifier",
            "hidden_layer_sizes": [128, 64],
            "alpha": 0.0001,
            "batch_size": 256,
            "learning_rate_init": 0.001,
            "max_iter": 150,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 12,
            "standardize": True,
            "seed_offset": 23,
        },
    }
    if study.get("target_probe") != expected_target_probe:
        raise RuntimeError("target probe differs from the locked runner")
    if study.get("leakage_attackers") != expected_attackers:
        raise RuntimeError("attacker portfolio differs from the locked runner")
    if int(store.manifest.get("n_examples", 0)) <= 0:
        raise RuntimeError("dataset manifest has no examples")
    return {
        "dataset_manifest_sha256": manifest_hash,
        "method": method,
        "runner_parameters": {
            "max_train": args.max_train,
            "max_construction": args.max_construction,
            "max_certification": args.max_certification,
            "max_external": args.max_external,
            "pca_dimension": args.dimension,
        },
    }


def validate_official_candidates(
    candidates: list[EditedCandidate],
    locked_configuration: dict[str, object],
) -> None:
    method = locked_configuration.get("method")
    if not isinstance(method, dict):
        raise RuntimeError("locked method record is missing")
    if len(candidates) != int(method.get("candidate_count", -1)):
        raise RuntimeError("official adapter returned an unexpected candidate count")
    expected_commit = method.get("upstream_commit")
    expected_remote = method.get("upstream_remote")
    for candidate in candidates:
        if candidate.provenance.get("commit") != expected_commit:
            raise RuntimeError(f"{candidate.key} came from the wrong upstream commit")
        if candidate.provenance.get("remote") != expected_remote:
            raise RuntimeError(f"{candidate.key} came from the wrong upstream remote")
        repository = Path(str(candidate.provenance.get("repository", "")))
        diff = subprocess.run(
            ["git", "-C", str(repository), "diff", "--quiet", "HEAD", "--"],
            check=False,
        )
        if diff.returncode != 0:
            raise RuntimeError(f"tracked files differ from upstream commit in {repository}")
        untracked = subprocess.run(
            ["git", "-C", str(repository), "ls-files", "--others", "--exclude-standard"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        material_untracked = [
            name
            for name in untracked
            if "__pycache__" not in name and not name.endswith((".pyc", ".DS_Store"))
        ]
        if material_untracked:
            raise RuntimeError(
                f"untracked upstream files could affect execution in {repository}: {material_untracked}"
            )


def random_cap(indices: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if cap <= 0 or len(indices) <= cap:
        return np.sort(np.asarray(indices, dtype=np.int64))
    return np.sort(rng.choice(indices, size=cap, replace=False).astype(np.int64))


def split_eraser_train_construction(
    indices: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    g: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    eraser_train: list[int] = []
    construction: list[int] = []
    groups = sorted({(int(y[index]), int(s[index]), int(g[index])) for index in indices})
    for group in groups:
        group_indices = np.asarray(
            [
                index
                for index in indices
                if (int(y[index]), int(s[index]), int(g[index])) == group
            ],
            dtype=np.int64,
        )
        rng.shuffle(group_indices)
        if len(group_indices) == 1:
            eraser_train.extend(group_indices.tolist())
            continue
        construction_count = max(1, int(round(0.2 * len(group_indices))))
        construction_count = min(construction_count, len(group_indices) - 1)
        construction.extend(group_indices[:construction_count].tolist())
        eraser_train.extend(group_indices[construction_count:].tolist())
    if not construction or not eraser_train:
        shuffled = np.asarray(indices, dtype=np.int64).copy()
        rng.shuffle(shuffled)
        construction_count = max(1, int(round(0.2 * len(shuffled))))
        construction_count = min(construction_count, len(shuffled) - 1)
        construction = shuffled[:construction_count].tolist()
        eraser_train = shuffled[construction_count:].tolist()
    return np.sort(np.asarray(eraser_train)), np.sort(np.asarray(construction))


def materialize(
    store: Store, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(store.z[indices], dtype=np.float32).copy(),
        np.asarray(store.y[indices], dtype=np.int64).copy(),
        np.asarray(store.s[indices], dtype=np.int64).copy(),
        np.asarray(store.g[indices], dtype=np.int64).copy(),
    )


def preprocess(
    train: np.ndarray,
    construction: np.ndarray,
    certification: np.ndarray,
    external: np.ndarray,
    *,
    dimension: int,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict[str, object]]:
    from sklearn.decomposition import PCA

    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[(~np.isfinite(std)) | (std < 1e-6)] = 1.0
    standardized = []
    for values in (train, construction, certification, external):
        transformed = (values - mean) / std
        transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        standardized.append(np.clip(transformed, -10.0, 10.0).astype(np.float32))
    target_dimension = min(dimension, standardized[0].shape[1], len(train) - 1)
    if target_dimension < standardized[0].shape[1]:
        reducer = PCA(n_components=target_dimension, svd_solver="randomized", random_state=seed)
        reducer.fit(standardized[0])
        standardized = [reducer.transform(values).astype(np.float32) for values in standardized]
        variance = float(reducer.explained_variance_ratio_.sum())
    else:
        variance = 1.0
    return tuple(standardized), {
        "standardized_from_train_only": True,
        "pca_fit_on_train_only": True,
        "input_dimension": int(train.shape[1]),
        "output_dimension": int(target_dimension),
        "explained_variance_ratio_sum": variance,
    }


def make_target_probe(seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
        random_state=seed,
    )


def make_attackers(seed: int, dimension: int) -> dict[str, object]:
    return {
        "linear": LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=seed + 11,
        ),
        "rbf": make_pipeline(
            Nystroem(
                kernel="rbf",
                gamma=1.0 / max(dimension, 1),
                n_components=min(256, max(32, dimension * 2)),
                random_state=seed + 13,
            ),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1500,
                random_state=seed + 17,
            ),
        ),
        "forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=8,
            random_state=seed + 19,
        ),
        "mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=150,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=12,
                random_state=seed + 23,
            ),
        ),
    }


def balanced_correctness(source: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(np.unique(source)) < 2:
        return None
    return float(balanced_accuracy_score(source, prediction))


def evaluate_candidate(
    candidate: EditedCandidate,
    identity_train: np.ndarray,
    identity_certification: np.ndarray,
    identity_external: np.ndarray,
    y_train: np.ndarray,
    y_certification: np.ndarray,
    y_external: np.ndarray,
    s_train: np.ndarray,
    s_certification: np.ndarray,
    s_external: np.ndarray,
    g_certification: np.ndarray,
    g_external: np.ndarray,
    *,
    seed: int,
    split_at: int,
    audit_path: Path,
) -> dict[str, object]:
    candidate_certification = candidate.external[:split_at]
    candidate_external = candidate.external[split_at:]
    identity_target = make_target_probe(seed + 101)
    identity_target.fit(identity_train, y_train)
    identity_cert_prediction = identity_target.predict(identity_certification)
    identity_external_prediction = identity_target.predict(identity_external)
    candidate_target = make_target_probe(seed + 101)
    candidate_target.fit(candidate.train, y_train)
    candidate_cert_prediction = candidate_target.predict(candidate_certification)
    candidate_external_prediction = candidate_target.predict(candidate_external)
    target_harm_certification = (
        (candidate_cert_prediction != y_certification).astype(np.int8)
        - (identity_cert_prediction != y_certification).astype(np.int8)
    )
    target_harm_external = (
        (candidate_external_prediction != y_external).astype(np.int8)
        - (identity_external_prediction != y_external).astype(np.int8)
    )

    arrays: dict[str, np.ndarray] = {
        "target_harm_certification": target_harm_certification,
        "target_harm_external": target_harm_external,
        "source_certification": s_certification.astype(np.int16),
        "source_external": s_external.astype(np.int16),
        "environment_certification": g_certification.astype(np.int16),
        "environment_external": g_external.astype(np.int16),
        "target_certification": y_certification.astype(np.int16),
        "target_external": y_external.astype(np.int16),
    }
    leakage_metrics: dict[str, object] = {}
    for name, attacker in make_attackers(seed, candidate.train.shape[1]).items():
        attacker.fit(candidate.train, s_train)
        cert_prediction = attacker.predict(candidate_certification)
        external_prediction = attacker.predict(candidate_external)
        arrays[f"leakage_correct_certification__{name}"] = (
            cert_prediction == s_certification
        ).astype(np.int8)
        arrays[f"leakage_correct_external__{name}"] = (
            external_prediction == s_external
        ).astype(np.int8)
        leakage_metrics[name] = {
            "certification_balanced_accuracy": balanced_correctness(
                s_certification, cert_prediction
            ),
            "external_balanced_accuracy": balanced_correctness(
                s_external, external_prediction
            ),
        }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(audit_path, **arrays)
    return {
        "candidate_key": candidate.key,
        "method": candidate.method,
        "strength": candidate.strength,
        "provenance": candidate.provenance,
        "audit_npz": str(audit_path),
        "audit_npz_sha256": sha256(audit_path),
        "metrics": {
            "certification_target_balanced_accuracy": float(
                balanced_accuracy_score(y_certification, candidate_cert_prediction)
            ),
            "external_target_balanced_accuracy": float(
                balanced_accuracy_score(y_external, candidate_external_prediction)
            ),
            "certification_mean_paired_target_harm": float(target_harm_certification.mean()),
            "external_mean_paired_target_harm": float(target_harm_external.mean()),
            "attackers": leakage_metrics,
        },
    }


def dispatch_candidates(
    method: str,
    train: np.ndarray,
    construction: np.ndarray,
    deployment: np.ndarray,
    y_train: np.ndarray,
    y_construction: np.ndarray,
    y_deployment: np.ndarray,
    s_train: np.ndarray,
    s_construction: np.ndarray,
    s_deployment: np.ndarray,
    *,
    seed: int,
    smoke: bool,
) -> list[EditedCandidate]:
    if method == "inlp":
        return inlp_candidates(
            train,
            construction,
            deployment,
            s_train,
            s_construction,
            ranks=[1, 2] if smoke else [1, 2, 4, 8],
            seed=seed,
        )
    if method == "leace":
        return [leace_candidate(train, construction, deployment, s_train)]
    if method == "rlace":
        ranks = [1] if smoke else [1, 4]
        iterations = 50 if smoke else 10_000
        return [
            rlace_candidate(
                train,
                construction,
                deployment,
                s_train,
                s_construction,
                rank=rank,
                seed=seed,
                iterations=iterations,
            )
            for rank in ranks
        ]
    if method == "taco":
        return taco_candidates(
            train,
            construction,
            deployment,
            y_train,
            y_construction,
            s_train,
            s_construction,
            removals=[1] if smoke else [1, 2, 3, 5],
            seed=seed,
            components=10 if smoke else 20,
            sobol_sampled=20 if smoke else 500,
            sobol_design=2 if smoke else 8,
            head_steps=20 if smoke else 250,
        )
    if method == "mance":
        return [
            mance_candidate(
                train,
                construction,
                deployment,
                y_train,
                y_construction,
                np.zeros_like(y_deployment),
                s_train,
                s_construction,
                np.zeros_like(s_deployment),
                seed=seed,
                epsilon=0.05,
                steps=1 if smoke else 3,
            )
        ]
    raise ValueError(f"unsupported method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["inlp", "leace", "rlace", "taco", "mance"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train", type=int, default=8000)
    parser.add_argument("--max-construction", type=int, default=2000)
    parser.add_argument("--max-certification", type=int, default=8000)
    parser.add_argument("--max-external", type=int, default=8000)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--external-output-dir", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--receipt-dir", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--prereg", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--claim-grade", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.claim_grade and args.smoke:
        raise ValueError("smoke runs cannot be claim-grade")
    if args.claim_grade:
        prereg, prereg_hash, commit = enforce_preregistration(args.prereg)
    else:
        prereg, prereg_hash, commit = {}, None, git_head()
    store = load_store(args.store_dir)
    locked_configuration = (
        validate_claim_configuration(args, prereg, store) if args.claim_grade else None
    )
    rng = np.random.default_rng(100_003 * args.seed + 2027)
    train_indices, construction_indices = split_eraser_train_construction(
        np.flatnonzero(store.split == SPLIT_TRAIN), store.y, store.s, store.g, rng
    )
    train_indices = random_cap(train_indices, args.max_train, rng)
    construction_indices = random_cap(construction_indices, args.max_construction, rng)
    certification_indices = random_cap(
        np.flatnonzero(store.split == SPLIT_VALIDATION), args.max_certification, rng
    )
    external_indices = random_cap(
        np.flatnonzero(store.split == SPLIT_EXTERNAL), args.max_external, rng
    )
    train, y_train, s_train, g_train = materialize(store, train_indices)
    construction, y_construction, s_construction, g_construction = materialize(
        store, construction_indices
    )
    certification, y_certification, s_certification, g_certification = materialize(
        store, certification_indices
    )
    external, y_external, s_external, g_external = materialize(store, external_indices)
    (train, construction, certification, external), preprocessing = preprocess(
        train,
        construction,
        certification,
        external,
        dimension=args.dimension,
        seed=args.seed,
    )
    deployment = np.concatenate([certification, external], axis=0)
    y_deployment = np.concatenate([y_certification, y_external])
    s_deployment = np.concatenate([s_certification, s_external])
    candidates = dispatch_candidates(
        args.method,
        train,
        construction,
        deployment,
        y_train,
        y_construction,
        y_deployment,
        s_train,
        s_construction,
        s_deployment,
        seed=args.seed,
        smoke=args.smoke,
    )
    if args.claim_grade:
        assert locked_configuration is not None
        validate_official_candidates(candidates, locked_configuration)

    run_key = f"{args.dataset}__{args.method}__seed-{args.seed}"
    audit_dir = args.external_output_dir / run_key
    results = []
    for index, candidate in enumerate(candidates):
        results.append(
            evaluate_candidate(
                candidate,
                train,
                certification,
                external,
                y_train,
                y_certification,
                y_external,
                s_train,
                s_certification,
                s_external,
                g_certification,
                g_external,
                seed=args.seed,
                split_at=len(certification),
                audit_path=audit_dir / f"candidate-{index:02d}.npz",
            )
        )
    receipt = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_grade": bool(args.claim_grade),
        "smoke": bool(args.smoke),
        "dataset": args.dataset,
        "store_dir": str(args.store_dir),
        "store_manifest_sha256": sha256(args.store_dir / "manifest.json"),
        "method_family": args.method,
        "method_name": METHOD_NAMES[args.method],
        "seed": args.seed,
        "git_commit": commit,
        "prereg_sha256": prereg_hash,
        "claim_configuration_verified": bool(args.claim_grade),
        "locked_configuration": locked_configuration,
        "split_policy": "eraser train/construction are disjoint group-stratified partitions of official train; certification is untouched official validation",
        "external_labels_locked_during_edit_construction": True,
        "indices": {
            "train": {"n": len(train_indices), "sha256": array_sha256(train_indices)},
            "construction": {"n": len(construction_indices), "sha256": array_sha256(construction_indices)},
            "certification": {"n": len(certification_indices), "sha256": array_sha256(certification_indices)},
            "external": {"n": len(external_indices), "sha256": array_sha256(external_indices)},
        },
        "preprocessing": preprocessing,
        "source_classes": {
            "train": sorted(int(value) for value in np.unique(s_train)),
            "construction": sorted(int(value) for value in np.unique(s_construction)),
            "certification": sorted(int(value) for value in np.unique(s_certification)),
            "external": sorted(int(value) for value in np.unique(s_external)),
        },
        "environment_classes": {
            "train": sorted(int(value) for value in np.unique(g_train)),
            "construction": sorted(int(value) for value in np.unique(g_construction)),
            "certification": sorted(int(value) for value in np.unique(g_certification)),
            "external": sorted(int(value) for value in np.unique(g_external)),
        },
        "candidates": results,
    }
    args.receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = args.receipt_dir / f"{run_key}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(receipt_path), "candidates": len(results)}, indent=2))


if __name__ == "__main__":
    main()
