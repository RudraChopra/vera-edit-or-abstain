"""Pinned upstream concept-erasure adapters for VERA's shared benchmark arrays."""

from __future__ import annotations

import copy
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


INLP_REPO = Path("/Volumes/Backups/FARO/external/nullspace_projection")
RLACE_REPO = Path("/Volumes/Backups/FARO/external/rlace-icml")
LEACE_REPO = Path("/Volumes/Backups/FARO/external/concept-erasure")
TACO_REPO = Path("/Volumes/Backups/FARO/external/TaCo")
MANCE_REPO = Path("/Volumes/Backups/FARO/external/mance")


@dataclass(frozen=True)
class EditedCandidate:
    method: str
    strength: str
    train: np.ndarray
    validation: np.ndarray
    external: np.ndarray
    provenance: dict[str, object]

    @property
    def key(self) -> str:
        return f"{self.method}::{self.strength}"


def git_metadata(repo: Path) -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {"repository": str(repo), "commit": run("rev-parse", "HEAD"), "remote": run("remote", "get-url", "origin")}


def _apply_projection(values: np.ndarray, projection: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) @ projection).astype(np.float32)


def inlp_candidates(
    train: np.ndarray,
    validation: np.ndarray,
    external: np.ndarray,
    source_train: np.ndarray,
    source_validation: np.ndarray,
    *,
    ranks: Iterable[int],
    seed: int,
) -> list[EditedCandidate]:
    ranks = sorted(set(int(rank) for rank in ranks))
    if not ranks or ranks[0] < 1:
        raise ValueError("INLP ranks must be positive")
    sys.path.insert(0, str(INLP_REPO))
    from src import debias  # type: ignore
    from sklearn.linear_model import SGDClassifier

    np.random.seed(seed)
    random.seed(seed)
    params = {
        "loss": "log_loss",
        "fit_intercept": True,
        "max_iter": 5000,
        "tol": 1e-4,
        "n_iter_no_change": 20,
        "alpha": 1e-4,
        "n_jobs": 8,
        "random_state": seed,
    }
    _, rowspaces, _ = debias.get_debiasing_projection(
        SGDClassifier,
        params,
        max(ranks),
        train.shape[1],
        True,
        -1.0,
        train,
        source_train,
        validation,
        source_validation,
        by_class=False,
        dropout_rate=0,
    )
    if len(rowspaces) < max(ranks):
        raise RuntimeError(f"official INLP returned only {len(rowspaces)} directions")
    metadata = git_metadata(INLP_REPO)
    candidates = []
    for rank in ranks:
        projection = debias.get_projection_to_intersection_of_nullspaces(
            rowspaces[:rank], train.shape[1]
        )
        candidates.append(
            EditedCandidate(
                method="INLP",
                strength=f"rank={rank}",
                train=_apply_projection(train, projection),
                validation=_apply_projection(validation, projection),
                external=_apply_projection(external, projection),
                provenance={
                    **metadata,
                    "official_entrypoint": "src.debias.get_debiasing_projection",
                    "compatibility_change": "SGDClassifier loss='log_loss' replaces removed alias loss='log'",
                    "rank": rank,
                },
            )
        )
    return candidates


def leace_candidate(
    train: np.ndarray,
    validation: np.ndarray,
    external: np.ndarray,
    source_train: np.ndarray,
) -> EditedCandidate:
    sys.path.insert(0, str(LEACE_REPO))
    from concept_erasure import LeaceFitter  # type: ignore

    classes, inverse = np.unique(source_train, return_inverse=True)
    one_hot = np.eye(len(classes), dtype=np.float32)[inverse]
    eraser = LeaceFitter.fit(
        torch.from_numpy(np.asarray(train, dtype=np.float32)),
        torch.from_numpy(one_hot),
        method="leace",
    ).eraser
    outputs = []
    with torch.no_grad():
        for values in (train, validation, external):
            outputs.append(
                eraser(torch.from_numpy(np.asarray(values, dtype=np.float32)))
                .cpu()
                .numpy()
                .astype(np.float32)
            )
    return EditedCandidate(
        method="LEACE",
        strength="closed_form",
        train=outputs[0],
        validation=outputs[1],
        external=outputs[2],
        provenance={
            **git_metadata(LEACE_REPO),
            "official_entrypoint": "concept_erasure.LeaceFitter.fit(method='leace')",
        },
    )


def rlace_candidate(
    train: np.ndarray,
    validation: np.ndarray,
    external: np.ndarray,
    source_train: np.ndarray,
    source_validation: np.ndarray,
    *,
    rank: int,
    seed: int,
    iterations: int,
    device: str = "cpu",
) -> EditedCandidate:
    sys.path.insert(0, str(RLACE_REPO))
    import rlace  # type: ignore

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    rlace.EVAL_CLF_PARAMS["loss"] = "log_loss"
    rlace.EVAL_CLF_PARAMS["iters_no_change"] = 15
    rlace.NUM_CLFS_IN_EVAL = 1
    output = rlace.solve_adv_game(
        train,
        source_train,
        validation,
        source_validation,
        rank=int(rank),
        device=device,
        out_iters=int(iterations),
        epsilon=0.002,
        batch_size=256,
        evalaute_every=max(250, min(1000, iterations // 10)),
        optimizer_class=torch.optim.SGD,
        optimizer_params_P={"lr": 0.005, "weight_decay": 1e-4, "momentum": 0.0},
        optimizer_params_predictor={"lr": 0.005, "weight_decay": 1e-5, "momentum": 0.9},
    )
    projection = np.asarray(output["P"], dtype=np.float64)
    return EditedCandidate(
        method="R-LACE",
        strength=f"rank={rank}",
        train=_apply_projection(train, projection),
        validation=_apply_projection(validation, projection),
        external=_apply_projection(external, projection),
        provenance={
            **git_metadata(RLACE_REPO),
            "official_entrypoint": "rlace.solve_adv_game",
            "compatibility_change": "SGDClassifier loss='log_loss' replaces removed alias loss='log'",
            "rank": int(rank),
            "iterations": int(iterations),
            "reported_upstream_score": float(output["score"]),
        },
    )


class _TaCoHead(torch.nn.Module):
    def __init__(self, dimension: int, classes: int):
        super().__init__()
        self.linear = torch.nn.Linear(dimension, classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)

    def end_model(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values)


def _train_taco_head(
    train: np.ndarray,
    labels: np.ndarray,
    validation: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int,
    steps: int,
    device: str,
) -> _TaCoHead:
    torch.manual_seed(seed)
    classes = int(max(labels.max(), validation_labels.max())) + 1
    model = _TaCoHead(train.shape[1], classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights /= weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights.astype(np.float32)).to(device)
    )
    x_train = torch.from_numpy(np.asarray(train, dtype=np.float32)).to(device)
    y_train = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(device)
    x_validation = torch.from_numpy(np.asarray(validation, dtype=np.float32)).to(device)
    y_validation = torch.from_numpy(np.asarray(validation_labels, dtype=np.int64)).to(device)
    best_accuracy = -1.0
    best_state = None
    for step in range(steps):
        indices = torch.randint(0, len(x_train), (min(512, len(x_train)),), device=device)
        loss = loss_fn(model(x_train[indices]), y_train[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step + 1 == steps:
            with torch.no_grad():
                accuracy = float((model(x_validation).argmax(1) == y_validation).float().mean())
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("TaCo head did not train")
    model.load_state_dict(best_state)
    model.eval()
    return model


def taco_candidates(
    train: np.ndarray,
    validation: np.ndarray,
    external: np.ndarray,
    target_train: np.ndarray,
    target_validation: np.ndarray,
    source_train: np.ndarray,
    source_validation: np.ndarray,
    *,
    removals: Iterable[int],
    seed: int,
    components: int = 20,
    sobol_sampled: int = 500,
    sobol_design: int = 8,
    head_steps: int = 250,
    device: str = "cpu",
) -> list[EditedCandidate]:
    sys.path.insert(0, str(TACO_REPO))
    from TaCo.concept_removal import (  # type: ignore
        build_gender_neutral_features,
        crop_concepts,
        sobol_importance_from_sample,
    )
    from sklearn.decomposition import PCA

    removals = sorted(set(int(value) for value in removals))
    if not removals or removals[-1] >= components:
        raise ValueError("TaCo removals must be in [1, components)")
    pca = PCA(n_components=components, random_state=seed)
    u_train = pca.fit_transform(train)
    u_validation = pca.transform(validation)
    u_external = pca.transform(external)
    w = pca.components_
    target_head = _train_taco_head(
        train,
        target_train,
        validation,
        target_validation,
        seed=seed + 101,
        steps=head_steps,
        device=device,
    )
    source_head = _train_taco_head(
        train,
        source_train,
        validation,
        source_validation,
        seed=seed + 211,
        steps=head_steps,
        device=device,
    )
    x_train_torch = torch.from_numpy(np.asarray(train, dtype=np.float32))
    target_importance, _ = sobol_importance_from_sample(
        x_train_torch,
        u_train,
        w,
        target_head,
        sampled=min(sobol_sampled, len(train)),
        num_components=components,
        sobol_nb_design=sobol_design,
        device=device,
    )
    source_importance, _ = sobol_importance_from_sample(
        x_train_torch,
        u_train,
        w,
        source_head,
        sampled=min(sobol_sampled, len(train)),
        num_components=components,
        sobol_nb_design=sobol_design,
        device=device,
    )
    angle = np.arctan(target_importance / np.maximum(source_importance, 1e-12)) * 180.0 / np.pi
    metadata = git_metadata(TACO_REPO)
    candidates = []
    for removal in removals:
        w_kept, keep = crop_concepts(w, angle, num_or_threshold=components - removal)
        candidates.append(
            EditedCandidate(
                method="TaCo",
                strength=f"components_removed={removal}",
                train=np.asarray(build_gender_neutral_features(u_train, w_kept, keep), dtype=np.float32),
                validation=np.asarray(
                    build_gender_neutral_features(u_validation, w_kept, keep), dtype=np.float32
                ),
                external=np.asarray(
                    build_gender_neutral_features(u_external, w_kept, keep), dtype=np.float32
                ),
                provenance={
                    **metadata,
                    "official_entrypoints": [
                        "TaCo.concept_removal.sobol_importance_from_sample",
                        "TaCo.concept_removal.crop_concepts",
                        "TaCo.concept_removal.build_gender_neutral_features",
                    ],
                    "protocol_adapter": "PCA is fit on train only to keep certification and external splits locked",
                    "components": components,
                    "components_removed": removal,
                    "sobol_sampled": min(sobol_sampled, len(train)),
                    "sobol_design": sobol_design,
                },
            )
        )
    return candidates


def mance_candidate(
    train: np.ndarray,
    validation: np.ndarray,
    external: np.ndarray,
    target_train: np.ndarray,
    target_validation: np.ndarray,
    target_external: np.ndarray,
    source_train: np.ndarray,
    source_validation: np.ndarray,
    source_external: np.ndarray,
    *,
    seed: int,
    epsilon: float,
    steps: int,
    device: str = "cpu",
) -> EditedCandidate:
    sys.path.insert(0, str(MANCE_REPO))
    from mance import MANCE  # type: ignore

    eraser = MANCE(
        variant="mance++",
        epsilon=float(epsilon),
        n_steps=int(steps),
        n_neighbors=8,
        scorer_hidden=128,
        scorer_steps=120,
        scorer_refit_every=3,
        eval_hidden=64,
        eval_steps=80,
        seed=seed,
        device=device,
        stop_at_floor=False,
        verbose=False,
    )
    result = eraser.fit_erase(
        train,
        source_train,
        validation,
        source_validation,
        external,
        np.zeros_like(source_external),
        control_train=target_train,
        control_val=target_validation,
        control_test=np.zeros_like(target_external),
    )
    return EditedCandidate(
        method="MANCE++",
        strength=f"epsilon={epsilon:g},steps={steps}",
        train=result.train.astype(np.float32),
        validation=result.val.astype(np.float32),
        external=result.test.astype(np.float32),
        provenance={
            **git_metadata(MANCE_REPO),
            "official_entrypoint": "mance.MANCE(variant='mance++').fit_erase",
            "epsilon": float(epsilon),
            "steps_requested": int(steps),
            "steps_completed": len(result.history) - 1,
            "n_neighbors": int(result.n_neighbors),
            "tangent_rank": int(result.rank),
            "history": result.history,
            "external_labels_hidden": True,
            "external_label_adapter": "dummy labels are supplied because stop_at_floor=False; edited representations do not depend on evaluation labels",
        },
    )
