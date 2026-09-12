from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flybrain-matplotlib")
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ablation import _load_config
from .experiment import load_assets


def _cosine(rows: np.ndarray) -> np.ndarray:
    rows = rows.astype(np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    normalized = np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)
    return normalized @ normalized.T


def _token_means(representations: np.ndarray, token_ids: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((size, representations.shape[1]), dtype=np.float32)
    counts = np.bincount(token_ids, minlength=size)
    for token in range(size):
        if counts[token]:
            means[token] = representations[token_ids == token].astype(np.float32).mean(axis=0)
    return means, counts


def _kmeans(rows: np.ndarray, usable: np.ndarray, seed: int) -> dict[int, list[int]]:
    chosen = np.flatnonzero(usable)
    if len(chosen) < 2:
        return {0: chosen.tolist()}
    k = min(4, len(chosen))
    rng = np.random.default_rng(seed)
    centers = rows[rng.choice(chosen, k, replace=False)].copy()
    labels = np.zeros(len(chosen), dtype=int)
    for _ in range(30):
        distances = ((rows[chosen, None] - centers[None, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = chosen[labels == cluster]
            if len(members):
                centers[cluster] = rows[members].mean(axis=0)
    return {cluster: chosen[labels == cluster].tolist() for cluster in range(k)}


def analyze(run_dir: Path, project_root: Path) -> dict:
    config = _load_config(run_dir)
    _, graph, books, vocabulary, _ = load_assets(project_root, config)
    with np.load(run_dir / "representations.npz") as raw:
        arrays = {name: raw[name].copy() for name in raw.files}
    inputs, targets = arrays.pop("inputs"), arrays.pop("targets")
    excluded_ids = {
        neuron for mapping in (books.input_ids, books.output_ids)
        for neurons in mapping.values() for neuron in neurons
    }
    internal = ~np.isin(graph.flywire_ids, list(excluded_ids))
    similarities, counts, means = {}, None, {}
    for condition, values in arrays.items():
        centered = values[:, internal].astype(np.float32) - config.dynamics.v_rest_mv
        condition_means, condition_counts = _token_means(centered, inputs, len(vocabulary.tokens))
        means[condition] = condition_means
        similarities[condition] = _cosine(condition_means)
        counts = condition_counts
    input_codes = np.zeros((len(vocabulary.tokens), len(graph.flywire_ids)), dtype=np.float32)
    lookup = {int(value): index for index, value in enumerate(graph.flywire_ids)}
    for token, neurons in books.input_ids.items():
        input_codes[token, [lookup[x] for x in neurons]] = 1
    similarities["input_codebook"] = _cosine(input_codes)
    context = np.zeros((len(vocabulary.tokens), len(vocabulary.tokens)), dtype=np.float32)
    for source, target in zip(inputs, targets):
        context[source, target] += 1
    context_similarity = _cosine(context)
    usable = counts >= 2
    triangle = np.triu_indices(len(vocabulary.tokens), 1)
    pair_mask = usable[triangle[0]] & usable[triangle[1]]
    correlations = {}
    for condition, matrix in similarities.items():
        x, y = matrix[triangle][pair_mask], context_similarity[triangle][pair_mask]
        correlations[condition] = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() and y.std() else None
    nearest = {}
    for token in np.flatnonzero(usable):
        order = np.argsort(-similarities["learned"][token], kind="stable")
        neighbors = [index for index in order if index != token and usable[index]][:5]
        nearest[vocabulary.tokens[token]] = [
            {"token": vocabulary.tokens[index], "cosine": float(similarities["learned"][token, index])}
            for index in neighbors
        ]
    clusters = _kmeans(means["learned"], usable, config.seeds[0])
    king_queen = {"available": False, "reason": "king and queen require at least five held-out contexts"}
    lookup_token = vocabulary.token_to_id
    if "king" in lookup_token and "queen" in lookup_token:
        king, queen = lookup_token["king"], lookup_token["queen"]
        if counts[king] >= 5 and counts[queen] >= 5:
            king_queen = {
                "available": True, "cosine": float(similarities["learned"][king, queen]),
                "king_contexts": int(counts[king]), "queen_contexts": int(counts[queen]),
            }
    report = {
        "contexts_per_token": {vocabulary.tokens[i]: int(counts[i]) for i in range(len(counts))},
        "context_similarity_correlation": correlations,
        "nearest_neighbors": nearest,
        "clusters": {
            str(cluster): [vocabulary.tokens[token] for token in tokens]
            for cluster, tokens in clusters.items()
        },
        "king_queen": king_queen,
        "exclusion": "all fixed input and output population neurons",
    }
    (run_dir / "semantic_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    labels = list(vocabulary.tokens)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, condition in zip(axes, ("learned", "frozen", "input_codebook")):
        image = axis.imshow(similarities[condition], vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_title(condition.replace("_", " "))
        axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        axis.set_yticks(range(len(labels)), labels, fontsize=7)
    figure.colorbar(image, ax=axes, shrink=0.75, label="cosine similarity")
    figure.savefig(run_dir / "semantic_cosine.png", dpi=160)
    plt.close(figure)
    return report
