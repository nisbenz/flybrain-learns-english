from __future__ import annotations

import numpy as np

from .graph import Connectome


def shuffled_connectome(graph: Connectome, seed: int) -> Connectome:
    """Permute destinations while retaining exact in/out degree sequences."""
    rng = np.random.default_rng(seed)
    post = rng.permutation(graph.post)
    magnitudes = rng.permutation(np.abs(graph.weights))
    source_sign = np.ones(len(graph.flywire_ids), dtype=np.float32)
    order = np.argsort(graph.pre, kind="stable")
    source_sign[graph.pre[order]] = np.sign(graph.weights[order])
    weights = source_sign[graph.pre] * magnitudes
    plastic = graph.plastic.copy()
    return Connectome(graph.flywire_ids.copy(), graph.pre.copy(), post, weights, plastic)


def random_connectome(graph: Connectome, seed: int) -> Connectome:
    """Draw a size/density-matched directed graph with source-consistent signs."""
    rng = np.random.default_rng(seed)
    edge_count, neuron_count = len(graph.pre), len(graph.flywire_ids)
    pre = rng.integers(0, neuron_count, edge_count, dtype=np.int64)
    post = rng.integers(0, neuron_count, edge_count, dtype=np.int64)
    source_sign = np.ones(neuron_count, dtype=np.float32)
    order = np.argsort(graph.pre, kind="stable")
    source_sign[graph.pre[order]] = np.sign(graph.weights[order])
    weights = source_sign[pre] * rng.permutation(np.abs(graph.weights))
    plastic_sources = np.unique(graph.pre[graph.plastic])
    plastic = np.isin(pre, plastic_sources)
    return Connectome(graph.flywire_ids.copy(), pre, post, weights, plastic)
