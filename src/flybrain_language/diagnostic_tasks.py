from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiagnosticTask:
    name: str
    sequences: tuple[tuple[int, ...], ...]
    labels: tuple[int, ...]


def make_task(name: str, examples: int, seed: int) -> DiagnosticTask:
    """Create a deterministic balanced task over three fixed token IDs."""
    if examples < 2 or examples % 2:
        raise ValueError("diagnostic examples must be a positive even number")
    if name == "association":
        templates = ((3,), (4,))
    elif name == "delayed_context":
        templates = ((3, 5), (4, 5))
    else:
        raise ValueError(f"unknown diagnostic task: {name}")
    labels = np.tile(np.arange(2, dtype=np.int64), examples // 2)
    order = np.random.default_rng(seed).permutation(examples)
    return DiagnosticTask(
        name=name,
        sequences=tuple(templates[int(labels[index])] for index in order),
        labels=tuple(int(labels[index]) for index in order),
    )


def direct_features(
    task: DiagnosticTask,
    vocabulary_size: int,
    dimensions: int,
    interface_seed: int,
    state_gain: float = 0.6,
    input_gain: float = 0.4,
) -> np.ndarray:
    """FLM-style matched bypass: fixed seeded codes and leaky state, no graph."""
    rng = np.random.default_rng(interface_seed)
    codes = rng.standard_normal((vocabulary_size, dimensions)).astype(np.float32)
    rows = []
    for sequence in task.sequences:
        state = np.zeros(dimensions, dtype=np.float32)
        for token in sequence:
            state = state_gain * state + input_gain * codes[token]
        rows.append(state / np.sqrt(np.mean(state * state) + 1e-6))
    return np.asarray(rows)
