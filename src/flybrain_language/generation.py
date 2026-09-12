from __future__ import annotations

import json
from pathlib import Path

import torch

from .ablation import _load_config
from .experiment import load_assets
from .metrics import local_populations
from .simulator import FlyLIFSimulator


def generate(run_dir: Path, project_root: Path, length: int, start: str = "<BOS>") -> list[str]:
    config = _load_config(run_dir)
    _, graph, books, vocabulary, _ = load_assets(project_root, config)
    try:
        token = vocabulary.token_to_id[start.lower() if not start.startswith("<") else start]
    except KeyError as error:
        raise ValueError(f"start token is outside the fixed vocabulary: {start}") from error
    raw = torch.load(run_dir / "checkpoints" / "best.pt", weights_only=True)
    simulator = FlyLIFSimulator(
        graph, config.dynamics, config.learning, int(raw["metadata"]["seed"]) + 910_000
    )
    simulator.weights.copy_(raw["weights"])
    inputs, outputs = local_populations(graph, books)
    simulator.reset()
    generated = [vocabulary.tokens[token]]
    for _ in range(length):
        result = simulator.run_word(
            inputs[token], outputs,
            presentation_ms=config.presentation_ms, gap_ms=config.gap_ms,
            prediction_ms=config.prediction_ms,
        )
        token = result["prediction"]
        generated.append(vocabulary.tokens[token])
    return generated
