from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .config import DynamicsConfig, ExperimentConfig, LearningConfig
from .controls import random_connectome, shuffled_connectome
from .experiment import load_assets
from .metrics import run_sequence
from .simulator import FlyLIFSimulator


def _load_config(run_dir: Path) -> ExperimentConfig:
    raw = json.loads((run_dir / "config.json").read_text())
    dynamics, learning = raw.pop("dynamics"), raw.pop("learning")
    return ExperimentConfig(
        **raw, dynamics=DynamicsConfig(**dynamics), learning=LearningConfig(**learning)
    )


def run_ablations(run_dir: Path, project_root: Path, pair_limit: int | None = None) -> dict:
    config = _load_config(run_dir)
    _, graph, books, _, splits = load_assets(project_root, config)
    checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", weights_only=True)
    seed = int(checkpoint["metadata"]["seed"])
    limit = min(pair_limit or config.train_pairs, config.train_pairs)
    results = {}
    for name, variant in (
        ("random_graph", random_connectome(graph, seed + 71)),
        ("shuffled_edges", shuffled_connectome(graph, seed + 79)),
    ):
        started = time.perf_counter()
        training = FlyLIFSimulator(variant, config.dynamics, config.learning, seed)
        interval = min(96, max(1, limit // 4))
        best_score, best_weights, best_pairs, train_metrics = -1.0, None, 0, None
        for start in range(0, limit, interval):
            length = min(interval, limit - start)
            train_metrics, _, _ = run_sequence(
                training, variant, books,
                splits["train"][start : start + length + 1], config, length, True,
            )
            validation = FlyLIFSimulator(
                variant, config.dynamics, config.learning, seed + 500_000
            )
            validation.weights.copy_(training.weights)
            validation_metrics, _, _ = run_sequence(
                validation, variant, books, splits["validation"], config,
                config.validation_pairs, False,
            )
            if validation_metrics["top1_accuracy"] > best_score:
                best_score = validation_metrics["top1_accuracy"]
                best_weights = training.weights.clone()
                best_pairs = start + length
        evaluation = FlyLIFSimulator(
            variant, config.dynamics, config.learning, seed + 900_000
        )
        evaluation.weights.copy_(best_weights)
        test_metrics, records, _ = run_sequence(
            evaluation, variant, books, splits["test"], config,
            config.test_pairs, False,
        )
        results[name] = {
            "training": train_metrics, "test": test_metrics,
            "train_pairs": limit, "best_pairs_seen": best_pairs,
            "best_validation_accuracy": best_score,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (run_dir / f"{name}_predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records)
        )
    (run_dir / "ablation_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    return results
