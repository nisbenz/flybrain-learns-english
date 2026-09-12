from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import config_dict, load_config
from .diagnostic_tasks import DiagnosticTask, make_task
from .experiment import load_assets
from .metrics import local_populations
from .probe import _sha256
from .simulator import FlyLIFSimulator


def _evaluate(simulator, inputs, outputs, task, config, learn=False) -> dict:
    rows = []
    for sequence, target in zip(task.sequences, task.labels):
        simulator.reset()
        result = None
        for index, token in enumerate(sequence):
            result = simulator.run_word(
                inputs[token], outputs,
                target=target if index == len(sequence) - 1 else None,
                learn=learn,
                presentation_ms=config.presentation_ms,
                gap_ms=config.gap_ms,
                prediction_ms=config.prediction_ms,
            )
        assert result is not None
        rows.append({
            "target": target, "prediction": result["prediction"],
            "silent": result["silent"], "tied": result["tied"],
        })
    hits = np.asarray([row["prediction"] == row["target"] for row in rows])
    labels = np.asarray(task.labels)
    return {
        "accuracy": float(hits.mean()),
        "per_class_accuracy": {
            str(label): float(hits[labels == label].mean()) for label in (0, 1)
        },
        "silent_fraction": float(np.mean([row["silent"] for row in rows])),
        "tie_fraction": float(np.mean([row["tied"] for row in rows])),
        "predictions": [row["prediction"] for row in rows],
    }


def run_plasticity_probe(
    config_path: Path,
    project_root: Path,
    task_name: str,
    seed: int,
    train_examples: int = 128,
    validation_examples: int = 32,
    test_examples: int = 64,
    noise_std_mv: float | None = None,
) -> Path:
    """Train existing internal synapses on a balanced two-choice task."""
    started = time.perf_counter()
    config = load_config(config_path)
    if noise_std_mv is not None:
        config.dynamics = replace(config.dynamics, noise_std_mv=noise_std_mv)
    data_dir, graph, books, vocabulary, _ = load_assets(project_root, config)
    inputs, all_outputs = local_populations(graph, books)
    outputs = all_outputs[[3, 4]]
    train_task = make_task(task_name, train_examples, seed + 101)
    validation_task = make_task(task_name, validation_examples, seed + 202)
    test_task = make_task(task_name, test_examples, seed + 303)
    simulator = FlyLIFSimulator(graph, config.dynamics, config.learning, seed)
    interval = min(32, train_examples)
    best_accuracy, best_weights, best_examples = -1.0, None, 0
    stages = []
    for start in range(0, train_examples, interval):
        stop = min(start + interval, train_examples)
        batch = DiagnosticTask(
            task_name, train_task.sequences[start:stop], train_task.labels[start:stop]
        )
        training = _evaluate(simulator, inputs, outputs, batch, config, learn=True)
        validation_simulator = FlyLIFSimulator(
            graph, config.dynamics, config.learning, seed + 500_000
        )
        validation_simulator.weights.copy_(simulator.weights)
        validation = _evaluate(
            validation_simulator, inputs, outputs, validation_task, config
        )
        stages.append({"examples_seen": stop, "training": training, "validation": validation})
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_examples = validation["accuracy"], stop
            best_weights = simulator.weights.clone()
    assert best_weights is not None
    results = {}
    for kind, weights in (("learned", best_weights), ("frozen", simulator.original_weights)):
        evaluation = FlyLIFSimulator(graph, config.dynamics, config.learning, seed + 900_000)
        evaluation.weights.copy_(weights)
        results[kind] = _evaluate(evaluation, inputs, outputs, test_task, config)
    run_name = f"plasticity-{config.name}-{task_name}-seed{seed}-noise{config.dynamics.noise_std_mv:g}"
    run_dir = project_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best.pt"
    torch.save({"weights": best_weights, "original_weights": simulator.original_weights}, checkpoint)
    changed = best_weights[simulator.plastic_indices] - simulator.original_weights[simulator.plastic_indices]
    report = {
        "schema_version": 1, "task": task_name, "seed": seed,
        "balanced": True, "chance": 0.5,
        "tokens": {"cue_0": vocabulary.tokens[3], "cue_1": vocabulary.tokens[4],
                   "shared": vocabulary.tokens[5], "output_token_ids": [3, 4]},
        "splits": {"train": train_examples, "validation": validation_examples,
                   "test": test_examples},
        "best_examples_seen": best_examples, "best_validation_accuracy": best_accuracy,
        "stages": stages, "results": results,
        "changed_plastic_fraction": float((changed != 0).float().mean()),
        "plastic_mean_absolute_change": float(changed.abs().mean()),
        "elapsed_seconds": time.perf_counter() - started,
        "resolved_config": config_dict(config),
        "artifacts": {"connectome_sha256": _sha256(data_dir / "connectome.npz"),
                      "codebooks_sha256": _sha256(data_dir / "codebooks.json"),
                      "checkpoint_sha256": _sha256(checkpoint)},
    }
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return run_dir
