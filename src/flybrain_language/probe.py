from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import config_dict, load_config
from .diagnostic_tasks import direct_features, make_task
from .experiment import load_assets
from .linear_probe import evaluate_probe, fit_probe
from .probe_features import graph_features, internal_indices


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_probe(
    config_path: Path,
    project_root: Path,
    task_name: str,
    seed: int,
    train_examples: int = 32,
    validation_examples: int = 16,
    test_examples: int = 32,
) -> Path:
    """Run a frozen-connectome diagnostic with matched learned readouts."""
    started = time.perf_counter()
    config = load_config(config_path)
    data_dir, graph, books, vocabulary, _ = load_assets(project_root, config)
    if len(vocabulary.tokens) < 6:
        raise ValueError("diagnostic tasks require at least six vocabulary entries")
    run_dir = project_root / "runs" / f"probe-{config.name}-{task_name}-seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "metrics.json"
    if result_path.exists():
        raise FileExistsError(f"completed diagnostic exists: {result_path}")
    sizes = (train_examples, validation_examples, test_examples)
    names = ("train", "validation", "test")
    tasks = {
        name: make_task(task_name, size, seed + offset)
        for name, size, offset in zip(names, sizes, (101, 202, 303))
    }
    dimension = len(internal_indices(graph, books))
    labels = {
        name: np.asarray(task.labels, dtype=np.int64) for name, task in tasks.items()
    }
    features, telemetry = {}, {}
    for kind, disconnected in (("connectome", False), ("disconnected", True)):
        features[kind], telemetry[kind] = {}, {}
        for split_index, name in enumerate(names):
            x, info = graph_features(
                graph, books, tasks[name], config,
                seed + 10_000 * (split_index + 1), disconnected,
            )
            features[kind][name], telemetry[kind][name] = x, info
    features["direct"] = {
        name: direct_features(task, len(vocabulary.tokens), dimension, seed + 404)
        for name, task in tasks.items()
    }
    results, checkpoint_hashes = {}, {}
    for index, kind in enumerate(("connectome", "direct", "disconnected")):
        state, fitting = fit_probe(
            features[kind]["train"], labels["train"],
            features[kind]["validation"], labels["validation"], seed + 505 + index,
        )
        checkpoint = run_dir / f"{kind}-readout.pt"
        torch.save(state, checkpoint)
        results[kind] = {
            "fitting": fitting,
            "test": evaluate_probe(state, features[kind]["test"], labels["test"]),
        }
        checkpoint_hashes[checkpoint.name] = _sha256(checkpoint)
    report = {
        "schema_version": 1,
        "task": task_name,
        "task_definition": {
            "class_0": list(tasks["train"].sequences[tasks["train"].labels.index(0)]),
            "class_1": list(tasks["train"].sequences[tasks["train"].labels.index(1)]),
            "token_strings": list(vocabulary.tokens),
            "balanced": True,
        },
        "seed": seed,
        "splits": dict(zip(names, sizes)),
        "uniform_chance": 0.5,
        "feature_dimension": dimension,
        "results": results,
        "telemetry": telemetry,
        "elapsed_seconds": time.perf_counter() - started,
        "resolved_config": config_dict(config),
        "artifacts": {
            "connectome_sha256": _sha256(data_dir / "connectome.npz"),
            "codebooks_sha256": _sha256(data_dir / "codebooks.json"),
            "readouts": checkpoint_hashes,
        },
        "claim_boundary": (
            "The readout is a diagnostic trained outside the anatomical model. "
            "Its accuracy is not evidence that internal synaptic plasticity learned the task."
        ),
    }
    result_path.write_text(json.dumps(report, indent=2) + "\n")
    return run_dir
