from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .codebook import load_codebooks
from .config import ExperimentConfig, config_dict, load_config
from .corpus import load_corpus
from .graph import Connectome
from .metrics import frequency_baseline, run_sequence, summarize_records
from .simulator import FlyLIFSimulator


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _records(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


def load_assets(project_root: Path, config: ExperimentConfig):
    data_dir = project_root / "data" / "processed" / config.name
    graph = Connectome.load(data_dir / "connectome.npz")
    books = load_codebooks(data_dir / "codebooks.json")
    vocabulary, splits = load_corpus(data_dir)
    return data_dir, graph, books, vocabulary, splits


def train(config_path: Path, project_root: Path, seed: int, pair_limit: int | None = None) -> Path:
    config = load_config(config_path)
    data_dir, graph, books, vocabulary, splits = load_assets(project_root, config)
    run_dir = project_root / "runs" / f"{config.name}-seed{seed}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _json(run_dir / "config.json", config_dict(config))
    simulator = FlyLIFSimulator(graph, config.dynamics, config.learning, seed)
    total = min(pair_limit or config.train_pairs, len(splits["train"]) - 1)
    interval = min(96, max(1, total // 4))
    stages, training_records, best_score, best_path = [], [], -1.0, None
    started = time.perf_counter()
    for start in range(0, total, interval):
        length = min(interval, total - start)
        ids = splits["train"][start : start + length + 1]
        train_metrics, records, _ = run_sequence(
            simulator, graph, books, ids, config, length, True
        )
        training_records.extend(records)
        candidate = checkpoint_dir / f"pairs-{start + length}.pt"
        simulator.checkpoint(candidate, {
            "pairs_seen": start + length, "seed": seed,
            "config": config_dict(config), "data_manifest": str(data_dir / "manifest.json"),
        })
        validation_simulator = FlyLIFSimulator(graph, config.dynamics, config.learning, seed + 500_000)
        validation_simulator.weights.copy_(simulator.weights)
        validation_metrics, _, _ = run_sequence(
            validation_simulator, graph, books, splits["validation"], config,
            config.validation_pairs, False,
        )
        stage = {
            "pairs_seen": start + length, "training": train_metrics,
            "validation": validation_metrics, "checkpoint": str(candidate),
        }
        stages.append(stage)
        score = validation_metrics["top1_accuracy"]
        if score > best_score:
            best_score, best_path = score, candidate
    if best_path is None:
        raise RuntimeError("training produced no checkpoint")
    best_metadata = simulator.restore(best_path)
    simulator.checkpoint(checkpoint_dir / "best.pt", best_metadata)
    metrics = summarize_records(training_records, len(vocabulary.tokens))
    metrics.update({
        "seed": seed, "pairs": total, "elapsed_seconds": time.perf_counter() - started,
        "best_validation_accuracy": best_score,
        "best_pairs_seen": best_metadata["pairs_seen"], "stages": stages,
        "changed_weight_fraction": float((simulator.weights != simulator.original_weights).float().mean()),
    })
    _json(run_dir / "training_metrics.json", metrics)
    _records(run_dir / "training_predictions.jsonl", training_records)
    return run_dir


def evaluate(run_dir: Path, project_root: Path) -> dict:
    config_raw = json.loads((run_dir / "config.json").read_text())
    dynamics = config_raw.pop("dynamics")
    learning = config_raw.pop("learning")
    config = ExperimentConfig(**config_raw)
    config.dynamics = type(config.dynamics)(**dynamics)
    config.learning = type(config.learning)(**learning)
    _, graph, books, vocabulary, splits = load_assets(project_root, config)
    checkpoint = torch.load(run_dir / "checkpoints" / "best.pt", map_location="cpu", weights_only=True)
    seed = int(checkpoint["metadata"]["seed"])
    results, predictions, reps = {}, {}, {}
    for name in ("learned", "frozen", "restored_original"):
        simulator = FlyLIFSimulator(graph, config.dynamics, config.learning, seed + 900_000)
        if name == "learned":
            simulator.weights.copy_(checkpoint["weights"])
        elif name == "restored_original":
            simulator.weights.copy_(checkpoint["original_weights"])
        metrics, records, representations = run_sequence(
            simulator, graph, books, splits["test"], config, config.test_pairs,
            False, collect_representations=True,
        )
        if name == "learned":
            changed = simulator.weights[simulator.plastic_indices] - simulator.original_weights[simulator.plastic_indices]
            metrics["plastic_weight_mean_absolute_change"] = float(changed.abs().mean())
            original = simulator.original_weights[simulator.plastic_indices].abs()
            current = simulator.weights[simulator.plastic_indices].abs()
            metrics["weight_bound_saturation"] = float((
                (current <= original * config.learning.min_fraction + 1e-7)
                | (current >= original * config.learning.max_fraction - 1e-7)
            ).float().mean())
        results[name], predictions[name], reps[name] = metrics, records, representations
    results["frequency"] = frequency_baseline(splits["train"], splits["test"], config)
    results["vocabulary"] = list(vocabulary.tokens)
    _json(run_dir / "evaluation_metrics.json", results)
    for name, records in predictions.items():
        _records(run_dir / f"{name}_predictions.jsonl", records)
    np.savez_compressed(
        run_dir / "representations.npz", learned=reps["learned"], frozen=reps["frozen"],
        restored_original=reps["restored_original"],
        targets=np.asarray([row["target"] for row in predictions["learned"]]),
        inputs=np.asarray([row["input"] for row in predictions["learned"]]),
    )
    return results
