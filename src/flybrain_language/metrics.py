from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import torch

from .codebook import Codebooks
from .config import ExperimentConfig
from .corpus import iter_contexts
from .graph import Connectome
from .simulator import FlyLIFSimulator


def local_populations(graph: Connectome, books: Codebooks) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    lookup = {int(flywire_id): index for index, flywire_id in enumerate(graph.flywire_ids)}
    inputs = {
        token: torch.tensor([lookup[neuron] for neuron in neurons], dtype=torch.long)
        for token, neurons in books.input_ids.items()
    }
    outputs = torch.tensor([
        [lookup[neuron] for neuron in books.output_ids[token]]
        for token in range(len(books.output_ids))
    ], dtype=torch.long)
    return inputs, outputs


def summarize_records(records: list[dict], vocabulary_size: int) -> dict:
    count = len(records)
    if not count:
        raise ValueError("cannot summarize an empty evaluation")
    correct = np.asarray([row["prediction"] == row["target"] for row in records])
    top5 = np.asarray([row["target"] in row["top5"] for row in records])
    retained = np.asarray([row["target"] >= 3 for row in records])
    losses = [-np.log(max(row["target_probability"], 1e-12)) for row in records]
    per_token = defaultdict(list)
    for row, hit in zip(records, correct):
        per_token[row["target"]].append(bool(hit))
    token_accuracy = {str(k): float(np.mean(v)) for k, v in sorted(per_token.items())}
    return {
        "pairs": count,
        "top1_accuracy": float(correct.mean()),
        "top5_accuracy": float(top5.mean()),
        "cross_entropy": float(np.mean(losses)),
        "perplexity": float(np.exp(np.mean(losses))),
        "perplexity_note": "activity probabilities use additive smoothing 0.25",
        "retained_word_accuracy": float(correct[retained].mean()) if retained.any() else None,
        "macro_accuracy": float(np.mean(list(token_accuracy.values()))),
        "unk_target_fraction": float(np.mean([row["target"] == 0 for row in records])),
        "silent_fraction": float(np.mean([row["silent"] for row in records])),
        "tie_fraction": float(np.mean([row["tied"] for row in records])),
        "per_token_accuracy": token_accuracy,
        "uniform_top1_chance": 1.0 / (vocabulary_size - 1),
        "uniform_top5_chance": min(5, vocabulary_size - 1) / (vocabulary_size - 1),
    }


def run_sequence(
    simulator: FlyLIFSimulator,
    graph: Connectome,
    books: Codebooks,
    ids: list[int],
    config: ExperimentConfig,
    pair_limit: int,
    learn: bool,
    collect_representations: bool = False,
) -> tuple[dict, list[dict], np.ndarray | None]:
    inputs, outputs = local_populations(graph, books)
    before_spikes = simulator.activity_counts.clone()
    before_steps = simulator.steps
    records, representations = [], []
    for context_index, context in enumerate(iter_contexts(ids, config.context_length, pair_limit)):
        simulator.reset()
        for input_token, target in context:
            result = simulator.run_word(
                inputs[input_token], outputs, target=target, learn=learn,
                presentation_ms=config.presentation_ms, gap_ms=config.gap_ms,
                prediction_ms=config.prediction_ms,
            )
            top5 = torch.argsort(result["scores"], descending=True, stable=True)[:5].tolist()
            records.append({
                "context": context_index, "input": input_token, "target": target,
                "prediction": result["prediction"], "top5": top5,
                "target_probability": float(result["probabilities"][target]),
                "silent": result["silent"], "tied": result["tied"],
            })
            if collect_representations:
                representations.append(result["representation"].numpy())
    metrics = summarize_records(records, len(books.output_ids))
    spike_delta = simulator.activity_counts - before_spikes
    step_delta = simulator.steps - before_steps
    seconds = step_delta * config.dynamics.dt_ms / 1000.0
    metrics.update({
        "mean_firing_rate_hz": float(spike_delta.sum()) / (len(spike_delta) * seconds),
        "active_neuron_fraction": float((spike_delta > 0).float().mean()),
        "simulation_steps": step_delta,
    })
    reps = np.asarray(representations, dtype=np.float16) if representations else None
    return metrics, records, reps


def frequency_baseline(training_ids: list[int], evaluation_ids: list[int], config: ExperimentConfig) -> dict:
    targets = [pair[1] for context in iter_contexts(training_ids, config.context_length) for pair in context]
    prediction = Counter(targets).most_common(1)[0][0]
    vocabulary_size = config.lexical_tokens + 3
    probability = np.full(vocabulary_size, 0.25)
    probability[prediction] += len(targets)
    probability /= probability.sum()
    records = []
    ranking = np.argsort(-probability, kind="stable")[:5].tolist()
    for context_index, context in enumerate(iter_contexts(
        evaluation_ids, config.context_length, config.test_pairs
    )):
        for input_token, target in context:
            records.append({
                "context": context_index, "input": input_token, "target": target,
                "prediction": prediction, "top5": ranking,
                "target_probability": float(probability[target]),
                "silent": False, "tied": False,
            })
    result = summarize_records(records, vocabulary_size)
    result["predicted_token_id"] = prediction
    return result
