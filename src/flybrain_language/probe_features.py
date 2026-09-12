from __future__ import annotations

import numpy as np
import torch

from .codebook import Codebooks
from .config import ExperimentConfig
from .diagnostic_tasks import DiagnosticTask
from .graph import Connectome
from .metrics import local_populations
from .simulator import FlyLIFSimulator


def internal_indices(graph: Connectome, books: Codebooks) -> torch.Tensor:
    lookup = {int(neuron): index for index, neuron in enumerate(graph.flywire_ids)}
    interface_ids = {
        neuron
        for mapping in (books.input_ids, books.output_ids)
        for population in mapping.values()
        for neuron in population
    }
    return torch.tensor(
        [index for neuron, index in lookup.items() if neuron not in interface_ids],
        dtype=torch.long,
    )


def graph_features(
    graph: Connectome,
    books: Codebooks,
    task: DiagnosticTask,
    config: ExperimentConfig,
    seed: int,
    disconnected: bool = False,
) -> tuple[np.ndarray, dict]:
    """Measure post-window state while keeping the anatomical model frozen."""
    variant = graph
    if disconnected:
        variant = Connectome(
            graph.flywire_ids.copy(), graph.pre.copy(), graph.post.copy(),
            np.zeros_like(graph.weights), graph.plastic.copy(),
        )
    simulator = FlyLIFSimulator(variant, config.dynamics, config.learning, seed)
    inputs, outputs = local_populations(variant, books)
    internal = internal_indices(variant, books)
    rows = []
    for sequence in task.sequences:
        simulator.reset()
        result = None
        for token in sequence:
            result = simulator.run_word(
                inputs[token], outputs,
                presentation_ms=config.presentation_ms,
                gap_ms=config.gap_ms,
                prediction_ms=config.prediction_ms,
            )
        assert result is not None
        state = result["representation"][internal] - config.dynamics.v_rest_mv
        state /= torch.sqrt(torch.mean(state.square()) + 1e-6)
        rows.append(state.numpy())
    simulated_seconds = simulator.steps * config.dynamics.dt_ms / 1000.0
    telemetry = {
        "internal_neurons": len(internal),
        "active_neuron_fraction": float((simulator.activity_counts > 0).float().mean()),
        "mean_firing_rate_hz": simulator.total_spikes / (
            simulator.neuron_count * simulated_seconds
        ),
        "disconnected": disconnected,
    }
    return np.asarray(rows, dtype=np.float32), telemetry
