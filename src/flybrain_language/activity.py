from __future__ import annotations

import json
import platform
import resource
import time
from pathlib import Path

import numpy as np

from .config import load_config
from .experiment import load_assets
from .metrics import local_populations
from .simulator import FlyLIFSimulator


def run_activity_check(config_path: Path, project_root: Path, output: Path) -> dict:
    config = load_config(config_path)
    _, graph, books, vocabulary, _ = load_assets(project_root, config)
    inputs, outputs = local_populations(graph, books)
    representations, active, firing_rates, elapsed = [], [], [], []
    for token in range(len(vocabulary.tokens)):
        simulator = FlyLIFSimulator(graph, config.dynamics, config.learning, seed=12345)
        started = time.perf_counter()
        result = simulator.run_word(
            inputs[token], outputs,
            presentation_ms=config.presentation_ms, gap_ms=config.gap_ms,
            prediction_ms=config.prediction_ms,
        )
        elapsed.append(time.perf_counter() - started)
        representations.append(result["representation"].numpy() - config.dynamics.v_rest_mv)
        active.append(int((simulator.activity_counts > 0).sum()))
        seconds = simulator.steps * config.dynamics.dt_ms / 1000
        firing_rates.append(float(simulator.total_spikes) / (simulator.neuron_count * seconds))
    representations = np.asarray(representations)
    distances = np.sqrt(((representations[:, None] - representations[None, :]) ** 2).sum(axis=2))
    off_diagonal = distances[np.triu_indices(len(representations), 1)]
    graph_bytes = sum(array.nbytes for array in (
        graph.flywire_ids, graph.pre, graph.post, graph.weights, graph.plastic
    ))
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    report = {
        "tokens": list(vocabulary.tokens), "neurons": len(graph.flywire_ids),
        "edges": len(graph.pre), "plastic_edges": int(graph.plastic.sum()),
        "active_neurons_per_token": active,
        "mean_firing_rate_hz_per_token": firing_rates,
        "minimum_pairwise_state_distance": float(off_diagonal.min()),
        "all_token_states_distinct": bool((off_diagonal > 0).all()),
        "mean_seconds_per_word": float(np.mean(elapsed)),
        "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "graph_array_mb": graph_bytes / 1024**2,
        "cpu": cpu_model, "logical_cpus": __import__("os").cpu_count(),
        "platform": platform.platform(),
        "cuda_available": bool(__import__("torch").cuda.is_available()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
