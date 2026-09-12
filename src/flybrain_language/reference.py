from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .config import DynamicsConfig, LearningConfig
from .graph import Connectome
from .simulator import FlyLIFSimulator


def _numpy_trace(config: DynamicsConfig, stimuli: np.ndarray, steps: int) -> np.ndarray:
    voltage = np.full(2, config.v_rest_mv)
    conductance = np.zeros(2)
    spikes = np.zeros(2)
    refractory = np.full(2, 10_000.0)
    delay_steps = round(config.delay_ms / config.dt_ms)
    delay = np.zeros((delay_steps + 1, 2))
    trace = []
    for step in range(steps):
        refractory = np.where(spikes > 0, 0, refractory + 1)
        recurrent = np.array([0.0, spikes[0]])
        available = refractory >= round(config.refractory_ms / config.dt_ms)
        new_conductance = conductance * (1 - config.dt_ms / config.tau_syn_ms) + delay[0] * available
        delay = np.roll(delay, -1, axis=0)
        delay[-1] = config.weight_scale_mv * recurrent
        voltage += stimuli[step]
        voltage += config.dt_ms / config.tau_mem_ms * (conductance - (voltage - config.v_rest_mv))
        spikes = (voltage > config.v_threshold_mv).astype(float)
        voltage = np.where(spikes > 0, config.v_reset_mv, voltage)
        conductance = np.where(spikes > 0, 0, new_conductance)
        trace.append(np.concatenate([voltage, conductance, spikes]))
    return np.asarray(trace)


def run_reference_check(raw_dir: Path, config: DynamicsConfig, output: Path) -> dict:
    source = (raw_dir / "reference" / "model.py").read_text()
    expected_fragments = {
        "v_rest_mv": r"'v_0'\s*:\s*-52\s*\*\s*mV",
        "v_threshold_mv": r"'v_th'\s*:\s*-45\s*\*\s*mV",
        "tau_mem_ms": r"'t_mbr'\s*:\s*20\s*\*\s*ms",
        "tau_syn_ms": r"'tau'\s*:\s*5\s*\*\s*ms",
        "delay_ms": r"'t_dly'\s*:\s*1\.8\s*\*\s*ms",
        "weight_scale_mv": r"'w_syn'\s*:\s*\.275\s*\*\s*mV",
    }
    source_matches = {name: bool(re.search(pattern, source)) for name, pattern in expected_fragments.items()}
    deterministic = replace(config, noise_std_mv=0.0, input_rate_hz=0.0)
    graph = Connectome(
        np.array([1, 2]), np.array([0]), np.array([1]),
        np.array([1.0], dtype=np.float32), np.array([False]),
    )
    simulator = FlyLIFSimulator(graph, deterministic, LearningConfig(), seed=0)
    steps = 60
    stimuli = np.zeros((steps, 2), dtype=np.float32)
    stimuli[0, 0] = 10.0
    observed = []
    for forced in stimuli:
        simulator._step(None, torch.from_numpy(forced))
        state = simulator.state
        observed.append(torch.cat([state.voltage, state.conductance, state.spikes]).numpy())
    expected = _numpy_trace(deterministic, stimuli, steps)
    max_error = float(np.max(np.abs(np.asarray(observed) - expected)))
    report = {
        "source_parameter_matches": source_matches,
        "deterministic_update_max_abs_error": max_error,
        "deterministic_parity": max_error < 1e-5,
        "stochastic_check": "excluded; activity propagation is reported separately",
    }
    if not all(source_matches.values()) or not report["deterministic_parity"]:
        raise AssertionError(f"reference parity failed: {report}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
