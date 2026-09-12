from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DynamicsConfig:
    dt_ms: float = 0.1
    tau_syn_ms: float = 5.0
    delay_ms: float = 1.8
    v_rest_mv: float = -52.0
    v_reset_mv: float = -52.0
    v_threshold_mv: float = -45.0
    tau_mem_ms: float = 20.0
    refractory_ms: float = 2.2
    weight_scale_mv: float = 0.275
    poisson_scale: float = 250.0
    input_rate_hz: float = 150.0
    noise_std_mv: float = 0.025


@dataclass
class LearningConfig:
    rate: float = 0.0005
    tau_pre_ms: float = 20.0
    tau_post_ms: float = 20.0
    tau_eligibility_ms: float = 500.0
    baseline_decay: float = 0.98
    regularization: float = 0.0001
    min_fraction: float = 0.05
    max_fraction: float = 3.0
    reward_correct: float = 1.0
    reward_incorrect: float = -0.25


@dataclass
class ExperimentConfig:
    name: str = "tiny"
    target_neurons: int | None = 2000
    lexical_tokens: int = 8
    input_cells: int = 64
    output_cells: int = 16
    max_hops: int = 3
    context_length: int = 32
    presentation_ms: float = 30.0
    gap_ms: float = 5.0
    prediction_ms: float = 75.0
    train_pairs: int = 384
    validation_pairs: int = 96
    test_pairs: int = 192
    seeds: list[int] = field(default_factory=lambda: [11, 23, 37])
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)


def _merge_dataclass(cls: type, values: dict[str, Any] | None):
    return cls(**(values or {}))


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    raw["dynamics"] = _merge_dataclass(DynamicsConfig, raw.get("dynamics"))
    raw["learning"] = _merge_dataclass(LearningConfig, raw.get("learning"))
    cfg = ExperimentConfig(**raw)
    if cfg.lexical_tokens < 1 or cfg.input_cells < 1 or cfg.output_cells < 1:
        raise ValueError("vocabulary and population sizes must be positive")
    if cfg.dynamics.delay_ms < cfg.dynamics.dt_ms:
        raise ValueError("synaptic delay must be at least one timestep")
    return cfg


def config_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)
