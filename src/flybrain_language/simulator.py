from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .config import DynamicsConfig, LearningConfig
from .graph import Connectome


@dataclass
class SimulationState:
    conductance: torch.Tensor
    delay: torch.Tensor
    spikes: torch.Tensor
    voltage: torch.Tensor
    refractory: torch.Tensor
    pre_trace: torch.Tensor
    post_trace: torch.Tensor
    eligibility: torch.Tensor


class FlyLIFSimulator:
    """Edge-list implementation of the Eon/Brian2 alpha-synapse LIF ordering."""

    def __init__(
        self,
        graph: Connectome,
        dynamics: DynamicsConfig,
        learning: LearningConfig,
        seed: int,
    ):
        self.graph = graph
        self.dynamics = dynamics
        self.learning = learning
        raw_pre = torch.from_numpy(graph.pre).long()
        raw_post = torch.from_numpy(graph.post).long()
        order = torch.argsort(raw_post * len(graph.flywire_ids) + raw_pre, stable=True)
        self.pre = raw_pre[order]
        self.post = raw_post[order]
        self.weights = torch.from_numpy(graph.weights).float()[order].clone()
        self.original_weights = self.weights.clone()
        plastic = torch.from_numpy(graph.plastic)[order]
        self.plastic_indices = plastic.nonzero().flatten()
        self.plastic_pre = self.pre[self.plastic_indices]
        self.plastic_post = self.post[self.plastic_indices]
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.reward_baseline = 0.0
        self.rewards_seen = 0
        self.total_spikes = 0
        self.steps = 0
        self.activity_counts = torch.zeros(self.neuron_count, dtype=torch.long)
        counts = torch.bincount(self.post, minlength=self.neuron_count)
        crow = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        self.sparse_weights = torch.sparse_csr_tensor(
            crow, self.pre, self.weights, size=(self.neuron_count, self.neuron_count)
        )
        self.reset()

    @property
    def neuron_count(self) -> int:
        return len(self.graph.flywire_ids)

    def reset(self) -> None:
        n = self.neuron_count
        delay_steps = round(self.dynamics.delay_ms / self.dynamics.dt_ms)
        self.state = SimulationState(
            torch.zeros(n), torch.zeros(delay_steps + 1, n), torch.zeros(n),
            torch.full((n,), self.dynamics.v_rest_mv), torch.full((n,), 10_000.0),
            torch.zeros(n), torch.zeros(n), torch.zeros(len(self.plastic_indices)),
        )

    def _step(
        self, stimulated: torch.Tensor | None, forced_voltage: torch.Tensor | None = None
    ) -> torch.Tensor:
        cfg, state = self.dynamics, self.state
        refractory_steps = round(cfg.refractory_ms / cfg.dt_ms)
        state.refractory = torch.where(
            state.spikes.bool(), torch.zeros_like(state.refractory), state.refractory + 1
        )
        recurrent = torch.mv(self.sparse_weights, state.spikes)
        available = (state.refractory >= refractory_steps).float()
        new_conductance = (
            state.conductance * (1 - cfg.dt_ms / cfg.tau_syn_ms)
            + state.delay[0] * available
        )
        state.delay = torch.roll(state.delay, -1, dims=0)
        state.delay[-1] = cfg.weight_scale_mv * recurrent
        voltage_stim = torch.zeros(self.neuron_count) if forced_voltage is None else forced_voltage.clone()
        if stimulated is not None and len(stimulated):
            draws = torch.rand(len(stimulated), generator=self.generator)
            fired = draws < cfg.input_rate_hz * cfg.dt_ms / 1000.0
            voltage_stim[stimulated] = fired.float() * cfg.poisson_scale * cfg.weight_scale_mv
        noise = torch.randn(self.neuron_count, generator=self.generator) * cfg.noise_std_mv
        state.voltage = state.voltage + voltage_stim + noise
        state.voltage = state.voltage + cfg.dt_ms / cfg.tau_mem_ms * (
            state.conductance - (state.voltage - cfg.v_rest_mv)
        )
        state.spikes = (state.voltage > cfg.v_threshold_mv).float()
        state.voltage = torch.where(
            state.spikes.bool(), torch.full_like(state.voltage, cfg.v_reset_mv), state.voltage
        )
        new_conductance = torch.where(state.spikes.bool(), 0.0, new_conductance)
        state.conductance = new_conductance
        self._update_eligibility()
        count = int(state.spikes.sum())
        self.total_spikes += count
        self.activity_counts.add_(state.spikes.long())
        self.steps += 1
        return state.spikes

    def _update_eligibility(self) -> None:
        cfg, state = self.dynamics, self.state
        pre_decay = torch.exp(torch.tensor(-cfg.dt_ms / self.learning.tau_pre_ms))
        post_decay = torch.exp(torch.tensor(-cfg.dt_ms / self.learning.tau_post_ms))
        eligibility_decay = torch.exp(torch.tensor(
            -cfg.dt_ms / self.learning.tau_eligibility_ms
        ))
        state.pre_trace.mul_(pre_decay).add_(state.spikes)
        state.post_trace.mul_(post_decay).add_(state.spikes)
        coincidence = (
            state.pre_trace[self.plastic_pre] * state.spikes[self.plastic_post]
            - 0.5 * state.post_trace[self.plastic_post] * state.spikes[self.plastic_pre]
        )
        state.eligibility.mul_(eligibility_decay).add_(coincidence)

    def apply_reward(self, correct: bool) -> float:
        cfg = self.learning
        reward = cfg.reward_correct if correct else cfg.reward_incorrect
        error = reward - self.reward_baseline
        indices = self.plastic_indices
        original = self.original_weights[indices]
        signs = original.sign()
        magnitude = self.weights[indices].abs()
        magnitude += cfg.rate * error * self.state.eligibility
        magnitude += cfg.regularization * (original.abs() - magnitude)
        magnitude.clamp_(original.abs() * cfg.min_fraction, original.abs() * cfg.max_fraction)
        self.weights[indices] = signs * magnitude
        self.reward_baseline = (
            cfg.baseline_decay * self.reward_baseline + (1 - cfg.baseline_decay) * reward
        )
        self.rewards_seen += 1
        return error

    def run_word(
        self,
        input_indices: torch.Tensor,
        output_indices: torch.Tensor,
        target: int | None = None,
        learn: bool = False,
        presentation_ms: float = 30.0,
        gap_ms: float = 5.0,
        prediction_ms: float = 75.0,
    ) -> dict:
        cfg = self.dynamics
        presentation_steps = round(presentation_ms / cfg.dt_ms)
        gap_steps = round(gap_ms / cfg.dt_ms)
        prediction_steps = round(prediction_ms / cfg.dt_ms)
        for _ in range(presentation_steps):
            self._step(input_indices)
        for _ in range(gap_steps):
            self._step(None)
        counts = torch.zeros(output_indices.shape[0])
        for _ in range(prediction_steps):
            spikes = self._step(None)
            counts += spikes[output_indices].float().mean(dim=1)
        prediction = int(torch.argmax(counts))
        maximum = counts.max()
        tied = int((counts == maximum).sum()) > 1
        silent = float(maximum) == 0.0
        probs = (counts + 0.25) / (counts.sum() + 0.25 * len(counts))
        reward_error = None
        if learn and target is not None:
            reward_error = self.apply_reward(prediction == target)
        return {
            "prediction": prediction, "scores": counts,
            "probabilities": probs, "tied": tied, "silent": silent,
            "reward_prediction_error": reward_error,
            "representation": self.state.voltage.clone(),
        }

    def checkpoint(self, path: Path, metadata: dict | None = None) -> None:
        torch.save({
            "weights": self.weights, "original_weights": self.original_weights,
            "state": self.state.__dict__, "generator_state": self.generator.get_state(),
            "reward_baseline": self.reward_baseline, "rewards_seen": self.rewards_seen,
            "total_spikes": self.total_spikes, "steps": self.steps,
            "activity_counts": self.activity_counts,
            "metadata": metadata or {},
        }, path)

    def restore(self, path: Path) -> dict:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        self.weights.copy_(raw["weights"])
        self.original_weights.copy_(raw["original_weights"])
        self.state = SimulationState(**raw["state"])
        self.generator.set_state(raw["generator_state"])
        self.reward_baseline = raw["reward_baseline"]
        self.rewards_seen = raw["rewards_seen"]
        self.total_spikes = raw["total_spikes"]
        self.steps = raw["steps"]
        self.activity_counts.copy_(raw["activity_counts"])
        return raw["metadata"]
