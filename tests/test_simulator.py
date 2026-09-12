from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from flybrain_language.metrics import run_sequence
from flybrain_language.simulator import FlyLIFSimulator


def test_silent_spike_count_tie_uses_lowest_token(small_graph, fast_config):
    simulator = FlyLIFSimulator(small_graph, fast_config.dynamics, fast_config.learning, seed=1)
    result = simulator.run_word(
        torch.tensor([0]), torch.tensor([[1], [2], [3]]),
        presentation_ms=1, gap_ms=0, prediction_ms=1,
    )
    assert result["prediction"] == 0
    assert result["silent"] is True
    assert result["tied"] is True


def test_reward_uses_previous_baseline_and_preserves_sign_bounds(small_graph, fast_config):
    simulator = FlyLIFSimulator(small_graph, fast_config.dynamics, fast_config.learning, seed=2)
    simulator.state.eligibility.fill_(100)
    first_error = simulator.apply_reward(True)
    assert first_error == 1.0
    assert abs(simulator.reward_baseline - 0.02) < 1e-12
    assert simulator.weights[0] == 3.0
    assert simulator.weights[1] == -1.5
    assert torch.equal(simulator.weights[2:], simulator.original_weights[2:])
    second_error = simulator.apply_reward(True)
    assert abs(second_error - 0.98) < 1e-8
    assert torch.equal(simulator.weights.sign(), simulator.original_weights.sign())


def test_external_stimulation_is_restricted(small_graph, fast_config):
    dynamics = replace(fast_config.dynamics, input_rate_hz=10_000)
    simulator = FlyLIFSimulator(small_graph, dynamics, fast_config.learning, seed=3)
    simulator.run_word(
        torch.tensor([0]), torch.tensor([[1], [2], [3]]),
        presentation_ms=1, gap_ms=0, prediction_ms=0,
    )
    assert simulator.activity_counts[0] == 1
    assert simulator.activity_counts[1:].sum() == 0


def test_state_continues_until_explicit_context_reset(small_graph, fast_config):
    simulator = FlyLIFSimulator(small_graph, fast_config.dynamics, fast_config.learning, seed=4)
    initial = simulator.state.voltage.clone()
    simulator._step(None, torch.tensor([1.0, 0.0, 0.0, 0.0]))
    after_first = simulator.state.voltage.clone()
    simulator._step(None, torch.zeros(4))
    assert not torch.equal(after_first, initial)
    assert not torch.equal(simulator.state.voltage, initial)
    simulator.reset()
    assert torch.equal(simulator.state.voltage, initial)


def test_checkpoint_continuation_is_exact(tmp_path: Path, small_graph, fast_config):
    dynamics = replace(fast_config.dynamics, noise_std_mv=0.1, input_rate_hz=500)
    first = FlyLIFSimulator(small_graph, dynamics, fast_config.learning, seed=5)
    populations = torch.tensor([[1], [2], [3]])
    first.run_word(torch.tensor([0]), populations, target=1, learn=True,
                   presentation_ms=2, gap_ms=1, prediction_ms=2)
    checkpoint = tmp_path / "state.pt"
    first.checkpoint(checkpoint, {"position": 1})
    expected = first.run_word(torch.tensor([0]), populations, target=2, learn=True,
                              presentation_ms=2, gap_ms=1, prediction_ms=2)
    second = FlyLIFSimulator(small_graph, dynamics, fast_config.learning, seed=999)
    assert second.restore(checkpoint) == {"position": 1}
    observed = second.run_word(torch.tensor([0]), populations, target=2, learn=True,
                               presentation_ms=2, gap_ms=1, prediction_ms=2)
    assert expected["prediction"] == observed["prediction"]
    assert torch.equal(expected["scores"], observed["scores"])
    assert torch.equal(first.weights, second.weights)
    assert torch.equal(first.state.voltage, second.state.voltage)


def test_frozen_evaluation_and_determinism(small_graph, small_books, fast_config):
    first = FlyLIFSimulator(small_graph, fast_config.dynamics, fast_config.learning, seed=6)
    second = FlyLIFSimulator(small_graph, fast_config.dynamics, fast_config.learning, seed=6)
    original = first.weights.clone()
    one, records_one, _ = run_sequence(
        first, small_graph, small_books, [1, 0, 2], fast_config, 2, False
    )
    two, records_two, _ = run_sequence(
        second, small_graph, small_books, [1, 0, 2], fast_config, 2, False
    )
    assert torch.equal(first.weights, original)
    assert records_one == records_two
    assert one == two


def test_connectome_round_trip(tmp_path: Path, small_graph):
    path = tmp_path / "graph.npz"
    small_graph.save(path)
    restored = type(small_graph).load(path)
    for field in ("flywire_ids", "pre", "post", "weights", "plastic"):
        assert np.array_equal(getattr(restored, field), getattr(small_graph, field))
