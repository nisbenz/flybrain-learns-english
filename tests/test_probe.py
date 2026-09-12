import json
import numpy as np

from flybrain_language.codebook import Codebooks
from flybrain_language.diagnostic_tasks import DiagnosticTask, direct_features, make_task
from flybrain_language.linear_probe import evaluate_probe, fit_probe
from flybrain_language.metrics import local_populations
from flybrain_language.plasticity_probe import _evaluate
from flybrain_language.probe_features import internal_indices
from flybrain_language.probe_reporting import summarize_probes
from flybrain_language.simulator import FlyLIFSimulator


def test_diagnostic_tasks_are_balanced_and_deterministic():
    first = make_task("association", 8, 12)
    second = make_task("association", 8, 12)
    assert first == second
    assert first.labels.count(0) == first.labels.count(1) == 4
    delayed = make_task("delayed_context", 8, 12)
    assert {sequence[-1] for sequence in delayed.sequences} == {5}


def test_direct_control_has_matched_dimension():
    task = make_task("delayed_context", 8, 13)
    features = direct_features(task, vocabulary_size=11, dimensions=17, interface_seed=2)
    assert features.shape == (8, 17)
    np.testing.assert_allclose(np.mean(features * features, axis=1), 1.0, atol=1e-5)


def test_internal_features_exclude_every_interface_neuron(small_graph):
    books = Codebooks(input_ids={0: (10,)}, output_ids={0: (13,)})
    assert internal_indices(small_graph, books).tolist() == [1, 2]


def test_probe_learns_separable_balanced_task():
    train = np.asarray([[-1.0, 0.0], [1.0, 0.0]] * 8, dtype=np.float32)
    labels = np.asarray([0, 1] * 8, dtype=np.int64)
    state, fitting = fit_probe(train, labels, train, labels, seed=3, epochs=30)
    result = evaluate_probe(state, train, labels)
    assert fitting["validation"]["accuracy"] == 1.0
    assert result["accuracy"] == 1.0


def test_probe_summary_reconstructs_held_out_targets(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    task = make_task("association", 4, 11 + 303)
    targets = list(task.labels)
    raw = {
        "task": "association", "seed": 11, "splits": {"test": 4},
        "results": {
            "connectome": {"test": {"accuracy": 1.0, "predictions": targets}},
            "direct": {"test": {"accuracy": 1.0, "predictions": targets}},
            "disconnected": {"test": {
                "accuracy": 0.0, "predictions": [1 - target for target in targets],
            }},
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(raw))
    result = summarize_probes([run_dir], tmp_path / "summary.json")
    association = result["tasks"]["association"]
    assert association["mean_accuracy"]["connectome"] == 1.0
    assert association["paired_hierarchical_bootstrap_95_ci"][
        "connectome_minus_disconnected"
    ] == [1.0, 1.0]


def test_plasticity_probe_scores_balanced_silent_outputs(
    small_graph, small_books, fast_config
):
    simulator = FlyLIFSimulator(
        small_graph, fast_config.dynamics, fast_config.learning, seed=4
    )
    inputs, outputs = local_populations(small_graph, small_books)
    task = DiagnosticTask("association", ((0,), (1,), (0,), (1,)), (0, 1, 0, 1))
    result = _evaluate(simulator, inputs, outputs[:2], task, fast_config)
    assert result["accuracy"] == 0.5
    assert result["silent_fraction"] == 1.0
