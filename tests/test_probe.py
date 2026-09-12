import numpy as np

from flybrain_language.codebook import Codebooks
from flybrain_language.diagnostic_tasks import direct_features, make_task
from flybrain_language.linear_probe import evaluate_probe, fit_probe
from flybrain_language.probe_features import internal_indices


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
