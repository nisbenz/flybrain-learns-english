import numpy as np
import pytest

from flybrain_language.codebook import Codebooks
from flybrain_language.config import DynamicsConfig, ExperimentConfig, LearningConfig
from flybrain_language.graph import Connectome


@pytest.fixture
def small_graph():
    return Connectome(
        flywire_ids=np.array([10, 11, 12, 13]),
        pre=np.array([0, 1, 2, 0]),
        post=np.array([1, 2, 3, 3]),
        weights=np.array([2.0, -1.0, 1.5, 1.0], dtype=np.float32),
        plastic=np.array([True, True, False, False]),
    )


@pytest.fixture
def small_books():
    return Codebooks(
        input_ids={0: (10,), 1: (11,), 2: (12,)},
        output_ids={0: (11,), 1: (12,), 2: (13,)},
    )


@pytest.fixture
def fast_config():
    return ExperimentConfig(
        lexical_tokens=0, input_cells=1, output_cells=1,
        context_length=2, presentation_ms=1, gap_ms=0, prediction_ms=1,
        train_pairs=2, validation_pairs=2, test_pairs=2,
        dynamics=DynamicsConfig(dt_ms=1, delay_ms=1, noise_std_mv=0, input_rate_hz=0),
        learning=LearningConfig(rate=0.1, min_fraction=0.5, max_fraction=1.5),
    )
