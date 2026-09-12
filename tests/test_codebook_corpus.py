from pathlib import Path

import numpy as np

from flybrain_language.codebook import generate_codebooks, load_codebooks
from flybrain_language.corpus import Vocabulary, contiguous_split, iter_contexts, tokenize
from flybrain_language.metrics import frequency_baseline


def test_fixed_codebooks_are_deterministic_and_outputs_disjoint(tmp_path: Path):
    kenyon = np.arange(1000, 1100)
    outputs = np.arange(2000, 2100)
    first = generate_codebooks(4, kenyon, outputs, 8, 5, seed=7)
    second = generate_codebooks(4, kenyon, outputs, 8, 5, seed=7)
    assert first == second
    flattened = [neuron for neurons in first.output_ids.values() for neuron in neurons]
    assert len(flattened) == len(set(flattened))
    path = tmp_path / "codebooks.json"
    first.save(path)
    assert load_codebooks(path) == first


def test_input_codes_do_not_encode_semantic_similarity():
    books = generate_codebooks(3, np.arange(100), np.arange(200, 260), 10, 4, seed=9)
    overlaps = [
        len(set(books.input_ids[left]) & set(books.input_ids[right]))
        for left in range(3) for right in range(left + 1, 3)
    ]
    assert max(overlaps) < 10


def test_tokenization_vocabulary_and_contiguous_split():
    assert tokenize("King, KING's!") == ["king", ",", "king's", "!"]
    parts = contiguous_split("0123456789")
    assert parts == {"train": "01234567", "validation": "8", "test": "9"}
    vocab = Vocabulary.build("b a b c", 2)
    assert vocab.tokens == ("<UNK>", "<BOS>", "<EOS>", "b", "a")
    assert vocab.encode("b missing") == [3, 0]


def test_contexts_exclude_bos_as_target_and_obey_limit():
    contexts = list(iter_contexts([1, 3, 1, 4, 2], context_length=2, pair_limit=2))
    assert contexts == [[(1, 3), (1, 4)]]


def test_frequency_baseline_uses_full_unigram_distribution(fast_config):
    result = frequency_baseline([1, 0, 0, 2], [1, 0, 2], fast_config)
    assert result["predicted_token_id"] == 0
    assert result["top1_accuracy"] == 0.5
    assert result["perplexity"] < 3
