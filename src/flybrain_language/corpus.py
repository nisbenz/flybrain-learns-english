from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[^\w\s]", re.IGNORECASE)
SPECIAL_TOKENS = ("<UNK>", "<BOS>", "<EOS>")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class Vocabulary:
    tokens: tuple[str, ...]

    @classmethod
    def build(cls, training_text: str, lexical_tokens: int) -> "Vocabulary":
        counts = Counter(tokenize(training_text))
        ranked = sorted(counts, key=lambda token: (-counts[token], token))
        return cls(SPECIAL_TOKENS + tuple(ranked[:lexical_tokens]))

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: index for index, token in enumerate(self.tokens)}

    def encode(self, text: str) -> list[int]:
        lookup = self.token_to_id
        unk = lookup["<UNK>"]
        return [lookup.get(token, unk) for token in tokenize(text)]

    def save(self, path: Path) -> None:
        body = {"tokens": self.tokens, "sha256": self.digest()}
        path.write_text(json.dumps(body, indent=2) + "\n")

    def digest(self) -> str:
        payload = json.dumps(self.tokens, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def contiguous_split(text: str) -> dict[str, str]:
    first = int(len(text) * 0.8)
    second = int(len(text) * 0.9)
    return {"train": text[:first], "validation": text[first:second], "test": text[second:]}


def prepare_corpus(raw_path: Path, output_dir: Path, lexical_tokens: int) -> dict:
    text = raw_path.read_text()
    splits = contiguous_split(text)
    vocab = Vocabulary.build(splits["train"], lexical_tokens)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(output_dir / "vocabulary.json")
    encoded = {}
    for name, split_text in splits.items():
        ids = [vocab.token_to_id["<BOS>"], *vocab.encode(split_text), vocab.token_to_id["<EOS>"]]
        encoded[name] = ids
    body = {
        "splits": encoded,
        "character_boundaries": [int(len(text) * 0.8), int(len(text) * 0.9)],
        "vocabulary_sha256": vocab.digest(),
        "token_counts": {name: len(ids) for name, ids in encoded.items()},
    }
    (output_dir / "corpus.json").write_text(json.dumps(body) + "\n")
    return {"vocabulary": vocab, **body}


def load_corpus(output_dir: Path) -> tuple[Vocabulary, dict[str, list[int]]]:
    vocab_raw = json.loads((output_dir / "vocabulary.json").read_text())
    vocab = Vocabulary(tuple(vocab_raw["tokens"]))
    if vocab.digest() != vocab_raw["sha256"]:
        raise ValueError("vocabulary checksum mismatch")
    corpus = json.loads((output_dir / "corpus.json").read_text())
    return vocab, corpus["splits"]


def iter_contexts(
    ids: list[int], context_length: int, pair_limit: int | None = None
) -> Iterator[list[tuple[int, int]]]:
    emitted = 0
    pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1) if ids[i + 1] != 1]
    for start in range(0, len(pairs), context_length):
        context = pairs[start : start + context_length]
        if pair_limit is not None:
            context = context[: max(0, pair_limit - emitted)]
        if not context:
            return
        yield context
        emitted += len(context)
        if pair_limit is not None and emitted >= pair_limit:
            return
