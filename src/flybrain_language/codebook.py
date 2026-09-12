from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Codebooks:
    input_ids: dict[int, tuple[int, ...]]
    output_ids: dict[int, tuple[int, ...]]

    def canonical(self) -> dict[str, dict[str, list[str]]]:
        return {
            "input_ids": {
                str(token): [str(neuron) for neuron in neurons]
                for token, neurons in sorted(self.input_ids.items())
            },
            "output_ids": {
                str(token): [str(neuron) for neuron in neurons]
                for token, neurons in sorted(self.output_ids.items())
            },
        }

    def digest(self) -> str:
        raw = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()

    def save(self, path: Path) -> None:
        body = self.canonical()
        body["sha256"] = self.digest()
        path.write_text(json.dumps(body, indent=2) + "\n")


def generate_codebooks(
    token_count: int,
    kenyon_ids: np.ndarray,
    output_candidates: np.ndarray,
    input_cells: int,
    output_cells: int,
    seed: int,
) -> Codebooks:
    kenyon_ids = np.sort(np.asarray(kenyon_ids, dtype=np.int64))
    output_candidates = np.asarray(output_candidates, dtype=np.int64)
    if len(kenyon_ids) < input_cells:
        raise ValueError("not enough annotated Kenyon cells for the input code")
    if len(output_candidates) < token_count * output_cells:
        raise ValueError("not enough disjoint output candidates")
    input_rng = np.random.default_rng(seed)
    output_rng = np.random.default_rng(seed + 1_000_003)
    inputs = {
        token: tuple(int(x) for x in np.sort(input_rng.choice(
            kenyon_ids, size=input_cells, replace=False
        )))
        for token in range(token_count)
    }
    permutation = output_rng.permutation(output_candidates)
    outputs = {
        token: tuple(int(x) for x in np.sort(
            permutation[token * output_cells : (token + 1) * output_cells]
        ))
        for token in range(token_count)
    }
    return Codebooks(inputs, outputs)


def load_codebooks(path: Path) -> Codebooks:
    raw = json.loads(path.read_text())
    books = Codebooks(
        {int(k): tuple(map(int, v)) for k, v in raw["input_ids"].items()},
        {int(k): tuple(map(int, v)) for k, v in raw["output_ids"].items()},
    )
    if books.digest() != raw["sha256"]:
        raise ValueError("codebook checksum mismatch")
    return books
