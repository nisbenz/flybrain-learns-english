from __future__ import annotations

import json
from pathlib import Path

from .config import config_dict, load_config
from .corpus import prepare_corpus
from .graph import prepare_connectome
from .sources import sha256_file


def prepare_assets(config_path: Path, project_root: Path) -> dict:
    config = load_config(config_path)
    raw = project_root / "data" / "raw"
    required = {
        "completeness": raw / "2025_Completeness_783.csv",
        "connectivity": raw / "2025_Connectivity_783.parquet",
        "annotations": raw / "Supplemental_file1_neuron_annotations.tsv",
        "corpus": raw / "tinyshakespeare.txt",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("run `flybrain-language download` first; missing: " + ", ".join(missing))
    output = project_root / "data" / "processed" / config.name
    corpus = prepare_corpus(required["corpus"], output, config.lexical_tokens)
    graph, books, graph_report = prepare_connectome(
        required["completeness"], required["connectivity"], required["annotations"],
        output, len(corpus["vocabulary"].tokens), config.input_cells,
        config.output_cells, config.target_neurons, config.max_hops, config.seeds[0],
    )
    report = {
        "config": config_dict(config),
        "raw_sha256": {name: sha256_file(path) for name, path in required.items()},
        "processed_sha256": {
            name: sha256_file(output / name)
            for name in ["connectome.npz", "codebooks.json", "vocabulary.json", "corpus.json"]
        },
        "graph": graph_report,
        "vocabulary_tokens": list(corpus["vocabulary"].tokens),
        "codebook_sha256": books.digest(),
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
