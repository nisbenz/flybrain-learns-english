from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    name: str
    revision: str
    url: str
    sha256: str
    local_group: str
    local_path: str
    output: str


FLY_COMMIT = "a3db62f9436074e485c0278290c2164ed6150808"
LIF_COMMIT = "c976c7a90b2ac5a472c028b5862974217e93573f"
ANNOTATION_COMMIT = "8587524c1748ce5ef2080822a2fc890fc03bf597"
SHAKESPEARE_COMMIT = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"

SOURCES = (
    Source(
        "FlyWire v783 neuron index", FLY_COMMIT,
        f"https://raw.githubusercontent.com/eonsystemspbc/fly-brain/{FLY_COMMIT}/data/2025_Completeness_783.csv",
        "52b0ac6094cd32c546f8d4c341e094376f48f4e791f8db9b166de5dff8199ea4",
        "fly", "data/2025_Completeness_783.csv", "2025_Completeness_783.csv",
    ),
    Source(
        "FlyWire v783 connectivity", FLY_COMMIT,
        f"https://raw.githubusercontent.com/eonsystemspbc/fly-brain/{FLY_COMMIT}/data/2025_Connectivity_783.parquet",
        "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347",
        "fly", "data/2025_Connectivity_783.parquet", "2025_Connectivity_783.parquet",
    ),
    Source(
        "FlyWire v783 annotations", ANNOTATION_COMMIT,
        f"https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{ANNOTATION_COMMIT}/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be",
        "annotations", "supplemental_files/Supplemental_file1_neuron_annotations.tsv",
        "Supplemental_file1_neuron_annotations.tsv",
    ),
    Source(
        "Brian2 reference model", LIF_COMMIT,
        f"https://raw.githubusercontent.com/eonsystemspbc/drosophila_brain_model_lif/{LIF_COMMIT}/model.py",
        "fc45837d7122c6ce2a7f3f2f23c515992e4b232aadb919efabb72337fac88e4e",
        "lif", "model.py", "reference/model.py",
    ),
    Source(
        "Tiny Shakespeare", SHAKESPEARE_COMMIT,
        f"https://raw.githubusercontent.com/karpathy/char-rnn/{SHAKESPEARE_COMMIT}/data/tinyshakespeare/input.txt",
        "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed",
        "corpus", "data/tinyshakespeare/input.txt", "tinyshakespeare.txt",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_sources(
    raw_dir: Path,
    local_roots: dict[str, Path | None] | None = None,
) -> list[dict[str, str]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_roots = local_roots or {}
    manifest = []
    for source in SOURCES:
        destination = raw_dir / source.output
        destination.parent.mkdir(parents=True, exist_ok=True)
        local_root = local_roots.get(source.local_group)
        local_source = local_root / source.local_path if local_root else None
        if local_source and local_source.exists():
            shutil.copyfile(local_source, destination)
        else:
            with urllib.request.urlopen(source.url) as response, destination.open("wb") as out:
                shutil.copyfileobj(response, out)
        actual = sha256_file(destination)
        if source.sha256 and actual != source.sha256:
            destination.unlink()
            raise ValueError(f"checksum mismatch for {source.name}: {actual}")
        row = asdict(source)
        row.update(path=str(destination), actual_sha256=actual)
        manifest.append(row)
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
