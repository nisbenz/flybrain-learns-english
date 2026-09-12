from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .anatomy import load_populations, validate_edge_signs
from .codebook import Codebooks, generate_codebooks


@dataclass
class Connectome:
    flywire_ids: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    weights: np.ndarray
    plastic: np.ndarray

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path, flywire_ids=self.flywire_ids, pre=self.pre, post=self.post,
            weights=self.weights, plastic=self.plastic,
        )

    @classmethod
    def load(cls, path: Path) -> "Connectome":
        with np.load(path) as raw:
            return cls(*(raw[name].copy() for name in (
                "flywire_ids", "pre", "post", "weights", "plastic"
            )))


def _reachable_layers(pre: np.ndarray, post: np.ndarray, starts: np.ndarray, hops: int):
    seen = set(map(int, starts))
    frontier = np.asarray(sorted(seen), dtype=np.int64)
    layers = [frontier]
    for _ in range(hops):
        reached = np.unique(post[np.isin(pre, frontier)])
        frontier = np.asarray(sorted(set(map(int, reached)) - seen), dtype=np.int64)
        layers.append(frontier)
        seen.update(map(int, frontier))
        if not len(frontier):
            break
    return layers


def _path_nodes(edges: pd.DataFrame, layers: list[np.ndarray], targets: set[int]) -> set[int]:
    selected = set(targets)
    current = set(targets)
    for depth in range(len(layers) - 1, 0, -1):
        layer_targets = (current | targets).intersection(map(int, layers[depth]))
        if not layer_targets:
            continue
        mask = edges.Postsynaptic_ID.isin(layer_targets) & edges.Presynaptic_ID.isin(layers[depth - 1])
        incoming = edges.loc[mask, ["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"]]
        parents = incoming.sort_values(
            ["Postsynaptic_ID", "Connectivity", "Presynaptic_ID"],
            ascending=[True, False, True],
        ).drop_duplicates("Postsynaptic_ID")
        current = set(map(int, parents.Presynaptic_ID))
        selected.update(current)
    return selected


def _fill_connected(edges: pd.DataFrame, selected: set[int], target: int) -> set[int]:
    if len(selected) >= target:
        return selected
    inside_pre = edges.Presynaptic_ID.isin(selected)
    inside_post = edges.Postsynaptic_ID.isin(selected)
    boundary = edges.loc[inside_pre ^ inside_post].copy()
    boundary["candidate"] = np.where(
        inside_pre[inside_pre ^ inside_post], boundary.Postsynaptic_ID, boundary.Presynaptic_ID
    )
    scores = boundary.groupby("candidate").Connectivity.sum().sort_values(ascending=False)
    for candidate in scores.index:
        selected.add(int(candidate))
        if len(selected) >= target:
            break
    return selected


def _corridor_mask(pre: np.ndarray, post: np.ndarray, mbons: set[int], outputs: set[int]) -> np.ndarray:
    forward = set(mbons)
    for _ in range(3):
        forward.update(map(int, post[np.isin(pre, list(forward))]))
    backward = set(outputs)
    for _ in range(3):
        backward.update(map(int, pre[np.isin(post, list(backward))]))
    corridor = forward.intersection(backward)
    return np.isin(pre, list(corridor)) & np.isin(post, list(corridor))


def prepare_connectome(
    completeness_path: Path,
    connectivity_path: Path,
    annotation_path: Path,
    output_dir: Path,
    token_count: int,
    input_cells: int,
    output_cells: int,
    target_neurons: int | None,
    max_hops: int,
    seed: int,
) -> tuple[Connectome, Codebooks, dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    complete = pd.read_csv(completeness_path, index_col=0)
    ids = complete.index.to_numpy(np.int64)
    edges = pd.read_parquet(connectivity_path)
    pre_id, post_id = edges.Presynaptic_ID.to_numpy(), edges.Postsynaptic_ID.to_numpy()
    if not np.array_equal(ids[edges.Presynaptic_Index.to_numpy()], pre_id):
        raise ValueError("presynaptic indices do not match FlyWire IDs")
    if not np.array_equal(ids[edges.Postsynaptic_Index.to_numpy()], post_id):
        raise ValueError("postsynaptic indices do not match FlyWire IDs")
    populations = load_populations(annotation_path)
    sign_report = validate_edge_signs(edges, populations.annotation)
    known = set(map(int, ids))
    kc_ids = np.asarray([x for x in populations.kenyon_ids if int(x) in known], dtype=np.int64)
    mbon_ids = np.asarray([x for x in populations.mbon_ids if int(x) in known], dtype=np.int64)
    layers = _reachable_layers(pre_id, post_id, mbon_ids, max_hops)
    candidates = np.unique(np.concatenate(layers[1:]))
    candidates = candidates[~np.isin(candidates, np.concatenate([kc_ids, mbon_ids]))]
    scores = edges.loc[edges.Postsynaptic_ID.isin(candidates)].groupby("Postsynaptic_ID").Connectivity.sum()
    ranked = pd.DataFrame({"id": scores.index, "score": scores.values}).sort_values(
        ["score", "id"], ascending=[False, True]
    ).id.to_numpy(np.int64)
    required_outputs = token_count * output_cells
    pool = ranked[: max(required_outputs * 4, required_outputs)]
    books = generate_codebooks(token_count, kc_ids, pool, input_cells, output_cells, seed)
    output_ids = set(x for values in books.output_ids.values() for x in values)
    input_ids = set(x for values in books.input_ids.values() for x in values)
    paths = _path_nodes(edges, layers, output_ids)
    selected = input_ids | output_ids | set(map(int, mbon_ids)) | paths
    if target_neurons is None:
        selected = known
    else:
        selected = _fill_connected(edges, selected, target_neurons)
    induced = edges.Presynaptic_ID.isin(selected) & edges.Postsynaptic_ID.isin(selected)
    kept = edges.loc[induced]
    selected_ids = np.asarray(sorted(selected), dtype=np.int64)
    lookup = pd.Series(np.arange(len(selected_ids)), index=selected_ids)
    local_pre = lookup.loc[kept.Presynaptic_ID].to_numpy(np.int64)
    local_post = lookup.loc[kept.Postsynaptic_ID].to_numpy(np.int64)
    plastic = np.isin(kept.Presynaptic_ID, list(input_ids)) | _corridor_mask(
        kept.Presynaptic_ID.to_numpy(), kept.Postsynaptic_ID.to_numpy(),
        set(map(int, mbon_ids)), output_ids,
    )
    graph = Connectome(
        selected_ids, local_pre, local_post,
        kept["Excitatory x Connectivity"].to_numpy(np.float32), plastic,
    )
    boundary = edges.Presynaptic_ID.isin(selected) ^ edges.Postsynaptic_ID.isin(selected)
    report = {
        "neurons": len(selected_ids), "edges": len(kept), "plastic_edges": int(plastic.sum()),
        "kenyon_pool": len(kc_ids), "mbons": len(mbon_ids),
        "lost_boundary_synapses": int(edges.loc[boundary, "Connectivity"].sum()),
        "sign_validation": sign_report,
    }
    graph.save(output_dir / "connectome.npz")
    books.save(output_dir / "codebooks.json")
    (output_dir / "graph_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return graph, books, report
