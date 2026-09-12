from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnnotatedPopulations:
    kenyon_ids: np.ndarray
    mbon_ids: np.ndarray
    annotation: pd.DataFrame


def load_populations(path: Path) -> AnnotatedPopulations:
    columns = [
        "root_id", "flow", "super_class", "cell_class", "cell_type",
        "hemibrain_type", "top_nt", "top_nt_conf", "side",
    ]
    table = pd.read_csv(path, sep="\t", usecols=columns, low_memory=False)
    table = table.drop_duplicates("root_id").set_index("root_id", drop=False)
    cell_class = table["cell_class"].fillna("").str.casefold()
    sides = table["side"].fillna("").str.casefold()
    kenyon = table.loc[(cell_class == "kenyon_cell") & sides.isin(["left", "right"])]
    mbon = table.loc[cell_class == "mbon"]
    if set(kenyon["side"].str.casefold()) != {"left", "right"}:
        raise ValueError("bilateral Kenyon-cell annotations are required")
    if len(kenyon) < 64 or mbon.empty:
        raise ValueError("required Kenyon-cell or MBON annotations are missing")
    return AnnotatedPopulations(
        np.sort(kenyon.root_id.to_numpy(np.int64)),
        np.sort(mbon.root_id.to_numpy(np.int64)),
        table,
    )


def validate_edge_signs(edges: pd.DataFrame, annotations: pd.DataFrame) -> dict:
    signs = edges.groupby("Presynaptic_ID", sort=False)["Excitatory"].agg(["min", "max"])
    if not signs[["min", "max"]].isin([-1, 1]).all().all():
        raise ValueError("connectivity contains unresolved excitatory signs")
    if (signs["min"] != signs["max"]).any():
        raise ValueError("a presynaptic neuron has mixed outgoing signs")
    source_ids = signs.index.intersection(annotations.index)
    nt = annotations.loc[source_ids, "top_nt"].fillna("unannotated")
    counts = nt.value_counts().to_dict()
    return {
        "edge_sign_source": "FlyWire v783 Excitatory column derived from neurotransmitter annotations",
        "presynaptic_neurons": int(len(signs)),
        "annotated_presynaptic_neurons": int(len(source_ids)),
        "neurotransmitter_counts": {str(k): int(v) for k, v in counts.items()},
    }
