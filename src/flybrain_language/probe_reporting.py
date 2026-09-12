from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .diagnostic_tasks import make_task


def _interval(values: list[float]) -> list[float]:
    return [float(x) for x in np.quantile(values, (0.025, 0.975))]


def _paired_bootstrap(runs: list[dict], draws: int = 10_000) -> dict:
    rng = np.random.default_rng(7301)
    samples = defaultdict(list)
    hits = []
    for run in runs:
        targets = np.asarray(make_task(
            run["task"], run["splits"]["test"], run["seed"] + 303
        ).labels)
        hits.append({
            kind: np.asarray(result["test"]["predictions"]) == targets
            for kind, result in run["results"].items()
        })
    for _ in range(draws):
        selected = rng.integers(0, len(runs), len(runs))
        means = defaultdict(list)
        for run_index in selected:
            indices = rng.integers(0, len(hits[run_index]["connectome"]),
                                   len(hits[run_index]["connectome"]))
            for kind, values in hits[run_index].items():
                means[kind].append(float(values[indices].mean()))
        for kind, values in means.items():
            samples[kind].append(float(np.mean(values)))
        samples["connectome_minus_disconnected"].append(
            float(np.mean(means["connectome"]) - np.mean(means["disconnected"]))
        )
        samples["connectome_minus_chance"].append(float(np.mean(means["connectome"]) - 0.5))
    return {name: _interval(values) for name, values in samples.items()}


def summarize_probes(run_dirs: list[Path], output: Path) -> dict:
    grouped = defaultdict(list)
    for run_dir in run_dirs:
        raw = json.loads((run_dir / "metrics.json").read_text())
        grouped[raw["task"]].append(raw)
    report = {"schema_version": 1, "tasks": {}}
    for task, runs in sorted(grouped.items()):
        runs.sort(key=lambda item: item["seed"])
        kinds = ("connectome", "direct", "disconnected")
        report["tasks"][task] = {
            "per_seed": [{
                "seed": run["seed"],
                **{kind: run["results"][kind]["test"]["accuracy"] for kind in kinds},
            } for run in runs],
            "mean_accuracy": {
                kind: float(np.mean([
                    run["results"][kind]["test"]["accuracy"] for run in runs
                ])) for kind in kinds
            },
            "paired_hierarchical_bootstrap_95_ci": _paired_bootstrap(runs),
            "chance": 0.5,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
