from __future__ import annotations

import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/flybrain-matplotlib")
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_json(path: Path):
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _compact(metrics: dict) -> dict:
    fields = (
        "top1_accuracy", "top5_accuracy", "cross_entropy", "perplexity",
        "retained_word_accuracy", "macro_accuracy", "unk_target_fraction",
        "silent_fraction", "tie_fraction", "mean_firing_rate_hz",
        "active_neuron_fraction", "plastic_weight_mean_absolute_change",
        "weight_bound_saturation",
    )
    return {field: metrics[field] for field in fields if field in metrics}


def _paired_ci(run_dirs: list[Path], comparator: str, retained: bool, seed: int = 2026) -> dict:
    blocks = []
    for run in run_dirs:
        learned = _load_jsonl(run / "learned_predictions.jsonl")
        if comparator == "frozen":
            other = _load_jsonl(run / "frozen_predictions.jsonl")
            other_hits = [row["prediction"] == row["target"] for row in other]
        else:
            other_hits = [row["target"] == 0 for row in learned]
        grouped = defaultdict(list)
        for index, row in enumerate(learned):
            if retained and row["target"] < 3:
                continue
            grouped[row["context"]].append(
                ((row["prediction"] == row["target"]), other_hits[index])
            )
        for values in grouped.values():
            blocks.append((sum(x[0] for x in values), sum(x[1] for x in values), len(values)))
    learned_hits, other_hits, total = np.asarray(blocks).sum(axis=0)
    observed = (learned_hits - other_hits) / total
    rng = np.random.default_rng(seed)
    samples = np.empty(5000)
    for index in range(len(samples)):
        selected = rng.integers(0, len(blocks), len(blocks))
        draw = np.asarray([blocks[item] for item in selected]).sum(axis=0)
        samples[index] = (draw[0] - draw[1]) / draw[2]
    return {
        "difference": float(observed),
        "context_bootstrap_95ci": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "contexts": len(blocks), "pairs": int(total),
    }


def summarize(run_dirs: list[Path], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = [_load_json(run / "evaluation_metrics.json") for run in run_dirs]
    training = [_load_json(run / "training_metrics.json") for run in run_dirs]
    semantic = [_load_json(run / "semantic_analysis.json") for run in run_dirs]
    conditions = ["learned", "frozen", "frequency"]
    fields = ["top1_accuracy", "top5_accuracy", "retained_word_accuracy", "macro_accuracy"]
    aggregate = {}
    for condition in conditions:
        aggregate[condition] = {}
        for field in fields:
            values = np.asarray([item[condition][field] for item in evaluations], dtype=float)
            aggregate[condition][field] = {
                "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1))
            }
    report = {
        "seeds": [item["seed"] for item in training],
        "aggregate": aggregate,
        "paired_inference": {
            "learned_minus_frozen_top1": _paired_ci(run_dirs, "frozen", False),
            "learned_minus_frequency_top1": _paired_ci(run_dirs, "frequency", False),
            "learned_minus_frozen_retained": _paired_ci(run_dirs, "frozen", True),
        },
        "per_seed": [{
            "seed": train["seed"],
            "best_validation_accuracy": train["best_validation_accuracy"],
            "best_pairs_seen": train["best_pairs_seen"],
            "training_seconds": train["elapsed_seconds"],
            "learned": _compact(evaluation["learned"]),
            "frozen": _compact(evaluation["frozen"]),
            "frequency": _compact(evaluation["frequency"]),
            "semantic_context_correlation": semantics["context_similarity_correlation"],
        } for train, evaluation, semantics in zip(training, evaluations, semantic)],
        "success": False,
        "success_reason": "learned top-1 did not exceed frozen and frequency baselines across seeds",
        "interpretation": "negative pilot result; no demonstrated next-word or semantic learning",
    }
    ablation_path = run_dirs[0] / "ablation_metrics.json"
    if ablation_path.exists():
        ablations = _load_json(ablation_path)
        report["ablations"] = {
            name: {"train_pairs": value["train_pairs"], **_compact(value["test"])}
            for name, value in ablations.items()
        }
    (output_dir / "pilot_summary.json").write_text(
        json.dumps(report, separators=(",", ":")) + "\n"
    )
    _performance_plot(aggregate, output_dir / "performance.png")
    _training_plot(training, output_dir / "training_stability.png")
    shutil.copyfile(run_dirs[0] / "semantic_cosine.png", output_dir / "semantic_cosine_seed11.png")
    return report


def _performance_plot(aggregate: dict, path: Path) -> None:
    fields = ["top1_accuracy", "top5_accuracy", "retained_word_accuracy", "macro_accuracy"]
    labels = ["top-1", "top-5", "retained", "macro"]
    x = np.arange(len(fields))
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for offset, condition in zip((-0.25, 0, 0.25), aggregate):
        means = [aggregate[condition][field]["mean"] for field in fields]
        errors = [aggregate[condition][field]["sample_sd"] for field in fields]
        axis.bar(x + offset, means, 0.24, yerr=errors, label=condition, capsize=3)
    axis.plot(x, [0.1, 0.5, 0.1, 0.1], "k--", linewidth=1, label="uniform chance")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("accuracy (mean ± sample SD across 3 seeds)")
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _training_plot(training: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for run in training:
        x = [stage["pairs_seen"] for stage in run["stages"]]
        y = [stage["validation"]["top1_accuracy"] for stage in run["stages"]]
        axis.plot(x, y, marker="o", label=f"seed {run['seed']}")
    axis.set(xlabel="training pairs", ylabel="validation top-1 accuracy", ylim=(0, 1))
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)
