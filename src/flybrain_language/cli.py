from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .ablation import run_ablations
from .config import load_config
from .experiment import evaluate, train
from .generation import generate
from .prepare import prepare_assets
from .reference import run_reference_check
from .semantic import analyze
from .sources import download_sources


def _path(value: str | None) -> Path | None:
    return Path(value).resolve() if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flybrain-language")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--from-local")
    download.add_argument("--annotations-dir")
    download.add_argument("--lif-dir")
    download.add_argument("--corpus-dir")
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    reference = commands.add_parser("reference-check")
    reference.add_argument("--config", type=Path, required=True)
    training = commands.add_parser("train")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--seed", type=int, required=True)
    training.add_argument("--pairs", type=int)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--run", type=Path, required=True)
    ablation = commands.add_parser("ablate")
    ablation.add_argument("--run", type=Path, required=True)
    ablation.add_argument("--pairs", type=int)
    generation = commands.add_parser("generate")
    generation.add_argument("--run", type=Path, required=True)
    generation.add_argument("--length", type=int, default=16)
    generation.add_argument("--start", default="<BOS>")
    semantic = commands.add_parser("analyze")
    semantic.add_argument("--run", type=Path, required=True)
    pilot = commands.add_parser("pilot")
    pilot.add_argument("--config", type=Path, required=True)
    pilot.add_argument("--budget-minutes", type=float, default=120)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "download":
        result = download_sources(root / "data" / "raw", {
            "fly": _path(args.from_local), "annotations": _path(args.annotations_dir),
            "lif": _path(args.lif_dir), "corpus": _path(args.corpus_dir),
        })
    elif args.command == "prepare":
        result = prepare_assets(args.config, root)
    elif args.command == "reference-check":
        config = load_config(args.config)
        result = run_reference_check(
            root / "data" / "raw", config.dynamics,
            root / "runs" / "reference_check.json",
        )
    elif args.command == "train":
        result = {"run": str(train(args.config, root, args.seed, args.pairs))}
    elif args.command == "evaluate":
        result = evaluate(args.run.resolve(), root)
    elif args.command == "ablate":
        result = run_ablations(args.run.resolve(), root, args.pairs)
    elif args.command == "generate":
        result = {"tokens": generate(args.run.resolve(), root, args.length, args.start)}
    elif args.command == "analyze":
        result = analyze(args.run.resolve(), root)
    else:
        result = _pilot(args.config, root, args.budget_minutes)
    print(json.dumps(result, indent=2))


def _pilot(config_path: Path, root: Path, budget_minutes: float) -> dict:
    started = time.perf_counter()
    deadline = started + budget_minutes * 60
    prepared = root / "data" / "processed" / load_config(config_path).name
    if not (prepared / "connectome.npz").exists():
        prepare_assets(config_path, root)
    completed = []
    for seed in load_config(config_path).seeds:
        if time.perf_counter() >= deadline:
            break
        run_dir = train(config_path, root, seed)
        evaluation = evaluate(run_dir, root)
        semantic = analyze(run_dir, root)
        completed.append({"seed": seed, "run": str(run_dir), "evaluation": evaluation, "semantic": semantic})
    if completed and time.perf_counter() < deadline:
        completed[0]["ablations"] = run_ablations(Path(completed[0]["run"]), root)
    return {
        "budget_minutes": budget_minutes,
        "elapsed_seconds": time.perf_counter() - started,
        "completed": completed,
        "budget_exhausted": time.perf_counter() >= deadline,
    }


if __name__ == "__main__":
    main()
