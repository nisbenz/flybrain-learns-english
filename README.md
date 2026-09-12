# Fly brain learns English

A runnable research prototype for testing next-word learning in a
FlyWire-v783-constrained spiking network. Tokens use immutable sparse Kenyon
cell codes, predictions use immutable disjoint spike-count populations, and
reward-modulated eligibility traces are the only trained parameters.

The checked-in source is lightweight. Connectome tables, annotations, the
corpus, checkpoints, and raw representations are downloaded or generated under
ignored directories.

## Quick start

```bash
uv sync --extra dev
uv run flybrain-language download --from-local /path/to/fly-brain \
  --annotations-dir /path/to/flywire_annotations
uv run flybrain-language prepare --config configs/tiny.yaml
uv run flybrain-language reference-check --config configs/tiny.yaml
uv run flybrain-language train --config configs/tiny.yaml --seed 11
uv run flybrain-language evaluate --run runs/tiny-seed11
uv run flybrain-language ablate --run runs/tiny-seed11
uv run flybrain-language analyze --run runs/tiny-seed11
uv run pytest
```

Omit the local repository arguments to fetch the pinned public files. The
`pilot` command prepares data and runs three seeds plus baselines within its
wall-clock budget:

```bash
uv run flybrain-language pilot --config configs/tiny.yaml --budget-minutes 120
```

See `REPORT.md` for executed results, limitations, and the distinction between
implemented modes and experiments that were actually run.
