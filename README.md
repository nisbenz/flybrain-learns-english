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
uv run flybrain-language activity-check --config configs/tiny.yaml
uv run flybrain-language train --config configs/tiny.yaml --seed 11
uv run flybrain-language evaluate --run runs/tiny-seed11
uv run flybrain-language ablate --run runs/tiny-seed11
uv run flybrain-language analyze --run runs/tiny-seed11
uv run flybrain-language summarize --runs \
  runs/tiny-seed11 runs/tiny-seed23 runs/tiny-seed37 --output results
uv run pytest
```

The FLM-inspired diagnostic curriculum separates representational capacity from
internal plasticity. Run the frozen-state readout and the stricter anatomical
plasticity task independently:

```bash
for task in association delayed_context; do
  for seed in 11 23 37; do
    uv run flybrain-language probe --config configs/tiny.yaml \
      --task "$task" --seed "$seed"
  done
done
uv run flybrain-language probe-summary --runs runs/probe-tiny-* \
  --output results/probe_summary.json

for seed in 11 23 37; do
  uv run flybrain-language plasticity-probe --config configs/tiny.yaml \
    --task association --seed "$seed"
done
uv run flybrain-language plasticity-summary \
  --runs runs/plasticity-tiny-association-seed*-noise0.025 \
  --output results/plasticity_probe_summary.json
```

`probe` trains a bias-free diagnostic readout while freezing the graph and
excluding every input/output population from its features. Its direct-input
and disconnected controls have exactly the same readout size. This diagnostic
does not satisfy the project's internal-plasticity learning criterion.
`plasticity-probe` retains the fixed spike-count decoder and trains only the
existing anatomical plastic-edge mask.

For the exact resolved environment used by the reported CPU run:

```bash
uv venv --python 3.12
uv pip sync requirements.lock --python .venv/bin/python
uv pip install --python .venv/bin/python --no-deps -e .
```

Omit the local repository arguments to fetch the pinned public files. The
`pilot` command prepares data and runs three seeds plus baselines within its
wall-clock budget:

```bash
uv run flybrain-language pilot --config configs/tiny.yaml --budget-minutes 120
```

See `REPORT.md` for executed results, limitations, and the distinction between
implemented modes and experiments that were actually run.

## Data and interfaces

`download` verifies pinned SHA-256 values before data are used. `prepare`
validates all connectivity indices against FlyWire IDs, checks that every edge
has a resolved sign and that each neuron has one outgoing sign, identifies
Kenyon cells and MBONs from v783 annotations, and saves immutable codebooks plus
a graph manifest. JSON stores FlyWire IDs as strings; arrays retain `int64`.

The simulator uses the upstream LIF constants and mutable sparse CSR
propagation. It keeps voltage, conductance, delay, refractory, trace, and RNG
state across words in a context. Only codebook neurons receive external input.
Predictions are normalized fixed-population spike counts with deterministic
token-ID tie breaking. No trainable decoder exists.

Detailed run artifacts live under `runs/` and are ignored because checkpoints
and representations can grow. Compact measured outputs are checked in under
`results/`:

- `pilot_summary.json`: three-seed metrics, paired context bootstrap intervals,
  and topology controls.
- `probe_summary.json`: balanced association and delayed-context readout tests.
- `plasticity_probe_summary.json`: the matching internal-plasticity test.
- `activity_check.json`: propagation, throughput, memory, and hardware.
- `performance.png`, `training_stability.png`, and `semantic_cosine_seed11.png`:
  small result plots, including the held-out state-similarity diagnostic.

The complete raw-source manifest is generated at `data/raw/manifest.json`; the
processed graph manifest records the selected neuron/edge counts, data hashes,
codebook hash, neurotransmitter coverage, and lost boundary connectivity.

## Reproducing individual modes

Replace `configs/tiny.yaml` with `configs/subgraph.yaml` for 32 lexical tokens
and approximately 8,000 neurons. `configs/full.yaml` selects all v783 neurons
and 128 lexical tokens. These larger modes were not run in the reported pilot
because the eight-token validation gate failed.
