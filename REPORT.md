# CPU pilot report

Run date: 2026-09-12. The primary result is negative: this implementation ran
successfully, but reward-modulated plasticity did not improve held-out
next-token prediction over the frozen FlyWire subgraph or the unigram baseline.
No semantic-learning claim is supported.

## Executed experiment

The run used FlyWire release 783, a 2,000-neuron path-preserving subgraph,
118,527 directed anatomical edges, and 28,609 restricted plastic edges. The
annotation pass found 5,177 bilateral Kenyon cells and 96 MBONs. Fixed codes
used 64 Kenyon cells per token and 16 disjoint output neurons per token. The
vocabulary was eight training-frequency lexical tokens plus `<UNK>`, `<BOS>`,
and `<EOS>`.

Every word used 30 ms stimulation, a 5 ms gap, and a 75 ms prediction window at
0.1 ms resolution. Training used 384 pairs per seed; checkpoints at 96, 192,
288, and 384 pairs were selected by 96 validation pairs. Final metrics used 192
held-out test pairs per seed for seeds 11, 23, and 37. State persisted within
32-token contexts and reset between contexts.

The deterministic two-neuron update trace agreed with an independent NumPy
translation of the pinned Brian2 ordering to a maximum absolute error of
`6.27e-6`. All 11 token codes produced distinct internal states. Individual
tokens activated 72–203 neurons, with mean firing rates from 1.51 to 3.66 Hz.
Peak process RSS in the activity check was 306.5 MB.

## Held-out measurements

Values below are means across three seeds; uncertainty is sample standard
deviation across seeds.

| Condition | Top-1 | Top-5 | Retained-word | Macro | Perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned FlyWire | 0.5052 ± 0.0180 | 0.7656 ± 0.0227 | 0.0164 ± 0.0164 | 0.0971 ± 0.0164 | 11.376 ± 0.043 |
| Frozen FlyWire | 0.5139 ± 0.0197 | 0.7656 ± 0.0239 | 0.0164 ± 0.0164 | 0.0985 ± 0.0169 | 11.397 ± 0.065 |
| Frequency | 0.6823 | 0.8854 | 0.0000 | 0.1111 | 3.458 |

The test prefix contains 68.23% `<UNK>` targets. Consequently, all-token
accuracy substantially overstates useful language behavior. Learned-minus-
frozen top-1 was −0.00868 with a context-bootstrap 95% interval of
[−0.02083, 0.00347]. Learned-minus-frequency was −0.17708,
[−0.20313, −0.15104]. On 183 retained-word pairs, learned-minus-frozen was
exactly 0. The success criterion therefore failed for every seed.

The learned network was silent in 75.17% of prediction windows and tied in many
windows, while averaging 1.56 Hz and activating about 39.7% of neurons over the
full test. Plastic weights changed by only about 0.0013–0.0019 synapse-count
units on average, and none reached configured bounds.

## Controls and representations

The size/density-matched random graph scored 0.0260 top-1 and fired at 63.1 Hz;
the degree-preserving shuffled graph scored 0.0104 and fired at 22.4 Hz. Both
controls selected their 96-pair checkpoint using the primary validation rule.
Their high firing rates show that topology randomization did not preserve the
stable activity regime, so they are useful failure controls rather than clean
evidence for an anatomical advantage. Restoring the learned network's original
weights exactly reproduced the frozen result for every seed.

Recorded primary training time totaled 1,220.8 seconds across three seeds; the
two validation-selected topology controls totaled 442.9 seconds. The activity
benchmark averaged 0.339 seconds per simulated word.

Learned internal-state/context-similarity correlations were −0.110, −0.077,
and −0.047. Frozen correlations were 0.009, −0.157, and −0.073. These values
show no consistent trained improvement. `king` and `queen` are outside this
eight-token vocabulary, so that comparison is explicitly unavailable.

## FLM-inspired simpler benchmarks

The diagnostic design follows FLM revision
`7251a8921db4f891c39bd75ee5ad827f7031a24b`: use balanced synthetic examples,
freeze the connectome while testing its state with a small validation-selected
readout, and compare a parameter-matched direct-input path and a graph with all
edges removed. This project does not use FLM's pretrained language model or
abstract rate recurrence in the anatomical learning experiment.

The association task distinguishes two single-token cues. The delayed-context
task distinguishes the same cues after both are followed by one shared token.
Each seed used 32 balanced training, 16 validation, and 32 held-out test
examples. Graph features contain 1,154 internal neurons after every input and
output neuron is excluded. The bias-free readout has 2,308 parameters in every
condition.

| Diagnostic | Frozen connectome probe | Direct input | Disconnected graph |
| --- | ---: | ---: | ---: |
| Association | 0.6979 | 1.0000 | 0.5104 |
| Delayed context | 0.5000 | 1.0000 | 0.5521 |

For association, connectome-minus-disconnected was 0.1875 with a paired
hierarchical-bootstrap 95% interval of [0.0625, 0.3229]. Connectome-minus-
chance was [0.0833, 0.3125]. This supports a narrow claim: present-cue identity
is decodable from frozen internal activity. Delayed-context accuracy was
exactly chance across seeds; its connectome-minus-chance interval was
[−0.1354, 0.1354]. The current word timing does not expose reliable memory of
the earlier cue.

The stricter association task retained fixed two-population spike-count output
and trained only the 28,609 anatomical plastic edges. It used 128 balanced
training, 32 validation, and 64 test examples per seed. Learned accuracy was
0.4583 versus 0.4531 frozen. Learned-minus-chance had a 95% interval of
[−0.1146, 0.0313], so internal learning failed even on this simpler task. The
successful linear probe is diagnostic evidence of available signal, not
evidence of synaptic learning.

## Limitations and next experiment

The vocabulary policy causes severe `<UNK>` imbalance, scalar reinforcement is
sparse, output populations are silent frequently, and a 2,000-neuron induced
subgraph discards 10,389,582 boundary synapses. The randomized controls also
leave the anatomical firing regime. These issues prevent a broad conclusion
about whether a full fly connectome can learn language.

The 32/128-token and full-connectome configurations are implemented but were
not run because the eight-token success gate failed. Projection-neuron input,
whole-network plasticity, and a conventional recurrent-spiking baseline remain
future experiments. A scientifically useful next pilot should reduce unknown
imbalance, calibrate output-population activity without language labels, and
compare local credit rules on the balanced association task before returning
to language. The probe results specifically favor a local supervised or
perturbation-based three-factor rule that can route an already decodable cue to
the fixed outputs.
