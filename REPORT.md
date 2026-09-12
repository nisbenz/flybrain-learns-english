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

The size/density-matched random graph scored 0.0208 top-1 and fired at 63.1 Hz;
the degree-preserving shuffled graph scored 0.0104 and fired at 22.5 Hz. Their
high firing rates show that topology randomization did not preserve the stable
activity regime, so they are useful failure controls rather than clean evidence
for an anatomical advantage. Restoring the learned network's original weights
exactly reproduced the frozen result for every seed.

Learned internal-state/context-similarity correlations were −0.110, −0.077,
and −0.047. Frozen correlations were 0.009, −0.157, and −0.073. These values
show no consistent trained improvement. `king` and `queen` are outside this
eight-token vocabulary, so that comparison is explicitly unavailable.

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
compare learning rules only after frozen output silence is resolved.
