# NeuralSplauts

A small PyTorch **CPU** prototype for bioacoustic localization using neural
spectrotemporal Gaussian splauts. It implements the
[implementation plan](Design/ADR/Implementation/001_splauts-implementation-plan.md).
The aim is to inspect acoustic representations, propagation gradients, spatial
ambiguity, and optimization behavior in reproducible synthetic experiments.

## Run

Use Python 3.11 or newer. From this directory:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m splauts demo --output outputs/single
python -m splauts demo --events 2 --refine --output outputs/multi
python -m splauts compare --grid-size 15 --output outputs/comparison
python -m unittest discover -s tests -v
```

The package uses PyTorch, NumPy, and Matplotlib; tests use the Python standard
library. Installing the package also provides the `splauts` command. All tensors
remain on CPU. The CLI defaults to one PyTorch thread because the matrices are
small; change this with `--threads` if appropriate for your machine.
Full comparison sweeps and multi-event refinement can take several minutes on
CPU. Reduce iteration counts and grid sizes for quick visual smoke checks.

`demo` defaults to four corner stations, one tonal call, 32 splauts, a
16-dimensional latent, 600 representation-training iterations, and 250
localization iterations. Examples of other experiments:

```powershell
python -m splauts demo --stations 6 --layout irregular --call sweep --noise 0.0001 --seed 12 --output outputs/sweep
python -m splauts demo --events 3 --splauts 64 --latent-dim 32 --output outputs/three
python -m splauts demo --train-steps 100 --localize-steps 100 --grid-size 11 --output outputs/quick
```

Short training runs are useful smoke checks, but can produce poor call fits and
misleading localization. Use separate output directories to preserve runs.

## Experiments and outputs

`demo` synthesizes 1–3 calls at known positions and onsets, fractionally delays
their waveforms to every station, applies distance attenuation, mixes them, and
adds seeded waveform noise. It trains a shared MLP with a learned embedding per
isolated call. Decoder output biases are initialized from **isolated training
spectrogram energy**, without station observations or position labels.

The decoder is then frozen and only event positions are optimized. Single events
start at the area centre. For multiple events, a 13 × 13 coordinate grid search
first improves reconstruction one associated event at a time, without position
labels; gradients then refine those estimates. Use `--coarse-grid-size 0` to
study direct inversion from the centre and its local minima. The neutral and
gradient-start positions are recorded in `metrics.json`; plotted trajectories
start after coarse initialization. The demo runs inversion twice from the same
initialization: without a position
loss, and with a position loss of weight 1. Ground-truth positions in the first
run are used exclusively to report errors. `--refine` adds successive stages
allowing onset, amplitude, and finally the embedding to change. These stages
start from the previous stage's estimate; their curves have separate iteration
axes starting at zero.

Every demo writes:

| File | Contents |
|---|---|
| `map.png` | All stations, true/estimated sources, and trajectories |
| `calls.png` | Clean STFT powers, learned fields, and individual Gaussian contours |
| `stations.png` | Observed/rendered spectrograms and signed residuals for every station |
| `landscape_*.png` | Conditional spatial reconstruction loss for each event |
| `losses.png` | Call MSE, station reconstruction loss, and location error |
| `metrics.json` | Configuration, ground truth, and separate metrics for each inversion |
| `experiment.pt` | Decoder state, embeddings, waveforms, power fields, axes, and trajectories |

Load the tensor bundle using `torch.load(path, weights_only=True)`. Reconstruct
the decoder with `SplautGenerator(Config(**metrics["config"]))` and load its
`generator` state dictionary. A spatial landscape varies one event's position
while holding its other parameters and all other events at their final
unsupervised estimates. It is a conditional loss surface, not a posterior or
proof of global uniqueness. The expensive grid resolution is adjustable with
`--grid-size`. Only the heatmap colour scale clips the upper 5% of losses, so
extreme source/station coincidences do not obscure interior valleys; the tensor
bundle preserves the unmodified values.

`compare` changes one variable at a time around a fixed single-event baseline:
corner/irregular/clustered geometry, zero/low/moderate noise, tonal/sweep/pulse
calls, and 4/6/8 stations. All cases share the same source, initial estimate,
seed, and decoder trained on the three isolated calls. It writes
`comparison.json`, `comparison.png`, and a spatial landscape for every case.
The clustered layout deliberately illustrates difficult geometry. All primary
simulation and localization entry points require at least four distinct,
non-collinear stations.

## Conventions and limits

- Coordinates are metres in a 300 × 300 m square; times are seconds, frequencies
  are Hz. Sound speed is 343 m/s, with perfect station synchronization.
- The simulator's amplitude gain is `1 / (distance + 1 m)`. The renderer's power
  gain is its square. Source waveform amplitude is squared before being passed
  as an event's renderer amplitude. `--noise` is an absolute station waveform
  standard deviation, after attenuation; it is not an SNR in dB.
- STFTs use 8 kHz audio, a normalized 512-sample Hann FFT, an 80-sample hop,
  centered windows, and zero padding. Power tensors are `[station, time,
  frequency]`; source fields are `[event, relative_time, frequency]`.
- Waveform delays use a phase ramp on a zero-padded FFT, preserving fractional
  samples without the frequency-dependent amplitude error of linear
  interpolation. The renderer directly evaluates Gaussian fields at continuous
  time coordinates; it never rounds propagation delays to frames.
- Gaussian centres are bounded by duration and the Nyquist band, widths are
  positive and bounded, and weights are nonnegative. Gaussian tails can extend
  beyond the event window; widths are bounded to control this extension.
- Training minimizes power MSE divided by a fixed training-peak squared for
  numerical scaling. Plots report raw call power MSE. Localization reconstruction
  loss is MSE divided by the mean squared observed power. Supervised position and
  onset terms use area and scene-duration scales respectively.
- Waveform mixing and additive power rendering differ for overlapping coherent
  calls. A finite Gaussian representation and STFT windowing introduce further
  approximation. Inspect residuals and location errors separately.
- Call identities/associations, durations, onsets, amplitudes, and learned call
  embeddings are known for initial inversion. This is direct event optimization,
  without automatic event detection or inference for unseen calls. Local minima,
  weak signals, geometry, and flexible joint refinement can all harm localization.

## Package layout and validation

`splauts/simulator` contains the independent waveform simulator;
`splauts/model` contains the event bank, shared MLP, and propagation renderer;
`splauts/training` contains acoustic fitting, inversion, and spatial grids;
`splauts/visualization` contains scripted Matplotlib diagnostics.

The tests cover fractional waveform delays and amplitude preservation, station
geometry, gains/noise/ground truth, coherent waveform overlap, decoder bounds,
finite-difference checks of position/onset/amplitude/all Gaussian parameters,
sub-frame propagation, permutation invariance and composition, call learning,
unsupervised-position recovery from exact and waveform observations, known
overlapping events, and optional joint refinement. Seeds are fixed; these checks
are controlled examples rather than a broad accuracy benchmark.
