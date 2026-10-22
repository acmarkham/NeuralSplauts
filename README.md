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

## Mixture-only self-supervision

This variant learns a fresh shared decoder directly from station mixtures,
without isolated calls, pretrained embeddings, call identities, associations,
positions, onsets, or source amplitudes. Run the requested 12-case matrix:

```powershell
python -m splauts mixtures --output outputs/mixtures
```

It tests **2/3/4/5 calls × 4/6/8 stations**. A smaller experiment:

```powershell
python -m splauts mixtures --events 3 --stations 4 8 --warmup-steps 250 --steps 400 --output outputs/mixture_subset
```

The fitter receives only `MixtureObservations(power, stations, times,
frequencies)`, a supplied call count, and physical/configuration priors. The
common 0.4 s duration is a prior; event count and duration discovery are not
tested. The procedural call family and all source labels are withheld. Each
case starts with a new randomly initialized MLP, not the decoder trained in
`demo`.

Initialization aligns the mixtures using candidate propagation delays,
back-projects a shared nonnegative source field, and greedily subtracts proposed
events. A 13 × 13 location grid and 500 Hz proposal window are data-independent
priors. The initial fields can contain other calls; they are provisional
back-projections of mixtures, **not clean training targets**. A 250-step warmup
learns these provisional shapes. During 400 subsequent mixture-optimization
steps, the first 75 hold shapes fixed while refining geometry, onset, and power;
then the decoder and event embeddings are also optimized.

Generated fields have unit RMS on the common event grid. Each event has a
positive, learned RMS power scale, initialized from its mixture back-projection.
This fixes the redundant decoder/amplitude scale and prevents a weak residual
candidate from overwhelming the scene. Proposal coordinates use grid-cell
centres to avoid initializing a source exactly on a station.

The objective combines equally weighted station-balanced power NMSE and
square-root-power error, plus a small spectral-extent penalty. Both reconstruction
terms use **only the observed mixtures**. The lowest mixture objective selects
the saved fit; source labels never select an iteration or initialization.
Source positions and appearance can still be ambiguous, especially for weak,
overlapping, or periodic calls.

`outputs/mixtures` contains:

| File | Contents |
|---|---|
| `results.json`, `results.csv` | All case settings, separate reconstruction and source/localization metrics |
| `summary.png` | The 4 × 3 matrix of location error, mixture NMSE, recovery fraction, and source-field NMSE |
| `calls_K_stations_M/components.png` | Mixture-derived initialization, learned source fields, and individual splauts |
| `calls_K_stations_M/station_mixtures.png` | Observed/rendered mixtures and every station's residual |
| `calls_K_stations_M/evaluation.png` | Post-fit source-position matching and error histories |
| `calls_K_stations_M/source_evaluation.png` | Withheld source fields versus inferred fields on absolute emission-time axes |
| `calls_K_stations_M/losses.png` | Initialization and mixture-only objective histories |
| `calls_K_stations_M/fit.pt`, `metrics.json` | Tensor checkpoint, assumptions, withheld inputs, proposals, estimates, and metrics |

Evaluation happens **after** fitting. It matches inferred events to true events
by minimum total position error over all permutations. It also compares inferred
emitted fields with withheld sources, so a good mixture reconstruction does not
automatically count as successful unmixing. Emission-centroid error is reported
because onset and relative splaut centres have a time-origin ambiguity. The
reported source NMSE compares fields on absolute emission-time axes, including
their recovered power scales.

Saved `fit.pt` files load with `weights_only=True`. Their `splauts` tensors are
normalized fields; render them with the saved `amplitudes`, positions, and
onsets. Re-decoding the saved MLP/latents requires `normalized_splauts` on the
saved event-relative grid. `evaluate_saved_sources` can regenerate post-fit
source comparisons from existing checkpoints without retraining or changing
model selection.

The experiment uses scene seed 7 and independent fit seed 101, zero added noise,
32 splauts per event, and 16 latent dimensions. Calls at higher counts extend the
same scene; the same calls are observed across station counts. Four stations use
the corners, and six/eight use the square perimeter. This is one controlled seed,
not an accuracy benchmark. The CLI uses four CPU threads for this variant and
flushes numerically negligible subnormal values to avoid slow Gaussian-tail
arithmetic.

Recorded seed-7 results are saved in
[mixture-only-seed7.json](Design/Experiments/mixture-only-seed7.json). Sources
localized within 10 m, using post-fit permutation matching:

| Calls | 4 stations | 6 stations | 8 stations |
|---|---|---|---|
| 2 | 2/2 | 2/2 | 2/2 |
| 3 | 2/3 | 2/3 | 2/3 |
| 4 | 3/4 | 3/4 | 3/4 |
| 5 | 4/5 | 4/5 | 4/5 |

Every tonal/sweep source localizes within 0.28 m. The weak pulse call is missed
in every scene containing it, including with eight stations. Mixture NMSE ranges
from 0.0086 to 0.1342, yet mean source-field NMSE reaches 5.16 in a failed
decomposition. These location counts therefore do not imply complete acoustic
unmixing. The initial unstable scaling formulation is preserved separately in
`outputs/mixtures_first_pass`; the tracked results retain its metrics as well.

## Experiments and outputs

### Supervised pretraining with a frozen decoder

```powershell
python -m splauts pretrained --output outputs/pretrained
```

This trains once on 96 isolated procedural calls (32 of each family, seed 23),
then freezes that **same decoder for all 12 held-out mixture cases**. Independent
isolated validation calls use seed 29; mixture scenes retain seed 7. Training
and test call specifications are checked for disjointness. Validation does not
select the checkpoint, and no test source position, onset, amplitude, identity,
call parameters, or true embedding is given to the localizer.

To run the stages separately or reuse the checkpoint:

```powershell
python -m splauts pretrained --pretrain-only --output outputs/call_prior
python -m splauts pretrained --prior outputs/call_prior/prior.pt --output outputs/frozen_reuse
python -m splauts pretrained --events 3 5 --stations 4 8 --output outputs/frozen_subset
```

Pretraining supervises unit-RMS source power fields with power and compressed
power losses. Acoustic coordinates describe frequency, bandwidth, modulation
rate, and pulse rate, plus a family code. The decoder uses the same shared MLP
and Gaussian representation, with frequency centres expressed relative to the
latent frequency coordinate. This makes frequency translation explicit while
the neural network learns relative spectrotemporal shape. Coordinates are
bounded to the pretraining domain during inversion; unused latent dimensions
remain zero.

A generic template bank is decoded from the frozen network, without test-call
parameters. Propagation-aware template correlation searches positions, emission
times, families, and acoustic codes in the station mixtures. Each candidate is
refined for 80 differentiable steps before subtraction. The candidates are then
jointly refined for 400 steps, optimizing only positions, onsets, RMS powers,
and continuous acoustic coordinates. Candidate families are inferred from the
mixtures and held fixed during continuous refinement. Count and the common
0.4 s duration remain supplied priors.

Decoder parameters have `requires_grad=False`, are excluded from every inversion
optimizer, and must accumulate no gradients. A SHA-256 fingerprint checks every
parameter and buffer before and after each fit. Gradients still pass through
the frozen decoder into the acoustic coordinates.

`outputs/pretrained/prior.pt` and `pretraining.json` preserve the model, training
specifications, generic latent bank, seeds, and representation metrics.
`pretraining.png` and `validation_calls.png` inspect training and independent
isolated-call reconstruction **using known codes for that diagnostic only**.
Per-case outputs follow the mixture-only layout above. The sweep also writes
`comparison.png` against the recorded mixture-only baseline when the scene
configuration matches. All clean test-source comparisons are post-fit audits.

This comparison changes the decoder prior/parameterization and candidate
initialization as well as freezing the network. It is not an isolated ablation
of freezing alone. A 25 × 25 position grid and 25 Hz template frequency spacing
are defaults for this variant; coarse banks can miss narrow calls, and greedy
candidate selection can duplicate a strong call or miss a weak one. More
training, a frozen appearance prior, or extra stations do not guarantee that
the inverse optimization reaches the correct solution.

The [recorded frozen-prior run](Design/Experiments/pretrained-frozen-seed7.json)
has training shape NMSE 0.0069 and independent validation NMSE 0.0172. All 12
decoder checksums are unchanged, and every mixture's calls are disjoint from
the isolated training examples. Sources localized within 10 m:

| Calls | 4 stations | 6 stations | 8 stations |
|---|---|---|---|
| 2 | 2/2 | 2/2 | 2/2 |
| 3 | 2/3 | 2/3 | 3/3 |
| 4 | 3/4 | 3/4 | 3/4 |
| 5 | 3/5 | 4/5 | 4/5 |

The three-call/eight-station case recovers the pulse source within about 0.17 m.
The five-call/four-station case loses a sweep source relative to the previous
variant. Overall recovery remains **33/42 sources**. Several frozen-prior fits
duplicate a strong call instead of recovering a weak one. Average mixture error
improves while average source-field error worsens; the appearance prior alone
does not resolve candidate selection and inverse-optimization failures. The
per-case source comparisons and `comparison.png` retain these failures.

### Harmonic atomic calls

```powershell
python -m splauts pretrained --harmonics 4 --splauts 96 --bank-step 7.5 --pretrain-steps 3000 --output outputs/harmonic4
```

Each waveform sums harmonics 1–4 of the **same instantaneous phase**, with
amplitudes proportional to `h**(-0.7)` and a shared envelope. Tonal FM, sweeps,
and pulse modulation therefore act on the complete call. Harmonic amplitudes
have unit sum of squares, keeping emitted energy approximately equal to the
single-harmonic control. This is an illustrative harmonic oscillator, not a
species-specific vocal-tract model.

`--harmonics`, `--harmonic-rolloff`, and `--frequency-scale` configure the call
domain. The default frequency scale is `min(1, 1.2 / harmonics)`, or 0.3 for four
harmonics. Both fundamental frequency and excursion are scaled. The four-order
prior covers fundamentals 240–750 Hz; scene draws use 270–690 Hz. The highest
harmonic and significant modulation sidebands must remain below the 4 kHz
Nyquist frequency; unsafe settings raise an error instead of dropping orders.
The original one-harmonic defaults and saved checkpoints remain supported.

The neural decoder allocates splauts across harmonic orders. Each centre is
expressed relative to its order times the latent fundamental; the MLP learns
the time structure, relative excursions, widths and power weights. With 96
splauts, each of four bands has 24. All bands share **one acoustic code, one
source position, one onset and one power scale**. Harmonics are never separate
source events. Harmonic count and rolloff are supplied domain assumptions;
family and continuous acoustic parameters are inferred from station mixtures.
The neural output is an atomic spectrotemporal **power field**; waveform phase
is supplied only by the procedural simulator, not generated by the network.
The spectral-extent penalty uses fundamental coordinates so intended spacing
between harmonics is not penalized.

`validation_calls.png` shows true held-out calls, frozen neural fields, and
individual splauts colored by harmonic order. This diagnostic uses known codes;
the subsequent localizer receives mixtures and the frozen decoder. Checkpoints
record harmonic settings and reject incompatible reuse.

For a paired single-harmonic control, use the same lower fundamentals, training
budget, bank spacing and total splaut capacity:

```powershell
python -m splauts pretrained --harmonics 1 --frequency-scale 0.3 --splauts 96 --bank-step 7.5 --pretrain-steps 3000 --output outputs/harmonic_control
```

These two runs retain the same sampled parameters, positions, onsets, amplitudes
and seeds across all 2/3/4/5-call × 4/6/8-station cases. Each prior is trained
independently on isolated calls and frozen throughout its sweep. Comparing to
the earlier high-frequency experiment would change the fundamental distribution
and decoder capacity as well as harmonic richness. The paired control avoids
those changes, though it still measures the complete learned-prior/inversion
pipeline on one scene seed.

If omitted, template-bank spacing scales with the fundamental domain (7.5 Hz
for four harmonics), keeping the search resolution comparable to the original.
Pass `--resume --prior outputs/harmonic4/prior.pt` with the same settings to
retain completed compatible cases after an interruption. Resume checks the
configuration, optimization budget and decoder fingerprint; the final source
audit verifies that saved observations reproduce.

Gaussian evaluation drops tails below `exp(-80)` (about `1.8e-35`), clamping
before exponentiation to avoid slow CPU underflow. Values and gradients inside
12.6 standard deviations are unchanged, and the dropped values are far below
the experiment's loss floors. This numerical guard applies to both variants.

After both sweeps, generate the paired report and plot with:

```powershell
python -m splauts.harmonic_experiments
```

The [recorded paired harmonic experiment](Design/Experiments/harmonic-atomic-seed7.json)
uses 96 isolated training calls, 24 independent validation calls, 3,000
pretraining steps and 96 total splauts per event in both variants. The
four-harmonic prior reaches validation shape NMSE **0.00966**, versus **0.00923**
for the control. Harmonic orders 1–4 separately have mean validation NMSE
0.0081, 0.0167, 0.0221 and 0.0216, so the upper bands are represented as well as
the fundamental. Inspect `outputs/harmonic4/validation_calls.png` for the
true calls, decoded fields and individual splauts.

Sources localized within 10 m by the four-harmonic variant:

| Calls | 4 stations | 6 stations | 8 stations |
|---|---|---|---|
| 2 | 2/2 | 2/2 | 2/2 |
| 3 | 2/3 | 2/3 | 3/3 |
| 4 | 3/4 | 3/4 | 4/4 |
| 5 | 4/5 | 4/5 | 4/5 |

Overall recovery is **35/42**, versus **36/42** for the paired single-harmonic
control. The control additionally recovers the pulse in the three-call,
six-station case; all other recovery counts match. Weak pulse alignment and
spurious/duplicate candidates remain failures. Mean-case source-field NMSE is
0.2133 versus 0.2167, while mixture NMSE is 0.0262 versus 0.0218. The richer
representation works, but does **not** improve localization on this seed.
All 24 fits retain their decoder fingerprints and use test calls disjoint from
pretraining. Source comparisons are post-fit audits. Full figures, checkpoints
and per-case metrics remain under `outputs/harmonic4`,
`outputs/harmonic_control` and `outputs/harmonic_comparison`.

### Moo contours and frequency-dependent attenuation

```powershell
python -m splauts.attenuation_experiments
```

This trains one frozen prior including **moo** calls, then runs 2/3/4/5 calls
at 4/6/8 fitting stations under three propagation conditions (36 fits total).
Scenes cycle moo/tonal/sweep/pulse, so every case includes a moo. All conditions
reuse the same decoder, emitted calls, geometry, onsets, amplitudes and noise
draws. The experiment uses four harmonics, 96 splauts per event, 0.6 s calls,
3 s recordings, seed 7, and fixed station waveform noise standard deviation
`0.0001`. The prior uses 128 isolated training calls (seed 23), 32 independent
validation calls (seed 29), and 3,000 training steps. Each fit uses a 17 × 17
position grid, 80 proposal-refinement steps and 400 joint-refinement steps.

A moo starts at `frequency - bandwidth`, rises smoothly to `frequency` during
the first quarter, sustains it for the middle half, and falls during the last
quarter. A cosine transition gives continuous frequency and slope. Its phase
is the analytic integral of that contour, and every harmonic follows the same
phase. A smooth attack/release surrounds a sustained envelope. The network
learns the complete emitted power field under one event code, position and
onset. Bandwidth controls pitch excursion; duration controls contour timing.
Existing call families and older three-family checkpoints retain their defaults.

The absorption law, in addition to geometric spreading, is:

```text
loss_dB(f, d) = alpha * (d / 1000 m) * (f / 1000 Hz)**p
waveform transmission = 10**(-loss_dB / 20)
power transmission    = 10**(-loss_dB / 10)
```

The default ablation uses `alpha=10 dB/km` at 1 kHz and `p=2`. For example,
at 200 m, the losses at 500 Hz, 1 kHz and 2 kHz are 0.5, 2 and 8 dB.
This is an illustrative absorption law, not a weather-calibrated atmosphere.
Waveform simulation applies its magnitude response on the padded FFT before
fractional propagation delay; dispersion beyond the travel delay is omitted.
The differentiable renderer applies the squared response at each STFT
frequency, with gradients through source-to-station distance. Proposal search
interpolates eight distance-binned spectral template banks; continuous gradient
refinement uses the exact transfer function.

| Condition | Waveform observations | Localizer propagation |
|---|---|---|
| `off` | Geometric spreading | Geometric spreading |
| `modeled` | Spreading + spectral absorption | Same spectral absorption law |
| `ignored` | **Identical observations to `modeled`** | Geometric spreading only |

`off` versus `modeled` measures the combined effects of added distance cues,
loss of high harmonics, interference changes and reduced received SNR.
`modeled` versus `ignored` measures the cost of omitting absorption from the
inverse model. Source energy is unchanged and station signals are not
renormalized. The absorption coefficient and exponent are supplied assumptions,
not estimated from the test calls.

Evaluation separates three questions: position recovery within 10 m; emitted
source-field NMSE against withheld source calls; and prediction at **four new
interior stations that never enter fitting or model selection**. Both noisy and
noise-free signal NMSE are reported at these stations, alongside reconstruction
error at fitting stations. Prediction uses the inferred calls and positions,
not true source parameters. All clean-source and new-station data are accessed
only after each fit is selected.

Outputs in `outputs/moo_attenuation` include `moo_and_absorption.png`, an audible
`moo_example.wav`, frozen prior diagnostics under `prior/`, per-condition
checkpoints and source comparisons, `heldout_prediction.pt` / `heldout_stations.png`
in each case directory, and `ablation.json` / `ablation.png`.
For separate stages or continuation:

```powershell
python -m splauts pretrained --moo --harmonics 4 --splauts 96 --call-duration 0.6 --scene-duration 3 --pretrain-steps 3000 --pretrain-only --output outputs/moo_attenuation/prior
python -m splauts.attenuation_experiments --prior outputs/moo_attenuation/prior/prior.pt
python -m splauts.attenuation_experiments --resume
python -m splauts.attenuation_experiments --audit-only
```

Use `--absorption-db-per-km`, `--absorption-power`, `--noise`, `--seed`,
`--events`, `--stations` and `--conditions` to vary the study, with a separate
`--output` for each protocol. The ordinary `pretrained` and `mixtures` commands
also expose `--moo`, duration options and absorption parameters. Their original
defaults remain unchanged.

The [recorded moo/attenuation ablation](Design/Experiments/moo-attenuation-seed7.json)
retains all 36 fits and their failures. Independent known-code validation NMSE
is **0.00613 for moo calls** and 0.0220 over all four families. Every decoder
fingerprint stays unchanged, and every test call is disjoint from pretraining.

| Propagation condition | Within 10 m | Mean position error (m) | Source-field NMSE | Fitted-station NMSE | Unseen-station NMSE |
|---|---|---|---|---|---|
| Off | 36/42 | 3.86 | 0.1297 | 0.0410 | 0.1694 |
| Absorption modeled | 36/42 | 4.83 | 0.1079 | 0.0375 | 0.1402 |
| Absorption ignored | 36/42 | 4.91 | 0.1356 | 0.0681 | 0.1251 |

Position means pool 42 source instances per condition; NMSE values average
the 12 case means. Unseen-station NMSE compares predictions with the clean
received signal; the fits use noisy station mixtures. Noisy prediction metrics
are also retained in the report. The same five underlying source events recur
across counts/station layouts, so these are not 42 independent scenes.

At this strength, modeled absorption reduces mean source-field and unseen-station
errors by about **17%** relative to no absorption, while localization recovery
stays unchanged and mean position error increases. Weak pulse alignment remains
the failure: all moo, tonal and sweep source instances localize within 0.7 m.

Ignoring absorption worsens emitted-source reconstruction and fitted-station
prediction relative to the correct model, but gives a lower *mean* unseen-station
error in this run. The correct model predicts better in **9/12 cases**. Two
four-station pulse-failure cases dominate that aggregate reversal. The report
includes paired changes and case-level errors to preserve this distinction.
The results describe this single seed, finite decoder and initialization/gradient
optimizer; added spectral distance cues do not guarantee better inverse solutions.

All 36 withheld-station diagnostics reproduce from their selected checkpoints.
`--audit-only` regenerates those artifacts and the summary without retraining or
selecting a different fit. The tests cover the moo contour/phase and harmonic
peaks, analytic dB scaling, waveform transfer, continuous position gradients,
distance/time template interpolation and a source recovered through absorbed
template search, alongside the previous regression tests.

### Known-call demos and comparisons

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

## Real elephant recording: first feasibility pass

The real-data runner selects the eight geographically nearest stations to M04
using CSV `TrackIndex` (one-based WAV channels), converts latitude/longitude to
local east/north metres, and streams the 41-channel floating-point WAV through
anti-aliased resampling to 512 Hz. Install the optional audio dependencies:

```powershell
rtk pip install -e ".[real]"
rtk proxy python -m splauts.real_experiments --output outputs/real_rumbles
rtk proxy python -m splauts.real_controls --output outputs/real_rumbles
rtk proxy python -m splauts.real_controls --injection --output outputs/real_rumbles
rtk proxy python -m splauts.real_tdoa --output outputs/real_rumbles
rtk proxy python -m splauts.real_experiments --report-only --output outputs/real_rumbles
```

Open `outputs/real_rumbles/report.html` for spectrograms, maps, validation scores,
and audio players. The default input is
`real_world/source_data/elephant rumble/multi_aud_20230308_104500.wav`, with the
matching CSV. Use a separate output folder for each configuration. A quick run:

```powershell
rtk proxy python -m splauts.real_experiments --duration 16 --starts 0 8 --steps 60 --warmup-steps 40 --output outputs/real_smoke
```

The default experiment covers all 120 seconds in 29 overlapping eight-second
windows at four-second intervals. Each has enough extra audio for the maximum
travel time within a station-centred square with a 250 m margin. Fits include
this padded support; metrics and audio use the central arrival-time window.
The model uses a 12–120 Hz band, 0.5 s Hann STFTs at 0.0625 s hops, and a
five-second event-duration prior. One- and two-source fits each learn a fresh
shared 48-splaut neural decoder, with 100 provisional-shape warmup and 200 joint
optimization steps. These are source-count capacity assumptions, not detected
animal counts. Audio is never loaded as the entire original 630 MB tensor.

Six stations enter proposal search and fitting: M04, M05, M07, M02, M15 and M09.
M03 and M11 are held out from initialization, fitting and checkpoint selection.
A recording-wide 20th-percentile noise floor is estimated separately at each
station/frequency, including held-out channels for background calibration.
There is one shared power scale, without independent channel normalization.
Propagation assumes synchronized clocks, 343 m/s sound speed, stationary
isotropic sources and waveform gain `1 / (distance + 50 m)`. The 50 m softening
is a modeling prior, not an estimated physical parameter or microphone calibration.

`fit.pt` contains the learned decoder, latent/event state, Gaussian parameters,
observations, per-source received powers, coordinates and axes, and loads with
`torch.load(..., weights_only=True)`. `candidates.csv` contains unlabelled source
positions and emission-time centroids; overlapping windows repeat candidates.
It must not be read as an animal census or a set of independently verified events.
Each case exports M04 mixture-phase masked audio and an exact waveform residual.
Mixture/components/residual share one gain. Both original-rate and 4× playback
copies are saved; accelerated playback changes frequency and duration.

`real_controls` reruns three nonoverlapping windows chosen by fitting error,
with a different initialization seed and with shuffled channel/coordinate
associations. The injection control independently synthesizes two overlapping
harmonic waveforms into the first 16 seconds of measured background at +12 dB
total band power, then fits mixtures without source labels. Its saved source
positions/audio are accessed only for evaluation. `real_tdoa` independently
maps mean normalized pairwise waveform correlation over a 101 × 101 position
grid in the 20–110 Hz band, using only the same six fitting stations. These maps
are localization diagnostics, not calibrated posteriors.

The first run establishes a tractable real-data pipeline and a successful
favorable injection control, but does **not** establish reliable separation or
localization of the original elephants. Station-local noise, flexible event
fields, unknown event counts, calibration, and amplitude/propagation mismatch
remain major problems. Bird-call separation is outside this low-frequency pass;
it needs higher-rate or multiresolution modeling. The recorded experiment and
its quantitative limitations are in
[real-rumbles-first-pass.json](Design/Experiments/real-rumbles-first-pass.json).

## Seismic elephant recording: dense-array feasibility pass

The seismic runner uses the native 500 Hz recording and a fixed propagation
speed of **330 m/s**. It selects stations within 500 m of the acoustic M04
coordinate, then makes a seeded random split: **16 fitting stations, 16 validation
stations, and 181 final test stations**. Only fitting channels enter source
initialization, decoder training, geometry refinement, or checkpoint selection.
Validation chooses the supplied source-count prior per window; test data does
not choose models or propagation settings.

The CSV lists **1,153 tracks**, but the WAV has **1,024 channels**. Available
one-based `TrackIndex` values are assumed to index the WAV directly. The runner
excludes all absent tracks, including 16 within the requested radius, and saves
the complete exclusion list. This is an explicit mapping assumption, not an
independent verification of the recording export.

```powershell
rtk proxy python -m splauts.seismic_experiments --output outputs/seismic_rumbles
rtk proxy python -m splauts.seismic_controls --output outputs/seismic_rumbles
rtk proxy python -m splauts.seismic_controls --injection --output outputs/seismic_rumbles
rtk proxy python -m splauts.seismic_controls --waveform-maps --output outputs/seismic_rumbles
rtk proxy python -m splauts.seismic_experiments --report-only --output outputs/seismic_rumbles
```

Open `outputs/seismic_rumbles/report.html` for candidate vibration audio,
source fields, station maps, unseen-station spectrogram predictions and controls.
`outputs/seismic_data` caches the selected channels and spectral observations,
so subsequent configurations do not decode the entire original WAV again.
`--resume` retains completed fits after checking recorded settings. Use a new
output directory after implementation changes. To reproduce the coarse
propagation comparison, using the same cached data and spatial split:

```powershell
rtk proxy python -m splauts.seismic_experiments --starts 28 36 44 --warmup-steps 80 --steps 150 --grid-size 13 --spreading 1 --output outputs/seismic_core_smoke
rtk proxy python -m splauts.seismic_experiments --starts 28 36 44 --warmup-steps 80 --steps 150 --grid-size 13 --spreading 2 --output outputs/seismic_core_spherical_smoke
```

The recorded experiment fits K=1 and K=2 in each of 29 overlapping eight-second
windows, with a four-second stride and a five-second source-duration prior.
Spectra use a one-second Hann window, 0.1 s hops, and the 8–80 Hz band. A fixed
20th-percentile spectral background is estimated separately per station across
the recording; validation/test channels participate only in their own stationary
background calibration. A common numerical power scale comes from fitting
stations. Channels are not individually normalized or retrospectively fitted
with test-station transient gains.

A fresh shared MLP produces 64 Gaussian splauts per event, with 100 provisional
mixture-shape warmup and 200 joint-fit iterations. Travel-time padding supplies
source-field context, but **only core-window arrivals enter proposal scoring and
the optimization objective**. This prevents a louder call in padding from taking
over the fit. The general mixture fitter now accepts an optional Boolean
`time_mask`; existing experiments retain their previous behavior when omitted.

Power spreading is `(distance + 25 m)**(-p)`. A coarse comparison of p=1
(surface-like) and p=2 (spherical) on the 28/36/44 s windows selected p=1 by
median validation NMSE, **0.899 versus 0.950**. This is a pragmatic propagation
choice, not a measurement of the seismic attenuation law. `Config` exposes
`power_spreading_exponent`; waveform simulation, both proposal searches and the
continuous renderer use it consistently. The legacy default remains p=2.

For native data, validation selected K=1 in 17 windows and K=2 in 12. Median
mean-test-station power NMSE is **0.949** (zero excess prediction is 1); 23 of 29
windows improve on that baseline. This is modest reconstruction, not clean
separation. In the three fixed harmonic windows, shuffled geometry worsens test
prediction in all three cases. Changing the decoder initialization moves matched
positions by at most about 10 m in these checks, but this is optimizer sensitivity,
not an accuracy bound. Some inferred fields show clear harmonic rumble structure;
others absorb station-local or non-rumble energy.

An additional check on the same three harmonic windows uses 96 splauts, 400
warmup steps, 800 joint-fit steps and a 17 × 17 proposal grid. It does not give
consistent validation or test improvements; adding optimization alone does not
resolve the native-data gap. These exploratory results remain separate from the
primary experiment, under `outputs/seismic_refined`:

```powershell
rtk proxy python -m splauts.seismic_experiments --starts 28 36 44 --warmup-steps 400 --steps 800 --splauts 96 --grid-size 17 --output outputs/seismic_refined
```

The independent injection adds two overlapping harmonic waveforms at +12 dB
total band power to measured seismic background. With known K/duration and
matched propagation, position errors are **7.8 m and 2.2 m**. Mean unseen-station
power NMSE is **0.0354**, and reference separation SI-SDR improves from
2.8 to 18.4 dB and from -7.0 to 10.0 dB. These favorable injected-call results
must not be read as measured elephant accuracy.

`selected.json` records the validation-selected fits. `candidates.csv` retains all
capacities and overlapping windows; `selected_candidates.csv` retains only the
selected capacities. These are candidate positions and emission centroids, not
independent calls or identified elephants. `fit.pt` saves the shared decoder,
event state, Gaussian parameters, fitting observations, axes and core time mask;
it loads with `weights_only=True`. Test predictions can be regenerated at any
saved station coordinate without fitting an extra gain. Reference audio uses
mixture-phase masks and preserves an exact residual, with one gain for all
components. Original-rate and 4× playback files are provided.

The [recorded seismic experiment](Design/Experiments/seismic-rumbles-first-pass.json)
preserves metrics and assumptions. Remaining issues include unverified channel
mapping, sensor/coupling differences, stationary noise versus transient sources,
event-count/duration discovery, source association across windows, and the lack
of clean sources or source-position ground truth. Dense held-out stations make
these limitations measurable; they do not by themselves resolve identifiability.

## Synthetic sensor self-noise and shared event discovery

This experiment implements a conservative **self + shared** decomposition.
Each fitting sensor starts with its entire observed spectrogram in its own
zero-delay self field, and all shared event gates start at exactly zero. A fresh
shared neural decoder proposes source fields; all Gaussians within one group
share a physical point location and onset. Five candidate groups are an upper
bound, not the true source count supplied to inference.

```powershell
rtk proxy python -m splauts.self_noise_experiments --workers 3
rtk proxy python -m splauts.self_noise_experiments --resume --workers 3
```

Open `outputs/self_noise_spatial/report.html` for the complete experiment,
ablations, source maps and true/inferred self/shared spectrograms. A shorter run:

```powershell
rtk proxy python -m splauts.self_noise_experiments --seeds 7 --noise-levels 1 --steps 300 --warmup-steps 100 --primary-only --output outputs/self_noise_quick
```

For observed station power `Y`, shared prediction `P`, and local power `S`:

```text
P = sum of physically propagated neural source fields
S = max(Y - P, 0)
L = 10 L_render + L_self + 0.08 L_space + 4 L_support
```

`S` is a direct per-sensor spectrogram with no source position or travel delay.
It is defined by energy conservation rather than an unrestricted optimization
that could reduce the self penalty simply by discarding observations. A good
shared source replaces self energy at multiple sensors. A bad shared source
overpredicts energy somewhere; nonnegative self cannot cancel that error.

`L_render` is station-balanced power NMSE of `P + S` versus `Y`.
`L_self` is the mean fraction of observed energy still in self, with fixed
per-station normalization. Shared gates `a_k` are measured in those same
station-balanced received-energy units, fixing source/decoder scale redundancy.

The spatial cost is a smooth occupied-location count:

```text
L_space = sum_k a_k / (tau + sum_j K(x_k, x_j) a_j)
K(x, y) = exp(-||x-y||^2 / (2 * r^2))
tau = 0.005; r = 15 m
```

Separated groups pay separately; coincident groups pay according to their
combined mass. Splitting one source into several time/spectral groups at the
same location therefore does not create extra location costs. This favors a
few compact sites while permitting multiple groups per site. Reported sites
cluster active groups within 15 m, independently of ground truth; a gate is
active above a fixed 0.03 received-energy threshold.

Self-energy and sparsity penalties alone can incorrectly promote a common
stationary sensor-noise spectrum into shared space. `L_support` addresses that
failure: it penalizes gate mass lacking approximately four agreeing sensors.
Agreement combines a spatially distributed rendered footprint with positive
normalized correlation of **temporally centered** spectral templates and station
observations. The default correlation threshold is 0.15 with a soft transition
of 0.03. Sensor gains cancel in this correlation. A common spectral baseline or
a transient at just one sensor is insufficient evidence of a shared event.
The tests explicitly cover both counterexamples.

Initialization aligns only fitting-station mixtures, removes a stationary median
for proposal construction, and estimates provisional common fields using a
lower station quartile. The actual self/shared losses use the **full original
observations**, not median-subtracted targets. A shared MLP warms on these
provisional mixtures, without isolated calls. Shared gates remain zero during
this representation warmup. Joint fitting gradually strengthens the sparse
activation penalty over its first half, and tests optional new gate activations
every 40 steps. Every checkpoint, including the all-self null model, is selected
using the same **fixed final objective**, not the relaxed optimization objective
or source labels.

The default 19-case study uses eight fitting stations, four unseen interior
stations, two overlapping harmonic moo-like sources, 500 Hz sampling,
0.5 s Hann FFTs, 0.05 s hops, a 6 s recording and the 8–120 Hz band. Each group
has 48 Gaussians and a supplied 2 s duration prior. Propagation uses 330 m/s and
power gain `1 / (distance + 25 m)^2`. Fits use 200 mixture-shape warmup steps and
600 joint steps. Seeds 7, 19 and 31 vary source positions and independent sensor
noise. Seed 7 was a pilot used during development; the other seeds are additional
controlled realizations of the same call family and geometry.

Each sensor has its own independent colored floor, harmonic call-like disturbance
and short burst. Noise is injected directly into that sensor with no propagation
or delay. Sensor RMS gains vary independently. Noise multipliers 0, 1 and 2 scale
waveforms relative to a common clean-signal RMS; they are not fixed per-sensor
SNRs. Primary cases add independently generated signal and noise **powers**, which
matches the renderer's additive model. Separate stress cases add waveforms before
the STFT, including interference that need not admit a nonnegative exact power
decomposition. Three noise-only controls and four loss ablations complete the
study: removing the self penalty, spatial cost, coherence penalty or reconstruction
penalty. The ground-truth noise and source fields never enter fitting.

An exact total reconstruction is not evidence of separation: the initial all-self
model already has zero error. Evaluation therefore measures source localization,
inferred self versus known sensor noise, and shared power prediction at **four
stations excluded from fitting**, compared with their known clean source powers.
Independent held-out noise is not predicted or retrospectively fitted away.
Checkpoints include the learned generator, Gaussian/event state, gates, full
optimization history, inferred self/shared fields and post-fit truth. They load
with `weights_only=True` and can regenerate predictions without training again.

The recorded primary power-additive results pool three realizations at each level:

| Noise RMS multiplier | Sources within 10 m | False/misplaced sites | Mean clean held-out shared NMSE |
|---|---|---|---|
| 0 | 6/6 | 0 | 0.002 |
| 1 | 6/6 | 1 | 0.144 |
| 2 | 2/6 | 4 | 0.208 |

At multiplier 1, all six matched location errors are below 4.8 m. Held-out NMSE
is 0.391 for pilot seed 7, which has an extra location, versus 0.022 and 0.021
for seeds 19 and 31. The mean self-noise NMSE at this level is 0.022. All three
noise-only controls select the all-self model with exactly zero shared energy.
The waveform-addition stress cases recover 5/6 sources within 10 m, with mean
clean held-out shared NMSE 0.169. Higher noise still produces substantial position
failures even when shared-power reconstruction appears reasonable.

The ablations expose the intended loss tradeoffs. Without the self penalty,
no sources activate and everything remains self, despite perfect reconstruction.
Without reconstruction cost, inferred shared energy reaches 3.6 times observed
energy and held-out NMSE rises to 5.52. Removing spatial sparsity increases
active sites from three to four on seed 7, but **improves** held-out NMSE from
0.391 to 0.080. Removing the coherence penalty also improves that single score
to 0.177. Thus fewer sites and stronger agreement are useful priors but are not
uniformly better predictors; finite-data overfitting and hyperparameter choice
remain visible failures. All ablations are retained.

See [self-noise-shared-first-pass.json](Design/Experiments/self-noise-shared-first-pass.json)
for recorded results, including false locations and high-noise failures. This is
a small controlled synthetic study. Four-sensor support, a point-source model,
duration and spatial-radius priors, finite candidate capacity, and power-domain
interference remain limitations; the experiment does not establish performance
for arbitrary noise or real elephant data.

## Real seismic self/shared trial: 30–50 seconds

Run the self/shared model on four overlapping 8 s windows, with 16 fitting,
16 validation and 181 evaluation-only stations near M04:

```powershell
python -m splauts.seismic_self_shared --resume
python -m splauts.seismic_self_shared --starts 38 --controls restart geometry_shuffle time_shuffle --resume
python -m splauts.seismic_self_shared --audit-only
python -m splauts.seismic_self_shared --report-only
```

This reuses `outputs/seismic_data`, verifies its source-file signature and station
ordering, and recomputes **full original spectral power** from cached native
waveforms. There is no floor subtraction before initializing self. Fits use
8–35 Hz, 330 m/s, power spreading exponent 1, five possible groups of 48 Gaussian
splauts, a 5 s duration prior, and the synthetic study's final loss weights.
Five seconds of propagation context are retained on each side, but only core
arrival frames enter proposals, normalization, support and the joint loss.
All fits start all-self; checkpoints minimize the fixed full loss.

The [local report](outputs/seismic_self_shared_low/report.html) contains station
maps, spectrogram decompositions, reference vibration audio at native and 4×
playback, unseen-station scores, checkpoints and control results. It retains a
broad-band 8–80 Hz comparison and two 38–46 s diagnostics:

```powershell
python -m splauts.seismic_self_shared --output outputs/seismic_self_shared_wide --starts 38 --high-hz 80 --resume
python -m splauts.seismic_self_shared --output outputs/seismic_self_shared_no_sparse --starts 38 --spatial-penalty 0 --resume
python -m splauts.seismic_self_shared --output outputs/seismic_self_shared_explore --starts 38 --exploration-steps 200 --resume
python -m splauts.seismic_self_shared --output outputs/seismic_self_shared_explore --starts 38 --controls geometry_shuffle time_shuffle --exploration-steps 200 --resume
```

The transferred optimizer stalls at all-self. Longer spatial exploration can
activate two sites while selecting with the full sparse loss, but does **not**
establish rumble separation: on 38–46 s, test raw NMSE improves to 0.865 while
dynamic NMSE remains 1.001 and temporal correlation is only 0.034. Removing
sparsity gives raw/dynamic NMSE 0.866/1.006. A geometry-shuffled continuation run
also activates two sites. The improvement largely explains slowly varying
background power; visible bursts at reference 13_05 remain in self. The
reported sites are not verified elephant positions. Self+shared reconstruction
at a held-out station is not used as separation evidence.

The extended run initially optimizes with zero spatial penalty, then ramps to
the original 0.08 penalty by step 500. Every iterate is still selected with the
fixed final loss, including the all-self alternative. This is a fit-driven
optimization diagnostic, not a model selected by test scores. The experiment
record is [seismic-self-shared-first-pass.json](Design/Experiments/seismic-self-shared-first-pass.json).
The unverified CSV TrackIndex/WAV ordering and lack of native source/location
ground truth remain limitations. A useful next experiment would reserve the
stationary floor for self and require a birth to predict time-varying power at
other fitting stations before accepting a location.

## Conventions and limits

- Coordinates are metres in a 300 × 300 m square; times are seconds, frequencies
  are Hz. Sound speed is 343 m/s, with perfect station synchronization.
- The default simulator amplitude gain is `1 / (distance + 1 m)`. The default renderer power
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

Mixture-only tests additionally verify learning without source-call access,
invariance to hidden-label changes, decoder/power normalization, five-call
simulation, and evaluation matching independent of component ordering.
