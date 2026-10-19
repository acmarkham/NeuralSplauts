# Implementation Plan: Neural Acoustic Splauts — CPU Prototype

**Status:** Proposed  
**Related decision:** ADR-001 — Neural Spectrotemporal Splauts for Bioacoustic Localization  
**Scope:** Small PyTorch prototype for visualization, experimentation, and validation.

## 1. Goal

Build a minimal end-to-end system that:

1. Generates synthetic animal calls using a procedural call generator inspired by Spore & Skog.
2. Places calls at known locations and emission times in a 300 m × 300 m area.
3. Simulates arrivals at **at least four recording stations**.
4. Trains a neural splaut generator to approximate calls as spectrotemporal intensity fields.
5. Differentiably renders these fields into each station's spectrogram.
6. Recovers source locations and visualizes reconstruction and optimization behavior.

The primary goal is **visual exploration of the solution space**, not production performance.

## 2. Technology and assumptions

| Component | Initial choice |
|---|---|
| Framework | PyTorch on CPU |
| Visualization | Matplotlib; optional Streamlit explorer |
| Simulation | Procedural animal-call generator |
| Area | 300 m × 300 m |
| Recording stations | **Minimum 4; default 4 in a non-collinear, spatially distributed configuration** |
| Sound speed | Constant, 343 m/s |
| Time synchronization | Perfect; no calibration errors |
| Attenuation | Distance-only |
| Noise | Additive Gaussian waveform noise |
| Representation | STFT power spectrogram |
| Neural generator | Small shared MLP |
| Splauts per event | Initially 16–32 |
| Event latent | Initially 16 dimensions |
| Losses | Simple MSE and optional supervised parameter losses |

Suggested starting STFT: 8 kHz sample rate, 512-sample FFT, 80-sample hop (10 ms).

No clock drift, reverberation, frequency-dependent attenuation, moving sources, or cross-station phase modelling.

**Station geometry requirement:** All primary localization experiments must use **at least four stations**. Place them around the observation area rather than in a line. Four stations are a design minimum, not a guarantee of accuracy: localization quality still depends on geometry, source position, signal-to-noise ratio, and event visibility. Experiments with fewer stations are optional diagnostics only, not acceptance tests.

## 3. Implementation components

### A. Synthetic scene simulator

Generate short scenes with 1–3 vocalizations. For each event, sample:

- Source location `(x, y)` and emission onset.
- Procedural call type and synthesis parameters.
- Source amplitude.

Generate source waveforms using the procedural call generator. For each of **M >= 4** stations at position `r_m`, calculate:

$$
\tau_{mk}=\frac{\|\mathbf p_k-\mathbf r_m\|}{343}.
$$

Apply fractional waveform delay, distance-dependent amplitude attenuation, sum events, and add Gaussian waveform noise. Convert each station waveform into a power spectrogram. Preserve clean source spectrograms, event parameters, and station geometry as ground truth.

Use a consistent attenuation convention: if the power gain is

$$
g(d)=\frac{1}{(d+d_0)^2},
$$

then the waveform simulator uses amplitude gain $\sqrt{g(d)}$. Choose a fixed $d_0>0$ to avoid singularities.

Waveform-domain simulation and additive power-domain rendering will not match exactly for overlapping calls; treat this as an intentional modelling approximation.

### B. Neural splaut generator

A small shared MLP maps an event latent `z_k` to `N` axis-aligned Gaussian **splauts**:

$$
F_\theta(z_k)\rightarrow\{(\mu_{ti},\mu_{fi},\sigma_{ti},\sigma_{fi},w_i)\}_{i=1}^{N}.
$$

Use bounded time and frequency centres, positive bounded widths, and nonnegative weights. The generated continuous spectrotemporal intensity is:

$$
S_k(u,f)=\sum_i w_i\exp\left[-\frac{(u-\mu_{ti})^2}{2\sigma_{ti}^2}-\frac{(f-\mu_{fi})^2}{2\sigma_{fi}^2}\right].
$$

All splauts of an event share its geographic position and onset. Splaut temporal centres are relative to onset and constrained to event duration.

### C. Differentiable propagation renderer

For event `k` at location `p_k` and onset `t_k`, render its contribution to station `m` as:

$$
\hat S_{mk}(t,f)=a_k\,g(\|\mathbf p_k-\mathbf r_m\|)\,S_k\left(t-t_k-\tau_{mk},f\right).
$$

Evaluate the Gaussian field directly at continuously shifted time coordinates; **do not round delays to spectrogram frames**. Sum contributions over events to produce one predicted power spectrogram per station.

Verify finite, nonzero gradients with respect to position, onset, and splaut parameters at representative nondegenerate test points.

### D. Training and inversion

**Experiment 1 — Supervised call representation.** Train the generator to reconstruct isolated procedural-call power spectrograms, using a learned per-example embedding or a simple encoder. Start with power-domain MSE:

$$
\mathcal L_{\mathrm{call}}=\operatorname{MSE}(S_{\mathrm{pred}},S_{\mathrm{true}}).
$$

**Experiment 2 — Source localization.** Freeze the call generator initially. Given ground-truth call identity/embedding and onset, optimize only source `(x, y)` to fit all station spectrograms. Then progressively allow onset, amplitude, and embedding to vary.

Use:

$$
\mathcal L=\lambda_s\operatorname{MSE}(\hat Y,Y)+\lambda_p\|\hat{\mathbf p}-\mathbf p_{\mathrm{true}}\|^2+\lambda_t(\hat t-t_{\mathrm{true}})^2.
$$

Normalize terms to comparable scales. Run localization **both with and without** the ground-truth location loss; the unsupervised-position run tests whether propagation-based reconstruction itself provides usable spatial gradients. Supervised labels can be used for training and evaluation, but must not conceal failures of the inverse renderer.

A learned event-detection/inference network is not required in this prototype; directly optimize event parameters.

## 4. Visualization

Provide plots or a small explorer showing:

- A 2D map with **at least four stations**, true and estimated source locations, and optimization trajectories.
- Procedural-call power spectrogram versus generated splaut field, including individual splauts.
- Observed and rendered spectrograms for every station.
- Per-station residual spectrograms.
- Reconstruction-loss heatmap over candidate `(x, y)` positions with other parameters fixed.
- Training loss and localization error over iterations.

The spatial loss heatmap is a priority: it should expose ambiguous solutions, local minima, and sensitivity to station placement.

## 5. Suggested project structure

```text
splauts/
├── simulator/
│   ├── calls.py          # Procedural animal calls
│   └── scene.py          # Multistation waveform simulation
├── model/
│   ├── generator.py      # Neural splaut MLP
│   ├── renderer.py       # Differentiable propagation
│   └── events.py         # Event parameters
├── training/
│   ├── train_calls.py    # Supervised call representation
│   └── localize.py       # Location inversion
├── visualization/
│   ├── plots.py
│   └── explorer.py
├── tests/
│   ├── test_delays.py
│   ├── test_gradients.py
│   └── test_station_geometry.py
└── config.py
```

## 6. Development milestones

| Milestone | Deliverable | Validation |
|---|---|---|
| 1. Simulator | Four-or-more-station synthetic scenes | Correct delays, gains, noise, and ground-truth labels |
| 2. Splaut renderer | Continuous Gaussian intensity field | Correct shapes and finite gradients |
| 3. Call learning | MLP represents procedural calls | Decreasing MSE and interpretable plots |
| 4. Localization | Recover coordinates using >=4 stations | Convergence toward ground truth without a position-supervision term |
| 5. Explorer | Interactive or scripted visualizations | Inspect maps, splauts, observations, residuals, and loss surfaces |
| 6. Multiple events | Overlapping-call reconstruction | Distinct locations recoverable with known event associations |

## 7. Initial experiments

| Variable | Suggested values |
|---|---|
| Stations | **4, 6, 8** (never fewer than 4 for primary localization evaluation) |
| Events | 1, 2, 3 |
| Splauts per call | 8, 16, 32, 64 |
| Latent dimension | 8, 16, 32 |
| Gaussian noise | None, low, moderate |
| Station geometry | Four corners, irregular perimeter, clustered (stress test) |
| Call type | Tonal chirp, sweep, pulse train |
| Inversion | Known call versus jointly estimated call |

Track localization error, spectrogram reconstruction error, and optimization stability. Keep simulation seeds fixed for reproducible visual comparisons.

## 8. Definition of done

The prototype is complete when it can:

1. Generate a synthetic scene with **at least four recording stations** and known event parameters.
2. Visualize ground-truth source positions and observed station spectrograms.
3. Represent isolated calls using neural splauts.
4. Recover a single source location by differentiating through the propagation renderer, without requiring a ground-truth position loss at inference time.
5. Plot the spatial reconstruction-loss landscape and compare station layouts, noise levels, and call types.
6. Demonstrate a basic multi-event example with known event associations.

**Out of scope:** production inference, real-world clock calibration, learned propagation, automatic event discovery, large-scale datasets, and GPU optimization.

The intended outcome is a small, understandable research prototype that tests whether neural splauts provide a useful representation for differentiable bioacoustic localization.
