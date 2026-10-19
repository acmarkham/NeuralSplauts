# ADR-001: Neural Spectrotemporal Splauts for Bioacoustic Localization

- **Status:** Proposed
- **Date:** 2026-10-19
- **Decision owners:** Bioacoustic localization project
- **Scope:** Neural architecture only; synthetic data and PyTorch implementation are separate decisions.

## Context

We have time-synchronized bioacoustic recordings from geographically distributed stations across a roughly 300 m area. Synchronization accuracy is approximately 10 ms. We want to decompose the recordings into a sparse set of localized vocalization events, recovering source position, emission onset, and acoustic characteristics by differentiably reconstructing each station's observations.

Independent time-frequency Gaussian components are too flexible: components belonging to one vocalization may drift apart in location or time. Cross-station carrier-phase reconstruction is not an initial requirement because synchronization and propagation uncertainty make it fragile at this scale.

We call an **audio splat** a **splaut** (plural **splauts**): a localized Gaussian primitive in relative time-frequency space.

## Decision

Adopt a **hierarchical neural acoustic splaut architecture**:

1. Represent each vocalization as one latent event with shared source position and emission onset.
2. Use a shared neural decoder (initially an MLP) to generate a bounded collection of nonnegative time-frequency Gaussian splauts from the event's latent acoustic embedding.
3. Render each event into every recording station using differentiable, continuous-valued acoustic travel-time delays and frequency-dependent power attenuation.
4. Sum event contributions with a capacity-constrained background model to reconstruct station power spectrograms.
5. Initially optimize event parameters directly; add a geometry-aware event inference network after validating the generative renderer.
6. Do not require phase coherence across recording stations in the initial architecture.

## Representation

### Stations and observations

Let station positions in a local metric coordinate system be

$$\mathcal R=\{\mathbf r_m=(x_m,y_m)\}_{m=1}^{M}.$$

Let $Y_m(t,f)\geq 0$ be the observed **power** spectrogram at station $m$. Spectrograms have a shared nominal time axis and a defined linear-frequency grid.

### Events

Represent a scene with up to $K$ candidate events:

$$\mathcal E=\{e_k\}_{k=1}^{K},\qquad e_k=(\mathbf p_k,t_k,d_k,z_k,a_k,q_k).$$

- $\mathbf p_k=(x_k,y_k)$: shared geographic source position.
- $t_k$: shared emission onset.
- $d_k>0$: event duration.
- $z_k\in\mathbb R^D$: acoustic latent embedding.
- $a_k\geq0$: source power scale.
- $q_k\in[0,1]$: candidate-event activation.

An event's splauts **do not** have independent geographic coordinates or absolute onset times.

### Neural splaut decoder

A shared decoder produces at most $N$ splauts per event:

$$F_\theta(z_k,d_k)\longrightarrow\{(\mu_{ki}^{t},\mu_{ki}^{f},\Sigma_{ki},w_{ki})\}_{i=1}^{N}.$$

Each splaut is a two-dimensional Gaussian in event-relative time $u$ and frequency $f$:

$$G_{ki}(u,f)=\exp\!\left[-\tfrac12(\mathbf v-\boldsymbol\mu_{ki})^T\Sigma_{ki}^{-1}(\mathbf v-\boldsymbol\mu_{ki})\right],$$

where $\mathbf v=(u,f)^T$ and $\boldsymbol\mu_{ki}=(\mu_{ki}^{t},\mu_{ki}^{f})^T$.

The emitted spectrotemporal intensity field is

$$S_k(u,f)=\sum_{i=1}^{N}w_{ki}G_{ki}(u,f),\qquad w_{ki}\geq0.$$

Decoder output transforms must ensure:

- $0\leq\mu_{ki}^{t}\leq d_k$ and frequencies inside the modelled band.
- Positive-definite $\Sigma_{ki}$, e.g. via Cholesky factors.
- Bounded temporal widths, plus penalties or truncation for Gaussian tails outside the event window. Bounding centres alone does not bound support.
- Nonnegative weights, with optional learned near-zero splaut activations.

A fixed maximum $N$ is the initial design; adaptive splaut count is deferred.

## Differentiable acoustic renderer

### Travel time

For event $k$ and station $m$, define

$$r_{mk}=\|\mathbf p_k-\mathbf r_m\|_2,\qquad \tau_{mk}=r_{mk}/c+b_m,$$

where $c$ is effective sound speed and $b_m$ is an optional relative station clock offset. Fix $b_1=0$ to establish a reference and regularize other offsets according to the synchronization uncertainty. Constant offsets are the initial assumption; drift is deferred.

### Propagation and continuous temporal projection

Use a nonnegative **power** gain $g_{mk}(f)$, initially a simple distance- and frequency-dependent attenuation law. Its geometric-spreading exponent must be consistent with a power, rather than waveform-amplitude, representation.

The event contribution at station $m$ is

$$\hat S_{mk}(t,f)=q_k a_k g_{mk}(f)S_k(t-t_k-\tau_{mk},f).$$

Evaluate the continuous Gaussian field at shifted time coordinates, without rounding $t_k+\tau_{mk}$ to STFT frames. This provides gradients from reconstruction loss to onset, source coordinates, decoder parameters, and optional clock offsets. Continuous interpolation does **not** eliminate clock uncertainty.

### Composition

$$\hat Y_m(t,f)=B_m(t,f)+\sum_{k=1}^{K}\hat S_{mk}(t,f),$$

where $B_m(t,f)\geq0$ is a restricted-capacity background component (e.g. smooth or low-rank). Prevent the background from absorbing structured calls.

**Approximation:** STFT powers of overlapping waveforms are not exactly additive due to interference terms. This model is an additive power approximation, not a phase-coherent physical waveform synthesizer. Evaluate its adequacy empirically.

## Neural modules and interfaces

1. **Event bank:** differentiable candidate event parameters $\mathcal E$.
2. **Splaut generator:** shared MLP $F_\theta$ mapping acoustic latent codes and duration to relative time-frequency Gaussian parameters.
3. **Propagation renderer:** station geometry, travel-time shifts, power attenuation, optional clock offsets.
4. **Compositor/background:** sums projected event fields and constrained background.
5. **Event inference network (later):** $E_\psi(\mathbf Y,\mathcal R)\to\hat{\mathcal E}$, using station-wise features and geometry-aware aggregation, invariant to station ordering. Event-set ordering must not affect the rendered scene.

Generative interface:

$$\hat{\mathbf Y}=\operatorname{Render}_\theta(\mathcal E,\mathcal R,\mathbf b).$$

Inverse interface:

$$\hat{\mathcal E}=E_\psi(\mathbf Y,\mathcal R).$$

The renderer must work independently of the inference network so direct latent optimization can be validated first.

## Objective functions

Support a composite loss:

$$\mathcal L=\lambda_{rec}\mathcal L_{rec}+\lambda_{sparse}\mathcal L_{sparse}+\lambda_{latent}\mathcal L_{latent}+\lambda_{clock}\mathcal L_{clock}+\lambda_{bg}\mathcal L_{bg}.$$

- **Reconstruction:** robust power-domain and/or compressed-power or log-power spectral discrepancies, with numerical floors; optionally multiple resolutions.
- **Event sparsity:** e.g. $\sum_k q_k$, with additional duplicate-event suppression if necessary. Continuous activations alone do not guarantee discrete selection.
- **Latent prior:** constrain $z_k$ to a plausible learned acoustic distribution.
- **Clock prior:** constrain $b_m$ to plausible relative timing uncertainty.
- **Background prior:** enforce smoothness/low rank/limited explanatory capacity.

Amplitude, attenuation, event activation, and background may be partially confounded; use fixed conventions and regularization. Localization identifiability also depends on station geometry, event audibility, and clock calibration.

## Architectural invariants

- Every splaut belonging to an event shares one source position and onset anchor.
- The event's emitted intensity field is station independent.
- Station arrival-time differences arise from geometry and explicit clock offsets.
- Source position and onset receive usable gradients through continuous time projection.
- Spectral intensity, weights, gains, and background remain nonnegative.
- Covariance matrices remain positive definite.
- Event duration and splaut extent are controlled.
- Reordering stations (with their metadata) or candidate events preserves the predicted scene.
- Unused events can be suppressed without affecting active events.

## Consequences and trade-offs

### Benefits

- Strong structural binding prevents different acoustic components of one event from drifting to unrelated locations.
- Compact event embeddings can encode coherent chirps, harmonics, pulses, and frequency sweeps.
- Continuous temporal projection allows gradient-based location estimation without frame-quantized delays.
- Sparse event representations remain inspectable and potentially interpretable.
- The architecture separates acoustic generation from propagation and inverse inference.

### Costs and risks

- Additive power rendering is not exact for overlapping coherent signals.
- Synchronization uncertainty (~10 ms) limits timing-based localization and should not be confused with spectrogram hop resolution.
- Station geometry and event visibility can make locations ambiguous.
- A flexible decoder/background may reconstruct recordings while producing incorrect locations.
- A fixed maximum event/splaut count may be inefficient or insufficient.
- Frequency-dependent attenuation may be difficult to identify jointly with source spectra and amplitudes.
- Decoder training on unrepresentative calls can bias inverse localization.

## Alternatives considered

1. **Independent geographic time-frequency splauts:** rejected initially because components of one call can drift apart spatially and temporally.
2. **Full waveform/complex phase-coherent renderer:** deferred; cross-station phase is fragile with current synchronization and propagation uncertainty, though within-station waveform timing may later be useful.
3. **Dense implicit $(x,y,t,f)$ neural field:** deferred because it is less naturally sparse and more computationally demanding.
4. **Directly predicting calls without a generative renderer:** rejected as the primary approach because it does not enforce multistation acoustic propagation by construction.

## Validation milestones

1. **Single-event forward rendering:** correct station-specific travel times and nonnegative projected fields.
2. **Gradient checks:** compare analytical/autodiff location and onset gradients against finite differences away from singularities.
3. **Single-event inverse optimization:** recover known locations and onsets from controlled observations.
4. **Multi-event composition:** separate overlapping events without excessive duplicates or spatial drift.
5. **Neural decoder:** learn coherent spectrotemporal structures from compact embeddings.
6. **Inference network:** propose events for unseen observations and station configurations.
7. **Joint refinement:** improve reconstruction and localization without violating invariants.

Evaluate localization error **separately** from spectrogram reconstruction error; a good reconstruction does not imply a correct geographic decomposition. Numerical thresholds and test distributions will be set in the synthetic-data plan.

## Deferred decisions

- Synthetic scene and recording generation, including station layouts, noise, propagation variation, and clock-error distributions.
- Concrete PyTorch modules, tensor shapes, batching, and performance optimization.
- Number of event candidates and splauts per event.
- Latent dimension (16–32 is only a starting hypothesis).
- Full versus axis-aligned time-frequency covariance (initial preference: full 2D).
- Spectrogram window/hop sizes and multiresolution losses.
- Fixed versus adaptive splaut allocation.
- Decoder pretraining and amortized inference training schedules.
- Static sources versus moving-animal trajectories.
- Calibration of source visibility, atmospheric propagation, and clock drift.

## Decision outcome

Proceed with an **event-bound neural splaut generator and differentiable spectrotemporal power renderer** as the architectural baseline. Keep synthetic-data design and concrete PyTorch implementation in subsequent documents. Revisit the additive-power and phase assumptions if validation reveals systematic localization failures.
