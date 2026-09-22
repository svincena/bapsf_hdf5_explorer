# Spectral Analysis

Open **Spectral Analysis…** beside Langmuir in the main browser. The existing
component assignments supply A and optionally B, in selector order. Set unused
components to **None**. Scalar/Absolute display modes treat these as scalar
signals; Magnitude/Vector modes identify vector components. The tool analyzes
the selected channel waveforms, not the absolute value or magnitude displayed
by the browser. Two-dimensional geometry is required for vector arrows.

The input is the browser's **currently processed dataset**, including applied
gain, integration, smoothing, baseline correction, and shot averaging. Unapplied
checkbox changes in the main window have no effect. The dialog displays that
processing history. Imported channels already share dimensions, shot numbers,
motion coordinates, and sample intervals; the importer rejects incompatible
channels. Spectral analysis additionally requires a uniform, finite time base.

## Workflow

1. In **Estimate**, choose the representative position/case/shot and interval.
   Drag on either input trace, use the navigation toolbar, or edit inclusive
   times/sample indices. These selections stay synchronized.
2. Set segment length, FFT length, overlap in samples, window, detrending, and
   segment averaging. Sampling rate is read from the time coordinates. FFT
   length may exceed segment length for zero padding. Maximum covariance lag
   controls the stored lag range, in samples.
3. **Process This Point/Shot** previews that location. **Process All** computes
   every position, case, and stored repeat in a background worker, with progress
   and cancellation. Nonfinite input or numerical failures at a location produce
   NaN there and a failure summary; other locations remain usable.
4. In **Display**, choose a quantity, complex representation, and frequency (or
   covariance lag). The numerical entry snaps to an actual bin and reports it.
   The start/end frequencies limit the spectrum view and the animation sweep.
   Use the location controls or click the spatial plot to move the spectrum's
   representative location. Plane plots retain the browser's spatial slices.
5. In **Animate**, choose the animation coordinate and field interpretation.
   Play/Pause, Step frame, frame rate, and loop controls operate on cached arrays.
   The title and frame indicator report actual frequency and artificial phase.

With **Average stored shots before spectra** disabled, each repeat is processed
independently. Enabled, the waveforms are averaged at each position/case before
estimation and the shot axis is removed. This is distinct from averaging spectra:
opposite-phase shots can cancel. If the main browser already averaged the shots,
the original repeats are unavailable to this dialog; reset main processing to
analyze them individually.

**Reset View** resets navigation while keeping the analysis interval and settings.
**Exit to Main** retains the dialog, controls, and calculated results. Reopening
with the same dataset and channel assignments reuses them. Changing inputs
creates a new dialog; estimator settings persist. Changing an estimator or the
interval invalidates spectra. Changing location, quantity, representation,
frequency, phase, color map, or animation mode does not run another FFT.

## Numerical conventions

All stored frequencies are in Hz, time/lag coordinates in seconds. The shared
Dataset currently has a single physical unit for all channels.

### Power and cross-power

The tool calls `scipy.signal.welch` and `scipy.signal.csd` with explicit window,
segment length, overlap, FFT length, detrending, one-sided output, and
`scaling="density"`. For a segment with window `w` and FFTs `X` and `Y`,

```
S_AA = average(d * abs(X)**2 / (fs * sum(w**2)))
S_AB = average(d * conj(X) * Y / (fs * sum(w**2)))
```

The one-sided factor `d` is 2 at interior bins, and 1 at DC and an even-length
FFT's Nyquist bin. Thus PSD has units `input-unit²/Hz`; summing the density times
bin spacing gives the window-weighted mean-square signal after detrending.
Zero padding increases the number of bins, not the independent resolution.
Windows are SciPy's periodic/DFT-even windows.

Mean and bias-corrected median segment estimates follow SciPy. For complex median
estimates, SciPy takes real and imaginary medians separately. These robust
estimates need not yield a positive-semidefinite spectral matrix; resulting
coherence can exceed one and is deliberately not clipped. Mean averaging is the
default. With only one segment, nonzero two-signal coherence is identically one;
multiple segments are necessary for a useful statistical estimate.

```
complex coherency = S_AB / sqrt(S_AA * S_BB)
magnitude-squared coherence = abs(complex coherency)**2
cross-phase = arg(S_AB)
```

**Positive cross-phase means B leads A**: for `A=cos(ωt)` and
`B=cos(ωt+θ)`, the measured phase is `+θ`. Phase is undefined where cross-power
vanishes; those values are NaN. Coherency is NaN when its power denominator is
zero. Phase displays can use radians or degrees. Complex cross-power and
coherency stay complex in memory; magnitude, phase, real, imaginary, and squared
magnitude are display transformations. Squared cross-power has squared PSD
units and is not another PSD.

### Covariance

Covariance is a **time-domain lag quantity**, not a frequency spectrum. Both
signals have their whole selected-interval means removed, independently of the
Welch detrending choice. With `N` selected samples,

```
R_AA[k] = sum_n A_centered[n] * A_centered[n+k] / N
R_AB[k] = sum_n A_centered[n] * B_centered[n+k] / N
```

Only overlapping samples contribute; the denominator remains `N` (the biased
covariance estimator). SciPy's FFT correlation computes the sums. At zero lag,
`R_AA` is population variance and `R_AB` is population covariance. Positive lag
means B follows A. Units are `input-unit²`. Frequency and phase animations are
disabled for covariance; select a lag to inspect its spatial structure.

### Scalar animations

**Frequency at fixed phase** indexes the requested available bins, optionally
skipping every Nth bin. **Phase at fixed frequency** uses N equally spaced phases
over one cycle, without duplicating the endpoint. The phase control supplies the
fixed phase or starting phase, respectively.

**Quantity representation** sweeps scalar magnitudes, phases, coherence, or other
selected display values. **Phase projection**, with amplitude **Quantity**, uses
`real(conj(Z) * exp(i*phase))` for complex cross-power or coherency. With amplitude
**A** or **B**, it instead uses

```
sqrt(S_AA or S_BB) * cos(phase - arg(S_AB))
```

Here the amplitude is an **amplitude spectral density**, in `input-unit/√Hz`,
not a reconstructed physical oscillation amplitude. The GUI identifies the
amplitude source. The animation is an artificial phase visualization and does
not claim to reconstruct the original waveform.

### Vector animations

Power spectra do not contain each component's phase relative to a common clock.
The tool therefore also stores **coherent peak phasors** from SciPy's STFT, using
the same segment/window/overlap/FFT/detrending parameters, no boundary extension,
and no padded final segment. For segment start `t_j` relative to the interval
start, STFT value `Z_j(f)` (scaled by `sum(w)`),

```
F(f) = d * mean_j(Z_j(f) * exp(-2πi*f*t_j))
```

This mean is always a complex arithmetic mean, independently of the Welch
mean/median choice. It preserves coherent phase across segment starts, positions,
and components. A bin-centered coherent cosine with peak amplitude `a` and phase
`θ` gives `F=a*exp(iθ)` at its positive-frequency bin. The same DC/Nyquist factor
exceptions apply. The common clock is the selected interval's first sample.

**Vector components** displays `real(F_A*exp(i*phase))` and
`real(F_B*exp(i*phase))` in the assigned horizontal/vertical directions, in input
units. Its positive artificial phase advances the ordinary cosine phasors in
time; this differs deliberately from the scalar *cross-phase referenced*
projection above. Separate component phases preserve polarization and spatial
phase variation. Incoherent fluctuations may average away in these coherent
phasors while remaining visible in Welch power. The plot explicitly says
“Coherent peak vector”; this is a phase-coherent estimator, not `sqrt(PSD)`.

## Storage and reuse

`spectral.Settings`, `spectral.process`, and `spectral.Result` are independent of
Qt. The result keeps the original geometry and case/shot axes, explicit frequency
and lag coordinates, complex cross-power, coherent component phasors, and a
failure list. Coherency and phase derive from cached spectra only when needed.
Point processing selects traces before repeat averaging. Full processing uses
bounded vectorized chunks to limit temporary STFT memory. Persistent output
memory still scales with the number of locations/shots and requested bins.

Only a requested real spatial frame is adapted to the existing singleton-time
Dataset convention already used by Langmuir. `plotting.render_derived` handles
maps, slices, curves, arrows, and artist updates; frequency never masquerades as
time in the stored spectral result. Animations generate frames on demand, with
no space × frequency × phase allocation. Fixed animation scales scan the selected
cached frames once. The standard plot toolbar supplies navigation and image
saving. Spectral results currently remain in memory; the main window's data and
MP4 exporters continue to operate on the main browser dataset.

Primary references: [SciPy Welch](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html),
[SciPy CSD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html),
and [SciPy correlate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html).
