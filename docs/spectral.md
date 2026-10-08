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

1. In **Trace**, choose the position/case/shot by axis indices, nearest spatial
   coordinates, flat trace index, or global shot number. Click **Locations** to
   choose a position interactively. In **Estimate**, set the analysis interval.
   Drag on either input trace, use the navigation toolbar, or edit inclusive
   times/sample indices. These selections stay synchronized.
2. Set segment length, FFT length, overlap in samples, window, detrending, and
   segment averaging. Sampling rate is read from the time coordinates. FFT
   length may exceed segment length for zero padding. Maximum covariance lag
   controls the stored lag range, in samples. **FFT workers** sets the maximum
   threads for batched FFTs, defaulting to half the detected logical CPUs,
   rounded down, with a minimum of one worker.
   Select one for serial FFTs. More workers can help larger jobs but may slow
   small jobs; compare processing times on representative data.
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
Changing FFT workers retains cached spectra and applies to the next processing
job. This setting persists when reopening the dialog or changing inputs.

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

The **Playback** slider below **Spatial field / animation** seeks directly to
a frequency or phase frame after **Process All**. It follows **Play** and
**Step frame**, displays the current frame number, and respects the selected
frequency bounds/bin stride or phase-frame count. Dragging or using the slider's
arrow, Home, and End keys pauses playback; **Play** continues from that frame.
The frame label in **Animate** reports its actual frequency and phase. Seeking
uses cached spectra and preserves the fixed animation color range. The slider
is disabled before batch results are available, for covariance, and when only
one frame is available.

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
Chunks remain sequential; SciPy FFT workers parallelize independent transforms
inside a batch. The worker setting is scoped to the analysis thread and restored
when processing finishes, fails, or is canceled. Cancellation remains available
between chunks. The per-trace covariance loop remains sequential.

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


## Single-trace spectrograms

Choose one channel for auto-power, or two channels for cross-power, then click
**Make spectrogram**. Two-channel results also retain each channel's auto-power.
The **Spectrogram** control tab has **Estimation** and **Display** pages. It uses
the shared analysis interval from **Estimate / Input traces**, with one stored
trace at the selected location, case and shot. The Welch **Average stored shots**
option does not apply to spectrograms. Any averaging already applied in the main
browser remains part of the input, as identified by its processing history.

**Trace** provides several synchronized ways to select that time series:

- Spatial coordinates snap to the nearest stored x/y/z coordinate. Axis indices
  explicitly select the position, case and shot; they are zero-based.
- The flat trace index uses the dataset's non-time dimension order, in C order
  (last listed axis varies fastest). It is an index in the mapped dataset, which
  can differ from the original digitizer row order.
- Global shot numbers select an exact stored trace when the shot map is
  available. Missing or ambiguous shot numbers are rejected.
- The **Locations — click to choose** plot shows channel A at the middle of the
  interval for the current case and shot. Clicking chooses the nearest stored
  location. This map is available before any spectral processing. Existing
  cached spatial spectral fields also accept location clicks.
- The persistent **Cycle through / Previous / Next** controls browse all traces,
  only spatial locations, only shots, or only cases. Location/shot/case cycling
  preserves the other indices. Wrap can be disabled to stop at an endpoint.

After the first calculation, **Update spectrogram when trace / settings change**
automatically computes the new selection in a background worker. Rapid changes
are combined into one update. The location map stays open during automatic
updates. Turn the option off to make each calculation explicitly. Color, phase,
frequency display and other display choices reuse cached arrays.

### Estimation options and normalization

- Segment length in **samples or milliseconds**, overlap in **samples or
  percent**, and FFT length. FFT length must cover the segment; overlap must
  be smaller than it. Zero padding gives finer frequency bins but does not
  improve the window's intrinsic resolution. The control reports window
  duration, hop duration and FFT bin spacing.
- Periodic windows (default) or symmetric windows: Hann, Hamming, Blackman,
  Blackman-Harris, Nuttall, flat top, boxcar, Bartlett, Tukey, Kaiser, Gaussian,
  and Chebyshev. Tukey exposes α, Kaiser β, Gaussian standard deviation in
  samples, and Chebyshev attenuation in dB.
- Per-window mean subtraction, linear detrending, or no detrending.
- **Density** in input-unit²/Hz or **spectrum** in input-unit². Each window
  produces a separate periodogram; windows are not averaged together in time.
- One-sided real-signal estimates or two-sided estimates centered on zero Hz.
- **None — complete windows** excludes partial windows. Zero, edge-value,
  even-reflection and odd-reflection boundary extension include windows
  centered at the interval start and along the selected hop grid. **Pad final
  incomplete hop** adds a final window when the grid does not fit exactly.
  With no boundary extension, its missing trailing samples are zero-padded;
  otherwise it uses the selected extension. Padded windows can be influenced by
  artificial edge data, including a fully padded final window for some hops.
- **FFT workers** controls SciPy's batched FFT threads, scoped to the background
  analysis job. Temporary window/FFT batches are bounded; cancel is checked
  between batches and progress reports completed and remaining windows.

For the windowed FFTs `A_j(f)` and `B_j(f)`, each time column stores

```
P_AA(f,j) = d * |A_j(f)|² / normalization
P_BB(f,j) = d * |B_j(f)|² / normalization
P_AB(f,j) = d * conj(A_j(f)) * B_j(f) / normalization
```

The normalization is `fs * sum(w²)` for density and `|sum(w)|²` for spectrum.
For one-sided estimates, `d=2` except at DC and an even FFT's Nyquist bin;
for two-sided estimates `d=1`. Positive cross-phase means B leads A, preserving
the existing CSD convention. Complete-window estimates averaged over time agree
with mean Welch/CSD for the same settings. No coherence estimate is implied by a
single unsmoothed time-frequency window.

Time coordinates identify the central sample, `floor(segment_samples/2)`, of
each window, using the imported time origin. This follows the ShortTimeFFT
sample-center convention; for odd-length segments it is half a sample earlier
than the legacy SciPy spectrogram timestamp. Padding can put centers beyond the
selected interval's endpoint. Frequency and time arrays stay explicit.

Only the selected trace(s) are transformed. Results have a 256 MiB allocation
limit; requests exceeding it explain how to reduce FFT length, overlap, or the
interval. This limit is for stored numerical results, not total plot memory.
The current result replaces the previous trace result, rather than accumulating
a full location × shot × case × time × frequency cube.

### Display and saving

Select auto-power A/B or cross-power. Cross-power offers magnitude, phase, real,
imaginary, and magnitude squared representations; phase may use degrees or
radians. Undefined phase at zero cross-power is masked. Color options include
linear, logarithmic, symmetric log (with threshold), and dB scales; reference
power, dB dynamic range, colormap, and automatic or fixed color limits are
editable. Signed and phase displays use linear or symmetric-log color, rather
than taking logarithms of signed values. Power dB is `10 log10(P/reference)`;
the reference uses the selected quantity's units. Display clipping does not
alter stored estimates. Frequency limits and a logarithmic frequency axis are
available; the latter requires a one-sided result and excludes DC.

The plot toolbar supports zoom, pan, and image export. **Save spectrogram data…**
creates an NPZ archive containing `time` (seconds), `frequency` (Hz),
`Auto_power_A`, and, for two channels, `Auto_power_B` and complex `Cross_power`.
Its JSON `metadata` records channel names, trace index, coordinates, global shot,
units, source, preprocessing history, estimator settings, and the phase/time
conventions. It can be loaded with `numpy.load(path, allow_pickle=False)`.
Saving is atomic and does not replace the main browser dataset.

The estimator uses SciPy window, detrending, and FFT primitives, with numerical
checks against [SciPy spectrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html),
[ShortTimeFFT](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.ShortTimeFFT.html),
and [CSD](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.csd.html).
