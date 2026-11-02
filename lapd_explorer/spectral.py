"""Spectral estimates independent of Qt; all coordinates use seconds and Hz.

Welch/CSD: one-sided density, d*conj(FFT(A))*FFT(B)/(fs*sum(w**2)),
averaged over segments. d=2 except DC and an even FFT's Nyquist bin.
Positive arg(S_AB) means B leads A. Covariances remove the interval mean
and use a fixed N denominator: R_AB[k] = sum A[n]*B[n+k]/N.
Coherent peak phasors are a separate estimator, described in docs/spectral.md.
"""
from dataclasses import dataclass, field, replace
from functools import cached_property
import copy
import os
import numpy as np
from scipy import fft, signal
from .model import Dataset
from .langmuir import interval_slice


SINGLE = ("Auto-power A", "Covariance A")
PAIRED = ("Auto-power B", "Cross-power", "Cross-covariance", "Complex coherency",
          "Magnitude-squared coherence", "Cross-phase")
COMPLEX = ("Cross-power", "Complex coherency")
REPRESENTATIONS = ("Magnitude", "Phase", "Real", "Imaginary", "Magnitude squared")


@dataclass
class Settings:
    interval: tuple | None = None
    nperseg: int = 256
    nfft: int = 256
    overlap: int = 128
    window: str = "hann"
    detrend: str = "constant"
    average: str = "mean"
    average_shots: bool = False
    phase_reference_confirmed: bool = False
    max_lag: int = 128
    fft_workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 1) // 2))


def default_segment_samples(samples):
    """Start with roughly half the record and two or more half-overlap windows.

    This is a resolution/averaging compromise, not a physical frequency choice.
    The GUI exposes duration and bandwidth so users can choose that explicitly.
    """
    return max(4, 2**int(np.floor(np.log2(max(4, 2*samples//3)))))


def validate_unmixed(data):
    # History fallback also protects portable datasets written before structured
    # provenance was introduced. Per-trace gain/integration/filtering is allowed.
    mixed_history = any(step.startswith("Spatial averaging:") or
                        (step.startswith("Average ") and "shot" in step.lower())
                        for step in data.history)
    if data.metadata.get("waveform_mixing") or mixed_history:
        raise ValueError("Spectral estimates require unmixed acquired traces. Reset waveform shot/spatial "
                         "averaging in the main browser; use Average spectra over matched shots here.")


def segment_support(samples, settings):
    """Segment count, approximate independent averages, and window bandwidth factor.

    The overlap correction assumes stationary Gaussian samples and does not
    establish independence between experimental shots or stationarity in time.
    """
    s = settings
    hop = s.nperseg-s.overlap
    count = 1 + (samples-s.nperseg)//hop
    window = signal.get_window(s.window, s.nperseg)
    energy = np.dot(window, window)
    lags = np.arange(1, min(count, (s.nperseg-1)//hop+1))
    offsets = lags*hop
    if len(offsets) > 16:
        # Near-total overlap must not turn a GUI support estimate into O(N²).
        rho = signal.correlate(window, window, method="fft")[s.nperseg-1+offsets]/energy
    else:
        rho = np.array([np.dot(window[:-offset], window[offset:])/energy for offset in offsets])
    correction = 1 + 2*np.sum((1-lags/count)*rho**2)
    return count, count/correction, energy/window.sum()**2


def sampling_rate(time):
    time = np.asarray(time)
    if time.ndim != 1 or len(time) < 4 or not np.all(np.isfinite(time)):
        raise ValueError("Spectral analysis needs at least four finite time coordinates.")
    dt = np.diff(time)
    if dt[0] <= 0 or not np.allclose(dt, dt[0], rtol=1e-5, atol=abs(dt[0])*1e-8):
        raise ValueError("Spectral analysis requires a uniformly sampled, increasing time base.")
    return 1 / dt[0]


def validate(data, names, settings):
    validate_unmixed(data)
    if len(names) not in (1, 2) or len(set(names)) != len(names):
        raise ValueError("Select one or two distinct channels; set the third component to None.")
    if any(n not in data.channels for n in names):
        raise ValueError("Select input channels in the main browser.")
    if any(data.channels[n].shape != data.shape for n in names):
        raise ValueError("Channels must share dimensions and the same time base.")
    fs = sampling_rate(data.coords["time"])
    s = settings
    window = interval_slice(data.coords["time"], s.interval, minimum=4)
    for name in ("nperseg", "nfft", "overlap", "max_lag", "fft_workers"):
        value = getattr(s, name)
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer.")
    if s.fft_workers < 1:
        raise ValueError("FFT workers must be at least 1.")
    if not 4 <= s.nperseg <= window.stop-window.start:
        raise ValueError("Segment length must be at least 4 and no larger than the selected sample count.")
    if s.nfft < s.nperseg:
        raise ValueError("FFT length must be at least the segment length.")
    if not 0 <= s.overlap < s.nperseg:
        raise ValueError("Overlap must satisfy 0 ≤ overlap < segment length.")
    if not 0 <= s.max_lag < window.stop-window.start:
        raise ValueError("Maximum covariance lag must be smaller than the selected sample count.")
    if s.window not in ("hann", "hamming", "blackman", "boxcar"):
        raise ValueError("Unsupported window function.")
    if s.detrend not in ("constant", "linear", "none") or s.average not in ("mean", "median"):
        raise ValueError("Unknown detrending or segment averaging method.")
    return fs, window


def selected_data(data, names, average_shots=False):
    """Geometry adapter only; never combine waveforms before spectral estimation."""
    channels = {n: data.channels[n] for n in names}
    if average_shots and "shot" in data.dims:
        key = tuple(0 if d == "shot" else slice(None) for d in data.dims)
        return replace(data, channels={n: a[key] for n, a in channels.items()},
                       dims=tuple(d for d in data.dims if d != "shot"),
                       coords={d: c for d, c in data.coords.items() if d != "shot"},
                       shot_numbers=None, history=data.history + ("Ensemble of matched per-shot spectra",))
    return replace(data, channels=channels)


def representation(values, kind="Magnitude", degrees=False):
    if kind == "Phase":
        return np.where(abs(values) > 0, np.angle(values, deg=degrees), np.nan)
    if kind == "Real":
        return values.real
    if kind == "Imaginary":
        return values.imag
    if kind == "Magnitude squared":
        return abs(values)**2
    if kind != "Magnitude":
        raise ValueError("Unknown complex representation.")
    return abs(values)


@dataclass
class Result:
    source: Dataset
    names: tuple
    settings: Settings
    frequency: np.ndarray
    lags: np.ndarray
    arrays: dict
    failures: list
    valid_shots: np.ndarray | None = None

    @cached_property
    def support(self):
        _, crop = validate(self.source, self.names, self.settings)
        return segment_support(crop.stop-crop.start, self.settings)

    def effective_averages(self):
        counts = np.ones(self.source.shape[:-1]) if self.valid_shots is None else self.valid_shots
        return counts * self.support[1]

    def coherence_threshold(self):
        """Approximate 95% pointwise null threshold, not a multiple-test correction."""
        effective = self.effective_averages()
        return 1 - np.exp(np.log(.05)/np.maximum(effective-1, np.finfo(float).eps))

    @property
    def dims(self):
        return self.source.dims[:-1] + ("frequency",)

    @property
    def coords(self):
        return {**{d: c for d, c in self.source.coords.items() if d != "time"},
                "frequency": self.frequency}

    @property
    def quantities(self):
        return SINGLE + (PAIRED if len(self.names) == 2 else ())

    def values(self, quantity, key=Ellipsis):
        if quantity not in self.quantities:
            raise ValueError("This quantity requires two selected channels.")
        if quantity == "Complex coherency" or quantity == "Magnitude-squared coherence":
            denominator = np.sqrt(self.arrays["Auto-power A"][key]) * np.sqrt(self.arrays["Auto-power B"][key])
            coherence = np.full(denominator.shape, np.nan + 1j*np.nan)
            supported = np.broadcast_to((self.effective_averages() > 1+1e-10)[..., None],
                                       self.arrays["Auto-power A"].shape)[key]
            if self.settings.average == "mean":
                np.divide(self.arrays["Cross-power"][key], denominator, out=coherence,
                          where=(denominator > 0) & supported)
            return abs(coherence)**2 if quantity == "Magnitude-squared coherence" else coherence
        if quantity == "Cross-phase":
            cross = self.arrays["Cross-power"][key]
            threshold = np.broadcast_to(self.coherence_threshold()[..., None],
                                        self.arrays["Auto-power A"].shape)[key]
            reliable = self.values("Magnitude-squared coherence", key) > threshold
            return np.where((abs(cross) > 0) & reliable, np.angle(cross), np.nan)
        return self.arrays[quantity][key]

    def coordinate(self, quantity):
        return self.lags if "covariance" in quantity.lower() else self.frequency

    def units(self, quantity, kind="Magnitude", degrees=False):
        if quantity == "Cross-phase" or (quantity in COMPLEX and kind == "Phase"):
            return "deg" if degrees else "rad"
        if "coheren" in quantity:
            return "dimensionless"
        unit = f"({self.source.units})²" + ("" if "covariance" in quantity.lower() else "/Hz")
        return f"({unit})²" if quantity in COMPLEX and kind == "Magnitude squared" else unit

    def displayed(self, quantity, kind="Magnitude", degrees=False, key=Ellipsis):
        values = self.values(quantity, key)
        if quantity in COMPLEX:
            if kind == "Phase":
                phase = self.values("Cross-phase", key)
                return np.rad2deg(phase) if degrees else phase
            return representation(values, kind, degrees)
        return np.rad2deg(values) if quantity == "Cross-phase" and degrees else values

    def frequency_index(self, value):
        if not np.isfinite(value) or not self.frequency[0] <= value <= self.frequency[-1]:
            raise ValueError("Requested frequency is outside the calculated range.")
        return int(np.argmin(abs(self.frequency-value)))

    def frequency_indices(self, start, stop, step=1):
        if (not np.all(np.isfinite([start, stop])) or start > stop or
                start < self.frequency[0] or stop > self.frequency[-1]):
            raise ValueError("Frequency interval must lie within the calculated range, with start ≤ end.")
        if not isinstance(step, (int, np.integer)) or step < 1:
            raise ValueError("Frequency bin step must be a positive integer.")
        indices = np.flatnonzero((self.frequency >= start) & (self.frequency <= stop))[::step]
        if not indices.size:
            raise ValueError("Frequency interval contains no calculated bins.")
        return indices

    def frame(self, quantity, bin_index, *, kind="Magnitude", degrees=False,
              phase=None, amplitude="Quantity", vector=False):
        """One cached spatial frame. Frequency and artificial phase are independent.

        Component projections use the same coherent phasors as vector frames.
        Coherent vector phasors use Re(F * exp(i*phase)), preserving relative
        and spatial phase with the same forward-time convention for both axes.
        """
        if not np.isfinite(phase if phase is not None else 0.):
            raise ValueError("Animation phase must be finite.")
        if (vector or (phase is not None and amplitude in ("A", "B"))) and not self.settings.phase_reference_confirmed:
            raise ValueError("Coherent component displays require calibrated compatible components and a "
                             "confirmed common phase reference across the compared records.")
        if vector:
            if len(self.names) != 2:
                raise ValueError("Vector animation requires two components.")
            channels = {n: np.real(self.arrays[f"Phasor {c}"][..., bin_index] *
                                   np.exp(1j*(phase or 0.))) for n, c in zip(self.names, "AB")}
            units = self.source.units
        elif phase is not None:
            if amplitude in ("A", "B"):
                if len(self.names) != 2:
                    raise ValueError("Coherent component amplitudes require two channels.")
                phasor = self.arrays[f"Phasor {amplitude}"][..., bin_index]
                values = np.real(phasor * np.exp(1j*phase))
                units = self.source.units
            elif quantity in COMPLEX:
                z = self.values(quantity, (..., bin_index))
                values = np.real(np.conj(z) * np.exp(1j*phase))
                units = self.units(quantity)
            else:
                raise ValueError("Phase projection needs a complex quantity or A/B amplitude with cross-phase.")
            channels = {quantity: values}
        else:
            channels = {quantity: self.displayed(quantity, kind, degrees, (..., bin_index))}
            units = self.units(quantity, kind, degrees)
        # Only this real spatial frame enters the legacy Dataset adapter. The
        # frequency/lag coordinates remain explicit in Result, never fake time.
        return replace(self.source, channels={n: a[..., None] for n, a in channels.items()},
                       coords=dict(self.source.coords, time=np.array([np.mean(self.settings.interval)])),
                       units=units)


def _estimates(traces, fs, s):
    kwargs = dict(fs=fs, window=s.window, nperseg=s.nperseg, noverlap=s.overlap,
                  nfft=s.nfft, detrend=False if s.detrend == "none" else s.detrend,
                  scaling="density", average=s.average, axis=-1)
    values = {}
    for label, trace in zip("AB", traces):
        _, values[f"Auto-power {label}"] = signal.welch(trace, **kwargs)
        # STFT uses sum(window) scaling; remove each segment's time-origin phase
        # before averaging. All locations/components share interval start t=0.
        f, _, z = signal.stft(trace, fs=fs, window=s.window, nperseg=s.nperseg,
                             noverlap=s.overlap, nfft=s.nfft, detrend=kwargs["detrend"],
                             boundary=None, padded=False, scaling="spectrum", axis=-1)
        starts = np.arange(z.shape[-1]) * (s.nperseg-s.overlap)/fs
        phasor = np.mean(z * np.exp(-2j*np.pi*f[:, None]*starts), axis=-1)
        factors = np.full(len(f), 2.)
        factors[0] = 1.
        if s.nfft % 2 == 0:
            factors[-1] = 1.
        values[f"Phasor {label}"] = phasor * factors
    a = traces[0] - traces[0].mean(axis=-1, keepdims=True)
    b = traces[-1] - traces[-1].mean(axis=-1, keepdims=True)
    n = a.shape[-1]
    # Correlate only along time, independently for each trace in the chunk.
    sl = slice(n-1-s.max_lag, n+s.max_lag)
    values["Covariance A"] = np.stack([signal.correlate(row, row, method="fft")[sl]/n for row in a])
    if len(traces) == 2:
        _, values["Cross-power"] = signal.csd(traces[0], traces[1], **kwargs)
        values["Cross-covariance"] = np.stack([
            signal.correlate(y, x, method="fft")[sl]/n for x, y in zip(a, b)])
    return values


def process(data, names, settings, progress=None, canceled=None, index=None):
    """Bounded vectorized batches; bad locations become NaN without losing others.

    index is in the original browser axes. Ensemble averaging combines only
    matched per-shot estimates, with identical valid-shot support for all terms.
    """
    fs, window = validate(data, names, settings)
    if canceled is not None and canceled():
        raise ValueError("Spectral processing canceled; previous results retained.")
    if index is not None:
        if len(index) != len(data.dims)-1:
            raise ValueError("Point index must identify every non-time axis.")
        if any(not isinstance(i, (int, np.integer)) or not 0 <= i < n
               for i, n in zip(index, data.shape[:-1])):
            raise ValueError("Point index is outside the stored range.")
        key = tuple(slice(None) if d == "shot" and settings.average_shots else i
                    for d, i in zip(data.dims[:-1], index))
        channels = {n: data.channels[n][key] for n in names}
        dims = (("shot",) if settings.average_shots and "shot" in data.dims else ()) + ("time",)
        records = Dataset(channels, dims, {d: data.coords[d] for d in dims}, units=data.units,
                          spatial_units=data.spatial_units, source=data.source, history=data.history,
                          metadata=dict(data.metadata))
    else:
        records = selected_data(data, names)
    source = selected_data(records, names, settings.average_shots)
    shot_axis = records.dims.index("shot") if settings.average_shots and "shot" in records.dims else None
    frequency = np.fft.rfftfreq(settings.nfft, 1/fs)
    lags = np.arange(-settings.max_lag, settings.max_lag+1)/fs
    shape = source.shape[:-1]
    arrays = {}
    for label in "AB"[:len(names)]:
        arrays[f"Auto-power {label}"] = np.zeros(shape+(len(frequency),))
        arrays[f"Phasor {label}"] = np.zeros(shape+(len(frequency),), dtype=complex)
    arrays["Covariance A"] = np.zeros(shape+(len(lags),))
    if len(names) == 2:
        arrays["Cross-power"] = np.zeros(shape+(len(frequency),), dtype=complex)
        arrays["Cross-covariance"] = np.zeros(shape+(len(lags),))
    counts = np.zeros(shape, dtype=int)
    record_shape = records.shape[:-1]
    total = int(np.prod(record_shape))
    segments = 1 + (window.stop-window.start-settings.nperseg)//(settings.nperseg-settings.overlap)
    chunk = max(1, min(64, 1_000_000//max(len(frequency)*segments, window.stop-window.start)))
    failures = []
    # SciPy's worker context is local to the calling thread. Enter it here,
    # inside the GUI's background worker, and restore it even on cancellation.
    with fft.set_workers(settings.fft_workers):
        for start in range(0, total, chunk):
            if canceled is not None and canceled():
                raise ValueError("Spectral processing canceled; previous results retained.")
            indices = [np.unravel_index(i, record_shape) for i in range(start, min(total, start+chunk))]
            traces = [np.stack([records.channels[n][i][window] for i in indices]) for n in names]
            valid = np.logical_and.reduce([np.all(np.isfinite(a), axis=-1) for a in traces])
            failures.extend((indices[i], "Nonfinite samples in selected interval") for i in np.flatnonzero(~valid))
            if valid.any():
                good = [i for i, ok in zip(indices, valid) if ok]
                finite_traces = [a[valid] for a in traces]

                def save(estimates, locations):
                    for row, i in enumerate(locations):
                        if not all(np.all(np.isfinite(a[row])) for a in estimates.values()):
                            failures.append((i, "Nonfinite spectral estimates (check signal scale)"))
                        else:
                            output_index = i if shot_axis is None else i[:shot_axis]+i[shot_axis+1:]
                            for key, values in estimates.items():
                                arrays[key][output_index] += values[row]
                            counts[output_index] += 1

                try:
                    save(_estimates(finite_traces, fs, settings), good)
                except (ValueError, FloatingPointError):
                    # An estimator failure in one trace must not discard its peers.
                    for row, i in enumerate(good):
                        try:
                            save(_estimates([a[row:row+1] for a in finite_traces], fs, settings), [i])
                        except (ValueError, FloatingPointError) as exc:
                            failures.append((i, str(exc)))
            if progress is not None:
                progress(min(total, start+chunk), total)
    for values in arrays.values():
        np.divide(values, counts[..., None], out=values, where=counts[..., None] > 0)
        values[counts == 0] = np.nan
    return Result(source, tuple(names), copy.deepcopy(settings), frequency, lags, arrays, failures, counts)
