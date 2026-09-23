"""Spectral estimates independent of Qt; all coordinates use seconds and Hz.

Welch/CSD: one-sided density, d*conj(FFT(A))*FFT(B)/(fs*sum(w**2)),
averaged over segments. d=2 except DC and an even FFT's Nyquist bin.
Positive arg(S_AB) means B leads A. Covariances remove the interval mean
and use a fixed N denominator: R_AB[k] = sum A[n]*B[n+k]/N.
Coherent peak phasors are a separate estimator, described in docs/spectral.md.
"""
from dataclasses import dataclass, replace
import copy
import numpy as np
from scipy import signal
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
    max_lag: int = 128


def sampling_rate(time):
    time = np.asarray(time)
    if time.ndim != 1 or len(time) < 4 or not np.all(np.isfinite(time)):
        raise ValueError("Spectral analysis needs at least four finite time coordinates.")
    dt = np.diff(time)
    if dt[0] <= 0 or not np.allclose(dt, dt[0], rtol=1e-5, atol=abs(dt[0])*1e-8):
        raise ValueError("Spectral analysis requires a uniformly sampled, increasing time base.")
    return 1 / dt[0]


def validate(data, names, settings):
    if len(names) not in (1, 2) or len(set(names)) != len(names):
        raise ValueError("Select one or two distinct channels; set the third component to None.")
    if any(n not in data.channels for n in names):
        raise ValueError("Select input channels in the main browser.")
    if any(data.channels[n].shape != data.shape for n in names):
        raise ValueError("Channels must share dimensions and the same time base.")
    fs = sampling_rate(data.coords["time"])
    s = settings
    window = interval_slice(data.coords["time"], s.interval, minimum=4)
    for name in ("nperseg", "nfft", "overlap", "max_lag"):
        value = getattr(s, name)
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer.")
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
    """Keep the browser's existing axes; optionally average repeats before estimation."""
    channels = {n: data.channels[n] for n in names}
    if average_shots and "shot" in data.dims:
        axis = data.dims.index("shot")
        return replace(data, channels={n: a.mean(axis=axis) for n, a in channels.items()},
                       dims=tuple(d for d in data.dims if d != "shot"),
                       coords={d: c for d, c in data.coords.items() if d != "shot"},
                       shot_numbers=None, history=data.history + ("Average shots before spectral estimation",))
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
            denominator = np.sqrt(self.arrays["Auto-power A"][key] * self.arrays["Auto-power B"][key])
            coherence = np.full(denominator.shape, np.nan + 1j*np.nan)
            np.divide(self.arrays["Cross-power"][key], denominator, out=coherence, where=denominator > 0)
            return abs(coherence)**2 if quantity == "Magnitude-squared coherence" else coherence
        if quantity == "Cross-phase":
            cross = self.arrays["Cross-power"][key]
            return np.where(abs(cross) > 0, np.angle(cross), np.nan)
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

        Cross-referenced scalar fields use |A| cos(phase - arg(S_AB)).
        Coherent vector phasors use Re(F * exp(i*phase)), preserving relative
        and spatial phase with the same forward-time convention for both axes.
        """
        if not np.isfinite(phase if phase is not None else 0.):
            raise ValueError("Animation phase must be finite.")
        if vector:
            if len(self.names) != 2:
                raise ValueError("Vector animation requires two components.")
            channels = {n: np.real(self.arrays[f"Phasor {c}"][..., bin_index] *
                                   np.exp(1j*(phase or 0.))) for n, c in zip(self.names, "AB")}
            units = self.source.units
        elif phase is not None:
            if amplitude in ("A", "B"):
                if len(self.names) != 2:
                    raise ValueError("Cross-phase referenced amplitudes require two channels.")
                power = self.arrays[f"Auto-power {amplitude}"][..., bin_index]
                phi = self.values("Cross-phase", (..., bin_index))
                values = np.sqrt(power) * np.cos(phase-phi)
                units = f"{self.source.units}/√Hz"
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

    index is in the original browser axes, before optional repeat averaging.
    """
    fs, window = validate(data, names, settings)
    if canceled is not None and canceled():
        raise ValueError("Spectral processing canceled; previous results retained.")
    if index is not None:
        if len(index) != len(data.dims)-1:
            raise ValueError("Point index must identify every non-time axis.")
        key = tuple(slice(None) if d == "shot" and settings.average_shots else i
                    for d, i in zip(data.dims[:-1], index))
        channels = {n: data.channels[n][key] for n in names}
        if settings.average_shots and "shot" in data.dims:
            channels = {n: a.mean(axis=0) for n, a in channels.items()}
        source = Dataset(channels, ("time",), {"time": data.coords["time"]}, units=data.units,
                         source=data.source, history=data.history)
    else:
        source = selected_data(data, names, settings.average_shots)
    frequency = np.fft.rfftfreq(settings.nfft, 1/fs)
    lags = np.arange(-settings.max_lag, settings.max_lag+1)/fs
    shape = source.shape[:-1]
    arrays = {}
    for label in "AB"[:len(names)]:
        arrays[f"Auto-power {label}"] = np.full(shape+(len(frequency),), np.nan)
        arrays[f"Phasor {label}"] = np.full(shape+(len(frequency),), np.nan+1j*np.nan)
    arrays["Covariance A"] = np.full(shape+(len(lags),), np.nan)
    if len(names) == 2:
        arrays["Cross-power"] = np.full(shape+(len(frequency),), np.nan+1j*np.nan)
        arrays["Cross-covariance"] = np.full(shape+(len(lags),), np.nan)
    total = int(np.prod(shape))
    segments = 1 + (window.stop-window.start-settings.nperseg)//(settings.nperseg-settings.overlap)
    chunk = max(1, min(64, 1_000_000//max(len(frequency)*segments, window.stop-window.start)))
    failures = []
    for start in range(0, total, chunk):
        if canceled is not None and canceled():
            raise ValueError("Spectral processing canceled; previous results retained.")
        indices = [np.unravel_index(i, shape) for i in range(start, min(total, start+chunk))]
        traces = [np.stack([source.channels[n][i][window] for i in indices]) for n in names]
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
                        for key, values in estimates.items():
                            arrays[key][i] = values[row]

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
    return Result(source, tuple(names), copy.deepcopy(settings), frequency, lags, arrays, failures)
