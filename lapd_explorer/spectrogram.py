"""Bounded, single-trace auto/cross spectrograms using batched SciPy FFTs.

The cross convention is conj(A)*B, as in the existing Welch/CSD analysis.
Time coordinates identify each window's central sample (floor(nperseg/2)),
consistent with ShortTimeFFT. Frequency is never stored as a time axis.
"""
from dataclasses import dataclass, field, asdict
import copy
import os
import json
import numpy as np
from scipy import fft, signal
from .spectral import sampling_rate, representation
from .langmuir import interval_slice


WINDOWS = ("hann", "hamming", "blackman", "blackmanharris", "nuttall", "flattop",
           "boxcar", "bartlett", "tukey", "kaiser", "gaussian", "chebwin")
PARAMETERS = {"tukey": ("Taper fraction (α)", .25), "kaiser": ("Kaiser β", 8.6),
              "gaussian": ("Gaussian σ (samples)", 32.), "chebwin": ("Attenuation (dB)", 100.)}
QUANTITIES = ("Auto-power A", "Auto-power B", "Cross-power")
BOUNDARIES = {"None — complete windows": "none", "Zeros": "zeros", "Edge values": "edge",
              "Even reflection": "even", "Odd reflection": "odd"}
MAX_OUTPUT_BYTES = 256 * 1024**2


@dataclass
class Settings:
    interval: tuple | None = None
    nperseg: int = 256
    overlap: int = 192
    nfft: int = 256
    window: str = "hann"
    window_parameter: float = .25
    symmetric_window: bool = False
    detrend: str = "constant"
    scaling: str = "density"
    sides: str = "onesided"
    boundary: str = "none"
    pad_end: bool = False
    fft_workers: int = field(default_factory=lambda: max(1, min(4, os.cpu_count() or 1)))


def trace_index(data, index):
    index = tuple(index)
    if len(index) != len(data.dims)-1 or any(
            not isinstance(i, (int, np.integer)) or isinstance(i, bool) or not 0 <= i < n
            for i, n in zip(index, data.shape[:-1])):
        raise ValueError("Trace index must identify every non-time axis within its stored range.")
    return tuple(int(i) for i in index)


def window_values(settings):
    s = settings
    if s.window not in WINDOWS:
        raise ValueError("Unknown spectrogram window.")
    param = s.window_parameter
    if s.window in PARAMETERS:
        if not np.isfinite(param) or (s.window == "tukey" and not 0 <= param <= 1) or (s.window != "tukey" and param <= 0):
            raise ValueError("Window parameter must be finite: Tukey α in [0,1], other parameters positive.")
    spec = (s.window, param) if s.window in PARAMETERS else s.window
    values = signal.get_window(spec, s.nperseg, fftbins=not s.symmetric_window)
    energy = np.sum(values**2)
    if not np.all(np.isfinite(values)) or energy <= 0 or (s.scaling == "spectrum" and abs(values.sum()) <= np.finfo(float).eps):
        raise ValueError("Window has no usable energy or spectrum normalization.")
    return values


def _layout(samples, settings):
    """Window starts, including only explicitly requested boundary padding."""
    s = settings
    hop = s.nperseg-s.overlap
    if s.boundary == "none":
        first, span = 0, samples-s.nperseg
    else:
        first, span = -(s.nperseg//2), samples-1
    count = 1 + (int(np.ceil(span/hop)) if s.pad_end else span//hop)
    return first, count


def layout(samples, settings):
    first, count = _layout(samples, settings)
    return first + np.arange(count, dtype=np.int64)*(settings.nperseg-settings.overlap)


def validate(data, names, settings, index=()):
    names = tuple(names)
    if len(names) not in (1, 2) or len(set(names)) != len(names) or any(n not in data.channels for n in names):
        raise ValueError("Select one or two distinct stored channels for a spectrogram.")
    index = trace_index(data, index)
    fs = sampling_rate(data.coords["time"])
    interval = settings.interval if settings.interval is not None else (data.coords["time"][0], data.coords["time"][-1])
    crop = interval_slice(data.coords["time"], interval, minimum=4)
    s = settings
    for key in ("nperseg", "overlap", "nfft", "fft_workers"):
        if not isinstance(getattr(s, key), (int, np.integer)) or isinstance(getattr(s, key), bool):
            raise ValueError(f"{key} must be an integer.")
    if not 4 <= s.nperseg <= crop.stop-crop.start:
        raise ValueError("Segment length must be at least 4 and fit inside the selected interval.")
    if not 0 <= s.overlap < s.nperseg:
        raise ValueError("Overlap must satisfy 0 ≤ overlap < segment length.")
    if s.nfft < s.nperseg or s.fft_workers < 1:
        raise ValueError("FFT length must cover the segment and FFT workers must be positive.")
    if s.detrend not in ("none", "constant", "linear") or s.scaling not in ("density", "spectrum"):
        raise ValueError("Unknown detrending or power scaling.")
    if s.sides not in ("onesided", "twosided") or s.boundary not in BOUNDARIES.values():
        raise ValueError("Unknown frequency sides or boundary padding.")
    return fs, crop, index, window_values(s)


@dataclass
class Result:
    names: tuple
    index: tuple
    location: dict
    global_shot: int | None
    units: str
    spatial_units: str
    source: str
    history: tuple
    settings: Settings
    frequency: np.ndarray
    time: np.ndarray
    arrays: dict

    def displayed(self, quantity, kind="Magnitude", degrees=False):
        if quantity not in self.arrays:
            raise ValueError("This spectrogram quantity requires two selected channels.")
        values = self.arrays[quantity]
        return representation(values, kind, degrees) if quantity == "Cross-power" else values

    def value_units(self, quantity, kind="Magnitude", degrees=False):
        if quantity == "Cross-power" and kind == "Phase":
            return "deg" if degrees else "rad"
        unit = f"({self.units})²" + ("/Hz" if self.settings.scaling == "density" else "")
        return f"({unit})²" if quantity == "Cross-power" and kind == "Magnitude squared" else unit


def _segments(trace, starts, length, padding):
    """Copy only this bounded group of windows, extending true interval edges."""
    lo, hi = int(starts[0]), int(starts[-1])+length
    i0, i1 = max(0, lo), min(len(trace), hi)
    left, right = max(0, -lo), max(0, hi-len(trace))
    # Reflection needs original samples beyond the window being calculated;
    # reflecting a short tail in isolation would introduce false repetition.
    if right:
        i0 = min(i0, max(0, len(trace)-right-1))
    if left:
        i1 = max(i1, min(len(trace), left+1))
    values = trace[i0:i1]
    if left or right:
        kwargs = {"zeros": dict(mode="constant"), "edge": dict(mode="edge"),
                  "even": dict(mode="reflect", reflect_type="even"),
                  "odd": dict(mode="reflect", reflect_type="odd")}[padding]
        values = np.pad(values, (left, right), **kwargs)
    hop = int(starts[1]-starts[0]) if len(starts) > 1 else 1
    offset = lo-i0+left
    return np.array(np.lib.stride_tricks.sliding_window_view(values, length)[offset::hop][:len(starts)], dtype=float, copy=True)


def process(data, names, settings, index=(), progress=None, canceled=None, *, max_output_bytes=MAX_OUTPUT_BYTES):
    fs, crop, index, window = validate(data, names, settings, index)
    names = tuple(names)
    s = copy.deepcopy(settings)
    count = crop.stop-crop.start
    _, windows = _layout(count, s)
    bins = s.nfft//2+1 if s.sides == "onesided" else s.nfft
    needed = windows*bins*(8 if len(names) == 1 else 32)
    if needed > max_output_bytes:
        raise ValueError(f"Spectrogram needs {needed/1024**2:.1f} MiB of results (limit {max_output_bytes/1024**2:g} MiB). "
                         "Shorten the interval, reduce FFT length, or reduce overlap.")
    starts = layout(count, s)
    frequency = fft.rfftfreq(s.nfft, 1/fs) if s.sides == "onesided" else fft.fftshift(fft.fftfreq(s.nfft, 1/fs))
    if s.interval is None:
        s.interval = (float(data.coords["time"][crop.start]), float(data.coords["time"][crop.stop-1]))
    if canceled and canceled():
        raise ValueError("Spectrogram processing canceled.")
    traces = [data.channels[n][index][crop] for n in names]
    if not all(np.all(np.isfinite(trace)) for trace in traces):
        raise ValueError("Nonfinite samples in the selected trace interval.")
    quantities = QUANTITIES[:1] if len(names) == 1 else QUANTITIES
    arrays = {q: np.empty((len(frequency), len(starts)), dtype=complex if q == "Cross-power" else float)
              for q in quantities}
    scale = 1/(fs*np.sum(window**2)) if s.scaling == "density" else 1/window.sum()**2
    factors = np.ones(len(frequency))
    if s.sides == "onesided":
        factors[1:] = 2
        if s.nfft % 2 == 0:
            factors[-1] = 1
    chunk = max(1, min(128, 1_000_000//max(s.nperseg, s.nfft)))
    if progress:
        progress(0, len(starts))
    with fft.set_workers(s.fft_workers):
        for first in range(0, len(starts), chunk):
            if canceled and canceled():
                raise ValueError("Spectrogram processing canceled.")
            last = min(len(starts), first+chunk)
            transforms = []
            for trace in traces:
                segments = _segments(trace, starts[first:last], s.nperseg, "zeros" if s.boundary == "none" else s.boundary)
                if s.detrend != "none":
                    segments = signal.detrend(segments, type=s.detrend, axis=-1, overwrite_data=True)
                segments *= window
                transform = fft.rfft(segments, n=s.nfft, axis=-1) if s.sides == "onesided" else fft.fftshift(fft.fft(segments, n=s.nfft, axis=-1), axes=-1)
                transforms.append(transform)
            for label, transform in zip("AB", transforms):
                arrays[f"Auto-power {label}"][:, first:last] = (abs(transform)**2 * scale*factors).T
            if len(names) == 2:
                arrays["Cross-power"][:, first:last] = (np.conj(transforms[0])*transforms[1]*scale*factors).T
            if not all(np.all(np.isfinite(a[:, first:last])) for a in arrays.values()):
                raise ValueError("Nonfinite spectrogram estimates; check the input signal scale.")
            if progress:
                progress(last, len(starts))
    time = data.coords["time"][crop.start] + (starts+s.nperseg//2)/fs
    shot = int(data.shot_numbers[index]) if data.shot_numbers is not None else None
    return Result(names, index, {d: float(data.coords[d][i]) for d, i in zip(data.dims[:-1], index)}, shot,
                  data.units, data.spatial_units, data.source, data.history, s, frequency, time, arrays)


def color_values(result, quantity, kind="Magnitude", degrees=False, scale="Linear", reference=1., dynamic_range=80.):
    """Display conversion only; phase/signed values keep their physical meaning."""
    values = result.displayed(quantity, kind, degrees)
    if scale == "dB":
        if quantity == "Cross-power" and kind not in ("Magnitude", "Magnitude squared"):
            raise ValueError("dB display requires nonnegative power or cross-power magnitude.")
        if not np.isfinite(reference) or reference <= 0 or not np.isfinite(dynamic_range) or dynamic_range <= 0:
            raise ValueError("dB reference and dynamic range must be finite and positive.")
        # These are powers (including |cross-power|), not amplitude spectra.
        values = 10*(np.log10(np.maximum(values, np.finfo(float).tiny))-np.log10(reference))
        peak = float(np.max(values))
        values = np.maximum(values, peak-dynamic_range)
        return values, f"dB re {reference:g} {result.value_units(quantity, kind, degrees)}"
    if scale not in ("Linear", "Logarithmic", "Symmetric log"):
        raise ValueError("Unknown spectrogram color scale.")
    if scale == "Logarithmic" and quantity == "Cross-power" and kind != "Magnitude" and kind != "Magnitude squared":
        raise ValueError("Logarithmic color requires power or cross-power magnitude; use symmetric log for signed values.")
    return values, result.value_units(quantity, kind, degrees)


def save(path, result):
    """Portable NPZ with complex estimates, explicit coordinates and JSON provenance."""
    metadata = dict(names=result.names, index=result.index, location=result.location,
                    global_shot=result.global_shot, units=result.units, spatial_units=result.spatial_units,
                    source=result.source, history=result.history, settings=asdict(result.settings),
                    cross_convention="conj(A)*B; positive phase means B leads A",
                    time_coordinate="window center sample; floor(nperseg/2)")
    with open(path, "wb") as target:
        np.savez_compressed(target, time=result.time, frequency=result.frequency,
                            metadata=json.dumps(metadata, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), **{q.replace("-", "_").replace(" ", "_"): a for q, a in result.arrays.items()})
