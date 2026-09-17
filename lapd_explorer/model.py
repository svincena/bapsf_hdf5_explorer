"""Scientific model independent of Qt. Time is always in seconds."""
from dataclasses import dataclass, field, replace
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import welch
from .smoothing import smooth_time_series


@dataclass
class Dataset:
    channels: dict[str, np.ndarray]
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]
    units: str = "V"
    spatial_units: str = "cm"
    source: str = ""
    history: tuple[str, ...] = ()
    shot_numbers: np.ndarray | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.dims = tuple(self.dims)
        if not self.channels or not self.dims or self.dims[-1] != "time":
            raise ValueError("At least one channel is required; time must be the final axis.")
        if len(set(self.dims)) != len(self.dims):
            raise ValueError("Axis names must be unique.")
        if not set(self.dims) <= {"x", "y", "z", "case", "shot", "time"}:
            raise ValueError("Supported axes: x, y, z, case, shot, time.")
        if len(self.spatial_dims) > 2:
            raise ValueError("A maximum of two spatial dimensions is supported.")
        shape = next(iter(self.channels.values())).shape
        if len(shape) != len(self.dims) or any(n < 1 for n in shape):
            raise ValueError("Dimension names must match a nonempty data shape.")
        for a in self.channels.values():
            if a.shape != shape or not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
                raise ValueError("Channels must be real numeric arrays with identical shapes.")
        for d, n in zip(self.dims, shape):
            c = np.asarray(self.coords[d], dtype=float)
            if c.shape != (n,) or not np.all(np.isfinite(c)):
                raise ValueError(f"Invalid coordinates for {d}.")
            if n > 1 and not np.all(np.diff(c) > 0):
                raise ValueError(f"Coordinates for {d} must increase strictly.")
            self.coords[d] = c
        if self.shot_numbers is not None and self.shot_numbers.shape != shape[:-1]:
            raise ValueError("Shot-number map must match all non-time axes.")

    @property
    def spatial_dims(self):
        return tuple(d for d in self.dims if d in "xyz")

    @property
    def shape(self):
        return next(iter(self.channels.values())).shape

    def selected(self, name, case=0, shot=0):
        """Select case/repeat, preserving spatial dimensions and time."""
        key = tuple(case if d == "case" else shot if d == "shot" else slice(None)
                    for d in self.dims)
        return self.channels[name][key]


def preprocess(data, *, average=False, baseline="None", integrate=False, gain=1.0, smoothing=None):
    """Apply per-trace baseline → integration → smoothing → gain → repeat averaging.

    Averaging never combines case or position axes. No inferred error bars are
    produced when only one stored trace is available.
    """
    if not np.isfinite(gain):
        raise ValueError("Gain must be finite.")
    out = {}
    steps = []
    t = data.coords["time"]
    for name, raw in data.channels.items():
        a = np.array(raw, dtype=float, copy=True)
        if baseline == "Remove mean":
            a -= np.mean(a, axis=-1, keepdims=True)
        elif baseline == "Linear detrend":
            if len(t) < 2 or not np.all(np.isfinite(a)):
                raise ValueError("Linear detrending needs at least two finite samples per trace.")
            # Fit against actual time coordinates, including nonuniform grids.
            tc = t - t.mean()
            mean = a.mean(axis=-1, keepdims=True)
            a -= mean + (np.sum((a - mean) * tc, axis=-1, keepdims=True)
                         / np.sum(tc ** 2)) * tc
        elif baseline != "None":
            raise ValueError("Unknown baseline operation.")
        if integrate:
            a = cumulative_trapezoid(a, t, axis=-1, initial=0)
        if smoothing is not None:
            a = smooth_time_series(a, time=t, **smoothing)
        a *= gain
        if average and "shot" in data.dims:
            a = np.mean(a, axis=data.dims.index("shot"))
        out[name] = a
    if baseline != "None":
        steps.append(baseline)
    if integrate:
        steps.append("Cumulative trapezoidal integration; initial value 0")
    if smoothing is not None:
        from .smoothing import DEFAULT_SMOOTHING
        settings = DEFAULT_SMOOTHING | smoothing
        keys = {"moving": ("window_size", "mode"), "savgol": ("window_size", "polyorder"),
                "gaussian": ("sigma", "mode"),
                "butterworth": ("butter_type", "cutoff", "butter_order")}[settings["method"]]
        if settings["method"] == "butterworth" and settings["butter_type"] == "bandpass":
            keys = ("butter_type", "cutoff", "cutoff_upper", "butter_order")
        details = ", ".join(f"{k}={settings[k]}" + (" Hz" if k in {"cutoff", "cutoff_upper"} else "")
                            for k in (*keys, "nan_policy"))
        steps.append(f"Time smoothing: {settings['method']} ({details})")
    if gain != 1:
        steps.append(f"Gain × {gain:g}")
    dims = data.dims
    coords = dict(data.coords)
    shots = data.shot_numbers
    if average and "shot" in dims:
        steps.append(f"Average {len(coords['shot'])} shots at each position/case")
        dims = tuple(d for d in dims if d != "shot")
        coords.pop("shot")
        shots = None
    return replace(data, channels=out, dims=dims, coords=coords,
                   units=f"{data.units}·s" if integrate else data.units,
                   shot_numbers=shots, history=data.history + tuple(steps))


def quantity(data, names, mode, case=0, shot=0):
    if not names:
        raise ValueError("Select at least one channel.")
    arrays = [data.selected(n, case, shot) for n in names]
    if mode == "Magnitude" or mode == "Vector":
        if len(arrays) not in (2, 3):
            raise ValueError("Select two or three channels for vector quantities.")
        return np.sqrt(sum(a * a for a in arrays))
    return np.abs(arrays[0]) if mode == "Absolute" else arrays[0]


def spectrum(trace, time):
    if len(time) < 4 or not np.all(np.isfinite(trace)):
        raise ValueError("PSD needs at least four finite samples.")
    dt = np.diff(time)
    if not np.allclose(dt, dt[0], rtol=1e-5, atol=abs(dt[0])*1e-8):
        raise ValueError("PSD requires uniformly spaced time samples.")
    return welch(trace, fs=1 / dt[0], nperseg=min(2048, len(time)))


def demo():
    rng = np.random.default_rng(12)
    x, y, t = np.linspace(-25, 25, 31), np.linspace(-20, 20, 25), np.linspace(0, 120e-6, 160)
    yy, xx, cc, ss, tt = np.meshgrid(y, x, np.arange(2), np.arange(4), t, indexing="ij")
    envelope = np.exp(-(xx**2 + yy**2) / 350)
    phase = 2*np.pi*(tt*35e3 - xx/45) + cc*0.7
    bx = envelope*np.cos(phase) + rng.normal(0, .04, xx.shape)
    by = envelope*np.sin(phase) + rng.normal(0, .04, xx.shape)
    return Dataset({"Bx": bx, "By": by}, ("y", "x", "case", "shot", "time"),
                   {"y": y, "x": x, "case": np.arange(2), "shot": np.arange(4), "time": t},
                   units="a.u.", source="Synthetic LAPD wave • demonstration",
                   shot_numbers=np.arange(1, np.prod(xx.shape[:-1])+1).reshape(xx.shape[:-1]))
