"""Independent time-axis filters for real arrays of any dimensionality."""
import operator
import numpy as np
from scipy.ndimage import uniform_filter1d, maximum_filter1d, gaussian_filter1d
from scipy.signal import savgol_filter, butter, sosfiltfilt


DEFAULT_SMOOTHING = dict(method="moving", window_size=11, polyorder=3,
                         sigma=2.0, cutoff=1000.0, cutoff_upper=10000.0,
                         butter_type="lowpass", butter_order=4,
                         mode="nearest", nan_policy="propagate")


def _integer(value, name, minimum):
    try:
        result = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer.") from None
    if isinstance(value, (bool, np.bool_)) or result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return result


def smooth_time_series(A, method="moving", *, time_axis=-1, window_size=11,
                       polyorder=3, sigma=2.0, cutoff=None, fs=None, dt=None,
                       butter_order=4, mode="nearest", nan_policy="propagate",
                       time=None, butter_type="lowpass", cutoff_upper=None):
    """Filter each trace without mixing any other axes or modifying the input.

    Moving/Savitzky–Golay windows and Gaussian sigma are in samples. These
    filters work in sample-index space even on nonuniform time grids.
    Butterworth requires uniform time (seconds), or exactly one of fs/dt.
    butter_type selects lowpass (default), highpass, or bandpass. cutoff is
    in Hz; bandpass also requires cutoff_upper in Hz, above cutoff.
    Short Butterworth traces are rejected rather than silently reducing padding.
    Nonfinite samples are treated as missing: propagate spreads them over the
    filter support (the whole trace for Butterworth); interp fills gaps using
    actual time when supplied, extending the nearest finite value at endpoints.
    Entirely missing traces remain NaN. Savitzky–Golay uses polynomial edges;
    mode controls only moving/Gaussian boundaries (constant means zero padding).
    """
    a = np.asarray(A)
    if a.ndim < 1 or not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
        raise ValueError("Input must be a real numeric array with at least one dimension.")
    axis = operator.index(time_axis)
    if not -a.ndim <= axis < a.ndim:
        raise ValueError("time_axis is outside the array dimensions.")
    axis %= a.ndim
    nt = a.shape[axis]
    if nt < 2:
        raise ValueError("Smoothing needs at least two time samples.")
    if nan_policy not in {"propagate", "interp"}:
        raise ValueError("nan_policy must be 'propagate' or 'interp'.")
    if mode not in {"nearest", "reflect", "mirror", "constant", "wrap"}:
        raise ValueError("Unknown boundary mode.")
    x = np.arange(nt, dtype=float)
    if time is not None:
        x = np.asarray(time, dtype=float)
        if x.shape != (nt,) or not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0):
            raise ValueError("Time coordinates must be finite, increasing, and match the time axis.")
    b = np.array(a, dtype=float, copy=True)
    b[~np.isfinite(b)] = np.nan
    if nan_policy == "interp":
        traces = np.moveaxis(b, axis, -1)
        # Explicit contiguous copy makes interpolation reliable for any axis/stride.
        rows = np.ascontiguousarray(traces).reshape(-1, nt)
        for row in rows:
            good = np.isfinite(row)
            if good.any() and not good.all():
                row[~good] = np.interp(x[~good], x[good], row[good])
        b = np.moveaxis(rows.reshape(traces.shape), -1, axis)
    if method == "moving":
        window_size = _integer(window_size, "Window size", 1)
        # Running sums on NaNs can poison the entire remainder of a trace.
        missing = ~np.isfinite(b)
        result = uniform_filter1d(np.where(missing, 0, b), window_size, axis=axis, mode=mode)
        result[maximum_filter1d(missing, window_size, axis=axis, mode=mode, cval=0)] = np.nan
        return result
    if method == "savgol":
        window_size = _integer(window_size, "Window size", 1)
        polyorder = _integer(polyorder, "Polynomial order", 0)
        if window_size % 2 != 1 or not polyorder < window_size <= nt:
            raise ValueError("Savitzky–Golay needs an odd window <= trace length and > polynomial order.")
        missing = ~np.isfinite(b)
        result = savgol_filter(np.where(missing, 0, b), window_size, polyorder,
                               axis=axis, mode="interp")
        affected = maximum_filter1d(missing, window_size, axis=axis, mode="constant", cval=0)
        # Polynomial edge fits depend on the whole first/last window, and
        # SciPy versions may reject NaNs in that fit. Restore missing support.
        edge_mask = np.moveaxis(affected, axis, -1)
        source_mask = np.moveaxis(missing, axis, -1)
        half = window_size // 2
        if half:
            edge_mask[..., :half] = source_mask[..., :window_size].any(axis=-1, keepdims=True)
            edge_mask[..., -half:] = source_mask[..., -window_size:].any(axis=-1, keepdims=True)
        result[np.moveaxis(edge_mask, -1, axis)] = np.nan
        return result
    if method == "gaussian":
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError("Gaussian sigma must be finite and positive.")
        return gaussian_filter1d(b, sigma, axis=axis, mode=mode)
    if method == "butterworth":
        butter_order = _integer(butter_order, "Butterworth order", 1)
        if butter_type not in {"lowpass", "highpass", "bandpass"}:
            raise ValueError("Butterworth type must be lowpass, highpass, or bandpass.")
        if time is not None:
            if fs is not None or dt is not None:
                raise ValueError("Supply time coordinates or fs/dt, not both.")
            intervals = np.diff(x)
            if not np.allclose(intervals, intervals[0], rtol=1e-5, atol=abs(intervals[0])*1e-8):
                raise ValueError("Butterworth smoothing requires uniformly spaced time samples.")
            dt = intervals[0]
        if (fs is None) == (dt is None):
            raise ValueError("Supply exactly one of fs or dt for Butterworth smoothing.")
        rate = fs if fs is not None else dt
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("Sampling frequency/interval must be finite and positive.")
        fs = float(fs) if fs is not None else 1.0 / dt
        if cutoff is None or not np.isfinite(cutoff) or not 0 < cutoff < fs/2:
            raise ValueError(f"Cutoff must satisfy 0 < cutoff < Nyquist ({fs/2:g} Hz).")
        frequencies = cutoff
        if butter_type == "bandpass":
            if (cutoff_upper is None or not np.isfinite(cutoff_upper)
                    or not cutoff < cutoff_upper < fs/2):
                raise ValueError(f"Band-pass cutoffs must satisfy 0 < lower < upper < Nyquist ({fs/2:g} Hz).")
            frequencies = (cutoff, cutoff_upper)
        # Band-pass designs have twice the order and SOS count of low/high-pass
        # designs, so derive the required padding from the actual sections.
        sos = butter(butter_order, frequencies, btype=butter_type, fs=fs, output="sos")
        padlen = 3 * (2*len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
        if nt <= padlen:
            raise ValueError(f"Butterworth order {butter_order} needs at least {padlen+1} time samples.")
        return sosfiltfilt(sos, b, axis=axis, padlen=padlen)
    raise ValueError("Unknown smoothing method; use moving, savgol, gaussian, or butterworth.")
