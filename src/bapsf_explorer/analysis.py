"""Non-destructive analysis along labeled dimensions. Time is always in seconds."""

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .data import _history


def subtract_baseline(data, start, stop):
    if not np.isfinite([start, stop]).all() or start > stop:
        raise ValueError("Baseline start must be at or before its end, in seconds.")
    window = data.sel(time=slice(start, stop))
    if window.sizes["time"] == 0:
        raise ValueError("The baseline interval contains no samples.")
    result = data - window.mean("time", skipna=True)
    result.attrs = _history(data, {"baseline_seconds": [start, stop]})
    return result


def filter_signal(data, kind, cutoff_hz, *, order=4):
    time = data.time.values
    if len(time) < 3:
        raise ValueError("Filtering requires at least three time samples.")
    dt = np.diff(time)
    # bapsflib can generate float32 time arrays; allow their rounding error.
    if not np.isfinite(dt).all() or dt[0] <= 0 or not np.allclose(dt, np.median(dt), rtol=2e-4, atol=0):
        raise ValueError("Filtering requires a uniformly sampled, increasing time axis.")
    fs = 1 / np.median(dt)
    cutoff = np.atleast_1d(cutoff_hz).astype(float)
    if kind not in ("lowpass", "highpass", "bandpass"):
        raise ValueError("Choose lowpass, highpass, or bandpass.")
    if cutoff.size != (2 if kind == "bandpass" else 1):
        raise ValueError("Bandpass needs two cutoffs; lowpass/highpass need one.")
    if not np.isfinite(cutoff).all() or np.any(cutoff <= 0) or np.any(cutoff >= fs / 2) or np.any(np.diff(cutoff) <= 0):
        raise ValueError(f"Cutoffs must increase and lie strictly between 0 and Nyquist ({fs / 2:g} Hz).")
    if not isinstance(order, int) or not 1 <= order <= 12:
        raise ValueError("Filter order must be an integer between 1 and 12.")
    arranged = data.transpose(..., "time")
    values = arranged.values.reshape(-1, len(time))
    valid = np.isfinite(values).all(axis=1)
    missing = np.isnan(values).all(axis=1)
    if not np.all(valid | missing):
        raise ValueError("A trace contains partial missing/invalid samples. Resolve gaps before filtering.")
    frequency = cutoff if kind == "bandpass" else float(cutoff[0])
    sos = butter(order, frequency, btype=kind, fs=fs, output="sos")
    output = np.full(values.shape, np.nan, dtype=float)
    if np.any(valid):
        try:
            # Bound temporary filtering buffers for large digitizer channels.
            indices = np.flatnonzero(valid)
            batch_size = max(1, (16 * 1024**2) // (len(time) * 8))
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                output[batch] = sosfiltfilt(sos, values[batch], axis=-1)
        except ValueError as exc:
            raise ValueError(f"The loaded time window is too short for this filter: {exc}") from exc
    result = arranged.copy(data=output.reshape(arranged.shape)).transpose(*data.dims)
    result.attrs = _history(data, {"filter": kind, "cutoff_hz": cutoff.tolist(), "order": order,
                                  "method": "Butterworth, forward-backward zero phase"})
    return result
