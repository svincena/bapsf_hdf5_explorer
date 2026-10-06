"""Temporal import selection in original sample indices, before thinning."""
import math
import numpy as np


def sample_slice(samples, dt, t0=0., sample_limits=None, time_limits=None, decimation=1):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(t0):
        raise ValueError("Sample interval must be positive and time origin finite.")
    if not isinstance(decimation, (int, np.integer)) or isinstance(decimation, bool) or decimation < 1:
        raise ValueError("Keep-every factor must be a positive integer.")
    if sample_limits is not None and time_limits is not None:
        raise ValueError("Choose either sample limits or time limits.")
    first, last = 0, samples - 1
    if time_limits is not None:
        lo, hi = time_limits
        if not np.isfinite(lo) or (hi is not None and not np.isfinite(hi)):
            raise ValueError("Time limits must be finite.")
        # Inclusive bounds, with tolerance for roundoff at exact samples.
        a = (lo - t0) / dt
        b = samples - 1 if hi is None else (hi - t0) / dt
        tolerance = np.finfo(float).eps * max(1., abs(a), abs(b), abs(t0 / dt)) * 16
        first, last = math.ceil(a - tolerance), math.floor(b + tolerance)
    elif sample_limits is not None:
        first, last = sample_limits
        last = samples - 1 if last is None else last
        if any(not isinstance(v, (int, np.integer)) or isinstance(v, bool) for v in (first, last)):
            raise ValueError("Sample limits must be integer indices.")
    if not 0 <= first <= last < samples:
        raise ValueError(f"Choose ordered limits within original samples 0–{samples - 1}.")
    if len(range(first, last + 1, decimation)) < 2:
        raise ValueError("The imported interval must retain at least two time samples.")
    return slice(first, last + 1, decimation)


def selection_metadata(selection, dt):
    return {"first original sample": selection.start,
            "last original sample limit (inclusive)": selection.stop - 1,
            "keep every Nth sample": selection.step,
            "original sample interval (s)": dt,
            "effective sample interval (s)": dt * selection.step,
            "effective sampling rate (Hz)": 1 / (dt * selection.step),
            "anti-alias filtering": False}


def preview_record(metadata, mode="index", index=None, shot=None, position=None, repeat=0, decimals=4):
    indices = metadata["indices"]
    if mode == "index":
        index = int(indices[len(indices) // 2]) if index is None else index
        matches = np.flatnonzero(indices == index)
    elif mode == "shot":
        if metadata["shots"] is None:
            raise ValueError("Global shot numbers are unavailable for raw datasets.")
        matches = np.flatnonzero(metadata["shots"] == shot)
    elif mode == "position":
        xyz = metadata["xyz"]
        axes = [i for i, value in enumerate(position) if value is not None]
        if xyz is None or not axes:
            raise ValueError("Enter at least one coordinate and select motion mapping.")
        valid = np.all(np.isfinite(xyz[:, axes]), axis=1)
        rows = np.flatnonzero(valid)
        if not len(rows):
            raise ValueError("Finite spatial coordinates are unavailable.")
        requested = np.asarray([position[i] for i in axes], dtype=float)
        if not np.all(np.isfinite(requested)):
            raise ValueError("Preview coordinates must be finite.")
        rounded = np.round(xyz[rows][:, axes], decimals)
        nearest = rounded[np.argmin(np.sum((rounded - requested) ** 2, axis=1))]
        matches = rows[np.all(rounded == nearest, axis=1)]
        if not 0 <= repeat < len(matches):
            raise ValueError(f"That position contains {len(matches)} records; choose an occurrence from 0 to {len(matches)-1}.")
        matches = matches[repeat:repeat + 1]
    else:
        raise ValueError("Unknown preview record selector.")
    if not len(matches):
        raise ValueError("No matching record in the selected record range.")
    row = int(matches[0])
    return int(indices[row]), row
