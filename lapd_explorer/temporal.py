"""Temporal import selection in original sample indices, before thinning."""
import math
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import numpy as np
from scipy.signal import firwin, resample_poly
from .cancellation import check_canceled


DOWNSAMPLING_METHODS = {"Polyphase FIR resampling": "polyphase",
                        "Simple decimation": "simple", "Block averaging": "average"}
SOURCE_CHUNK_SAMPLES = 65536


def retained_samples(selection, downsampling="polyphase"):
    length = selection.stop - selection.start
    return length // selection.step if downsampling == "average" else len(range(selection.start, selection.stop, selection.step))


def sample_times(selection, dt, t0=0., downsampling="polyphase"):
    offset = (selection.step - 1) / 2 if downsampling == "average" else 0
    return t0 + (selection.start + offset + np.arange(retained_samples(selection, downsampling))*selection.step)*dt


def sample_slice(samples, dt, t0=0., sample_limits=None, time_limits=None, decimation=1,
                 downsampling="polyphase"):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(t0):
        raise ValueError("Sample interval must be positive and time origin finite.")
    if not isinstance(decimation, (int, np.integer)) or isinstance(decimation, bool) or decimation < 1:
        raise ValueError("Keep-every factor must be a positive integer.")
    if downsampling not in DOWNSAMPLING_METHODS.values():
        raise ValueError("Unknown downsampling method.")
    if downsampling == "polyphase" and decimation > 50000:
        raise ValueError("Polyphase FIR supports factors up to 50,000; choose a smaller factor or another method.")
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
    selection = slice(first, last + 1, decimation)
    if retained_samples(selection, downsampling) < 2:
        raise ValueError("The imported interval must retain at least two time samples.")
    return selection


def source_chunk_samples(selection, downsampling):
    """Upper bound on the samples read per record in a filtered batch."""
    if selection.step == 1 or downsampling == "simple":
        return min(retained_samples(selection, downsampling), max(1, SOURCE_CHUNK_SAMPLES // selection.step))
    halo = 20*selection.step if downsampling == "polyphase" else 0
    return min(selection.stop-selection.start, SOURCE_CHUNK_SAMPLES + halo)


def resample_into(read, output, selection, downsampling="polyphase", *, chunk_samples=SOURCE_CHUNK_SAMPLES, progress=None, canceled=None):
    """Parallelize substantial numeric work across records; keep HDF5 reads serial."""
    check_canceled(canceled)
    workers = min(4, os.cpu_count() or 1, len(output))
    parallel = selection.step > 1 and downsampling == "polyphase" and output.size*selection.step >= 262144 and workers > 1
    with ThreadPoolExecutor(max_workers=workers) if parallel else nullcontext() as pool:
        _resample_into(read, output, selection, downsampling, chunk_samples=chunk_samples, pool=pool, progress=progress, canceled=canceled)


def _resample_into(read, output, selection, downsampling, *, chunk_samples, pool, progress, canceled):
    """Fill a record batch from bounded disk reads, keeping one global sample phase.

    FIR chunks include the entire filter support on both sides. Only the
    selected interval's true endpoints use zero padding, matching resample_poly
    on the complete cropped trace. Averaging keeps complete blocks only.
    """
    factor = selection.step
    count = output.shape[-1]
    check_canceled(canceled)
    if factor == 1 or downsampling == "simple":
        outputs_per_chunk = max(1, chunk_samples // factor)
        for first in range(0, count, outputs_per_chunk):
            check_canceled(canceled)
            last = min(count, first+outputs_per_chunk)
            part = slice(selection.start+first*factor, min(selection.stop, selection.start+last*factor), factor)
            values = read(part)
            check_canceled(canceled)
            output[:, first:last] = values
            del values
            if progress:
                progress(last)
        check_canceled(canceled)
        return
    outputs_per_chunk = max(1, chunk_samples // factor)
    if downsampling == "polyphase":
        half = 10*factor
        coefficients = firwin(2*half+1, 1/factor, window=("kaiser", 5.0))
        for first in range(0, count, outputs_per_chunk):
            check_canceled(canceled)
            last = min(count, first+outputs_per_chunk)
            center = selection.start + first*factor
            lo = max(selection.start, center-half)
            hi = min(selection.stop, selection.start+(last-1)*factor+half+1)
            values = read(slice(lo, hi, 1))
            check_canceled(canceled)
            skip = (center-lo)//factor
            rows = np.array_split(np.arange(len(output)), min(4, len(output))) if pool is not None else [np.arange(len(output))]
            def filter_rows(indices):
                check_canceled(canceled)
                return resample_poly(values[indices[0]:indices[-1]+1], 1, factor, axis=-1, window=coefficients)[:, skip:skip+last-first]
            results = pool.map(filter_rows, rows) if pool is not None else map(filter_rows, rows)
            for indices, filtered in zip(rows, results):
                check_canceled(canceled)
                output[indices[0]:indices[-1]+1, first:last] = filtered
            del values, filtered
            if progress:
                progress(last)
    elif downsampling == "average":
        for first in range(0, count, outputs_per_chunk):
            check_canceled(canceled)
            last = min(count, first+outputs_per_chunk)
            lo, hi = selection.start+first*factor, selection.start+last*factor
            if factor <= chunk_samples:
                values = read(slice(lo, hi, 1))
                check_canceled(canceled)
                output[:, first:last] = values.reshape(len(output), last-first, factor).mean(axis=-1, dtype=np.float64)
                del values
            else:
                # Even a single averaging block may exceed the read budget.
                total = np.zeros(len(output), dtype=np.float64)
                for start in range(lo, hi, chunk_samples):
                    check_canceled(canceled)
                    total += read(slice(start, min(hi, start+chunk_samples), 1)).sum(axis=-1, dtype=np.float64)
                output[:, first] = total/factor
            if progress:
                progress(last)
    else:
        raise ValueError("Unknown downsampling method.")
    check_canceled(canceled)


def selection_metadata(selection, dt, downsampling="polyphase"):
    active = selection.step > 1
    info = {"first original sample": selection.start,
            "last original sample limit (inclusive)": selection.stop - 1,
            "keep every Nth sample": selection.step,
            "original sample interval (s)": dt,
            "effective sample interval (s)": dt * selection.step,
            "effective sampling rate (Hz)": 1 / (dt * selection.step),
            "downsampling method": downsampling if active else "none",
            "anti-alias filtering": active and downsampling != "simple"}
    if active and downsampling == "polyphase":
        info.update({"FIR taps": 20*selection.step+1, "FIR window": "Kaiser, beta=5",
                     "FIR cutoff (Hz)": 1/(2*dt*selection.step), "interval edge padding": "zero"})
    elif active and downsampling == "average":
        info.update({"filter": "boxcar; weak anti-alias suppression", "time coordinates": "block centers",
                     "discarded trailing samples": (selection.stop-selection.start) % selection.step})
    return info


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
