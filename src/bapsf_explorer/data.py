"""GUI-independent shot representation and explicit, lossless reshaping."""

from collections import defaultdict
from math import prod
import json

import numpy as np
import xarray as xr


def record_array(signal, time, shot_ids, *, positions=None, name="Signal", attrs=None):
    signal = np.asarray(signal)
    time = np.asarray(time, dtype=float)
    shot_ids = np.asarray(shot_ids)
    if signal.ndim != 2 or signal.shape != (shot_ids.size, time.size):
        raise ValueError("Expected signal shape (record, time), with matching shot IDs and time.")
    if not signal.size or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Data must be nonempty and time must be finite and strictly increasing.")
    coords = {"record": np.arange(len(shot_ids)), "shot_id": ("record", shot_ids), "time": time}
    for axis, values in (positions or {}).items():
        if axis in coords or axis == "shot":
            raise ValueError(f"Reserved coordinate name: {axis}")
        coords[axis] = ("record", np.asarray(values, dtype=float))
    result = xr.DataArray(signal, dims=("record", "time"), coords=coords, name=name, attrs=attrs or {})
    result.time.attrs["units"] = "s"
    for axis in positions or {}:
        result[axis].attrs["units"] = result.attrs.get("position_units", "native")
    return result


def _history(data, event):
    attrs = dict(data.attrs)
    events = json.loads(attrs.get("history", "[]"))
    events.append(event)
    attrs["history"] = json.dumps(events)
    return attrs


def acquisition_reshape(raw, dimensions):
    """Reshape in C acquisition order; the rightmost supplied axis varies fastest.

    dimensions is an ordered mapping, e.g. {"y": 4, "x": 8, "case": 2, "shot": 5}.
    Coordinate indices are deliberately NOT represented as physical positions.
    Original per-record coordinates are retained as multidimensional coordinates.
    """
    if raw.dims != ("record", "time"):
        raise ValueError("Reshaping requires the original (record, time) data.")
    if not dimensions or any(not k.isidentifier() or k in ("time", "record", "shot_id") for k in dimensions):
        raise ValueError("Provide named dimensions, excluding time, record, and shot_id.")
    sizes = tuple(dimensions.values())
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in sizes):
        raise ValueError("Dimension sizes must be positive integers.")
    if prod(sizes) != raw.sizes["record"]:
        raise ValueError(f"Dimension product {prod(sizes)} must equal {raw.sizes['record']} loaded records.")
    dims = tuple(dimensions)
    coords = {dim: np.arange(n) for dim, n in dimensions.items()}
    coords["time"] = raw.time
    for key, coordinate in raw.coords.items():
        if coordinate.dims == ("record",) and key != "record":
            target = f"position_{key}" if key in dimensions else key
            coords[target] = (dims, coordinate.values.reshape(sizes), dict(coordinate.attrs))
    result = xr.DataArray(raw.values.reshape(*sizes, raw.sizes["time"]),
                          dims=(*dims, "time"), coords=coords, name=raw.name,
                          attrs=_history(raw, {"reshape": dimensions, "order": "C"}))
    for dim in dims:
        result[dim].attrs["units"] = "index"
    return result


def motion_reshape(raw, axes=None, *, decimals=4, cases=None, max_bytes=512 * 1024**2):
    """Group records by measured position and optional explicit per-record case labels.

    Repeats at each position/case become `shot`, in acquisition order. Missing
    cells/repeats are NaN and shot_id=-1; no acquisition records are discarded.
    Rounding is an explicit coordinate quantization in native position units.
    """
    if raw.dims != ("record", "time"):
        raise ValueError("Motion grouping requires original (record, time) data.")
    if axes is None:
        if "motion_axes" in raw.attrs:
            axes = json.loads(raw.attrs["motion_axes"])
        else:
            axes = [a for a in ("y", "x", "z") if a in raw.coords
                    and np.isfinite(raw[a]).all() and np.unique(np.round(raw[a].values, decimals)).size > 1]
    axes = list(axes)
    if not axes and cases is None and "motion_axes" in raw.attrs:
        return acquisition_reshape(raw, {"shot": raw.sizes["record"]})
    if not axes or len(set(axes)) != len(axes) or any(a in ("record", "time", "shot", "case", "shot_id") for a in axes):
        raise ValueError("Choose distinct varying spatial coordinates for motion grouping.")
    vectors = []
    for axis in axes:
        if axis not in raw.coords or raw[axis].dims != ("record",):
            raise ValueError(f"No per-record coordinate for {axis}.")
        values = np.round(raw[axis].values, decimals)
        if not np.isfinite(values).all():
            raise ValueError(f"Coordinate {axis} contains missing positions; choose a matched motion configuration.")
        vectors.append(values)
    if cases is not None:
        cases = np.asarray(cases)
        if cases.shape != (raw.sizes["record"],):
            raise ValueError("Supply one case label per loaded record.")
        if cases.dtype.kind in "fc" and not np.isfinite(cases).all():
            raise ValueError("Case labels must be finite.")
        axes.append("case")
        vectors.append(cases)
    unique = [np.unique(v) for v in vectors]
    inverse = [np.searchsorted(u, v) for u, v in zip(unique, vectors)]
    groups = defaultdict(list)
    for row, key in enumerate(zip(*inverse)):
        groups[key].append(row)
    repeats = max(map(len, groups.values()))
    shape = (*[len(u) for u in unique], repeats)
    dtype = np.result_type(raw.dtype, np.float32)
    estimate = prod(shape) * (raw.sizes["time"] * dtype.itemsize + 8)
    if estimate > max_bytes:
        raise ValueError(f"Motion grid needs {estimate / 1024**2:.0f} MiB. Select fewer records/axes or adjust coordinate rounding.")
    values = np.full((*shape, raw.sizes["time"]), np.nan, dtype=dtype)
    ids = np.full(shape, -1, dtype=np.int64)
    for key, rows in groups.items():
        values[key][:len(rows)] = raw.values[rows]
        ids[key][:len(rows)] = raw.shot_id.values[rows]
    coords = dict(zip(axes, unique))
    coords.update(shot=np.arange(repeats), time=raw.time, shot_id=((*axes, "shot"), ids))
    for name, coordinate in raw.coords.items():
        if name in coords or name == "record" or coordinate.dims != ("record",):
            continue
        if not np.issubdtype(coordinate.dtype, np.number):
            continue
        retained = np.full(shape, np.nan)
        for key, rows in groups.items():
            retained[key][:len(rows)] = coordinate.values[rows]
        coords[name] = ((*axes, "shot"), retained, dict(coordinate.attrs))
    result = xr.DataArray(values, dims=(*axes, "shot", "time"), coords=coords, name=raw.name,
                          attrs=_history(raw, {"motion_axes": axes, "round_decimals": decimals,
                                               "repeat_order": "acquisition"}))
    for axis in axes:
        if axis in raw.coords:
            result[axis].attrs = dict(raw[axis].attrs)
    result.attrs["missing_cells"] = int(np.count_nonzero(ids == -1))
    return result


def demo_data():
    """Small reproducible traveling perturbation, two cases, four repeats."""
    rng = np.random.default_rng(71)
    y, x, case, shot = np.meshgrid(np.linspace(-12, 12, 13), np.linspace(-15, 15, 17),
                                  np.arange(2), np.arange(4), indexing="ij")
    time = np.arange(600) * 2e-6
    xf, yf, cf = x.ravel()[:, None], y.ravel()[:, None], case.ravel()[:, None]
    envelope = np.exp(-((xf / 12)**2 + (yf / 10)**2))
    wave = envelope * np.sin(2 * np.pi * (5000 * time - xf / 35) + cf * 0.8)
    signal = (0.3 + 0.03 * yf + wave + rng.normal(0, 0.08, wave.shape)).astype("float32")
    result = record_array(signal, time, np.arange(len(signal)) + 1001,
                          positions={"x": x.ravel(), "y": y.ravel()}, name="Ion saturation • synthetic",
                          attrs={"units": "V", "position_units": "cm", "source": "Synthetic demo"})
    return result.assign_coords(case=("record", case.ravel()))
