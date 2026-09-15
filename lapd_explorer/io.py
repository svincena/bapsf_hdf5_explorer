"""Read-only acquisition adapters and portable processed-data export."""
from dataclasses import dataclass
import json
import numpy as np
import h5py
from .model import Dataset


@dataclass
class Records:
    channels: dict
    time: np.ndarray
    shots: np.ndarray
    xyz: np.ndarray | None
    source: str
    metadata: dict


def inspect_file(path):
    from bapsflib import lapd
    channels, controls, datasets = [], [], []
    with h5py.File(path, "r") as f:
        if f.attrs.get("lapd_explorer_format", "") == "1":
            return {"portable": True, "channels": [], "controls": [], "datasets": []}
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim and obj.dtype.kind in "fiu":
                datasets.append((name, obj.shape))
        f.visititems(visit)
    error = ""
    try:
        with lapd.File(path, mode="r") as f:
            for device, mapper in f.digitizers.items():
                for config_name, config in mapper.configs.items():
                    for adc in config.get("adc", []):
                        for board, channel_group, info in config.get(adc, []):
                            group = (channel_group,) if np.isscalar(channel_group) else channel_group
                            for channel in group:
                                channels.append(dict(digitizer=device, config_name=config_name,
                                                     adc=adc, board=int(board), channel=int(channel)))
            for name, mapper in f.controls.items():
                if "motion" in str(mapper.contype).lower():
                    controls.extend((name, cfg) for cfg in mapper.configs)
    except Exception as exc:
        error = str(exc)
    return dict(portable=False, channels=channels, controls=controls, datasets=datasets, error=error)


def read_lapd(path, selections, control=None, start=0, stop=None, t0=0.0, position_source="target"):
    from bapsflib import lapd
    arrays, reference, dt, xyz, infos = {}, None, None, None, {}
    with lapd.File(path, mode="r") as f:
        infos["acquisition"] = dict(f.info)
        for name, spec in selections.items():
            opts = {k: v for k, v in spec.items() if k not in ("board", "channel") and v != ""}
            r = f.read_data(spec["board"], spec["channel"], **opts,
                            index=slice(start, stop),
                            add_controls=[tuple(control)] if control else None)
            shot = np.asarray(r["shotnum"])
            signal = np.array(r["signal"], copy=True)
            step = r.dt
            if step is None:
                raise ValueError("Digitizer sample interval is missing. Use the raw HDF5 importer with an explicit interval.")
            step = float(step.to_value("s") if hasattr(step, "to_value") else step)
            if step <= 0 or not np.isfinite(step):
                raise ValueError("Invalid digitizer sample interval.")
            if reference is not None and (not np.array_equal(shot, reference)
                    or signal.shape != next(iter(arrays.values())).shape
                    or not np.isclose(step, dt, rtol=1e-10, atol=0)):
                raise ValueError("Vector channels must have identical shot numbers, sample counts and time steps.")
            reference, dt = shot.copy(), step
            arrays[name] = signal
            position_field = "xyz_target" if position_source == "target" and "xyz_target" in r.dtype.names else "xyz"
            current_xyz = np.asarray(r[position_field], dtype=float)
            if xyz is not None and not np.allclose(current_xyz, xyz, equal_nan=True):
                raise ValueError("Channel motion coordinates do not match.")
            xyz = current_xyz.copy()
            infos[name] = dict(r.info)
            infos[name]["position field used"] = position_field
            infos[name]["measured xyz"] = np.asarray(r["xyz"]).tolist()
            if "xyz_target" in r.dtype.names:
                infos[name]["target xyz"] = np.asarray(r["xyz_target"]).tolist()
    if reference is None or len(reference) == 0:
        raise ValueError("No records in the selected range.")
    if len(np.unique(reference)) != len(reference):
        raise ValueError("Duplicate global shot numbers; select a single acquisition configuration.")
    return Records(arrays, t0 + np.arange(signal.shape[-1])*dt, reference,
                   xyz, str(path), infos)


def map_motion(records, cases=1, repeats=1, repeat_order="case,shot", decimals=4):
    """Group by position, then explicitly map per-position acquisition order.

    Reject incomplete grids and unmatched repeat counts rather than silently
    combining missing positions or treating parameter scans as statistics.
    """
    xyz = records.xyz
    if xyz is None or not np.all(np.isfinite(xyz)):
        raise ValueError("Finite motion coordinates are unavailable. Use manual dimensions.")
    xyz = np.round(xyz, decimals)
    axes = [i for i in range(3) if len(np.unique(xyz[:, i])) > 1]
    if len(axes) > 2:
        raise ValueError("Motion varies in 3 spatial axes. Select a planar subset before importing.")
    axes = list(reversed(axes))  # y,x; z,x; z,y
    dims = ["xyz"[i] for i in axes]
    coords = {d: np.unique(xyz[:, i]) for d, i in zip(dims, axes)}
    shape = tuple(len(coords[d]) for d in dims)
    if cases < 1 or repeats < 1:
        raise ValueError("Cases and stored repeats must be positive.")
    expected = int(np.prod(shape)) * cases * repeats
    if expected != len(records.shots):
        raise ValueError(f"Grid requires {expected} records; found {len(records.shots)}. Set cases/repeats explicitly or use manual mapping.")
    output = {n: np.empty(shape + (cases, repeats, len(records.time)), dtype=a.dtype)
              for n, a in records.channels.items()}
    shotmap = np.empty(shape + (cases, repeats), dtype=records.shots.dtype)
    buckets = {}
    for row in range(len(xyz)):
        key = tuple(int(np.searchsorted(coords[d], xyz[row, i])) for d, i in zip(dims, axes))
        buckets.setdefault(key, []).append(row)
    if len(buckets) != int(np.prod(shape)):
        raise ValueError("Motion points do not form a complete rectangular grid.")
    for key, rows in buckets.items():
        if len(rows) != cases*repeats:
            raise ValueError("Unequal records per position; select a balanced acquisition subset.")
        for n, a in records.channels.items():
            block = a[rows]
            if repeat_order == "case,shot":
                output[n][key] = block.reshape(cases, repeats, -1)
            elif repeat_order == "shot,case":
                output[n][key] = block.reshape(repeats, cases, -1).transpose(1, 0, 2)
            else:
                raise ValueError("Unknown per-position acquisition order.")
        block = records.shots[rows]
        shotmap[key] = (block.reshape(cases, repeats) if repeat_order == "case,shot"
                        else block.reshape(repeats, cases).T)
    dims += ["case", "shot", "time"]
    coords.update(case=np.arange(cases), shot=np.arange(repeats), time=records.time)
    # Singleton case/repeat axes carry no extra statistical information.
    for d in ("case", "shot"):
        if len(coords[d]) == 1:
            axis = dims.index(d)
            output = {n: np.squeeze(a, axis=axis) for n, a in output.items()}
            shotmap = np.squeeze(shotmap, axis=axis)
            dims.remove(d)
            coords.pop(d)
    return Dataset(output, tuple(dims), coords, source=records.source,
                   shot_numbers=shotmap, metadata=records.metadata,
                   history=(f"Motion grid rounded to {decimals} decimals; per-position order {repeat_order}",))


def map_manual(records, axes, spatial_units="cm"):
    """axes = [(name, size, start, stop), ...] in slowest-to-fastest order."""
    dims = tuple(a[0] for a in axes) + ("time",)
    sizes = tuple(a[1] for a in axes)
    if int(np.prod(sizes)) != len(records.shots):
        raise ValueError(f"Dimension product {int(np.prod(sizes))} does not match {len(records.shots)} records.")
    coords = {name: np.linspace(start, stop, size) for name, size, start, stop in axes}
    coords["time"] = records.time
    arrays = {n: a.reshape(sizes + (len(records.time),)) for n, a in records.channels.items()}
    return Dataset(arrays, dims, coords, spatial_units=spatial_units, source=records.source,
                   shot_numbers=records.shots.reshape(sizes), metadata=records.metadata,
                   history=("Manual mapping, C order (last listed axis varies fastest)",))


def read_raw(path, paths, axes, dt, t0=0, spatial_units="cm", units="V"):
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(t0):
        raise ValueError("Sample interval must be positive and start time finite.")
    arrays = {}
    with h5py.File(path, "r") as f:
        for name, p in paths.items():
            a = np.asarray(f[p], dtype=float)
            if a.ndim < 1:
                raise ValueError("A time-series dataset is required.")
            arrays[name] = a.reshape(-1, a.shape[-1])
    shape = next(iter(arrays.values())).shape
    if any(a.shape != shape for a in arrays.values()):
        raise ValueError("Raw channels must have matching record/time shapes.")
    records = Records(arrays, t0+np.arange(shape[-1])*dt, np.arange(shape[0]), None, str(path),
                      {"raw datasets": paths, "shot numbers": "record indices; global shot numbers unavailable"})
    result = map_manual(records, axes, spatial_units)
    result.units = units
    result.shot_numbers = None
    return result


def save_dataset(path, data):
    with h5py.File(path, "w") as f:
        f.attrs.update(lapd_explorer_format="1", dims=json.dumps(data.dims), units=data.units,
                       spatial_units=data.spatial_units, source=data.source,
                       history=json.dumps(data.history), metadata=json.dumps(data.metadata, default=str))
        channels = f.create_group("channels")
        for i, (name, a) in enumerate(data.channels.items()):
            ds = channels.create_dataset(str(i), data=a, compression="gzip")
            ds.attrs["name"] = name
        coords = f.create_group("coordinates")
        for d, c in data.coords.items():
            coords.create_dataset(d, data=c)
        if data.shot_numbers is not None:
            f.create_dataset("shot_numbers", data=data.shot_numbers)


def load_dataset(path):
    with h5py.File(path, "r") as f:
        if f.attrs.get("lapd_explorer_format") != "1":
            raise ValueError("Not a LAPD Explorer export.")
        return Dataset({ds.attrs["name"]: ds[:] for ds in f["channels"].values()},
                       tuple(json.loads(f.attrs["dims"])),
                       {d: c[:] for d, c in f["coordinates"].items()},
                       units=f.attrs["units"], spatial_units=f.attrs["spatial_units"],
                       source=f.attrs["source"], history=tuple(json.loads(f.attrs["history"])),
                       metadata=json.loads(f.attrs["metadata"]),
                       shot_numbers=f["shot_numbers"][()] if "shot_numbers" in f else None)
